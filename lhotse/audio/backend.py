import logging
import os
import re
import sys
import traceback
from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO, IOBase
from pathlib import Path
from subprocess import PIPE, CalledProcessError, run
from typing import Any, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from lhotse.audio.utils import (
    AudioLoadingError,
    VideoInfo,
    verbose_audio_loading_exceptions,
)
from lhotse.augmentation import Resample
from lhotse.utils import Pathlike, Seconds, compute_num_samples, is_torchaudio_available

_FFMPEG_TORCHAUDIO_INFO_ENABLED: bool = is_torchaudio_available()
CURRENT_AUDIO_BACKEND: Optional["AudioBackend"] = None


def available_audio_backends() -> List[str]:
    """
    Return a list of names of available audio backends, including "default".
    """
    return ["default"] + sorted(AudioBackend.KNOWN_BACKENDS.keys())


@contextmanager
def audio_backend(backend: Union["AudioBackend", str]):
    """
    Context manager that sets Lhotse's audio backend to the specified value
    and restores the previous audio backend at the end of its scope.

    Example::

        >>> with audio_backend("LibsndfileBackend"):
        ...     some_audio_loading_fn()
    """
    previous = get_current_audio_backend()
    set_current_audio_backend(backend)
    yield
    set_current_audio_backend(previous)


def get_current_audio_backend() -> "AudioBackend":
    """
    Return the audio backend currently set by the user, or default.
    """
    global CURRENT_AUDIO_BACKEND

    # First check if the user has programmatically overridden the audio backend.
    if CURRENT_AUDIO_BACKEND is not None:
        return CURRENT_AUDIO_BACKEND

    # Then, check if the user has overridden the audio backend via an env var.
    maybe_backend = os.environ.get("LHOTSE_AUDIO_BACKEND")
    if maybe_backend is not None:
        set_current_audio_backend(maybe_backend)
        return CURRENT_AUDIO_BACKEND

    # Lastly, fall back to the default backend.
    set_current_audio_backend("default")
    return CURRENT_AUDIO_BACKEND


def set_current_audio_backend(backend: Union["AudioBackend", str]) -> None:
    """
    Force Lhotse to use a specific audio backend to read every audio file,
    overriding the default behaviour of educated guessing + trial-and-error.

    Example forcing Lhotse to use ``audioread`` library for every audio loading operation::

        >>> set_current_audio_backend(AudioreadBackend())
    """
    global CURRENT_AUDIO_BACKEND
    if backend == "default":
        backend = get_default_audio_backend()
    elif isinstance(backend, str):
        backend = AudioBackend.new(backend)
    else:
        assert isinstance(
            backend, AudioBackend
        ), f"Expected str or AudioBackend, got: {backend}"
    CURRENT_AUDIO_BACKEND = backend


@lru_cache(maxsize=1)
def get_default_audio_backend() -> "AudioBackend":
    """
    Return a backend that can be used to read all audio formats supported by Lhotse.

    It first looks for special cases that need very specific handling
    (such as: opus, sphere/shorten, in-memory buffers)
    and tries to match them against relevant audio backends.

    Then, it tries to use several audio loading libraries (torchaudio, soundfile, audioread).
    In case the first fails, it tries the next one, and so on.
    """
    return CompositeAudioBackend(
        [
            # First handle special cases: OPUS and SPHERE (SPHERE may be encoded with shorten,
            #   which can only be decoded by binaries "shorten" and "sph2pipe").
            FfmpegSubprocessOpusBackend(),
            Sph2pipeSubprocessBackend(),
            # New FFMPEG backend available only in torchaudio 2.0.x+
            TorchaudioFFMPEGBackend(),
            # Prefer libsndfile for in-memory buffers only
            LibsndfileBackend(),
            # Torchaudio should be able to deal with most audio types...
            TorchaudioDefaultBackend(),
            # ... if not, try audioread...
            AudioreadBackend(),
            # ... oops.
        ]
    )


def set_ffmpeg_torchaudio_info_enabled(enabled: bool) -> None:
    """
    Override Lhotse's global setting for whether to use ffmpeg-torchaudio to
    compute the duration of audio files. If disabled, we fall back to using a different
    backend such as sox_io or soundfile.

    .. note:: See this issue for more details: https://github.com/lhotse-speech/lhotse/issues/1026

    Example::

        >>> import lhotse
        >>> lhotse.set_ffmpeg_torchaudio_info_enabled(False)  # don't use ffmpeg-torchaudio

    :param enabled: Whether to use torchaudio to compute audio file duration.
    """
    global _FFMPEG_TORCHAUDIO_INFO_ENABLED
    if enabled != _FFMPEG_TORCHAUDIO_INFO_ENABLED:
        logging.info(
            "The user overrided the global setting for whether to use ffmpeg-torchaudio "
            "to compute the duration of audio files. "
            f"Old setting: {_FFMPEG_TORCHAUDIO_INFO_ENABLED}. "
            f"New setting: {enabled}."
        )
    _FFMPEG_TORCHAUDIO_INFO_ENABLED = enabled


def get_ffmpeg_torchaudio_info_enabled() -> bool:
    """
    Return FFMPEG_TORCHAUDIO_INFO_ENABLED, which is Lhotse's global setting for whether to
    use ffmpeg-torchaudio to compute the duration of audio files.

    Example::

        >>> import lhotse
        >>> lhotse.get_ffmpeg_torchaudio_info_enabled()
    """
    return _FFMPEG_TORCHAUDIO_INFO_ENABLED


FileObject = Any  # Alias for file-like objects


class AudioBackend:
    """
    Internal Lhotse abstraction. An AudioBackend defines three methods:
    one for reading audio, and two filters that help determine if it should be used.

    ``handles_special_case`` means this backend should be exclusively
    used for a given type of input path/file.

    ``is_applicable`` means this backend most likely can be used for a given type of input path/file,
    but it may also fail. Its purpose is more to filter out formats that definitely are not supported.
    """

    KNOWN_BACKENDS = {}

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in AudioBackend.KNOWN_BACKENDS:
            AudioBackend.KNOWN_BACKENDS[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def new(cls, name: str) -> "AudioBackend":
        if name not in cls.KNOWN_BACKENDS:
            raise RuntimeError(f"Unknown audio backend name: {name}")
        return cls.KNOWN_BACKENDS[name]()

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        raise NotImplementedError()

    def info(self, path_or_fd: Union[Pathlike, FileObject]):
        raise NotImplementedError()

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return False

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return True


class FfmpegSubprocessOpusBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ):
        assert isinstance(
            path_or_fd, (str, Path)
        ), f"Cannot use an ffmpeg subprocess to read from path of type: '{type(path_or_fd)}'"
        return read_opus_ffmpeg(
            path=path_or_fd,
            offset=offset,
            duration=duration,
            force_opus_sampling_rate=force_opus_sampling_rate,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith(
            ".opus"
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return self.handles_special_case(path_or_fd)


class Sph2pipeSubprocessBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        assert isinstance(
            path_or_fd, (str, Path)
        ), f"Cannot use an sph2pipe subprocess to read from path of type: '{type(path_or_fd)}'"
        return read_sph(
            sph_path=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith(
            ".sph"
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return self.handles_special_case(path_or_fd)


class FfmpegTorchaudioStreamerBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return torchaudio_ffmpeg_load(
            path_or_fileobj=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return (
            is_torchaudio_available()
            and torchaudio_supports_ffmpeg()
            and isinstance(path_or_fd, BytesIO)
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        # Technically it's applicable with regular files as well, but for now
        # we're not enabling that feature.
        return (
            is_torchaudio_available()
            and torchaudio_supports_ffmpeg()
            and isinstance(path_or_fd, BytesIO)
        )


class TorchaudioDefaultBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return torchaudio_load(
            path_or_fd=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return is_torchaudio_available()


class TorchaudioFFMPEGBackend(AudioBackend):
    """
    A new FFMPEG backend available in torchaudio 2.0.
    It should be free from many issues of soundfile and sox_io backends.
    """

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return torchaudio_2_ffmpeg_load(
            path_or_fd=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        """
        FFMPEG backend requires at least Torchaudio 2.0.
        For version == 2.0.x, we also need env var TORCHAUDIO_USE_BACKEND_DISPATCHER=1
        For version >= 2.1.x, this will already be the default.
        """
        return is_torchaudio_available() and torchaudio_2_0_ffmpeg_enabled()


class LibsndfileBackend(AudioBackend):
    """
    A backend that uses PySoundFile.

    .. note:: PySoundFile has issues on MacOS because of the way its CFFI bindings are implemented.
        For now, we disable it on this platform.
        See: https://github.com/bastibe/python-soundfile/issues/331
    """

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return soundfile_load(
            path_or_fd=path_or_fd,
            offset=offset,
            duration=duration,
        )

    def handles_special_case(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return (
            not (sys.platform == "darwin")
            and isinstance(path_or_fd, BytesIO)
            and not torchaudio_2_0_ffmpeg_enabled()  # FFMPEG is preferable to this hack.
        )

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return True


class AudioreadBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        return audioread_load(
            path_or_file=path_or_fd,
            offset=offset,
            duration=duration,
        )


class CompositeAudioBackend(AudioBackend):
    """
    Combines multiple audio backends.
    It will try each out sequentially, and back off to the next one in the list if the current one fails.
    It uses the special filter methods to prioritize special case backends,
    and skip backends that are not applicable.
    """

    def __init__(self, backends: List[AudioBackend]):
        self.backends = backends

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        candidates = []
        for b in self.backends:
            if b.handles_special_case(path_or_fd):
                candidates.append(b)

        assert len(candidates) < 2, (
            f"CompositeAudioBackend has more than one sub-backend that "
            f"handles a given special case for input '{path_or_fd}'"
        )

        if len(candidates) == 1:
            try:
                return candidates[0].read_audio(
                    path_or_fd=path_or_fd,
                    offset=offset,
                    duration=duration,
                    force_opus_sampling_rate=force_opus_sampling_rate,
                )
            except Exception as e:
                raise AudioLoadingError(
                    f"Reading audio from '{path_or_fd}' failed. Details: {type(e)}: {str(e)}"
                )

        exceptions = []
        for b in self.backends:
            if b.is_applicable(path_or_fd):
                try:
                    return b.read_audio(
                        path_or_fd=path_or_fd,
                        offset=offset,
                        duration=duration,
                        force_opus_sampling_rate=force_opus_sampling_rate,
                    )
                except Exception as e:
                    msg = f"Exception #{len(exceptions)} ({type(b)}): "
                    if verbose_audio_loading_exceptions():
                        exceptions.append(f"{msg}{traceback.format_exc()}")
                    else:
                        exceptions.append(f"{msg}{type(e)}: {str(e)}")

        if not exceptions:
            raise AudioLoadingError(
                f"No applicable backend found for input: '{path_or_fd}'"
            )
        else:
            NL = "\n"
            maybe_info = (
                ""
                if verbose_audio_loading_exceptions()
                else "\nSet LHOTSE_AUDIO_LOADING_EXCEPTION_VERBOSE=1 environment variable for full stack traces."
            )
            raise AudioLoadingError(
                f"Reading audio from '{path_or_fd}' failed. Details:{NL}{NL.join(exceptions)}{maybe_info}"
            )


class LibsndfileCompatibleAudioInfo(NamedTuple):
    channels: int
    frames: int
    samplerate: int
    duration: float
    video: Optional[VideoInfo] = None


@lru_cache(maxsize=1)
def torchaudio_supports_ffmpeg() -> bool:
    """
    Returns ``True`` when torchaudio version is at least 0.12.0, which
    has support for FFMPEG streamer API.
    """
    # If user has disabled ffmpeg-torchaudio, we don't need to check the version.
    if not _FFMPEG_TORCHAUDIO_INFO_ENABLED:
        return False

    import torchaudio
    from packaging import version

    return version.parse(torchaudio.__version__) >= version.parse("0.12.0")


@lru_cache(maxsize=1)
def torchaudio_2_0_ffmpeg_enabled() -> bool:
    """
    Returns ``True`` when torchaudio.load supports "ffmpeg" backend.
    This requires either version 2.1.x+ or 2.0.x with env var TORCHAUDIO_USE_BACKEND_DISPATCHER=1.
    """
    if not is_torchaudio_available():
        return False

    import torchaudio
    from packaging import version

    ver = version.parse(torchaudio.__version__)
    if ver >= version.parse("2.1.0"):
        # Enabled by default, disable with TORCHAUDIO_USE_BACKEND_DISPATCHER=0
        return os.environ.get("TORCHAUDIO_USE_BACKEND_DISPATCHER", "1") == "1"
    if ver >= version.parse("2.0"):
        # Disabled by default, enable with TORCHAUDIO_USE_BACKEND_DISPATCHER=1
        return os.environ.get("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0") == "1"
    return False


@lru_cache(maxsize=1)
def torchaudio_soundfile_supports_format() -> bool:
    """
    Returns ``True`` when torchaudio version is at least 0.9.0, which
    has support for ``format`` keyword arg in ``torchaudio.save()``.
    """
    import torchaudio
    from packaging import version

    return version.parse(torchaudio.__version__) >= version.parse("0.9.0")


def torchaudio_info(
    path_or_fileobj: Union[Path, str, BytesIO]
) -> LibsndfileCompatibleAudioInfo:
    """
    Return an audio info data structure that's a compatible subset of ``pysoundfile.info()``
    that we need to create a ``Recording`` manifest.
    """
    import torchaudio

    if torchaudio_2_0_ffmpeg_enabled():
        # Torchaudio 2.0 with official "ffmpeg" backend should solve all the special cases below.
        info = torchaudio.info(path_or_fileobj, backend="ffmpeg")
        return LibsndfileCompatibleAudioInfo(
            channels=info.num_channels,
            frames=info.num_frames,
            samplerate=int(info.sample_rate),
            duration=info.num_frames / info.sample_rate,
        )

    is_mpeg = isinstance(path_or_fileobj, (str, Path)) and any(
        str(path_or_fileobj).endswith(ext) for ext in (".mp3", ".m4a")
    )
    is_fileobj = isinstance(path_or_fileobj, BytesIO)
    if (is_mpeg or is_fileobj) and torchaudio_supports_ffmpeg():
        # Torchaudio 0.12 has a new StreamReader API that uses ffmpeg.
        #
        # They dropped support for using sox bindings in torchaudio.info
        # for MP3 files and implicitly delegate the call to ffmpeg.
        # Unfortunately, they always return num_frames/num_samples = 0,
        # as explained here: https://github.com/pytorch/audio/issues/2524
        # We have to work around by streaming the MP3 and counting the number
        # of samples.
        #
        # Unfortunately torchaudio also has issues with reading from file objects
        # sometimes, which apparently we can work around by using StreamReader API.
        # See:
        # - https://github.com/pytorch/audio/issues/2524#issuecomment-1223901818
        # - https://github.com/pytorch/audio/issues/2662
        from torchaudio.io import StreamReader

        streamer = StreamReader(
            src=str(path_or_fileobj) if is_mpeg else path_or_fileobj
        )
        assert streamer.num_src_streams == 1, (
            "Lhotse doesn't support files with more than one source stream yet "
            "(not to be confused with multi-channel)."
        )
        info = streamer.get_src_stream_info(streamer.default_audio_stream)
        streamer.add_basic_audio_stream(
            frames_per_chunk=int(info.sample_rate),
        )
        tot_samples = 0
        for (chunk,) in streamer.stream():
            tot_samples += chunk.shape[0]
        return LibsndfileCompatibleAudioInfo(
            channels=info.num_channels,
            frames=tot_samples,
            samplerate=int(info.sample_rate),
            duration=tot_samples / info.sample_rate,
        )

    info = torchaudio.info(path_or_fileobj)
    return LibsndfileCompatibleAudioInfo(
        channels=info.num_channels,
        frames=info.num_frames,
        samplerate=int(info.sample_rate),
        duration=info.num_frames / info.sample_rate,
    )


def torchaudio_ffmpeg_streamer_info(
    path_or_fileobj: Union[Path, str, BytesIO]
) -> LibsndfileCompatibleAudioInfo:
    from torchaudio.io import StreamReader

    is_fileobj = not isinstance(path_or_fileobj, Path)
    is_mpeg = not is_fileobj and any(
        str(path_or_fileobj).endswith(ext) for ext in (".mp3", ".mp4", ".m4a")
    )
    if not is_fileobj:
        path_or_fileobj = str(path_or_fileobj)
    stream = StreamReader(path_or_fileobj)

    # Collect the information about available video and audio streams.
    num_streams = stream.num_src_streams
    audio_streams = {}
    video_streams = {}
    for stream_idx in range(num_streams):
        info = stream.get_src_stream_info(stream_idx)
        if info.media_type == "video":
            video_streams[stream_idx] = info
        elif info.media_type == "audio":
            audio_streams[stream_idx] = info
        else:
            raise RuntimeError(f"Unexpected media_type: {info}")

    assert (
        len(video_streams) < 2
    ), f"Lhotse currently does not support more than one video stream in a file (found {len(video_streams)})."
    assert len(audio_streams) < 2, (
        f"Lhotse currently does not support files with more than a single FFMPEG "
        f"audio stream yet (found {len(audio_streams)}). "
        f"Note that this is not the same as multi-channel which is generally supported."
    )

    meta = {}

    if video_streams:
        ((video_stream_idx, video_stream),) = list(video_streams.items())
        tot_frames = video_stream.num_frames

        if tot_frames == 0:  # num frames not available in header/metadata
            stream.add_basic_video_stream(
                round(video_stream.frame_rate), stream_index=video_stream_idx
            )
            for (chunk,) in stream.stream():
                tot_frames += chunk.shape[0]
            stream.remove_stream(0)

        meta["video"] = VideoInfo(
            fps=video_stream.frame_rate,
            height=video_stream.height,
            width=video_stream.width,
            num_frames=tot_frames,
        )

    if audio_streams:
        ((audio_stream_idx, audio_stream),) = list(audio_streams.items())
        stream.add_basic_audio_stream(
            frames_per_chunk=int(audio_stream.sample_rate),
            stream_index=audio_stream_idx,
        )

        def _try_read_num_samples():
            if is_mpeg or is_fileobj:
                # These cases often have insufficient or corrupted metadata, so we might need to scan
                # the full audio stream to learn the actual number of frames. If video is available,
                # we can quickly verify before performing the costly reading.
                video_info = meta.get("video", None)
                if video_info is not None:
                    audio_duration = audio_stream.num_frames / audio_stream.sample_rate
                    # for now 1ms tolerance
                    if abs(audio_duration - video_info.duration) < 1e-3:
                        return audio_stream.num_frames
                return 0
            else:
                return audio_stream.num_frames

        tot_samples = _try_read_num_samples()
        if tot_samples == 0:
            # There was a mismatch between video and audio duration in metadata,
            # we'll have to read the file to figure it out.
            for (chunk,) in stream.stream():
                tot_samples += chunk.shape[0]

        meta.update(
            channels=audio_stream.num_channels,
            frames=tot_samples,
            samplerate=int(audio_stream.sample_rate),
            duration=tot_samples / audio_stream.sample_rate,
        )

    return LibsndfileCompatibleAudioInfo(**meta)


def torchaudio_load(
    path_or_fd: Pathlike, offset: Seconds = 0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    import torchaudio

    # Need to grab the "info" about sampling rate before reading to compute
    # the number of samples provided in offset / num_frames.
    frame_offset = 0
    num_frames = -1
    if offset > 0 or duration is not None:
        audio_info = torchaudio_info(path_or_fd)
        if offset > 0:
            frame_offset = compute_num_samples(offset, audio_info.samplerate)
        if duration is not None:
            num_frames = compute_num_samples(duration, audio_info.samplerate)
    if isinstance(path_or_fd, IOBase):
        # Set seek pointer to the beginning of the file as torchaudio.info
        # might have left it at the end of the header
        path_or_fd.seek(0)
    audio, sampling_rate = torchaudio.load(
        path_or_fd,
        frame_offset=frame_offset,
        num_frames=num_frames,
    )
    return audio.numpy(), int(sampling_rate)


def torchaudio_2_ffmpeg_load(
    path_or_fd: Pathlike, offset: Seconds = 0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    import torchaudio

    # Need to grab the "info" about sampling rate before reading to compute
    # the number of samples provided in offset / num_frames.
    frame_offset = 0
    num_frames = -1
    if offset > 0 or duration is not None:
        audio_info = torchaudio.info(path_or_fd, backend="ffmpeg")
        if offset > 0:
            frame_offset = compute_num_samples(offset, audio_info.sample_rate)
        if duration is not None:
            num_frames = compute_num_samples(duration, audio_info.sample_rate)
    if isinstance(path_or_fd, IOBase):
        # Set seek pointer to the beginning of the file as torchaudio.info
        # might have left it at the end of the header
        path_or_fd.seek(0)
    audio, sampling_rate = torchaudio.load(
        path_or_fd,
        frame_offset=frame_offset,
        num_frames=num_frames,
        backend="ffmpeg",
    )
    return audio.numpy(), int(sampling_rate)


def torchaudio_ffmpeg_load(
    path_or_fileobj: Union[Path, str, BytesIO],
    offset: Seconds = 0,
    duration: Optional[Seconds] = None,
) -> Tuple[np.ndarray, int]:
    import torchaudio

    if not torchaudio_supports_ffmpeg():
        raise RuntimeError(
            "Using FFMPEG streamer backend for reading is supported only "
            "with PyTorch 1.12+ and torchaudio 0.12+"
        )

    if isinstance(path_or_fileobj, Path):
        path_or_fileobj = str(path_or_fileobj)

    streamer = torchaudio.io.StreamReader(src=path_or_fileobj)
    assert streamer.num_src_streams == 1, (
        "Lhotse doesn't support files with more than one source stream yet "
        "(not to be confused with multi-channel)."
    )
    info = streamer.get_src_stream_info(streamer.default_audio_stream)
    sampling_rate = int(info.sample_rate)

    if duration is not None:
        # Try to read whole audio in a single chunk.
        streamer.add_basic_audio_stream(
            frames_per_chunk=compute_num_samples(duration, sampling_rate)
        )
        streamer.seek(offset)
        (audio,) = next(streamer.stream())
        audio = audio.transpose(0, 1)
    else:
        # Read in 1 second chunks and concatenate (we don't know how much audio is incoming)
        streamer.add_basic_audio_stream(frames_per_chunk=sampling_rate)
        streamer.seek(offset)
        audio = torch.cat([t.transpose(0, 1) for t, in streamer.stream()], dim=0)

    # Return shape (num_channels, num_samples)
    return audio.numpy(), sampling_rate


def soundfile_load(
    path_or_fd: Pathlike, offset: Seconds = 0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    with sf.SoundFile(path_or_fd) as sf_desc:
        sampling_rate = sf_desc.samplerate
        if offset > 0:
            # Seek to the start of the target read
            sf_desc.seek(compute_num_samples(offset, sampling_rate))
        if duration is not None:
            frame_duration = compute_num_samples(duration, sampling_rate)
        else:
            frame_duration = -1
        # Load the target number of frames, and transpose to match librosa form
        return (
            sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=False).T,
            int(sampling_rate),
        )


def audioread_info(path: Pathlike) -> LibsndfileCompatibleAudioInfo:
    """
    Return an audio info data structure that's a compatible subset of ``pysoundfile.info()``
    that we need to create a ``Recording`` manifest.
    """
    import audioread

    # We just read the file and compute the number of samples
    # -- no other method seems fully reliable...
    with audioread.audio_open(
        str(path), backends=_available_audioread_backends()
    ) as input_file:
        shape = audioread_load(input_file)[0].shape
        if len(shape) == 1:
            num_samples = shape[0]
        else:
            num_samples = shape[1]
        return LibsndfileCompatibleAudioInfo(
            channels=input_file.channels,
            frames=num_samples,
            samplerate=int(input_file.samplerate),
            duration=num_samples / input_file.samplerate,
        )


@lru_cache(maxsize=1)
def _available_audioread_backends():
    """
    Reduces the overhead of ``audioread.audio_open()`` when called repeatedly
    by caching the results of scanning for FFMPEG etc.
    """
    import audioread

    backends = audioread.available_backends()
    logging.info(f"Using audioread. Available backends: {backends}")
    return backends


def audioread_load(
    path_or_file: Union[Pathlike, FileObject],
    offset: Seconds = 0.0,
    duration: Seconds = None,
    dtype=np.float32,
):
    """Load an audio buffer using audioread.
    This loads one block at a time, and then concatenates the results.

    This function is based on librosa:
    https://github.com/librosa/librosa/blob/main/librosa/core/audio.py#L180
    """
    import audioread

    @contextmanager
    def file_handle():
        if isinstance(path_or_file, (str, Path)):
            yield audioread.audio_open(
                path_or_file, backends=_available_audioread_backends()
            )
        else:
            yield path_or_file

    y = []
    with file_handle() as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = _buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, int(sr_native)


def _buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    This function is based on librosa:
    https://github.com/librosa/librosa/blob/main/librosa/util/utils.py#L1312

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``
    dtype : numeric type
        The target output type (default: 32-bit float)
    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def opus_info(
    path: Pathlike, force_opus_sampling_rate: Optional[int] = None
) -> LibsndfileCompatibleAudioInfo:
    samples, sampling_rate = read_opus(
        path, force_opus_sampling_rate=force_opus_sampling_rate
    )
    return LibsndfileCompatibleAudioInfo(
        channels=samples.shape[0],
        frames=samples.shape[1],
        samplerate=int(sampling_rate),
        duration=samples.shape[1] / sampling_rate,
    )


def read_opus(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files either using torchaudio or ffmpeg.
    Torchaudio is faster, but if unavailable for some reason,
    we fallback to a slower ffmpeg-based implementation.

    :return: a tuple of audio samples and the sampling rate.
    """
    # TODO: Revisit using torchaudio backend for OPUS
    #       once it's more thoroughly benchmarked against ffmpeg
    #       and has a competitive I/O speed.
    #       See: https://github.com/pytorch/audio/issues/1994
    # try:
    #     return read_opus_torchaudio(
    #         path=path,
    #         offset=offset,
    #         duration=duration,
    #         force_opus_sampling_rate=force_opus_sampling_rate,
    #     )
    # except:
    return read_opus_ffmpeg(
        path=path,
        offset=offset,
        duration=duration,
        force_opus_sampling_rate=force_opus_sampling_rate,
    )


def read_opus_torchaudio(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files using torchaudio.
    This is just running ``tochaudio.load()``, but we take care of extra resampling if needed.

    :return: a tuple of audio samples and the sampling rate.
    """
    audio, sampling_rate = torchaudio_load(
        path_or_fd=path, offset=offset, duration=duration
    )

    if force_opus_sampling_rate is None or force_opus_sampling_rate == sampling_rate:
        return audio, sampling_rate

    resampler = Resample(
        source_sampling_rate=sampling_rate,
        target_sampling_rate=force_opus_sampling_rate,
    )
    resampled_audio = resampler(audio)
    return resampled_audio, force_opus_sampling_rate


def read_opus_ffmpeg(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files using ffmpeg in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.
    Optionally, we can force ffmpeg to resample to the true sampling rate (if we know it up-front).

    :return: a tuple of audio samples and the sampling rate.
    """
    # Construct the ffmpeg command depending on the arguments passed.
    cmd = "ffmpeg -threads 1"
    sampling_rate = 48000
    # Note: we have to add offset and duration options (-ss and -t) BEFORE specifying the input
    #       (-i), otherwise ffmpeg will decode everything and trim afterwards...
    if offset > 0:
        cmd += f" -ss {offset}"
    if duration is not None:
        cmd += f" -t {duration}"
    # Add the input specifier after offset and duration.
    cmd += f" -i '{path}'"
    # Optionally resample the output.
    if force_opus_sampling_rate is not None:
        sampling_rate = force_opus_sampling_rate
    cmd += f" -ar {sampling_rate}"
    # Read audio samples directly as float32.
    cmd += " -f f32le -threads 1 pipe:1"
    # Actual audio reading.
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    raw_audio = proc.stdout
    audio = np.frombuffer(raw_audio, dtype=np.float32)
    # Determine if the recording is mono or stereo and decode accordingly.
    try:
        channel_string = parse_channel_from_ffmpeg_output(proc.stderr)
        if channel_string == "stereo":
            new_audio = np.empty((2, audio.shape[0] // 2), dtype=np.float32)
            new_audio[0, :] = audio[::2]
            new_audio[1, :] = audio[1::2]
            audio = new_audio
        elif channel_string == "mono":
            audio = audio.reshape(1, -1)
        else:
            raise NotImplementedError(
                f"Unknown channel description from ffmpeg: {channel_string}"
            )
    except ValueError as e:
        raise AudioLoadingError(
            f"{e}\nThe ffmpeg command for which the program failed is: '{cmd}', error code: {proc.returncode}"
        )
    return audio, sampling_rate


def parse_channel_from_ffmpeg_output(ffmpeg_stderr: bytes) -> str:
    # ffmpeg will output line such as the following, amongst others:
    # "Stream #0:0: Audio: pcm_f32le, 16000 Hz, mono, flt, 512 kb/s"
    # but sometimes it can be "Stream #0:0(eng):", which we handle with regexp
    pattern = re.compile(r"^\s*Stream #0:0.*: Audio: pcm_f32le.+(mono|stereo).+\s*$")
    for line in ffmpeg_stderr.splitlines():
        try:
            line = line.decode()
        except UnicodeDecodeError:
            # Why can we get UnicodeDecoderError from ffmpeg output?
            # Because some files may contain the metadata, including a short description of the recording,
            # which may be encoded in arbitrarily encoding different than ASCII/UTF-8, such as latin-1,
            # and Python will not automatically recognize that.
            # We simply ignore these lines as they won't have any relevant information for us.
            continue
        match = pattern.match(line)
        if match is not None:
            return match.group(1)
    raise ValueError(
        f"Could not determine the number of channels for OPUS file from the following ffmpeg output "
        f"(shown as bytestring due to avoid possible encoding issues):\n{str(ffmpeg_stderr)}"
    )


def soundfile_info(path: Pathlike) -> LibsndfileCompatibleAudioInfo:
    import soundfile as sf

    info_ = sf.info(str(path))
    return LibsndfileCompatibleAudioInfo(
        channels=info_.channels,
        frames=info_.frames,
        samplerate=info_.samplerate,
        duration=info_.duration,
    )


def sph_info(path: Pathlike) -> LibsndfileCompatibleAudioInfo:
    samples, sampling_rate = read_sph(path)
    return LibsndfileCompatibleAudioInfo(
        channels=samples.shape[0],
        frames=samples.shape[1],
        samplerate=int(sampling_rate),
        duration=samples.shape[1] / sampling_rate,
    )


def read_sph(
    sph_path: Pathlike, offset: Seconds = 0.0, duration: Optional[Seconds] = None
) -> Tuple[np.ndarray, int]:
    """
    Reads SPH files using sph2pipe in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.

    :return: a tuple of audio samples and the sampling rate.
    """

    sph_path = Path(sph_path)

    # Construct the sph2pipe command depending on the arguments passed.
    cmd = f"sph2pipe -f wav -p -t {offset}:"

    if duration is not None:
        cmd += f"{round(offset + duration, 5)}"
    # Add the input specifier after offset and duration.
    cmd += f" {sph_path}"

    # Actual audio reading.
    try:
        proc = BytesIO(
            run(cmd, shell=True, check=True, stdout=PIPE, stderr=PIPE).stdout
        )
    except CalledProcessError as e:
        if e.returncode == 127:
            raise ValueError(
                "It seems that 'sph2pipe' binary is not installed; "
                "did you run 'lhotse install-sph2pipe'?"
            )
        else:
            raise

    import soundfile as sf

    with sf.SoundFile(proc) as sf_desc:
        audio, sampling_rate = sf_desc.read(dtype=np.float32), sf_desc.samplerate
        audio = audio.reshape(1, -1) if sf_desc.channels == 1 else audio.T

    return audio, sampling_rate


def save_flac_file(
    dest: Union[str, Path, BytesIO],
    src: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    *args,
    **kwargs,
):
    if is_torchaudio_available():
        torchaudio_save_flac_safe(
            dest=dest, src=src, sample_rate=sample_rate, *args, **kwargs
        )
    else:
        import soundfile as sf

        kwargs.pop("bits_per_sample", None)  # ignore this arg when not using torchaudio
        if torch.is_tensor(src):
            src = src.numpy()
        src = src.squeeze(0)
        sf.write(file=dest, data=src, samplerate=sample_rate, format="FLAC")


def torchaudio_save_flac_safe(
    dest: Union[str, Path, BytesIO],
    src: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    *args,
    **kwargs,
):
    import torchaudio

    src = torch.as_tensor(src)
    saving_flac = kwargs.get("format") == "flac" or (
        not isinstance(dest, BytesIO) and str(dest).endswith(".flac")
    )
    if torchaudio_soundfile_supports_format() and saving_flac:
        # Prefer saving with soundfile backend whenever possible to avoid issue:
        # https://github.com/pytorch/audio/issues/2662
        # Saving with sox_io backend to FLAC may corrupt the file.
        torchaudio.backend.soundfile_backend.save(
            dest,
            src,
            sample_rate=sample_rate,
            format=kwargs.pop("format", "flac"),
            bits_per_sample=kwargs.pop("bits_per_sample", 16),
            *args,
            **kwargs,
        )
    else:
        torchaudio.backend.sox_io_backend.save(
            dest, src, sample_rate=sample_rate, *args, **kwargs
        )


def read_audio(
    path_or_fd: Union[Pathlike, FileObject],
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    return get_current_audio_backend().read_audio(
        path_or_fd=path_or_fd,
        offset=offset,
        duration=duration,
        force_opus_sampling_rate=force_opus_sampling_rate,
    )


def info(
    path: Union[Pathlike, BytesIO],
    force_opus_sampling_rate: Optional[int] = None,
    force_read_audio: bool = False,
) -> LibsndfileCompatibleAudioInfo:

    is_path = isinstance(path, (Path, str))

    if is_path and Path(path).suffix.lower() == ".sph":
        # We handle SPHERE as another special case because some old codecs (i.e. "shorten" codec)
        # can't be handled by neither pysoundfile nor pyaudioread.
        return sph_info(path)

    if is_path and Path(path).suffix.lower() == ".opus":
        # We handle OPUS as a special case because we might need to force a certain sampling rate.
        return opus_info(path, force_opus_sampling_rate=force_opus_sampling_rate)

    if force_read_audio:
        # This is a reliable fallback for situations when the user knows that audio files do not
        # have duration metadata in their headers.
        # We will use "audioread" backend that spawns an ffmpeg process, reads the audio,
        # and computes the duration.
        assert (
            is_path
        ), f"info(obj, force_read_audio=True) is not supported for object of type: {type(path)}"
        return audioread_info(path)

    try:
        if torchaudio_2_0_ffmpeg_enabled():
            return torchaudio_ffmpeg_streamer_info(path)
        else:  # hacky but easy way to proceed...
            raise Exception("Skipping - torchaudio ffmpeg streamer unavailable")
    except:
        try:
            return torchaudio_info(path)
        except:
            try:
                return soundfile_info(path)
            except:
                return audioread_info(path)
    # If all fail, then Python 3 will display all exception messages.
