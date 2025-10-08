import atexit
import contextlib
import ctypes
import ctypes.util
import os
from typing import Optional, Tuple

import numpy as np

__all__ = ["libsox", "libsox_rate", "libsox_available"]

libsox = None
sox_free = None
LIBSOX_INITIALIZED = False
SOX_SUCCESS = 0
SOX_ENCODING_FLOAT = 3


class sox_signalinfo_t(ctypes.Structure):
    _fields_ = [
        ("rate", ctypes.c_double),
        ("channels", ctypes.c_uint),
        ("precision", ctypes.c_uint),
        ("length", ctypes.c_uint64),
        ("mult", ctypes.POINTER(ctypes.c_double)),
    ]


class sox_encodinginfo_t(ctypes.Structure):
    _fields_ = [
        ("encoding", ctypes.c_uint),
        ("bits_per_sample", ctypes.c_uint),
        ("compression", ctypes.c_double),
        ("reverse_bytes", ctypes.c_uint),
        ("reverse_nibbles", ctypes.c_uint),
        ("reverse_bits", ctypes.c_uint),
        ("opposite_endian", ctypes.c_uint),
    ]


class sox_format_t(ctypes.Structure):
    _fields_ = [
        ("filename", ctypes.c_char_p),
        ("signal", sox_signalinfo_t),
        ("encoding", sox_encodinginfo_t),
        ("filetype", ctypes.c_char_p),
    ]


class sox_effect_handler_t(ctypes.Structure):
    pass


class sox_effects_chain_t(ctypes.Structure):
    pass


class sox_effect_t(ctypes.Structure):
    pass


def libsox_available() -> bool:
    return ctypes.util.find_library("sox") is not None


def libsox_import() -> None:
    global libsox
    global sox_free
    global LIBSOX_INITIALIZED
    if LIBSOX_INITIALIZED:
        return

    if not libsox_available():
        raise RuntimeError("libsox not available but import requested")

    libsox_ = ctypes.CDLL(ctypes.util.find_library("sox"))
    libc_ = ctypes.CDLL(ctypes.util.find_library("c"))

    sox_free_ = libc_.free
    sox_free_.argtypes = [ctypes.c_void_p]
    sox_free_.restype = None

    libsox_.sox_init.restype = ctypes.c_int
    libsox_.sox_quit.restype = ctypes.c_int

    libsox_.sox_open_mem_read.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(sox_signalinfo_t),
        ctypes.POINTER(sox_encodinginfo_t),
        ctypes.c_char_p,
    ]
    libsox_.sox_open_mem_read.restype = ctypes.POINTER(sox_format_t)

    libsox_.sox_open_mem_write.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(sox_signalinfo_t),
        ctypes.POINTER(sox_encodinginfo_t),
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    libsox_.sox_open_mem_write.restype = ctypes.POINTER(sox_format_t)

    libsox_.sox_close.argtypes = [ctypes.POINTER(sox_format_t)]
    libsox_.sox_close.restype = ctypes.c_int

    libsox_.sox_create_effects_chain.argtypes = [
        ctypes.POINTER(sox_encodinginfo_t),
        ctypes.POINTER(sox_encodinginfo_t),
    ]
    libsox_.sox_create_effects_chain.restype = ctypes.POINTER(sox_effects_chain_t)

    libsox_.sox_delete_effects_chain.argtypes = [ctypes.POINTER(sox_effects_chain_t)]

    libsox_.sox_find_effect.argtypes = [ctypes.c_char_p]
    libsox_.sox_find_effect.restype = ctypes.POINTER(sox_effect_handler_t)

    libsox_.sox_create_effect.argtypes = [ctypes.POINTER(sox_effect_handler_t)]
    libsox_.sox_create_effect.restype = ctypes.POINTER(sox_effect_t)

    libsox_.sox_effect_options.argtypes = [
        ctypes.POINTER(sox_effect_t),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    libsox_.sox_effect_options.restype = ctypes.c_int

    libsox_.sox_add_effect.argtypes = [
        ctypes.POINTER(sox_effects_chain_t),
        ctypes.POINTER(sox_effect_t),
        ctypes.POINTER(sox_signalinfo_t),
        ctypes.POINTER(sox_signalinfo_t),
    ]
    libsox_.sox_add_effect.restype = ctypes.c_int

    libsox_.sox_flow_effects.argtypes = [
        ctypes.POINTER(sox_effects_chain_t),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    libsox_.sox_flow_effects.restype = ctypes.c_int

    if libsox_.sox_init() != SOX_SUCCESS:
        raise RuntimeError("Failed to initialize SoX.")

    libsox = libsox_
    sox_free = sox_free_
    LIBSOX_INITIALIZED = True


def libsox_cleanup() -> None:
    global LIBSOX_INITIALIZED
    if not LIBSOX_INITIALIZED:
        return

    libsox.sox_quit()


atexit.register(libsox_cleanup)


def libsox_rate(
    audio: np.ndarray,
    sample_rate: int,
    target_rate: int,
    quality: str = "v",
    phase: str = "I",
    bandwidth: float = 99.0,
) -> Tuple[np.ndarray, int]:
    if not LIBSOX_INITIALIZED:
        libsox_import()

    if audio.ndim != 1 or audio.dtype != np.float32:
        raise ValueError("Audio must be a 1D float32 numpy array.")

    in_ft = None
    out_ft = None
    chain = None
    try:
        input_signal = sox_signalinfo_t(
            rate=float(sample_rate), channels=1, precision=32, length=len(audio)
        )
        input_encoding = sox_encodinginfo_t(
            encoding=SOX_ENCODING_FLOAT, bits_per_sample=32
        )
        in_ft = libsox.sox_open_mem_read(
            audio.ctypes.data_as(ctypes.c_void_p),
            audio.nbytes,
            ctypes.byref(input_signal),
            ctypes.byref(input_encoding),
            b"raw",
        )
        if not in_ft:
            raise RuntimeError("Failed to open memory for reading.")

        original_input_signal = sox_signalinfo_t()
        ctypes.memmove(
            ctypes.byref(original_input_signal),
            ctypes.byref(in_ft.contents.signal),
            ctypes.sizeof(sox_signalinfo_t),
        )

        max_output_samples = round(
            len(audio) * float(target_rate) / float(sample_rate) * 1.05
        )

        max_output_size = max_output_samples * 4
        output_buffer = (ctypes.c_char * max_output_size)()

        output_signal = sox_signalinfo_t(rate=target_rate, channels=1, precision=32)
        output_encoding = sox_encodinginfo_t(
            encoding=SOX_ENCODING_FLOAT, bits_per_sample=32
        )
        out_ft = libsox.sox_open_mem_write(
            output_buffer,
            max_output_size,
            ctypes.byref(output_signal),
            ctypes.byref(output_encoding),
            b"raw",
            None,
        )
        if not out_ft:
            raise RuntimeError("Failed to open memory for writing.")

        chain = libsox.sox_create_effects_chain(
            ctypes.byref(in_ft.contents.encoding),
            ctypes.byref(out_ft.contents.encoding),
        )
        if not chain:
            raise RuntimeError("Failed to create effects chain.")

        intermediate_signal = sox_signalinfo_t()
        ctypes.memmove(
            ctypes.byref(intermediate_signal),
            ctypes.byref(in_ft.contents.signal),
            ctypes.sizeof(sox_signalinfo_t),
        )

        e = libsox.sox_create_effect(libsox.sox_find_effect(b"input"))
        args = (ctypes.c_char_p * 1)(ctypes.cast(in_ft, ctypes.c_char_p))
        libsox.sox_effect_options(e, 1, args)
        libsox.sox_add_effect(
            chain,
            e,
            ctypes.byref(intermediate_signal),
            ctypes.byref(intermediate_signal),
        )
        sox_free(e)

        e = libsox.sox_create_effect(libsox.sox_find_effect(b"rate"))

        rate_args_list = [
            f"-{quality}",
            f"-{phase}",
            "-b",
            str(bandwidth),
            str(target_rate),
        ]

        num_rate_args = len(rate_args_list)
        args = (ctypes.c_char_p * num_rate_args)(
            *[s.encode("utf-8") for s in rate_args_list]
        )

        libsox.sox_effect_options(e, num_rate_args, args)
        libsox.sox_add_effect(
            chain,
            e,
            ctypes.byref(intermediate_signal),
            ctypes.byref(intermediate_signal),
        )
        sox_free(e)

        e = libsox.sox_create_effect(libsox.sox_find_effect(b"output"))
        args = (ctypes.c_char_p * 1)(ctypes.cast(out_ft, ctypes.c_char_p))
        libsox.sox_effect_options(e, 1, args)
        libsox.sox_add_effect(
            chain,
            e,
            ctypes.byref(intermediate_signal),
            ctypes.byref(out_ft.contents.signal),
        )
        sox_free(e)

        libsox.sox_flow_effects(chain, None, None)

    finally:
        if chain:
            libsox.sox_delete_effects_chain(chain)
        if out_ft:
            libsox.sox_close(out_ft)

        if in_ft:
            ctypes.memmove(
                ctypes.byref(in_ft.contents.signal),
                ctypes.byref(original_input_signal),
                ctypes.sizeof(sox_signalinfo_t),
            )
            libsox.sox_close(in_ft)

    num_output_samples = round(len(audio) * target_rate / sample_rate)

    # Ensure a standard, writeable numpy array is returned by making a copy.
    return np.frombuffer(
        output_buffer.raw[: num_output_samples * 4], dtype=np.float32
    ).copy(), int(target_rate)
