"""
This dataset consists of transcripts for 663 podcasts from the This American Life radio program from 1995 to 2020, covering 637 hours of audio (57.7 minutes per conversation) and an average of 18 unique speakers per conversation.

We hope that this dataset can serve as a new benchmark for the difficult tasks of speech transcription, speaker diarization, and dialog modeling on long, open-domain, multi-speaker conversations.

To learn more, please read our paper at: https://arxiv.org/pdf/2005.08072.pdf, and check the README.txt.
"""
import glob
import json
import logging
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional, Union
from urllib.error import HTTPError

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download


def scrape_urls(website_url, output_path, year_range=(1995, 2021)):
    if not is_module_available("bs4"):
        raise ImportError("Please 'pip install beautifulsoup4' first.")

    import requests
    from bs4 import BeautifulSoup

    urls = {}
    for year in range(*year_range):
        print(f"Scraping {year}...")
        url = f"{website_url}/archive?year={year}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        page_urls = set()
        for a in soup.find_all("a", href=True, class_="goto-episode"):
            if a["href"].startswith("/"):
                page_urls.add(f"{website_url}{a['href']}")

        print(f"Found {len(page_urls)} episodes in {year}.")

        for episode_url in tqdm(page_urls):
            episode_id = int(episode_url.split("/")[-2])
            response = requests.get(episode_url)
            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True, download=True):
                urls[f"ep-{episode_id}"] = a["href"]

    print(f"Saving results ({len(urls)} episodes)...")
    with open(output_path, "w") as f:
        json.dump(urls, f)


def included_episodes(target_dir: Pathlike) -> Iterable[str]:
    for subset in ["train", "valid", "test"]:
        with open(Path(target_dir) / f"{subset}-transcripts-aligned.json") as f:
            for episode_id in json.load(f).keys():
                yield episode_id


def download_this_american_life(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    metadata_url="https://ipfs.io/ipfs/bafybeidyt3ch6t4dtu2ehdriod3jvuh34qu4pwjyoba2jrjpmqwckkr6q4/this_american_life.zip",
    website_url="https://thisamericanlife.org",
):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "metadata.zip"
    completed_detector = target_dir / "README.txt"

    if not completed_detector.is_file() or force_download:
        resumable_download(metadata_url, zip_path, force_download=force_download)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print("Extracting...")
            zip_ref.extractall(target_dir)

        zip_path.unlink()

    # This American Life website was updated since the dataset annotations were published.
    # The links in the HTML page included are no longer valid and need to be re-scraped.
    urls_path = target_dir / "urls.json"
    if not urls_path.is_file():
        scrape_urls(website_url, urls_path)

    with open(urls_path) as f:
        urls = json.load(f)

    audio_dir = target_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    for ep_id in included_episodes(target_dir):
        print(f"Downloading episode {ep_id}... ({urls[ep_id]})")

        try:
            resumable_download(
                urls[ep_id], audio_dir / f"{ep_id}.mp3", force_download=force_download
            )
        except HTTPError as e:
            # Some episodes are no longer available on the website (like removed for anonymity reason, ep-374).
            print(f"Failed to download {ep_id}: {e}. Skipping...")
            continue

    print("Done!")


def prepare_this_american_life(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
):
    manifests = {}
    for subset in ["train", "dev", "test"]:
        manifests[subset] = prepare_this_american_life_subset(
            corpus_dir=corpus_dir,
            subset=subset,
            output_dir=output_dir,
        )

    return manifests


def prepare_this_american_life_subset(
    corpus_dir: Pathlike,
    subset: str,
    output_dir: Optional[Pathlike] = None,
):
    if not is_module_available("nltk"):
        raise ImportError("Please 'pip install nltk' first.")

    from nltk import word_tokenize

    corpus_dir = Path(corpus_dir).absolute()

    file_subset = "valid" if subset == "dev" else subset
    with open(Path(corpus_dir) / f"{file_subset}-transcripts-aligned.json") as f:
        transcripts = json.load(f)

    recordings = []
    supervisions = []
    pbar = tqdm(transcripts.items())
    for ep_id, transcript in pbar:
        pbar.set_description(desc=f"Processing {subset} subset ({ep_id})")
        audio_path = corpus_dir / "audio" / f"{ep_id}.mp3"
        if not audio_path.is_file():
            logging.warning(f"File {audio_path} not found - skipping.")
            continue

        recordings.append(Recording.from_file(audio_path))

        for utt_ix, utt in enumerate(transcript):
            text = utt["utterance"]
            words = word_tokenize(text)
            if len(words) != utt["n_words"]:
                logging.warning(
                    f"Transcript mismatch for {ep_id}-{utt_ix}: {utt['n_words']} words in the transcript, {len(words)} tokens in the text."
                )

            alignments = [
                AlignmentItem(words[int(ix)], start, end - start)
                for start, end, ix in utt["alignments"]
                if ix < len(words)
            ]
            segment = SupervisionSegment(
                id=f"{ep_id}-{utt_ix}",
                recording_id=ep_id,
                start=utt["utterance_start"],
                duration=utt["utterance_end"] - utt["utterance_start"],
                channel=0,
                text=text,
                language="en",
                speaker=utt["speaker"],
            )
            segment = segment.with_alignment("word", alignments)
            supervisions.append(segment)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recording_set.to_file(
            output_dir / f"this-american-life_recordings_{subset}.jsonl.gz"
        )
        supervision_set.to_file(
            output_dir / f"this-american-life_supervisions_{subset}.jsonl.gz"
        )

    return {"recordings": recording_set, "supervisions": supervision_set}
