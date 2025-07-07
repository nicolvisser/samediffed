from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Union

import polars as pl
import tgt


@dataclass
class Item:
    file: str
    onset: float
    offset: float
    label: str
    speaker: int


def get_filename(path: Path) -> str:
    return path.stem


def get_speaker_id(path: Path) -> int:
    return int(path.parent.parent.name)


def update_word_counts(item: Item, word_counts: Dict) -> None:
    if item.label not in word_counts:
        word_counts[item.label] = 0
    word_counts[item.label] += 1


def char_len_filter(item: Item, min_chars: int = 5) -> bool:
    # number of characters must be 5 or more
    return len(item.label) >= min_chars


def duration_filter(item: Item, min_dur: float = 0.5) -> bool:
    # duration must be 0.5 seconds or longer
    return item.offset - item.onset >= min_dur


def word_count_filter(item: Item, word_counts: Dict, min_count: int = 2) -> bool:
    # word must occur at least twice in the dataset
    return word_counts[item.label] >= min_count


def get_item_iter(textgrid_dir: Path, tier_name: str) -> Iterable[Item]:
    textgrid_files = sorted(textgrid_dir.rglob("*.TextGrid"))
    assert len(textgrid_files) > 0, f"No .TextGrid files found in {textgrid_dir}"
    print(f"Found {len(textgrid_files)} .TextGrid files.")
    for textgrid_file in textgrid_files:
        filename = get_filename(textgrid_file)
        speaker_id = get_speaker_id(textgrid_file)
        tg = tgt.io.read_textgrid(textgrid_file, include_empty_intervals=False)
        for interval in tg.get_tier_by_name(tier_name):
            yield Item(
                file=filename,
                speaker=speaker_id,
                label=interval.text,
                onset=interval.start_time,
                offset=interval.end_time,
            )


def item_to_dict(item: Item) -> Dict:
    return {
        "#file": item.file,
        "onset": item.onset,
        "offset": item.offset,
        "#word": item.label,
        "speaker": item.speaker,
    }


def make_samediff_item_file(
    textgrid_dir: Union[str, Path],
    item_file_path: Union[str, Path],
    tier_name: str = "words",
    min_chars: int = 5,
    min_dur: float = 0.5,
    min_count: int = 2,
) -> None:
    textgrid_dir = Path(textgrid_dir)
    item_file_path = Path(item_file_path)

    # make Item objects from textgrid files
    items = list(get_item_iter(textgrid_dir, tier_name))

    # apply filters
    items = filter(partial(char_len_filter, min_chars=min_chars), items)
    items = filter(partial(duration_filter, min_dur=min_dur), items)
    items = list(items)

    # count word occurences and then filter
    word_counts = {}
    for item in items:
        update_word_counts(item, word_counts)
    items = filter(
        partial(word_count_filter, word_counts=word_counts, min_count=min_count), items
    )

    # map to headings
    items = map(item_to_dict, items)

    df = pl.DataFrame(items)
    df = df.sort(["speaker", "#file", "onset"])
    print(df.head(10))

    item_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(item_file_path, separator=" ")
    print(f"Saved {len(df)} rows to {item_file_path.absolute()}")


if __name__ == "__main__":
    make_samediff_item_file(
        textgrid_dir="/mnt/wsl/newt/alignments/mfa/us-arpa/LibriSpeech/dev-clean",
        item_file_path="samediff.item",
        tier_name="words",
        min_chars=5,
        min_dur=0.5,
        min_count=2,
    )
