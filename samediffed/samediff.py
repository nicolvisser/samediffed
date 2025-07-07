from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from samediffed import _rusted as rusted

plt.rcParams["font.family"] = "monospace"


@dataclass
class Results:
    swsp: pl.DataFrame
    swdp: pl.DataFrame
    dwsp: pl.DataFrame
    dwdp: pl.DataFrame


@dataclass
class EvaluatedTask:
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    average_precision: float
    precision_recall_breakeven: float


@dataclass
class EvaluatedResults:
    sw: EvaluatedTask
    swsp: EvaluatedTask
    swdp: EvaluatedTask

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        data = {
            "AP SW": self.sw.average_precision,
            "AP SWSP": self.swsp.average_precision,
            "AP SWDP": self.swdp.average_precision,
            "PRB SW": self.sw.precision_recall_breakeven,
            "PRB SWSP": self.swsp.precision_recall_breakeven,
            "PRB SWDP": self.swdp.precision_recall_breakeven,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


def setup_tasks(
    item_file_path: Union[str, Path],
    units_dir: Union[str, Path],
    frequency: float = 50,
    extension: str = ".npy",
    units_maker: Callable = lambda path: np.load(path),
) -> pl.DataFrame:
    item_file_path = Path(item_file_path)
    units_dir = Path(units_dir)

    df = pl.read_csv(item_file_path, separator=" ")

    stems = df["#file"].unique().to_list()
    paths = {
        p.stem: str(p.absolute()) for p in sorted(units_dir.rglob(f"*{extension}"))
    }

    missing_stems = [stem for stem in stems if stem not in paths]
    if missing_stems:
        raise ValueError(f"The following files could not be found: {missing_stems}")

    df = df.with_columns(pl.col("#file").replace(paths).alias("path"))

    def onset_to_start_index(onset: float) -> int:
        return math.floor(onset * frequency)

    def offset_to_end_index(offset: float) -> int:
        return math.ceil(offset * frequency)

    df = df.with_columns(
        [
            pl.col("onset")
            .map_elements(onset_to_start_index, return_dtype=pl.Int64)
            .alias("start_index"),
            pl.col("offset")
            .map_elements(offset_to_end_index, return_dtype=pl.Int64)
            .alias("end_index"),
        ]
    )

    def units_loader(
        path: Path, start_index: int, end_index: int, deduplicate: bool = True
    ):
        # load the features given the start and end indices
        units = units_maker(path)
        assert units.ndim == 1
        units = units[start_index:end_index]
        if deduplicate:
            # remove consecutive duplicates
            does_not_repeat = np.concatenate(
                [[True], units[1:] != units[:-1]], dtype=np.bool
            )
            units = units[does_not_repeat]
        return units

    def row_to_units(row):
        return units_loader(row["path"], row["start_index"], row["end_index"])

    # Compute the units for each row using a list comprehension with progress bar
    units = [row_to_units(row) for row in tqdm(df.to_dicts(), desc="Loading units")]
    df = df.with_columns(pl.Series("units", units))

    # Sort by word alphabetically before adding row indices
    df = df.sort("#word")

    # Add row indices for filtering symmetric pairs
    df = df.with_row_index("row_idx")

    return df


def run_tasks(task_df: pl.DataFrame) -> Results:
    lf = task_df.lazy()

    lf1 = lf.with_columns(
        [
            pl.col("#word").alias("word1"),
            pl.col("speaker").alias("speaker1"),
            pl.col("units").alias("units1"),
            pl.col("row_idx").alias("row_idx1"),
        ]
    ).select(["word1", "speaker1", "units1", "row_idx1"])

    lf2 = lf.with_columns(
        [
            pl.col("#word").alias("word2"),
            pl.col("speaker").alias("speaker2"),
            pl.col("units").alias("units2"),
            pl.col("row_idx").alias("row_idx2"),
        ]
    ).select(["word2", "speaker2", "units2", "row_idx2"])

    # Cartesian product (cross join) - only upper triangle to avoid redundant symmetric computations
    pairs = lf1.join(lf2, how="cross").filter(pl.col("row_idx1") < pl.col("row_idx2"))

    SWSP = pairs.filter(
        (pl.col("word1") == pl.col("word2"))
        & (pl.col("speaker1") == pl.col("speaker2"))
    )
    SWDP = pairs.filter(
        (pl.col("word1") == pl.col("word2"))
        & (pl.col("speaker1") != pl.col("speaker2"))
    )
    DWSP = pairs.filter(
        (pl.col("word1") != pl.col("word2"))
        & (pl.col("speaker1") == pl.col("speaker2"))
    )
    DWDP = pairs.filter(
        (pl.col("word1") != pl.col("word2"))
        & (pl.col("speaker1") != pl.col("speaker2"))
    )

    def add_edit_distance_batch(lf, batch_size=10000, desc="Computing edit distances"):
        # Collect the relevant columns using streaming mode
        df_pairs = lf.select(
            [
                "word1",
                "word2",
                "speaker1",
                "speaker2",
                "units1",
                "units2",
                "row_idx1",
                "row_idx2",
            ]
        ).collect(engine="streaming")
        n = len(df_pairs)
        edit_distances = []
        with tqdm(total=n, desc=desc) as pbar:
            for i in range(0, n, batch_size):
                batch1 = df_pairs["units1"][i : i + batch_size].to_list()
                batch2 = df_pairs["units2"][i : i + batch_size].to_list()
                edit_distances.extend(
                    rusted.normalized_edit_distance_polars(batch1, batch2)
                )
                pbar.update(len(batch1))
        df_pairs = df_pairs.with_columns(
            pl.Series("normalized_edit_distance", edit_distances, dtype=pl.Float64)
        )
        return df_pairs

    SWSP_df = add_edit_distance_batch(SWSP, desc="Computing SWSP pairs")
    SWDP_df = add_edit_distance_batch(SWDP, desc="Computing SWDP pairs")
    DWSP_df = add_edit_distance_batch(DWSP, desc="Computing DWSP pairs")
    DWDP_df = add_edit_distance_batch(DWDP, desc="Computing DWDP pairs")

    return Results(SWSP_df, SWDP_df, DWSP_df, DWDP_df)


def print_samples(results: Results):
    # Example: print a small sample from each

    print("SWSP sample:")
    print(
        results.swsp.select(
            ["word1", "word2", "speaker1", "speaker2", "normalized_edit_distance"]
        ).head(5)
    )
    print("SWDP sample:")
    print(
        results.swdp.select(
            ["word1", "word2", "speaker1", "speaker2", "normalized_edit_distance"]
        ).head(5)
    )
    print("DWSP sample:")
    print(
        results.dwsp.select(
            ["word1", "word2", "speaker1", "speaker2", "normalized_edit_distance"]
        ).head(5)
    )
    print("DWDP sample:")
    print(
        results.dwdp.select(
            ["word1", "word2", "speaker1", "speaker2", "normalized_edit_distance"]
        ).head(5)
    )


def evaluate_results(results: Results) -> EvaluatedResults:
    def get_scores_and_labels(df, label):
        scores = df["normalized_edit_distance"].to_numpy()
        labels = np.full_like(scores, label, dtype=int)
        return scores, labels

    y_score_swsp, y_true_swsp = get_scores_and_labels(results.swsp, 1)
    y_score_swdp, y_true_swdp = get_scores_and_labels(results.swdp, 1)
    y_score_dwsp, y_true_dwsp = get_scores_and_labels(results.dwsp, 0)
    y_score_dwdp, y_true_dwdp = get_scores_and_labels(results.dwdp, 0)

    # Task SW: All pairs
    y_true_sw = np.concatenate([y_true_swsp, y_true_swdp, y_true_dwsp, y_true_dwdp])
    y_score_sw = np.concatenate(
        [y_score_swsp, y_score_swdp, y_score_dwsp, y_score_dwdp]
    )
    p_sw, r_sw, t_sw = precision_recall_curve(y_true_sw, -y_score_sw)
    ap_sw = np.abs(np.trapezoid(p_sw, r_sw)).item()
    prb_sw = p_sw[np.argmin(np.abs(r_sw - p_sw))].item()

    # Task SWSP: Same speaker only
    y_true_swsp = np.concatenate([y_true_swsp, y_true_dwsp])
    y_score_swsp = np.concatenate([y_score_swsp, y_score_dwsp])
    p_swsp, r_swsp, t_swsp = precision_recall_curve(y_true_swsp, -y_score_swsp)
    ap_swsp = np.abs(np.trapezoid(p_swsp, r_swsp)).item()
    prb_swsp = p_sw[np.argmin(np.abs(r_swsp - p_swsp))].item()

    # Task SWDP: Different speaker only
    y_true_swdp = np.concatenate([y_true_swdp, y_true_dwdp])
    y_score_swdp = np.concatenate([y_score_swdp, y_score_dwdp])
    p_swdp, r_swdp, t_swdp = precision_recall_curve(y_true_swdp, -y_score_swdp)
    ap_swdp = np.abs(np.trapezoid(p_swdp, r_swdp)).item()
    prb_swdp = p_sw[np.argmin(np.abs(r_swdp - p_swdp))].item()

    return EvaluatedResults(
        sw=EvaluatedTask(
            precision=p_sw,
            recall=r_sw,
            thresholds=t_sw,
            average_precision=ap_sw,
            precision_recall_breakeven=prb_sw,
        ),
        swsp=EvaluatedTask(
            precision=p_swsp,
            recall=r_swsp,
            thresholds=t_swsp,
            average_precision=ap_swsp,
            precision_recall_breakeven=prb_swsp,
        ),
        swdp=EvaluatedTask(
            precision=p_swdp,
            recall=r_swdp,
            thresholds=t_swdp,
            average_precision=ap_swdp,
            precision_recall_breakeven=prb_swdp,
        ),
    )


def plot_precision_vs_recall(evaluated_results: EvaluatedResults) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        evaluated_results.sw.recall,
        evaluated_results.sw.precision,
        label="SW",
        color="black",
    )
    ax.plot(
        evaluated_results.swsp.recall,
        evaluated_results.swsp.precision,
        label="SWSP",
        color="green",
    )
    ax.plot(
        evaluated_results.swdp.recall,
        evaluated_results.swdp.precision,
        label="SWDP",
        color="red",
    )

    # Fill under the curves
    ax.fill_between(
        evaluated_results.sw.recall,
        evaluated_results.sw.precision,
        alpha=0.2,
        color="black",
    )
    ax.fill_between(
        evaluated_results.swsp.recall,
        evaluated_results.swsp.precision,
        alpha=0.2,
        color="green",
    )
    ax.fill_between(
        evaluated_results.swdp.recall,
        evaluated_results.swdp.precision,
        alpha=0.2,
        color="red",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(title="Task type")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_density(results: Results, subsample_max_n=10000, subsample_seed=42) -> Figure:
    # Subsample up to 10,000 points for each group
    def subsample(series, max_n=subsample_max_n, seed=subsample_seed):
        arr = series.to_numpy()
        if len(arr) > max_n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(arr), size=max_n, replace=False)
            arr = arr[idx]
        return arr

    swsp_sample = subsample(results.swsp["normalized_edit_distance"])
    swdp_sample = subsample(results.swdp["normalized_edit_distance"])
    dwsp_sample = subsample(results.dwsp["normalized_edit_distance"])
    dwdp_sample = subsample(results.dwdp["normalized_edit_distance"])

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        swsp_sample,
        label="SWSP",
        fill=True,
        color="green",
        alpha=0.3,
        linewidth=0.2,
        ax=ax,
    )
    sns.kdeplot(
        swdp_sample,
        label="SWDP",
        fill=True,
        color="blue",
        alpha=0.3,
        linewidth=0.2,
        ax=ax,
    )
    sns.kdeplot(
        dwsp_sample,
        label="DWSP",
        fill=True,
        color="yellow",
        alpha=0.3,
        linewidth=0.2,
        ax=ax,
    )
    sns.kdeplot(
        dwdp_sample,
        label="DWDP",
        fill=True,
        color="red",
        alpha=0.3,
        linewidth=0.2,
        ax=ax,
    )
    ax.set_xlabel("Normalized Edit Distance")
    ax.set_ylabel("Density")
    ax.set_title("Estimated Probability Density of Normalized Edit Distances")
    ax.legend(title="Pair Type")
    ax.grid(True)
    fig.tight_layout()
    return fig


def show_interactive_examples(
    task_df: pl.DataFrame,
    results: Results,
    waveforms_dir: Union[str, Path],
    waveform_extension: str = ".flac",
    num_samples: int = 2,
    seed: int = 42,
) -> None:
    from IPython.display import display, Audio
    import soundfile as sf
    from pathlib import Path

    # Helper to join sampled pairs with task_df to get file/onset/offset info
    def enrich_with_metadata(sampled_df, idx1, idx2):
        meta_cols = ["#file", "onset", "offset", "units"]
        task_df1 = task_df.select(["row_idx"] + meta_cols).rename(
            {
                "row_idx": idx1,
                "#file": "file1",
                "onset": "onset1",
                "offset": "offset1",
                "units": "units1_seq",
            }
        )
        task_df2 = task_df.select(["row_idx"] + meta_cols).rename(
            {
                "row_idx": idx2,
                "#file": "file2",
                "onset": "onset2",
                "offset": "offset2",
                "units": "units2_seq",
            }
        )
        enriched = sampled_df.join(task_df1, on=idx1).join(task_df2, on=idx2)
        return enriched

    # Sample from each results dataframe
    swsp_sample = results.swsp.sample(num_samples, seed=seed)
    swdp_sample = results.swdp.sample(num_samples, seed=seed)
    dwsp_sample = results.dwsp.sample(num_samples, seed=seed)
    dwdp_sample = results.dwdp.sample(num_samples, seed=seed)

    # Enrich with metadata
    swsp_enriched = enrich_with_metadata(swsp_sample, "row_idx1", "row_idx2")
    swdp_enriched = enrich_with_metadata(swdp_sample, "row_idx1", "row_idx2")
    dwsp_enriched = enrich_with_metadata(dwsp_sample, "row_idx1", "row_idx2")
    dwdp_enriched = enrich_with_metadata(dwdp_sample, "row_idx1", "row_idx2")

    # Helper to display audio and info for each pair
    def display_pair(row, waveforms_dir, waveform_extension):
        # Find file1 and file2 by matching stem with rglob
        stem1 = row["file1"]
        stem2 = row["file2"]
        # waveform_extension is like '*.flac'
        print(f"{stem1}{Path(waveform_extension).suffix}")
        matches1 = list(Path(waveforms_dir).rglob(f"{stem1}{waveform_extension}"))
        matches2 = list(Path(waveforms_dir).rglob(f"{stem2}{waveform_extension}"))
        if not matches1:
            print(f"Audio file for {stem1} not found.")
            return
        if not matches2:
            print(f"Audio file for {stem2} not found.")
            return
        file1 = matches1[0]
        file2 = matches2[0]
        # Get sample rates first
        with sf.SoundFile(file1) as f1:
            sr1 = f1.samplerate
        with sf.SoundFile(file2) as f2:
            sr2 = f2.samplerate
        # Compute start/stop in samples
        start1 = int(row["onset1"] * sr1)
        stop1 = int(row["offset1"] * sr1)
        start2 = int(row["onset2"] * sr2)
        stop2 = int(row["offset2"] * sr2)
        # Load audio segments
        audio1, _ = sf.read(file1, start=start1, stop=stop1)
        audio2, _ = sf.read(file2, start=start2, stop=stop2)
        # Display info and audio
        print(
            f"Pair: {row['file1']} ({row['onset1']:.2f}-{row['offset1']:.2f}) <-> {row['file2']} ({row['onset2']:.2f}-{row['offset2']:.2f})"
        )
        print(f"Normalized Edit Distance: {row['normalized_edit_distance']:.3f}")
        print(f"Units 1: {row['units1_seq']}")
        print(f"Units 2: {row['units2_seq']}")
        display(Audio(audio1, rate=sr1))
        display(Audio(audio2, rate=sr2))
        print("-" * 60)

    # For each group, display samples
    for label, enriched in zip(
        ["SWSP", "SWDP", "DWSP", "DWDP"],
        [swsp_enriched, swdp_enriched, dwsp_enriched, dwdp_enriched],
    ):
        print(f"\n=== {label} Examples ===")
        for row in enriched.to_dicts():
            display_pair(row, waveforms_dir, waveform_extension)
