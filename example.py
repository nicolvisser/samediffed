from samediffed import (
    setup_tasks,
    run_tasks,
    plot_density,
    evaluate_results,
    plot_precision_vs_recall,
    show_interactive_examples,
)
import numpy as np


if __name__ == "__main__":

    # load data and prepare tasks
    task_df = setup_tasks(
        item_file_path="data/librispeech-samediff-words-dev-clean.item",
        units_dir="/mnt/wsl/newt/units_duped/hubert-discrete/LibriSpeech/dev-clean",
        frequency=50,
        extension=".npy",
        units_maker=lambda path: np.load(path),
    )

    # compute pairwise edit distances
    results = run_tasks(task_df)

    # compute evaluation metrics
    evaluated_results = evaluate_results(results)

    # print and save results
    print(f"AP SW: {evaluated_results.sw.average_precision}")
    evaluated_results.save_json("output/results.json")

    # [Optional]: plot the density of the normalized edit distances
    fig_density = plot_density(results)
    fig_density.savefig("output/density.svg", format="svg")

    # [Optional]: plot the precision vs recall curve
    fig_pr = plot_precision_vs_recall(evaluated_results)
    fig_pr.savefig("output/precision-recall.svg", format="svg")

    # [Optional]: show examples where you can listen to the snippets
    # Note: You need to do this in a Jupyter or interactive environment!
    show_interactive_examples(
        task_df,
        results,
        waveforms_dir="/mnt/wsl/newt/datasets/LibriSpeech/dev-clean",
        waveform_extension=".flac",
        num_samples=1,
        seed=42,
    )
