#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from pathlib import Path

"""
Outputs the following stats for all of the conversation sets in the generated dataset.
* Number of conversations
* Number of turns
* <sil> per turn breakdown counts
* Number of phrase pairs
* <sil> vs non-sil thought-response pair breakdown counts
* Mean phrases per turn
* Mean <sil> vs non-sil phrases
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATED_DATA_DIR = REPO_ROOT / "generated_data"
DEFAULT_OUTPUT_PATH = DEFAULT_GENERATED_DATA_DIR / "dataset_stats.csv"


def get_file_list(path_in):
    return sorted(Path(path_in).glob("*.jsonl"))

def parse_jsonl(jsonl_path):
    conversation_count = 0
    turns_counts = []
    phrases_counts = []
    sil_counts = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conversation = obj["conversation"]
            conversation_count += 1
            length = len(conversation)
            turns_counts.append(length)
            for jj in range(length):
                turn = conversation[jj]
                phrases = len(turn["thoughts"])
                phrases_counts.append(phrases)
                sil_counts.append(turn["thoughts"].count("<sil>"))
    return {"conversation_count": conversation_count, "turns_counts": turns_counts, "phrases_counts": phrases_counts, "sil_counts": sil_counts}

def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize(label, conversation_count, turns_counts, phrases_counts, sil_counts):
    nonsil_counts = [p - s for p, s in zip(phrases_counts, sil_counts)]
    total_turns = sum(turns_counts)
    total_phrases = sum(phrases_counts)
    total_sil = sum(sil_counts)
    total_nonsil = total_phrases - total_sil
    turns_mean, turns_std = _mean_std(turns_counts)
    phrases_mean, phrases_std = _mean_std(phrases_counts)
    sil_mean, sil_std = _mean_std(sil_counts)
    nonsil_mean, nonsil_std = _mean_std(nonsil_counts)
    return {
        "dataset": label,
        "num_conversations": conversation_count,
        "num_turns": total_turns,
        "turns_per_conv_mean": turns_mean,
        "turns_per_conv_std": turns_std,
        "num_phrase_pairs": total_phrases,
        "phrases_per_turn_mean": phrases_mean,
        "phrases_per_turn_std": phrases_std,
        "num_sil": total_sil,
        "sil_per_turn_mean": sil_mean,
        "sil_per_turn_std": sil_std,
        "num_nonsil": total_nonsil,
        "nonsil_per_turn_mean": nonsil_mean,
        "nonsil_per_turn_std": nonsil_std,
    }


def print_row(row):
    print(f"\n=== {row['dataset']} ===")
    print(f"Number of conversations: {row['num_conversations']}")
    print(f"Number of turns: {row['num_turns']}")
    print(f"    Turns per conversation: {row['turns_per_conv_mean']:.3f} +/- {row['turns_per_conv_std']:.3f}")
    print(f"Number of phrase pairs: {row['num_phrase_pairs']}")
    print(f"    Phrase pairs per turn: {row['phrases_per_turn_mean']:.3f} +/- {row['phrases_per_turn_std']:.3f}")
    print(f"Total <sil> phrases: {row['num_sil']}")
    print(f"    <sil> per turn: {row['sil_per_turn_mean']:.3f} +/- {row['sil_per_turn_std']:.3f}")
    print(f"Total non-sil phrases: {row['num_nonsil']}")
    print(f"    non-sil per turn: {row['nonsil_per_turn_mean']:.3f} +/- {row['nonsil_per_turn_std']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize generated ConvFill JSONL files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_GENERATED_DATA_DIR,
        help="Directory containing generated JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV summary path.",
    )
    args = parser.parse_args()

    file_list = get_file_list(args.input_dir)
    if not file_list:
        raise SystemExit(f"No JSONL files found in {args.input_dir}")

    rows = []
    all_turns = []
    all_phrases = []
    all_sil = []
    total_conversations = 0

    for jsonl_path in file_list:
        stats = parse_jsonl(jsonl_path)
        row = summarize(
            jsonl_path.name,
            stats["conversation_count"],
            stats["turns_counts"],
            stats["phrases_counts"],
            stats["sil_counts"],
        )
        rows.append(row)
        print_row(row)

        total_conversations += stats["conversation_count"]
        all_turns.extend(stats["turns_counts"])
        all_phrases.extend(stats["phrases_counts"])
        all_sil.extend(stats["sil_counts"])

    total_row = summarize("TOTAL", total_conversations, all_turns, all_phrases, all_sil)
    rows.append(total_row)
    print_row(total_row)

    csv_path = args.output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
