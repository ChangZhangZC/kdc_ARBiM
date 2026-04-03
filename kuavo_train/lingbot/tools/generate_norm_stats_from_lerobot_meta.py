import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot meta/stats.json into a LingBot-compatible norm_stats file."
    )
    parser.add_argument("--stats-json", required=True, help="Path to LeRobot meta/stats.json")
    parser.add_argument("--output", required=True, help="Path to output LingBot norm stats json")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["observation.state", "action"],
        help="Keys to keep from LeRobot stats.json",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats_json)
    output_path = Path(args.output)

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    missing = [key for key in args.keys if key not in stats]
    if missing:
        raise KeyError(f"Missing keys in {stats_path}: {missing}")

    count = None
    for key in args.keys:
        key_count = stats[key].get("count")
        if isinstance(key_count, list) and key_count:
            count = int(key_count[0])
            break

    payload = {
        "norm_stats": {key: stats[key] for key in args.keys},
        "count": count,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote LingBot norm stats to: {output_path}")


if __name__ == "__main__":
    main()
