#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import rosbag


KEYWORDS = (
    "reward",
    "score",
    "success",
    "simulator",
    "mujoco",
    "pose",
    "object",
    "task",
)


def _fmt_frequency(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f} Hz"
    except (TypeError, ValueError):
        return str(value)


def _print_topics(bag: rosbag.Bag) -> list[str]:
    info = bag.get_type_and_topic_info()
    topics = sorted(info.topics.items())

    print("\n=== TOPICS ===")
    print(f"{'topic':<60} {'type':<40} {'count':>10} {'frequency':>14}")
    print("-" * 130)
    for topic, topic_info in topics:
        print(
            f"{topic:<60} "
            f"{topic_info.msg_type:<40} "
            f"{topic_info.message_count:>10} "
            f"{_fmt_frequency(topic_info.frequency):>14}"
        )

    return [topic for topic, _ in topics]


def _print_candidates(topics: list[str]) -> None:
    candidates = [
        topic
        for topic in topics
        if any(keyword in topic.lower() for keyword in KEYWORDS)
    ]

    print("\n=== REWARD / SIMULATOR CANDIDATES ===")
    if not candidates:
        print("No obvious reward/simulator-related topics found by name.")
        return
    for topic in candidates:
        print(topic)


def _print_samples(bag: rosbag.Bag, topic: str, max_messages: int) -> None:
    print(f"\n=== SAMPLE MESSAGES: {topic} ===")
    count = 0
    for _, msg, stamp in bag.read_messages(topics=[topic]):
        print(f"\n[{count}] t={stamp.to_sec():.9f}")
        print(msg)
        count += 1
        if count >= max_messages:
            break

    if count == 0:
        print("No messages found for this topic.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a ROS1 bag before designing ARBiM reward extraction."
    )
    parser.add_argument("bag", help="Path to a ROS1 .bag file.")
    parser.add_argument(
        "--sample-topic",
        action="append",
        default=[],
        help="Print sample messages from this topic. Can be specified multiple times.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=3,
        help="Maximum sample messages printed per --sample-topic. Default: 3.",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"Rosbag not found: {bag_path}")
    if args.max_messages <= 0:
        raise ValueError("--max-messages must be > 0")

    with rosbag.Bag(str(bag_path), "r") as bag:
        start = bag.get_start_time()
        end = bag.get_end_time()

        print("=== BAG ===")
        print(f"path     : {bag_path}")
        print(f"start    : {start:.9f}")
        print(f"end      : {end:.9f}")
        print(f"duration : {end - start:.3f} s")

        topics = _print_topics(bag)
        _print_candidates(topics)

        topic_set = set(topics)
        for topic in args.sample_topic:
            if topic not in topic_set:
                print(f"\n[WARN] Topic not found: {topic}")
                continue
            _print_samples(bag, topic, args.max_messages)


if __name__ == "__main__":
    main()
