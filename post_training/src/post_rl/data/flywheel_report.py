"""Summarize one completed Data Wheel round from conversion metadata."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import zarr


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _lerobot_stats(root: Path) -> tuple[int, int]:
    info = _read_json(root / "meta" / "info.json")
    return int(info["total_episodes"]), int(info["total_frames"])


def _zarr_stats(root: Path) -> tuple[int, int]:
    dataset = zarr.open_group(str(root), mode="r")
    return len(dataset["meta"]["episode_ends"]), int(dataset["data"]["action"].shape[0])


def _check(actual: tuple[int, int], expected: tuple[int, int], label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: actual episodes/frames={actual}, expected={expected}"
        )


def write_report(
    staging: list[Path],
    prev_lerobot: Path,
    prev_zarr: Path,
    rollout_lerobot: Path,
    rollout_zarr: Path,
    merged_lerobot: Path,
    merged_zarr: Path,
    output: Path,
) -> None:
    staging_meta = _read_json(
        rollout_lerobot / "meta" / "arbim_staging_provenance.json"
    )
    merged_meta = _read_json(
        merged_lerobot / "meta" / "arbim_merge_provenance.json"
    )
    accepted = staging_meta["episodes"]
    rejected = staging_meta["skipped_invalid"]
    valid_episodes = len(accepted)
    invalid_episodes = len(rejected)
    captured_episodes = valid_episodes + invalid_episodes
    if staging_meta["source_count"] != valid_episodes:
        raise ValueError("Staging provenance source_count mismatch")
    valid_frames = sum(int(ep["num_frames"]) for ep in accepted)
    invalid_frames = staging_meta.get("skipped_invalid_frames")
    captured_frames = valid_frames + int(invalid_frames) if invalid_frames is not None else None

    prior_il = _lerobot_stats(prev_lerobot)
    added_il = _lerobot_stats(rollout_lerobot)
    combined_il = _lerobot_stats(merged_lerobot)
    prior_rl = _zarr_stats(prev_zarr)
    added_rl = _zarr_stats(rollout_zarr)
    combined_rl = _zarr_stats(merged_zarr)
    _check(added_il, (valid_episodes, valid_frames), "rollout LeRobot")
    _check(added_rl, (valid_episodes, valid_frames), "rollout Zarr")
    _check(combined_il, tuple(map(sum, zip(prior_il, added_il))), "merged LeRobot")
    _check(combined_rl, tuple(map(sum, zip(prior_rl, added_rl))), "merged Zarr")
    _check(
        (int(merged_meta["total_episodes"]), int(merged_meta["total_frames"])),
        combined_il,
        "merged LeRobot provenance",
    )
    if len(merged_meta["episodes"]) != combined_il[0]:
        raise ValueError("Merged LeRobot episode provenance count mismatch")

    def fmt(stats: tuple[int, int]) -> str:
        return f"{stats[0]} episodes, {stats[1]} frames"

    lines = [
        "ARBiM Data Wheel — Round Summary",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "[Rollout selection]",
        f"Successful Staging rollouts (saved): {captured_episodes}",
        f"Reward valid / exported:           {valid_episodes}",
        f"Reward invalid / excluded:        {invalid_episodes}",
        f"Valid action frames (T):          {valid_frames}",
        f"Invalid action frames (T):        {invalid_frames if invalid_frames is not None else 'not recorded'}",
        f"Total saved action frames (T):    {captured_frames if captured_frames is not None else 'not recorded'}",
        "Note: final T+1 observations remain in Staging; frames count action-aligned T.",
        "Note: unsuccessful/aborted rollouts without Staging files are not counted.",
        "",
        "[LeRobot / IL]",
        f"Previous:  {fmt(prior_il)}",
        f"Added:     {fmt(added_il)}",
        f"Merged:    {fmt(combined_il)}",
        "",
        "[Zarr / Offline RL]",
        f"Previous:  {fmt(prior_rl)}",
        f"Added:     {fmt(added_rl)}",
        f"Merged:    {fmt(combined_rl)}",
        "Note: one Zarr transition row corresponds to one action-aligned frame.",
        "",
        "[Paths]",
        f"Staging sources ({len(staging)}):",
        *(f"  - {source}" for source in staging),
        f"Rollout LeRobot: {rollout_lerobot}",
        f"Rollout Zarr:    {rollout_zarr}",
        f"Merged LeRobot:  {merged_lerobot}",
        f"Merged Zarr:     {merged_zarr}",
    ]
    if rejected:
        lines += ["", "[Excluded Reward-invalid Staging files]"]
        details = {
            item["path"]: item for item in staging_meta.get("skipped_invalid_details", [])
        }
        for source in rejected:
            issue = details.get(source, {})
            reason = issue.get("reward_error") or "unspecified"
            length = issue.get("num_steps", "?")
            lines.append(f"- {source} | {length} frames | {reason}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSummary saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", nargs="+", required=True, type=Path)
    for name in (
        "prev_lerobot", "prev_zarr", "rollout_lerobot",
        "rollout_zarr", "merged_lerobot", "merged_zarr", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    write_report(**vars(args))


if __name__ == "__main__":
    main()
