from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POST_RL_SRC = REPO_ROOT / "post_training" / "src"
for path in (REPO_ROOT, POST_RL_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from post_rl.data.reward_zarr_migration import (
    clone_latent_cache_for_reward_clone,
    migrate_reward_zarr,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clone an existing Offline RL Zarr dataset and replace only reward/return "
            "with Reward V2: +1 at the later of the two action-gripper releases. "
            "Episode length, observations, actions, done, and timeout remain unchanged."
        )
    )
    parser.add_argument("--source-zarr", required=True)
    parser.add_argument("--target-zarr", required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--left-gripper-index", type=int, default=7)
    parser.add_argument("--right-gripper-index", type=int, default=15)
    parser.add_argument(
        "--left-open-side",
        choices=("auto", "low", "high"),
        default="auto",
    )
    parser.add_argument(
        "--right-open-side",
        choices=("auto", "low", "high"),
        default="auto",
    )
    parser.add_argument("--open-infer-frames", type=int, default=5)
    parser.add_argument("--min-dwell", type=int, default=3)
    parser.add_argument("--expected-cycles", type=int, default=1)
    parser.add_argument(
        "--copy-mode",
        choices=("auto", "copy", "reflink"),
        default="auto",
        help=(
            "auto tries GNU cp --reflink=auto and safely falls back to a normal copy. "
            "No hard links are used for Zarr data."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--source-latent-cache",
        default=None,
        help=(
            "Optional existing frozen ACT latent cache. Because Reward V2 changes only "
            "reward/return, the observation latents are identical and can be reused."
        ),
    )
    parser.add_argument(
        "--target-latent-cache",
        default=None,
        help=(
            "Optional target cache directory. Default: "
            "<target-zarr>.act_latent_cache/<source-cache-basename>."
        ),
    )
    parser.add_argument(
        "--latent-link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help=(
            "How to materialize immutable obs_latent.npy/next_indices.npy. hardlink is "
            "recommended on the same filesystem and avoids duplicating the very large cache."
        ),
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to save the migration summary JSON.",
    )
    args = parser.parse_args()

    summary = migrate_reward_zarr(
        args.source_zarr,
        args.target_zarr,
        gamma=args.gamma,
        horizon=args.horizon,
        left_gripper_index=args.left_gripper_index,
        right_gripper_index=args.right_gripper_index,
        left_open_side=args.left_open_side,
        right_open_side=args.right_open_side,
        open_infer_frames=args.open_infer_frames,
        min_dwell=args.min_dwell,
        expected_cycles=args.expected_cycles,
        overwrite=args.overwrite,
        copy_mode=args.copy_mode,
    )

    latent_summary = None
    if args.source_latent_cache:
        latent_summary = clone_latent_cache_for_reward_clone(
            args.source_latent_cache,
            args.source_zarr,
            args.target_zarr,
            target_cache=args.target_latent_cache,
            link_mode=args.latent_link_mode,
            overwrite=args.overwrite,
        )
        summary["latent_cache"] = latent_summary

    print("\n=== Reward V2 Zarr migration ===")
    print(f"Source:   {summary['source_zarr']}")
    print(f"Target:   {summary['target_zarr']}")
    print(f"Frames:   {summary['frames']}")
    print(f"Episodes: {summary['episodes']}")
    print(f"Reward +1 count: {summary['reward_count']}")
    tail = summary["tail_after_reward_frames"]
    print(
        "Tail after reward frames min/mean/median/max: "
        f"{tail['min']:.0f} / {tail['mean']:.2f} / "
        f"{tail['median']:.2f} / {tail['max']:.0f}"
    )
    print(
        f"Final H={args.horizon} chunk contains reward: "
        f"{100.0 * summary['final_h_chunk_contains_reward_fraction']:.1f}% episodes"
    )
    if summary["final_h_chunk_reward_offset_mean"] is not None:
        print(
            "Reward offset inside final H chunk mean: "
            f"{summary['final_h_chunk_reward_offset_mean']:.2f}"
        )

    if latent_summary is not None:
        print("\nLatent cache:")
        print(f"  Source: {latent_summary['source_cache']}")
        print(f"  Target: {latent_summary['target_cache']}")
        print(f"  Mode:   {latent_summary['link_mode']}")
        print(f"  Shape:  {latent_summary['latent_shape']}")

    if args.summary_json:
        path = pathlib.Path(args.summary_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(summary, file, indent=2)
        print(f"\nSummary: {path}")

    print(
        "\nNOTE: this is a reward-timing ablation, not a terminal-boundary change. "
        "Because the post-release tail is preserved, the final H-step chunk may still "
        "contain the +1 reward; the statistic above makes that explicit."
    )


if __name__ == "__main__":
    main()
