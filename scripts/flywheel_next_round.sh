#!/usr/bin/env bash
# ARBiM Data Wheel: one new rollout round -> 4 training-ready artifacts.
# Run with a trusted Staging NPY and the PREVIOUS round's merged LeRobot + Zarr.
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage:
  bash scripts/flywheel_next_round.sh \
    --staging PATH \
    --previous-lerobot PATH \
    --previous-zarr PATH \
    --output-root NEW_DIRECTORY [options]

Required:
  --staging PATH              A directory of *_staging.npy files or one staging file.
  --previous-lerobot PATH     Previous round's merged IL LeRobot (full RGB+Depth).
  --previous-zarr PATH        Previous round's merged Offline RL Zarr.
  --output-root PATH          NEW directory; never overwritten.

Options:
  --zarr-depth auto|on|off    Default: auto (inherit previous Zarr); mismatch fails.
                             All LeRobot archives ALWAYS retain three Depth cameras.
  --invalid-reward error|skip
                             Default: error; skip omits Reward-invalid successes
                             from BOTH the new IL and RL datasets.
  --expected-rollouts N       Require exactly N Reward-valid staging episodes.
  --expected-fps N            Verify previous LeRobot metadata FPS (e.g. 10).
  --task TEXT                 Default: ARBiM Data Wheel.
  --rollout-repo-id ID         Default: arbim/flywheel_rollout.
  --merged-repo-id ID          Default: arbim/flywheel_merged.
  --batch-size N              LeRobot->Zarr decode batch size (default: 50).
  --jpeg-quality N            RGB JPEG quality in PPO Zarr (default: 95).
  --return-gamma FLOAT        Merged PPO return gamma (default: 0.99).
  --python EXECUTABLE         Default: python (active arbim environment).
  --dry-run                   Preflight and print planned outputs; write nothing.
  -h, --help                  Show this help.

Outputs:
  NEW_DIRECTORY/rollout_lerobot
  NEW_DIRECTORY/rollout_reward_v2.zarr
  NEW_DIRECTORY/merged_il_lerobot
  NEW_DIRECTORY/merged_offline_rl.zarr
  NEW_DIRECTORY/configs/rollout_to_zarr.yaml
  NEW_DIRECTORY/configs/merge_zarr.yaml

Notes:
  1. The previous Zarr is reused, not regenerated.
  2. Only Reward-valid successful rollouts are used by this unified IL+RL flow.
  3. Use a unique output-root each round; preserve the Staging NPY as archive.
  4. Staging does not record measured FPS: verify the MuJoCo ros_rate separately.
HELP
}

STAGING=""
PREV_LEROBOT=""
PREV_ZARR=""
OUTPUT_ROOT=""
ZARR_DEPTH="auto"
INVALID_REWARD="error"
EXPECTED_ROLLOUTS=""
EXPECTED_FPS=""
TASK="ARBiM Data Wheel"
ROLLOUT_REPO_ID="arbim/flywheel_rollout"
MERGED_REPO_ID="arbim/flywheel_merged"
BATCH_SIZE="50"
JPEG_QUALITY="95"
RETURN_GAMMA="0.99"
PYTHON_BIN="python"
DRY_RUN=0

need_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staging|--staging-dir) need_value "$@"; STAGING="$2"; shift 2 ;;
    --previous-lerobot|--prev-lerobot) need_value "$@"; PREV_LEROBOT="$2"; shift 2 ;;
    --previous-zarr|--prev-zarr) need_value "$@"; PREV_ZARR="$2"; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --zarr-depth) need_value "$@"; ZARR_DEPTH="$2"; shift 2 ;;
    --invalid-reward) need_value "$@"; INVALID_REWARD="$2"; shift 2 ;;
    --expected-rollouts) need_value "$@"; EXPECTED_ROLLOUTS="$2"; shift 2 ;;
    --expected-fps) need_value "$@"; EXPECTED_FPS="$2"; shift 2 ;;
    --task) need_value "$@"; TASK="$2"; shift 2 ;;
    --rollout-repo-id) need_value "$@"; ROLLOUT_REPO_ID="$2"; shift 2 ;;
    --merged-repo-id) need_value "$@"; MERGED_REPO_ID="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --jpeg-quality) need_value "$@"; JPEG_QUALITY="$2"; shift 2 ;;
    --return-gamma) need_value "$@"; RETURN_GAMMA="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for item in "$STAGING" "$PREV_LEROBOT" "$PREV_ZARR" "$OUTPUT_ROOT"; do
  [[ -n "$item" ]] || { echo "Four required paths must be provided." >&2; usage >&2; exit 2; }
done
[[ "$ZARR_DEPTH" == "auto" || "$ZARR_DEPTH" == "on" || "$ZARR_DEPTH" == "off" ]] ||
  { echo "Invalid --zarr-depth: $ZARR_DEPTH" >&2; exit 2; }
[[ "$INVALID_REWARD" == "error" || "$INVALID_REWARD" == "skip" ]] ||
  { echo "Invalid --invalid-reward: $INVALID_REWARD" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null ||
  { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/post_training/src:$REPO_ROOT/third_party/lerobot/src"

# Validate all inputs BEFORE creating the destination. Only trusted Staging
# files should be supplied (np.load(..., allow_pickle=True)).
PREFLIGHT="$("$PYTHON_BIN" - \
  "$STAGING" "$PREV_LEROBOT" "$PREV_ZARR" "$OUTPUT_ROOT" \
  "$ZARR_DEPTH" "$INVALID_REWARD" "$EXPECTED_ROLLOUTS" \
  "$EXPECTED_FPS" "$BATCH_SIZE" "$JPEG_QUALITY" "$RETURN_GAMMA" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
import zarr

(staging_arg, il_arg, rl_arg, output_arg, depth_mode, invalid_mode,
 expected_rollouts, expected_fps, batch_size, quality, gamma) = sys.argv[1:]
staging, old_il, old_rl, dest = (
    Path(p).expanduser().resolve()
    for p in (staging_arg, il_arg, rl_arg, output_arg)
)
if dest.exists():
    raise FileExistsError(f"Output directory already exists: {dest}")
for src in (staging, old_il, old_rl):
    if dest == src or dest in src.parents or src in dest.parents:
        raise ValueError(f"Output directory must not contain, or be inside, input: {src}")
if not (old_il / "meta" / "info.json").is_file():
    raise FileNotFoundError(f"Previous LeRobot meta/info.json missing: {old_il}")
if not old_rl.is_dir():
    raise FileNotFoundError(f"Previous Zarr missing: {old_rl}")
if not staging.exists():
    raise FileNotFoundError(f"Staging not found: {staging}")

info = json.loads((old_il / "meta" / "info.json").read_text())
depth_keys = ("observation.depth_h", "observation.depth_l", "observation.depth_r")
if not all(key in info.get("features", {}) for key in depth_keys):
    raise ValueError("Previous LeRobot is not the full-Depth archive (3 Depth cameras required)")
fps = float(info["fps"])
if not math.isfinite(fps) or fps <= 0 or not fps.is_integer():
    raise ValueError(f"Invalid previous LeRobot FPS: {fps}")
if expected_fps and fps != float(expected_fps):
    raise ValueError(f"LeRobot FPS={fps:g} differs from expected {expected_fps}")
il_frames = int(info["total_frames"])
il_episodes = int(info["total_episodes"])

old = zarr.open_group(str(old_rl), mode="r")
if old.attrs.get("rgb_storage") != "jpeg_bytes_v1":
    raise ValueError("Previous Zarr is not JPEG-backed ARBiM Offline RL data")
if old.attrs.get("reward_mode") != "last_gripper_release_sparse":
    raise ValueError("Previous Zarr must use Reward V2: last_gripper_release_sparse")
data = old["data"]
depth_names = ("head_depth", "wrist_left_depth", "wrist_right_depth")
present = [k for name in depth_names for k in (name, f"next_{name}") if k in data]
if len(present) not in (0, 6):
    raise ValueError(f"Previous Zarr has partial Depth fields: {present}")
old_depth = len(present) == 6
if depth_mode != "auto" and (depth_mode == "on") != old_depth:
    raise ValueError("--zarr-depth differs from previous Zarr; merge cannot mix Depth modes")
use_depth = old_depth if depth_mode == "auto" else depth_mode == "on"
rl_frames = int(data["action"].shape[0])
rl_ends = np.asarray(old["meta"]["episode_ends"][:])
rl_episodes = len(rl_ends)
if rl_episodes < 1 or int(rl_ends[-1]) != rl_frames:
    raise ValueError("Previous Zarr episode boundaries are invalid")
if (il_frames, il_episodes) != (rl_frames, rl_episodes):
    raise ValueError(
        "Previous merged LeRobot and Zarr have different frame/episode counts: "
        f"IL={il_episodes}/{il_frames}, RL={rl_episodes}/{rl_frames}; "
        "check that both paths belong to the same round"
    )
for label, number, lo, hi in (
    ("batch-size", int(batch_size), 1, 100000),
    ("jpeg-quality", int(quality), 1, 100),
):
    if not lo <= number <= hi:
        raise ValueError(f"Invalid {label}: {number}")
if not math.isfinite(float(gamma)) or not 0 < float(gamma) <= 1:
    raise ValueError("return-gamma must be in (0,1]")

files = sorted(staging.glob("*_staging.npy")) if staging.is_dir() else [staging]
if not files:
    raise ValueError(f"No Staging files found: {staging}")
valid = 0
invalid = []
new_frames = 0
for path in files:
    sample = np.load(path, allow_pickle=True).item()
    if sample.get("__format__") != "arbim_rollout_staging_v2" or not sample.get("success"):
        raise ValueError(f"Not a successful Staging V2 file: {path}")
    if sample.get("reward_mode") != "last_gripper_release_sparse":
        raise ValueError(f"Staging Reward mode mismatch: {path}")
    if sample.get("use_depth") is not True:
        raise ValueError(f"All 3 Depth cameras are mandatory for Staging: {path}")
    n = int(sample["num_steps"])
    if np.asarray(sample["actions"]).shape != (n, 16) or len(sample["observations"]) != n + 1:
        raise ValueError(f"Invalid action/observation lengths: {path}")
    if sample.get("reward_status") != "valid":
        invalid.append(path.name)
        continue
    r = np.asarray(sample.get("reward"), dtype=np.float32)
    points = np.flatnonzero(r)
    if r.shape != (n,) or len(points) != 1 or r[points[0]] != 1 or points[0] >= n - 1:
        raise ValueError(f"Invalid nonterminal Reward V2: {path}")
    valid += 1
    new_frames += n
if invalid and invalid_mode == "error":
    raise ValueError(
        f"{len(invalid)} successful Staging episode(s) have invalid Reward V2: "
        + ", ".join(invalid)
        + ". Use --invalid-reward skip to exclude them from BOTH IL and RL."
    )
if not valid:
    raise ValueError("No Reward-valid Staging episodes remain")
if expected_rollouts and valid != int(expected_rollouts):
    raise ValueError(f"Expected {expected_rollouts} Reward-valid rollouts, got {valid}")
print(
    f"Preflight PASS: history={il_episodes} episodes/{il_frames} frames, "
    f"new valid={valid} episodes/{new_frames} frames, "
    f"invalid skipped={len(invalid)}, FPS={fps:g}, "
    f"Zarr Depth={'on' if use_depth else 'off'}",
    file=sys.stderr,
)
print("true" if use_depth else "false")
PY
)"

ROLLOUT_IL="$OUTPUT_ROOT/rollout_lerobot"
ROLLOUT_ZARR="$OUTPUT_ROOT/rollout_reward_v2.zarr"
MERGED_IL="$OUTPUT_ROOT/merged_il_lerobot"
MERGED_ZARR="$OUTPUT_ROOT/merged_offline_rl.zarr"
CFG_DIR="$OUTPUT_ROOT/configs"

echo "Output 1: $ROLLOUT_IL"
echo "Output 2: $ROLLOUT_ZARR"
echo "Output 3: $MERGED_IL"
echo "Output 4: $MERGED_ZARR"
echo "PPO Zarr Depth: $PREFLIGHT (LeRobot always includes 3 Depth cameras)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN PASSED: inputs validated, no files created."
  exit 0
fi

mkdir -p "$CFG_DIR"
trap 'echo "FAILED at line $LINENO. Partial results, if any, remain in $OUTPUT_ROOT; use a NEW output root when retrying." >&2' ERR

echo "[1/5] Staging -> Rollout LeRobot (Reward V2-valid subset, full Depth)"
"$PYTHON_BIN" post_training/src/post_rl/data/staging_to_lerobot.py \
  "$STAGING" \
  --template-lerobot-root "$PREV_LEROBOT" \
  --output-root "$ROLLOUT_IL" \
  --repo-id "$ROLLOUT_REPO_ID" \
  --task "$TASK" \
  --require-valid-reward

# Write actual resolved configs into output-root for reproducibility.
"$PYTHON_BIN" - "$ROLLOUT_IL" "$ROLLOUT_ZARR" "$PREV_ZARR" \
  "$MERGED_ZARR" "$CFG_DIR" "$PREFLIGHT" \
  "$BATCH_SIZE" "$JPEG_QUALITY" "$RETURN_GAMMA" <<'PY'
import sys
from pathlib import Path
import yaml

(new_il, new_zarr, old_zarr, merged_zarr, cfg_dir,
 use_depth, batch_size, quality, gamma) = sys.argv[1:]
cfg = Path(cfg_dir)
convert = {
    "mode": "lerobot_to_zarr",
    "lerobot_root": str(Path(new_il).resolve()),
    "zarr_output_path": str(Path(new_zarr).resolve()),
    "overwrite": False,
    "use_depth": use_depth == "true",
    "stream_batch_size": int(batch_size),
    "jpeg_quality": int(quality),
    "reward_mode": "last_gripper_release_sparse",
    "reward_source": "recorded",
}
merge = {
    "mode": "merge_zarr",
    "overwrite": False,
    "zarr_output_path": str(Path(merged_zarr).resolve()),
    "return_gamma": float(gamma),
    "zarr_sources": [
        {"name": "previous_merged_rl", "path": str(Path(old_zarr).resolve())},
        {"name": "new_rollout", "path": str(Path(new_zarr).resolve())},
    ],
}
for filename, data in (("rollout_to_zarr.yaml", convert), ("merge_zarr.yaml", merge)):
    (cfg / filename).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

echo "[2/5] Rollout LeRobot -> new PPO Zarr (recorded Reward V2)"
"$PYTHON_BIN" post_training/src/post_rl/data/data_prepare.py \
  --config "$CFG_DIR/rollout_to_zarr.yaml"

echo "[3/5] Previous LeRobot + new Rollout LeRobot -> merged IL LeRobot"
"$PYTHON_BIN" post_training/src/post_rl/data/lerobot_merge.py \
  "$PREV_LEROBOT" "$ROLLOUT_IL" \
  --output-root "$MERGED_IL" \
  --repo-id "$MERGED_REPO_ID" \
  --fallback-task "$TASK"

echo "[4/5] Previous PPO Zarr + new Rollout Zarr -> merged PPO Zarr"
"$PYTHON_BIN" post_training/src/post_rl/data/data_prepare.py \
  --config "$CFG_DIR/merge_zarr.yaml"

echo "[5/5] Verify frame counts, Reward V2 and Depth contracts"
"$PYTHON_BIN" - "$PREV_LEROBOT" "$PREV_ZARR" \
  "$ROLLOUT_IL" "$ROLLOUT_ZARR" "$MERGED_IL" "$MERGED_ZARR" "$PREFLIGHT" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import zarr

old_il, old_rl, new_il, new_rl, merged_il, merged_rl = map(Path, sys.argv[1:7])
use_depth = sys.argv[7] == "true"
def lerobot_counts(p):
    info = json.loads((p / "meta" / "info.json").read_text())
    assert all(k in info["features"] for k in (
        "observation.depth_h", "observation.depth_l", "observation.depth_r"
    )), f"Missing Depth in LeRobot {p}"
    return int(info["total_episodes"]), int(info["total_frames"])
def zarr_counts(p):
    z = zarr.open_group(str(p), mode="r")
    frames = int(z["data"]["action"].shape[0])
    ends = np.asarray(z["meta"]["episode_ends"][:])
    assert len(ends) > 0 and int(ends[-1]) == frames
    assert z.attrs.get("reward_mode") == "last_gripper_release_sparse"
    actual_depth = all(k in z["data"] for name in (
        "head_depth", "wrist_left_depth", "wrist_right_depth"
    ) for k in (name, f"next_{name}"))
    assert actual_depth == use_depth, f"Wrong Zarr Depth mode at {p}"
    return len(ends), frames
i0, i1, im = map(lerobot_counts, (old_il, new_il, merged_il))
z0, z1, zm = map(zarr_counts, (old_rl, new_rl, merged_rl))
assert im == (i0[0] + i1[0], i0[1] + i1[1]), (i0, i1, im)
assert zm == (z0[0] + z1[0], z0[1] + z1[1]), (z0, z1, zm)
assert i1 == z1 and im == zm, (i1, z1, im, zm)
# Verify preserved Reward V2 positions in the appended PPO segment.
rollout = zarr.open_group(str(new_rl), mode="r")
combined = zarr.open_group(str(merged_rl), mode="r")
rollout_rewards = np.asarray(rollout["data"]["reward"][:])
combined_rewards = np.asarray(combined["data"]["reward"][-z1[1]:])
np.testing.assert_array_equal(combined_rewards, rollout_rewards)

# LeRobot v0.4.2 shard mapping has caused silent video-boundary corruption
# before: actually decode RGB/Depth near the newly appended dataset boundary.
from post_rl.data.data_prepare import load_lerobot_dataset
dataset = load_lerobot_dataset(merged_il)
video_keys = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
    "observation.depth_h",
    "observation.depth_l",
    "observation.depth_r",
)
for index in (0, i0[1] - 1, i0[1], im[1] - 1):
    frame = dataset[index]
    for key in video_keys:
        image = np.asarray(frame[key])
        assert image.ndim == 3 and image.shape[0] == 3
        assert np.isfinite(image).all(), (index, key)
print(f"DATA WHEEL NEXT ROUND PASSED: {im[0]} episodes, {im[1]} frames")
PY

echo "Rollout LeRobot: $ROLLOUT_IL"
echo "Rollout PPO Zarr: $ROLLOUT_ZARR"
echo "Merged IL LeRobot: $MERGED_IL"
echo "Merged PPO Zarr: $MERGED_ZARR"
