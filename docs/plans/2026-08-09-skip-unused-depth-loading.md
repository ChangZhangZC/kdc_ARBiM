# Skip Unused Depth Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent QuavoAct training from reading or decoding depth observations when `policy.custom.use_depth` is `false`, while preserving existing depth-enabled behavior.

**Architecture:** Filter depth features before policy/preprocessor construction and pass the excluded keys into the repository's dataset wrapper. The wrapper filters parquet columns and video timestamp queries so disabled depth data never reaches video decoding. Both single-process and Accelerate training entrypoints use the same helpers to avoid divergent behavior.

**Tech Stack:** Python, PyTorch `Dataset`/`DataLoader`, Hugging Face `datasets`, LeRobot v0.4.2, Hydra/OmegaConf, pytest.

---

### Task 1: Add regression tests for feature and video-key filtering

**Files:**
- Create: `tests/kuavo_train/test_depth_loading_filter.py`
- Modify: `kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py`

**Step 1: Write failing tests**

Add focused tests that verify:

```python
def test_disabled_depth_is_removed_from_policy_features(): ...
def test_enabled_depth_keeps_policy_features(): ...
def test_disabled_depth_is_not_in_video_timestamp_queries(): ...
def test_disabled_depth_column_is_removed_from_hf_dataset(): ...
```

Use lightweight fake metadata/HF dataset objects or mocks; do not require a real robot dataset or decode actual videos.

**Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=third_party/lerobot/src:. pytest -q tests/kuavo_train/test_depth_loading_filter.py
```

Expected: failure because the filtering helpers and dataset behavior do not exist yet.

### Task 2: Implement filtering in the dataset wrapper

**Files:**
- Modify: `kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py`
- Test: `tests/kuavo_train/test_depth_loading_filter.py`

**Step 1: Add a shared policy-feature filter**

Add a helper that returns a copy of policy features with `FeatureType.DEPTH` entries removed only when `use_depth` is false. It must also return or expose the excluded feature keys so the loader receives the same decision.

**Step 2: Extend `CustomLeRobotDataset`**

Accept an optional set of excluded feature keys. Before calling the parent initializer, store that set. Ensure loaded HF/parquet columns matching excluded keys are removed, and ensure video timestamp queries iterate only active video keys. Keep behavior identical when no keys are excluded.

**Step 3: Run the focused tests**

Run:

```bash
PYTHONPATH=third_party/lerobot/src:. pytest -q tests/kuavo_train/test_depth_loading_filter.py
```

Expected: all focused tests pass.

### Task 3: Wire both QuavoAct training entrypoints to the optimized loader

**Files:**
- Modify: `kuavo_train/train_policy.py`
- Modify: `kuavo_train/train_policy_with_accelerate.py`
- Modify: `kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py`
- Test: `tests/kuavo_train/test_depth_loading_filter.py`

**Step 1: Filter features before policy construction**

Read `policy.custom.use_depth` with a safe false default. Filter depth entries before constructing `input_features`, while leaving action/output features untouched.

**Step 2: Restrict delta timestamps to active policy features**

Build observation/action delta timestamps only for keys present in the active policy input/output feature dictionaries. Disabled depth keys must not enter `delta_indices`.

**Step 3: Use the custom dataset loader**

Replace direct `LeRobotDataset` construction in both training entrypoints with `CustomLeRobotDataset`, passing the excluded depth keys. Do not modify the vendored `third_party/lerobot` submodule.

**Step 4: Add source-level integration assertions**

Add a test that confirms both entrypoints use `CustomLeRobotDataset` and the shared filtering path, without importing or launching full training.

### Task 4: Verify the hotfix

**Files:**
- Verify: `kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py`
- Verify: `kuavo_train/train_policy.py`
- Verify: `kuavo_train/train_policy_with_accelerate.py`
- Verify: `tests/kuavo_train/test_depth_loading_filter.py`

**Step 1: Run focused regression tests**

```bash
PYTHONPATH=third_party/lerobot/src:. pytest -q tests/kuavo_train/test_depth_loading_filter.py
```

Expected: all tests pass.

**Step 2: Compile changed Python files**

```bash
python -m py_compile kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py kuavo_train/train_policy.py kuavo_train/train_policy_with_accelerate.py tests/kuavo_train/test_depth_loading_filter.py
```

Expected: exit code 0.

**Step 3: Inspect the final diff**

```bash
git diff --check
git diff --stat
```

Expected: no whitespace errors and changes limited to the planned files.
