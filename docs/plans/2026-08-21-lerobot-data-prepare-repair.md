# LeRobot Data Preparation Repair Implementation Plan

**Goal:** Make the standalone LeRobot-to-NPY conversion function save a validated output and preserve episode boundaries correctly.

**Architecture:** Keep the new LeRobot-oriented output schema. Read every dataset frame once, group frames by their actual `episode_index`, and normalize tensors to NumPy arrays before serializing.

**Tech Stack:** Python, NumPy, LeRobotDataset.

---

### Task 1: Repair the converter

**Files:**
- Modify: `RL Post Training/data/data_prepare.py:48-156`

**Step 1:** Load a local dataset using its local root and optional Hub repository id.

**Step 2:** Group input frames by their actual episode index in one dataset pass.

**Step 3:** Validate required observations, actions, RGB/depth modalities, and convert values to NumPy.

**Step 4:** Save `processed` and use standard logging without undeclared dependencies.

### Task 2: Static verification

**Files:**
- Verify: `RL Post Training/data/data_prepare.py`

**Step 1:** Parse the module with Python AST only; do not execute the target script or load a dataset.

**Step 2:** Inspect the diff and run GitNexus change detection.
