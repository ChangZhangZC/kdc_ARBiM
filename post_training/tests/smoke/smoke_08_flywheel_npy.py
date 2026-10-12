"""Batch 04: staging -> Processed NPY V2 -> build_db smoke test."""

import pathlib
import sys
import tempfile
import unittest
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from kuavo_deploy.src.eval.rollout_recorder import RolloutRecorder, DEPTH_KEYS
from post_rl.data.rollout_to_npy import convert_staging_to_npy, PROCESSED_FORMAT


def obs(value, image_size=16, use_depth=False):
    sample = {"observation.state": np.full((1, 16), value, dtype=np.float32)}
    for name in (
        "observation.images.head_cam_h",
        "observation.images.wrist_cam_l",
        "observation.images.wrist_cam_r",
    ):
        pixels = np.full((1, 3, image_size, image_size), 0.25, dtype=np.float32)
        sample[name] = pixels
    if use_depth:
        depth = np.linspace(0, 1, image_size * image_size, dtype=np.float32)
        depth = depth.reshape(1, 1, image_size, image_size)
        for key in DEPTH_KEYS:
            sample[key] = depth.copy()
    return sample


def actions():
    result = np.zeros((14, 16), dtype=np.float32)
    result[3:6, 7] = 1.0
    result[4:8, 15] = 1.0
    return result


def save_rollout(folder, episode, *, valid=True, image_size=16, use_depth=False):
    rec = RolloutRecorder(folder, episode)
    rec.start_episode(obs(episode * 100, image_size, use_depth=use_depth))
    for step, command in enumerate(actions() if valid else np.zeros((7, 16))):
        rec.append_step(command, obs(episode * 100 + step + 1, image_size, use_depth=use_depth))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return rec.finish_episode(True)


class TestStagingToProcessed(unittest.TestCase):
    def test_two_episodes_and_invalid_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            stage = directory / "staging"
            first = save_rollout(stage, 1)
            invalid = save_rollout(stage, 2, valid=False)
            second = save_rollout(stage, 3)
            output = directory / "processed.npy"
            result = convert_staging_to_npy([stage], output, batch_size=5)
            self.assertEqual(result["num_frames"], 28)
            self.assertEqual(result["num_episodes"], 2)
            self.assertEqual(result["num_chunks"], 6)
            self.assertEqual(result["skipped_invalid"], 1)
            self.assertEqual(result["skipped_paths"], [str(invalid)])
            self.assertTrue(first.exists() and second.exists())
            self.assertTrue(output.exists())

            with output.open("rb") as file:
                header = np.load(file, allow_pickle=True).item()
                self.assertEqual(header["__format__"], PROCESSED_FORMAT)
                self.assertEqual(header["num_frames"], 28)
                self.assertEqual(header["num_episodes"], 2)
                self.assertEqual(header["rgb_storage"], "jpeg")
                self.assertFalse(header["use_depth"])
                self.assertEqual(header["state_shape"], (16,))
                self.assertEqual(header["action_shape"], (16,))
                rows = []
                while True:
                    try:
                        chunk = np.load(file, allow_pickle=True).item()
                    except EOFError:
                        break
                    rows.extend([
                        (
                            np.asarray(chunk["agent_pos"][i]),
                            np.asarray(chunk["action"][i]),
                            chunk["rgb"][i],
                            float(chunk["reward"][i]),
                            bool(chunk["done"][i]),
                            bool(chunk["timeout"][i]),
                        )
                        for i in range(len(chunk["action"]))
                    ])
                self.assertEqual(len(rows), 28)
            self.assertEqual([i for i, row in enumerate(rows) if row[4]], [13, 27])
            self.assertEqual([i for i, row in enumerate(rows) if row[3] == 1], [8, 22])
            self.assertEqual(float(rows[0][0][0]), 100)
            self.assertEqual(float(rows[13][0][0]), 113)
            self.assertEqual(float(rows[14][0][0]), 300)
            self.assertEqual(float(rows[27][0][0]), 313)
            self.assertEqual(float(rows[27][3]), 0.0)
            self.assertTrue(all(row[4] == row[5] for row in rows))
            self.assertTrue(all(set(row[2]) == set(header["rgb_shapes"]) for row in rows))
            self.assertEqual(rows[0][2]["observation.images.head_cam_h"],
                             np.load(first, allow_pickle=True).item()["observations"][0]
                             ["rgb"]["observation.images.head_cam_h"])

            with self.assertRaises(FileExistsError):
                convert_staging_to_npy([stage], output, batch_size=5)

    def test_no_valid_reward_produces_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            save_rollout(folder, 1, valid=False)
            output = folder / "output.npy"
            with self.assertRaisesRegex(ValueError, "No successful staging episodes"):
                convert_staging_to_npy([folder], output)
            self.assertFalse(output.exists())

    def test_corrupt_valid_reward_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            original = save_rollout(folder, 1)
            data = np.load(original, allow_pickle=True).item()
            data["reward"][-1] = 1
            with original.open("wb") as out:
                np.save(out, data, allow_pickle=True)
            with self.assertRaisesRegex(ValueError, "single nonterminal"):
                convert_staging_to_npy([folder], folder / "processed.npy")

    def test_cross_episode_camera_dimension_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            save_rollout(folder, 1, image_size=16)
            save_rollout(folder, 2, image_size=20)
            with self.assertRaisesRegex(ValueError, "mismatched camera"):
                convert_staging_to_npy([folder], folder / "processed.npy")


    def test_depth_manifest_frames_and_zarr_match_lerobot_format(self):
        from post_rl.data import data_prepare as dp
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            stage_dir = directory / "staging"
            stage_file = save_rollout(stage_dir, 1, use_depth=True)
            processed = directory / "depth.npy"
            convert_staging_to_npy([stage_dir], processed, batch_size=6, use_depth=True)
            staging = np.load(stage_file, allow_pickle=True).item()
            self.assertTrue(staging["use_depth"])

            header = dp._read_source_header(processed, use_depth=True)
            self.assertTrue(header["streamed"])
            for key in DEPTH_KEYS:
                self.assertEqual(header["depth_shapes"][key], (3, 16, 16))
                self.assertEqual(header["depth_dtypes"][key], np.dtype("float32"))
            frames = list(dp._iter_processed_frames(processed, use_depth=True))
            self.assertEqual(len(frames), 14)
            quantized = (
                np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
                * 255
            ).astype(np.uint8).astype(np.float32) / 255.0
            expected = np.repeat(quantized[None], 3, axis=0)
            for frame in frames:
                for key in DEPTH_KEYS:
                    depth = frame["depth"][key]
                    self.assertEqual(depth.shape, (3, 16, 16))
                    self.assertEqual(depth.dtype, np.float32)
                    np.testing.assert_array_equal(depth, expected)

            result = dp.run_build_db({
                "zarr_output_path": str(directory / "depth.zarr"),
                "teleop_sources": [{"name": "rollout", "path": str(processed),
                                    "kind": "rollout_npy"}],
                "use_depth": True, "overwrite": False, "stream_batch_size": 8,
            })
            self.assertEqual(result["num_frames"], 14)
            z = zarr.open_group(str(directory / "depth.zarr"), mode="r")
            for key in ("head_depth", "wrist_left_depth", "wrist_right_depth"):
                self.assertEqual(z["data"][key].shape, (14, 3, 16, 16))
                np.testing.assert_array_equal(z["data"][key][0], expected)
                self.assertEqual(z["data"][f"next_{key}"].shape, (14, 3, 16, 16))

    def test_depth_export_requires_all_rollouts_to_have_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            save_rollout(directory, 1, use_depth=False)
            save_rollout(directory, 2, use_depth=True)
            with self.assertRaisesRegex(ValueError, "Depth requested"):
                convert_staging_to_npy(
                    [directory], directory / "mixed.npy", use_depth=True
                )

    def test_rgb_only_export_from_depth_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            save_rollout(directory, 1, use_depth=True)
            output = directory / "rgb_only.npy"
            convert_staging_to_npy([directory], output)
            with output.open("rb") as handle:
                header = np.load(handle, allow_pickle=True).item()
                chunk = np.load(handle, allow_pickle=True).item()
            self.assertFalse(header["use_depth"])
            self.assertNotIn("depth_shapes", header)
            self.assertNotIn("depth", chunk)

    def test_incomplete_depth_staging_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            original = save_rollout(directory, 1, use_depth=True)
            staged = np.load(original, allow_pickle=True).item()
            del staged["observations"][0]["depth"]["observation.depth_h"]
            with original.open("wb") as fp:
                np.save(fp, staged, allow_pickle=True)
            with self.assertRaisesRegex(ValueError, "missing Depth camera"):
                convert_staging_to_npy([original], directory / "processed.npy", use_depth=True)

    def test_readers_and_build_db_compatibility(self):
        from post_rl.data import data_prepare as dp
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            save_rollout(folder, 1)
            save_rollout(folder, 2)
            output = folder / "processed.npy"
            convert_staging_to_npy([folder], output, batch_size=6)
            header = dp._read_source_header(output, use_depth=False)
            self.assertTrue(header["streamed"])
            self.assertEqual(header["stream_format"], dp.STREAM_NPY_FORMAT)
            rows = list(dp._iter_processed_frames(output, use_depth=False))
            self.assertEqual(len(rows), 28)
            self.assertEqual([i for i, row in enumerate(rows) if row["done"]], [13, 27])
            self.assertEqual([i for i, row in enumerate(rows) if row["reward"] == 1], [8, 22])
            result = dp.run_build_db({
                "zarr_output_path": str(folder / "merged.zarr"),
                "teleop_sources": [{"name": "rollout", "path": str(output)}],
                "use_depth": False, "overwrite": False, "stream_batch_size": 7,
                "jpeg_quality": 95,
            })
            self.assertEqual(result["num_frames"], 28)
            self.assertEqual(result["num_episodes"], 2)
            self.assertEqual(result["episode_ends"].tolist(), [14, 28])
            z = zarr.open_group(str(folder / "merged.zarr"), mode="r")
            done = np.asarray(z["data/done"][:], dtype=bool).reshape(-1)
            nxt = np.asarray(z["data/next_index"][:], dtype=np.int64)
            self.assertEqual(np.flatnonzero(done).tolist(), [13, 27])
            self.assertEqual(nxt[13], 13)
            self.assertEqual(nxt[14], 15)
            self.assertEqual(nxt[27], 27)
            np.testing.assert_allclose(z["data/return"][13], [0.0])


if __name__ == "__main__":
    unittest.main()
