from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for path in (
    REPO_ROOT,
    REPO_ROOT / "third_party" / "lerobot" / "src",
    REPO_ROOT / "post_training" / "src",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches  # noqa: E402,F401
from kuavo_deploy.config import load_kuavo_config  # noqa: E402
from kuavo_deploy.src.scripts.script_auto_test import ArmMove  # noqa: E402
from post_rl.critic.networks import ACTCriticEncoder  # noqa: E402
from post_rl.data.latent_cache import FrozenLatentCache  # noqa: E402


def _resolve_processor_dir(checkpoint: pathlib.Path) -> pathlib.Path:
    for candidate in (checkpoint, checkpoint.parent):
        required = (
            candidate / "policy_preprocessor.json",
            candidate / "policy_postprocessor.json",
        )
        if all(path.is_file() for path in required):
            return candidate
    raise FileNotFoundError(f"Could not resolve processor bundle for {checkpoint}")


def _processor_fingerprint(processor_dir: pathlib.Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        [*processor_dir.glob("policy_preprocessor*"), *processor_dir.glob("policy_postprocessor*")],
        key=lambda path: path.name,
    )
    for path in files:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_action_norm(processor_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with (processor_dir / "policy_preprocessor.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    normalizer = next(
        (
            step
            for step in config.get("steps", [])
            if step.get("registry_name") == "normalizer_processor"
            or str(step.get("class", "")).endswith("NormalizerProcessorStep")
        ),
        None,
    )
    if normalizer is None or not normalizer.get("state_file"):
        raise KeyError("policy_preprocessor.json does not define a normalizer state_file")
    state = load_file(str(processor_dir / normalizer["state_file"]), device="cpu")
    mean = state["action.mean"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    std = state["action.std"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    if np.any(std <= 0):
        raise ValueError("action.std must be strictly positive")
    return mean, std


def _fingerprint_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _pool_cache_rows(cache, indices: np.ndarray, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        tokens = np.asarray(cache.obs[batch_indices], dtype=np.float32)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected cached ACT latent [B,S,D], got {tokens.shape}")
        rows.append(tokens.mean(axis=1, dtype=np.float32))
    if not rows:
        raise RuntimeError("No cached latent rows selected")
    return np.concatenate(rows, axis=0)


def _knn_scores(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    batch_size: int,
) -> np.ndarray:
    scores = []
    with torch.inference_mode():
        for start in range(0, query.shape[0], batch_size):
            batch = query[start : start + batch_size]
            distances = torch.cdist(batch, reference, p=2)
            nearest = torch.topk(distances, k=k, dim=1, largest=False).values
            scores.append(nearest.mean(dim=1).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float64, copy=False)


def _prepare_reference(
    cache: FrozenLatentCache,
    *,
    device: torch.device,
    reference_size: int,
    calibration_size: int,
    k: int,
    seed: int,
    cache_batch_size: int,
    knn_batch_size: int,
):
    total = len(cache.obs)
    needed = reference_size + calibration_size
    if reference_size < k:
        raise ValueError(f"reference_size={reference_size} must be >= k={k}")
    if needed > total:
        raise ValueError(
            f"reference_size + calibration_size = {needed} exceeds cache size {total}"
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(total, size=needed, replace=False)
    reference_indices = np.sort(selected[:reference_size])
    calibration_indices = np.sort(selected[reference_size:])

    print(
        f"Pooling cached demonstration latents: reference={reference_size}, "
        f"calibration={calibration_size}, total_cache={total}"
    )
    reference_np = _pool_cache_rows(cache, reference_indices, cache_batch_size)
    calibration_np = _pool_cache_rows(cache, calibration_indices, cache_batch_size)

    feature_mean = reference_np.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = reference_np.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, np.float32(1e-6))
    reference_z = (reference_np - feature_mean) / feature_std
    calibration_z = (calibration_np - feature_mean) / feature_std

    reference = torch.from_numpy(reference_z).to(device=device, dtype=torch.float32)
    calibration = torch.from_numpy(calibration_z).to(device=device, dtype=torch.float32)
    calibration_scores = _knn_scores(
        calibration,
        reference,
        k=k,
        batch_size=knn_batch_size,
    )
    calibration_scores.sort()
    quantiles = {
        "p50": float(np.percentile(calibration_scores, 50)),
        "p90": float(np.percentile(calibration_scores, 90)),
        "p95": float(np.percentile(calibration_scores, 95)),
        "p99": float(np.percentile(calibration_scores, 99)),
        "max": float(calibration_scores[-1]),
    }
    return reference, feature_mean, feature_std, calibration_scores, quantiles


class LatentOODPolicy:
    """Execute one ACT policy while scoring every live observation against demo latent cache."""

    def __init__(
        self,
        policy,
        encoder: ACTCriticEncoder,
        *,
        reference: torch.Tensor,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        calibration_scores: np.ndarray,
        k: int,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        csv_path: pathlib.Path,
        log_every: int,
        execute_name: str,
    ) -> None:
        self.policy = policy
        self.encoder = encoder
        self.config = policy.config
        self.reference = reference
        self.feature_mean = torch.from_numpy(feature_mean).to(reference.device)
        self.feature_std = torch.from_numpy(feature_std).to(reference.device)
        self.calibration_scores = calibration_scores
        self.k = int(k)
        self.action_mean = np.asarray(action_mean, dtype=np.float64)
        self.action_std = np.asarray(action_std, dtype=np.float64)
        self.execute_name = execute_name
        self.log_every = max(int(log_every), 1)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self._file = csv_path.open("w", newline="", buffering=1)
        self._writer = None
        self._episode = -1
        self._step = 0
        self._has_steps = False
        self._prev_action = None
        self._scores = []
        self._percentiles = []

    def eval(self):
        self.policy.eval()
        self.encoder.eval()
        return self

    def to(self, device):
        self.policy.to(device)
        self.encoder.to(device)
        return self

    def reset(self):
        self.policy.reset()
        if self._episode < 0:
            self._episode = 0
        elif self._has_steps:
            self._episode += 1
        self._step = 0
        self._has_steps = False
        self._prev_action = None

    @torch.inference_mode()
    def select_action(self, observation):
        action = self.policy.select_action(observation)
        pooled_latent = self.encoder(observation)
        if pooled_latent.ndim != 2 or pooled_latent.shape[0] != 1:
            raise RuntimeError(
                f"Expected mean-pooled ACT latent [1,D], got {tuple(pooled_latent.shape)}"
            )
        if pooled_latent.shape[1] != self.reference.shape[1]:
            raise RuntimeError(
                f"Live latent dim {pooled_latent.shape[1]} != reference dim {self.reference.shape[1]}"
            )

        query = (pooled_latent.float() - self.feature_mean) / self.feature_std
        distances = torch.cdist(query, self.reference, p=2)
        nearest = torch.topk(distances, k=self.k, dim=1, largest=False).values
        knn_distance = float(nearest.mean().item())
        percentile = float(
            100.0
            * np.searchsorted(self.calibration_scores, knn_distance, side="right")
            / len(self.calibration_scores)
        )

        action_norm = action[0].detach().float().cpu().numpy().astype(np.float64)
        action_phys = action_norm * self.action_std + self.action_mean
        step_delta = (
            float("nan")
            if self._prev_action is None
            else float(np.linalg.norm(action_phys - self._prev_action))
        )
        row = {
            "episode": self._episode,
            "step": self._step,
            "executed_policy": self.execute_name,
            "latent_knn_distance": knn_distance,
            "latent_demo_percentile": percentile,
            "latent_above_demo_p95": int(percentile >= 95.0),
            "latent_above_demo_p99": int(percentile >= 99.0),
            "executed_step_delta_l2": step_delta,
        }
        for index, value in enumerate(action_phys):
            row[f"executed_action_{index}"] = float(value)

        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

        self._scores.append(knn_distance)
        self._percentiles.append(percentile)
        if self._step % self.log_every == 0:
            print(
                f"[latent-ood] episode={self._episode} step={self._step} "
                f"policy={self.execute_name} knn={knn_distance:.6g} "
                f"demo_percentile={percentile:.2f}% step_delta={step_delta:.6g}"
            )

        self._prev_action = action_phys.copy()
        self._step += 1
        self._has_steps = True
        return action

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        self._file.close()
        print(f"\nLatent OOD CSV: {self.csv_path}")
        if self._scores:
            print(
                f"Mean latent kNN distance: {float(np.mean(self._scores)):.6g}; "
                f"mean demo percentile: {float(np.mean(self._percentiles)):.2f}%"
            )
            print(
                f"Max demo percentile: {float(np.max(self._percentiles)):.2f}%; "
                f"frames >=P95: {int(np.sum(np.asarray(self._percentiles) >= 95.0))}/"
                f"{len(self._percentiles)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score live Kuavo rollout observations against the already-cached demonstration ACT "
            "latent distribution. The metric uses the same mean token readout as the V1 critic, "
            "feature standardization from a demo reference split, and kNN calibration on held-out "
            "demo latents."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--il-checkpoint", required=True)
    parser.add_argument("--postrl-checkpoint", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--execute", choices=("postrl", "il"), default="postrl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--reference-size", type=int, default=4096)
    parser.add_argument("--calibration-size", type=int, default=1024)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--knn-batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="post_training/outputs/latent_ood")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if args.reference_size < 1 or args.calibration_size < 1:
        raise ValueError("reference/calibration sizes must be >= 1")
    if args.cache_batch_size < 1 or args.knn_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")

    il_checkpoint = pathlib.Path(args.il_checkpoint).expanduser().resolve()
    postrl_checkpoint = pathlib.Path(args.postrl_checkpoint).expanduser().resolve()
    for label, path in (("IL", il_checkpoint), ("Post-RL", postrl_checkpoint)):
        if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"{label} deterministic checkpoint is incomplete: {path}")

    il_processor = _resolve_processor_dir(il_checkpoint)
    postrl_processor = _resolve_processor_dir(postrl_checkpoint)
    il_fp = _processor_fingerprint(il_processor)
    postrl_fp = _processor_fingerprint(postrl_processor)
    if il_fp != postrl_fp:
        raise RuntimeError("IL and Post-RL processor bundles differ")
    action_mean, action_std = _load_action_norm(il_processor)

    config_path = pathlib.Path(args.config).expanduser().resolve()
    config = load_kuavo_config(config_path)
    if args.device is not None:
        config.inference.device = args.device
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError("--episodes must be >= 1")
        config.inference.eval_episodes = int(args.episodes)
    config.inference.policy_type = "act"
    execute_checkpoint = postrl_checkpoint if args.execute == "postrl" else il_checkpoint
    config.inference.pretrained_path = str(execute_checkpoint)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.inference.method = f"{config.inference.method}_latent_ood_{args.execute}"
    config.inference.timestamp = f"{config.inference.timestamp}_{stamp}"
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "latent_ood.csv"

    cache_dir = pathlib.Path(args.latent_cache_dir).expanduser().resolve()
    cache = FrozenLatentCache(str(cache_dir))
    cache_metadata = dict(cache.metadata)

    _arm = ArmMove(config)
    from kuavo_deploy.src.eval import sim_auto_test as sim_eval

    device = torch.device(config.inference.device)
    base_setup_policy = sim_eval.setup_policy
    policy = base_setup_policy(execute_checkpoint, "act", config.inference, device)

    # The V1 cache was produced through ACTCriticEncoder / ACTStateEncoder. Reuse exactly
    # that representation and mean-token readout. copy_model=False avoids a second ACT copy.
    encoder = ACTCriticEncoder(policy.model, copy_model=False).to(device).eval()
    encoder_sha = _fingerprint_module(encoder)
    cache_encoder_sha = str(cache_metadata.get("encoder_sha256", ""))
    if not cache_encoder_sha:
        raise RuntimeError("Latent cache metadata does not contain encoder_sha256")
    if encoder_sha != cache_encoder_sha:
        raise RuntimeError(
            "Live ACT encoder does not match latent cache encoder: "
            f"live={encoder_sha[:12]}..., cache={cache_encoder_sha[:12]}..."
        )
    if cache.obs.shape[-1] != encoder.output_dim:
        raise RuntimeError(
            f"Cache latent dim {cache.obs.shape[-1]} != ACT encoder dim {encoder.output_dim}"
        )

    reference, feature_mean, feature_std, calibration_scores, quantiles = _prepare_reference(
        cache,
        device=device,
        reference_size=args.reference_size,
        calibration_size=args.calibration_size,
        k=args.k,
        seed=args.seed,
        cache_batch_size=args.cache_batch_size,
        knn_batch_size=args.knn_batch_size,
    )

    summary = {
        "config": str(config_path),
        "il_checkpoint": str(il_checkpoint),
        "postrl_checkpoint": str(postrl_checkpoint),
        "executed_policy": args.execute,
        "executed_checkpoint": str(execute_checkpoint),
        "processor_fingerprint": il_fp,
        "latent_cache_dir": str(cache_dir),
        "latent_cache_metadata": cache_metadata,
        "encoder_sha256": encoder_sha,
        "readout": "mean_over_ACT_encoder_tokens",
        "distance": "euclidean_after_per_feature_demo_standardization",
        "knn_k": int(args.k),
        "reference_size": int(args.reference_size),
        "calibration_size": int(args.calibration_size),
        "calibration_knn_quantiles": quantiles,
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    diagnostic = LatentOODPolicy(
        policy,
        encoder,
        reference=reference,
        feature_mean=feature_mean,
        feature_std=feature_std,
        calibration_scores=calibration_scores,
        k=args.k,
        action_mean=action_mean,
        action_std=action_std,
        csv_path=csv_path,
        log_every=args.log_every,
        execute_name=args.execute,
    ).eval().to(device)

    def _setup(_pretrained_path, policy_type, cfg, device=device):
        if policy_type != "act":
            raise ValueError("Latent OOD diagnostic supports ACT only")
        return diagnostic

    sim_eval.setup_policy = _setup
    try:
        print(f"Processor bundles are bit-identical: {il_fp[:12]}...")
        print(f"Latent cache encoder matches live ACT: {encoder_sha[:12]}...")
        print(f"Cache source dataset: {cache_metadata.get('source_dataset')}")
        print(
            "Demo calibration kNN: "
            f"P50={quantiles['p50']:.6g}, P95={quantiles['p95']:.6g}, "
            f"P99={quantiles['p99']:.6g}"
        )
        print(f"Executing {args.execute}; per-step latent diagnostics: {csv_path}")
        sim_eval.kuavo_eval_autotest(config)
    finally:
        sim_eval.setup_policy = base_setup_policy
        diagnostic.close()


if __name__ == "__main__":
    main()
