"""
Wrapper around `python -m verl.model_merger merge` that stubs out the
`deepspeed` package before any verl/transformers/accelerate import.

Why:
    transformers.modeling_utils.save_pretrained() -> unwrap_model() ->
    accelerate.utils.other.extract_model_from_parallel() does
        from deepspeed import DeepSpeedEngine
        if isinstance(model, DeepSpeedEngine): ...
    Importing `deepspeed` triggers its op_builder which requires CUDA_HOME
    (system CUDA toolkit, not the conda runtime). On boxes without
    `nvcc` this raises MissingCUDAException and the merge dies.

    We don't actually need DeepSpeed for FSDP-shard -> HF conversion, so
    we install a no-op stub module BEFORE the first verl import. The
    isinstance check then returns False and the unwrap path falls through
    to the standard transformers branch.

Usage::

    python scripts/merge_fipo_ckpt.py \
        --local_dir /path/to/global_step_X/actor \
        --target_dir models/fipo_step_X_merged
"""
from __future__ import annotations

import sys
import types


def _install_deepspeed_stub() -> None:
    """Install a no-op deepspeed stub.

    accelerate.utils.imports.is_deepspeed_available() calls
    importlib.util.find_spec("deepspeed"), which requires the module to
    have a non-None __spec__. A bare types.ModuleType has __spec__=None
    and triggers `ValueError: deepspeed.__spec__ is None`. We attach a
    minimal ModuleSpec so find_spec returns truthy, then the downstream
    `from deepspeed import DeepSpeedEngine` resolves to our stub class
    and `isinstance(model, DeepSpeedEngine)` is always False.
    """
    if "deepspeed" in sys.modules:
        return
    from importlib.machinery import ModuleSpec

    ds = types.ModuleType("deepspeed")
    ds.__spec__ = ModuleSpec("deepspeed", loader=None)

    class _DeepSpeedEngine:  # noqa: D401 - intentional stub
        """Stub class so isinstance(model, DeepSpeedEngine) is always False."""

    ds.DeepSpeedEngine = _DeepSpeedEngine
    sys.modules["deepspeed"] = ds


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FSDP shard -> HF merger (deepspeed-free)")
    parser.add_argument("--local_dir", required=True, help="Path to global_step_X/actor")
    parser.add_argument("--target_dir", required=True, help="Output HF model dir")
    parser.add_argument(
        "--use_cpu_initialization",
        action="store_true",
        default=True,
        help="Avoid putting the full materialised model on GPU during merge",
    )
    parser.add_argument("--no-cpu-init", dest="use_cpu_initialization", action="store_false")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    _install_deepspeed_stub()

    # Import verl AFTER the stub is in sys.modules
    from verl.model_merger.__main__ import main as merger_main

    sys.argv = [
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        args.local_dir,
        "--target_dir",
        args.target_dir,
    ]
    if args.use_cpu_initialization:
        sys.argv.append("--use_cpu_initialization")
    if args.trust_remote_code:
        sys.argv.append("--trust-remote-code")

    print(f"[merge_fipo_ckpt] launching: {' '.join(sys.argv)}", flush=True)
    merger_main()


if __name__ == "__main__":
    main()
