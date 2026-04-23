"""
Auto-import FIPO patches into every Python process that has the project root
on sys.path. Picked up automatically by site.py during interpreter init,
including Ray worker subprocesses spawned by verl-latest.

Why this exists:
    verl-latest's policy_loss registry (verl.trainer.ppo.core_algos.POLICY_LOSS_REGISTRY)
    is module-level state. The driver imports our forward-ported `future_kl`
    via main_fipo.py, but Ray workers spawn fresh interpreters that do NOT
    inherit the driver's sys.modules — their registry is empty until they
    explicitly import our patch.

    sitecustomize is the cleanest non-invasive hook: site.py imports it on
    every interpreter startup, so every Ray worker registers `future_kl`
    before verl ever queries the registry.

We swallow ImportError so that running `python` in a non-VLM env (e.g.
plain shell) doesn't blow up.
"""
try:
    # future_kl_loss MUST be loaded by every actor worker via sitecustomize,
    # because verl's POLICY_LOSS_REGISTRY is per-process module state.
    from src.stage3_fipo.verl_patches import future_kl_loss  # noqa: F401
except Exception:  # noqa: BLE001 - never break interpreter startup
    pass

# reward_manager is intentionally NOT auto-imported here. It is loaded on
# demand by verl's `source=importlib` path only inside RewardLoopWorker
# processes, which avoids actor/ref/rollout workers needlessly downloading
# the sentence-encoder.


def _install_verl_worker_gpu_remap() -> None:
    """
    Patch verl.single_controller.base.worker.Worker._setup_env_cuda_visible_devices
    so each Ray actor pins itself to the correct **torch-relative** GPU index
    when CUDA_VISIBLE_DEVICES is non-contiguous (e.g. "1,2,3,4,5,6" to skip a
    flaky GPU 0).

    Why this exists:
        Upstream calls
            get_torch_device().set_device(int(local_rank))
        where `local_rank` is the *physical* GPU id reported by
        ray.get_runtime_context().get_accelerator_ids()["GPU"][0] (e.g. "6").
        torch.cuda only sees indices 0..N-1 inside the CUDA_VISIBLE_DEVICES
        mask, so set_device(6) either:
          - raises "invalid device ordinal" when 6 >= device_count, or worse
          - silently lands on the wrong physical card when 6 < device_count,
            causing two ranks to collide on the same GPU
            (NCCL "Duplicate GPU detected").

    The fix maps physical id -> torch index via CUDA_VISIBLE_DEVICES.split(","),
    leaving the rest of the upstream logic untouched. Only the `set_device`
    call site changes, so it cannot regress single-tenant runs where
    CVD="0,1,...,N-1" is already aligned.

    Activated only when FIPO_PATCH_VERL=1 (set by run_fipo_v1.sh) so other
    Python processes on the box don't pay the verl-import cost.
    """
    import os

    if os.environ.get("FIPO_PATCH_VERL", "0") != "1":
        return

    try:
        import ray
        from verl.single_controller.base.worker import Worker
        from verl.utils.device import get_torch_device, is_npu_available
        from verl.utils.ray_utils import ray_noset_visible_devices
    except Exception:  # noqa: BLE001 - never break interpreter startup
        return

    if getattr(Worker, "_fipo_gpu_remap_patched", False):
        return

    def _patched(self):  # type: ignore[no-untyped-def]
        rocr_val = os.environ.get("ROCR_VISIBLE_DEVICES", None)
        hip_val = os.environ.get("HIP_VISIBLE_DEVICES", None)
        cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if hip_val:
            val = os.environ.pop("HIP_VISIBLE_DEVICES")
            if cuda_val:
                assert val == cuda_val, (
                    f"HIP_VISIBLE_DEVICES={val} disagrees with CUDA_VISIBLE_DEVICES={cuda_val}"
                )
            else:
                cuda_val = val
                os.environ["CUDA_VISIBLE_DEVICES"] = val
        if rocr_val:
            if cuda_val:
                raise ValueError(
                    "Don't set ROCR_VISIBLE_DEVICES alongside HIP/CUDA_VISIBLE_DEVICES."
                )
            cuda_val = os.environ.pop("ROCR_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val

        if ray_noset_visible_devices():
            device_name = "NPU" if is_npu_available else "GPU"
            local_rank = ray.get_runtime_context().get_accelerator_ids()[device_name][0]
            os.environ["LOCAL_RANK"] = local_rank
            physical_id = int(local_rank)
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            ids = [int(x) for x in cvd.split(",") if x.strip()]
            # Remap physical -> torch index. Falls back to identity when CVD
            # is unset (= same as upstream behaviour).
            torch_idx = ids.index(physical_id) if (ids and physical_id in ids) else physical_id
            get_torch_device().set_device(torch_idx)

    Worker._setup_env_cuda_visible_devices = _patched
    Worker._fipo_gpu_remap_patched = True


_install_verl_worker_gpu_remap()
