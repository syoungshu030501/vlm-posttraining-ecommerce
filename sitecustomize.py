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


def _install_cuda_set_device_remap() -> None:
    """
    Patch torch.cuda.set_device to accept the *physical* GPU id reported by
    ray.get_runtime_context().get_accelerator_ids() even when CUDA_VISIBLE_DEVICES
    is non-contiguous or doesn't start at 0.

    Why:
        verl/single_controller/base/worker.py:281 calls
            get_torch_device().set_device(int(local_rank))
        where `local_rank` is the physical GPU id (e.g. "7") reported by ray.
        torch.cuda only sees indices 0..N-1 *inside* the CUDA_VISIBLE_DEVICES
        mask, so set_device(7) raises "invalid device ordinal" whenever
        CUDA_VISIBLE_DEVICES="0,3,4,5,6,7" or similar.

    The patch is a no-op for the common case (idx already in [0, device_count)),
    so it cannot break vanilla code paths.

    Only activated when FIPO_PATCH_VERL=1 (set by run_fipo_v1.sh) to avoid
    surprising other Python processes on the box.
    """
    import os

    if os.environ.get("FIPO_PATCH_VERL", "0") != "1":
        return
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.cuda, "_fipo_set_device_remapped", False):
        return

    _orig = torch.cuda.set_device

    def _safe_set_device(device):  # noqa: ANN001 - matches torch signature
        if isinstance(device, torch.device):
            idx = device.index
        else:
            idx = int(device)
        n = torch.cuda.device_count()
        if idx is None or 0 <= idx < n:
            return _orig(device)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [int(x) for x in cvd.split(",") if x.strip()]
        if idx in ids:
            return _orig(ids.index(idx))
        return _orig(device)

    torch.cuda.set_device = _safe_set_device
    torch.cuda._fipo_set_device_remapped = True


_install_cuda_set_device_remap()
