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
