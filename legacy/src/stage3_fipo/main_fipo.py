"""
FIPO entry point — wraps verl.trainer.main_ppo, but first registers our
project-specific patches into verl's registries:

- src.stage3_fipo.verl_patches.future_kl_loss   -> POLICY_LOSS_REGISTRY["future_kl"]
- src.stage3_fipo.verl_patches.reward_manager   -> REWARD_MANAGER_REGISTRY["vlm_audit_v2"]

Run via run_fipo_v1.sh, do not invoke directly (Hydra config path resolution
needs verl-latest on PYTHONPATH).
"""
import os
import sys

# Side-effect imports MUST happen before main() so that @register_policy_loss
# and @register decorators populate the verl registries (driver process).
# Ray spawn workers inherit the driver's PYTHONPATH (set by run_fipo_v1.sh),
# and Python's site machinery auto-imports sitecustomize.py from project root,
# which re-imports future_kl_loss and registers it in every worker process.
import src.stage3_fipo.verl_patches.future_kl_loss  # noqa: F401
import src.stage3_fipo.verl_patches.reward_manager  # noqa: F401

# Default to HF mirror for the bge encoder download (idempotent).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
