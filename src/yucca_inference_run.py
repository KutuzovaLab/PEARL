import sys
import torch

def _install_compat_shim():
    # 1. Allowlist the builtins yucca's hyperparameters legitimately contain.
    try:
        torch.serialization.add_safe_globals([int, float, str, bool, dict, list, tuple, set])
    except AttributeError:
        return  # torch < 2.4: no weights_only default, nothing to do

    # 2. Lightning does not expose weights_only, so restore the pre-2.6 default for
    #    checkpoints loaded in this process.
    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load

def main():
    _install_compat_shim()
    from yucca.pipeline.run.run_inference import main as yucca_main
    return yucca_main()

if __name__ == "__main__":
    sys.exit(main())
