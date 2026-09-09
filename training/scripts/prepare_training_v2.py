"""Create a frozen split for the wrist-normalized feature contract."""
import importlib
import sys

sys.argv[0] = "prepare_training_v2.py"
module = importlib.import_module("prepare_training")
module.fingerprint = importlib.import_module("baseline_v2").fingerprint
module.CONTRACT = importlib.import_module("baseline_v2").CONTRACT

if __name__ == "__main__":
    module.main()
