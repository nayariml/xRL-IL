"""
AMP_for_hardware is expected next to xRL. To use another checkout:
LEGGED_GYM_ROOT=~/projects/AMP_for_hardware python xRL/scripts/train.py
"""
import importlib.util
import os
import sys

XRL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(os.path.dirname(XRL_DIR), "AMP_for_hardware")
LEGGED_GYM_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("LEGGED_GYM_ROOT", DEFAULT_ROOT)))

sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != XRL_DIR]

for path in (os.path.join(LEGGED_GYM_ROOT, "rsl_rl"), LEGGED_GYM_ROOT):
    if os.path.isdir(path):
        sys.path.insert(0, path)

# `import xRL` is this checkout, whatever its folder is called and whatever else is on the path
spec = importlib.util.spec_from_file_location(
    "xRL", os.path.join(XRL_DIR, "__init__.py"), submodule_search_locations=[XRL_DIR])
sys.modules["xRL"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["xRL"])
