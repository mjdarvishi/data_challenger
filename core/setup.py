import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
import importlib
import numpy as np

# =========================================================
# PATH MANAGEMENT
# =========================================================

MODEL_REPOS_DIRNAME = "external_models"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _default_base_dir() -> Path:
    # Keep cloned model repos under a dedicated project directory.
    return _project_root() / MODEL_REPOS_DIRNAME


def _resolve_repo_dir(base_path: Path, repo_name: str) -> Path:
    preferred = base_path / repo_name
    legacy = _project_root() / repo_name

    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def _migrate_repo_if_needed(base_path: Path, repo_name: str) -> Path:
    target = base_path / repo_name
    legacy = _project_root() / repo_name

    if target.exists():
        return target

    if legacy.exists() and legacy != target:
        print(f"-> moving {repo_name} into {base_path}")
        shutil.move(str(legacy), str(target))

    return target

def _ensure_model_repo_paths(base_dir=None):
    if base_dir is None:
        base_dir = _default_base_dir()
    base_path = Path(base_dir)

    itransformer_dir = _resolve_repo_dir(base_path, "iTransformer")
    autoformer_dir = _resolve_repo_dir(base_path, "Autoformer")
    dlinear_dir = _resolve_repo_dir(base_path, "LTSF-Linear")

    legacy_itransformer_dir = _project_root() / "iTransformer"
    legacy_autoformer_dir = _project_root() / "Autoformer"
    legacy_dlinear_dir = _project_root() / "LTSF-Linear"

    preferred_paths = [
        itransformer_dir,
        itransformer_dir / "iTransformer",
        autoformer_dir,
        dlinear_dir,
        legacy_itransformer_dir,
        legacy_itransformer_dir / "iTransformer",
        legacy_autoformer_dir,
        legacy_dlinear_dir,
    ]

    # remove duplicates from sys.path
    for repo_path in preferred_paths:
        repo_str = str(repo_path)
        while repo_str in sys.path:
            sys.path.remove(repo_str)

    # insert correctly
    for repo_path in reversed(preferred_paths):
        if repo_path.exists():
            sys.path.insert(0, str(repo_path))

    return itransformer_dir, autoformer_dir, dlinear_dir


def _run_cmd(cmd):
    print("->", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _prioritize_repo_path(repo_dir: Path):
    repo_str = str(repo_dir)
    while repo_str in sys.path:
        sys.path.remove(repo_str)
    if repo_dir.exists():
        sys.path.insert(0, repo_str)


def _purge_conflicting_package_cache(repo_dir: Path, package_prefixes: tuple[str, ...]):
    repo_dir_str = str(repo_dir)
    for mod_name, mod in list(sys.modules.items()):
        if not any(
            mod_name == prefix or mod_name.startswith(prefix + ".")
            for prefix in package_prefixes
        ):
            continue

        mod_file = str(getattr(mod, "__file__", "") or "")
        if mod_file and repo_dir_str not in mod_file:
            del sys.modules[mod_name]


def _purge_conflicting_utils_cache(itransformer_dir: Path):
    for name in ["utils", "utils.masking"]:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = str(getattr(mod, "__file__", "") or "")
        if str(itransformer_dir) not in mod_file:
            del sys.modules[name]


# =========================================================
# MODEL IMPORTS
# =========================================================

def _require_local_repo(repo_name: str) -> Path:
    repo_dir = _default_base_dir() / repo_name
    if not repo_dir.exists():
        raise ImportError(
            f"Missing local repo: {repo_dir}. Place {repo_name} under {_default_base_dir()}."
        )
    return repo_dir

def import_itransformer_model():
    itransformer_dir = _require_local_repo("iTransformer")
    _prioritize_repo_path(itransformer_dir)
    _purge_conflicting_package_cache(itransformer_dir, ("layers", "utils", "model"))
    _purge_conflicting_utils_cache(itransformer_dir)
    importlib.invalidate_caches()

    try:
        module = importlib.import_module("model.iTransformer")
        return module.Model
    except Exception as exc:
        raise ImportError("iTransformer import failed") from exc


def import_autoformer_model():
    autoformer_dir = _require_local_repo("Autoformer")
    _prioritize_repo_path(autoformer_dir)
    _purge_conflicting_package_cache(autoformer_dir, ("layers", "utils", "models", "model"))
    importlib.invalidate_caches()

    for module_name in ("model.Autoformer", "models.Autoformer"):
        try:
            module = importlib.import_module(module_name)
            return module.Model
        except Exception:
            continue

    raise ImportError(
        f"Autoformer import failed. Expected model/Autoformer.py or models/Autoformer.py under {autoformer_dir}."
    )


# =========================================================
# MAIN SETUP
# =========================================================

def setup_models(base_dir=None, seed=42):
    """
    Clean setup for:
    - iTransformer
    - Autoformer
    - LTSF-Linear (DLinear)

    SAMFormer REMOVED (not compatible with adversarial training)
    """
    import torch

    # -------------------------
    # reproducibility
    # -------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if base_dir is None:
        base_dir = _default_base_dir()
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. Core tooling
    # -------------------------
    _run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "pip",
        "setuptools",
        "wheel"
    ])

    # -------------------------
    # 2. Scientific stack
    # -------------------------
    _run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "scikit-learn==1.5.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "einops>=0.8.0",
        "tqdm>=4.66.0",
        "optuna>=3.6.0",
        "pytorch-lightning>=2.2.0",
    ])

    # -------------------------
    # 3. PyTorch (CUDA 12.1)
    # -------------------------
    _run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ])

    # -------------------------
    # 4. Clone iTransformer
    # -------------------------
    itransformer_dir = _migrate_repo_if_needed(base_path, "iTransformer")

    if not itransformer_dir.exists():
        _run_cmd([
            "git",
            "clone",
            "https://github.com/thuml/iTransformer.git",
            str(itransformer_dir),
        ])
    else:
        print("-> iTransformer already exists")

    # -------------------------
    # 5. Clone Autoformer
    # -------------------------
    autoformer_dir = _migrate_repo_if_needed(base_path, "Autoformer")

    if not autoformer_dir.exists():
        _run_cmd([
            "git",
            "clone",
            "https://github.com/thuml/Autoformer.git",
            str(autoformer_dir),
        ])
    else:
        print("-> Autoformer already exists")

    # -------------------------
    # 6. Clone LTSF-Linear (DLinear)
    # -------------------------
    dlinear_dir = _migrate_repo_if_needed(base_path, "LTSF-Linear")

    if not dlinear_dir.exists():
        _run_cmd([
            "git",
            "clone",
            "https://github.com/cure-lab/LTSF-Linear.git",
            str(dlinear_dir),
        ])
    else:
        print("-> LTSF-Linear already exists")

    # -------------------------
    # 7. sys.path setup
    # -------------------------
    for repo_path in [itransformer_dir, autoformer_dir, dlinear_dir]:
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    # -------------------------
    # 8. sanity imports
    # -------------------------
    try:
        import numpy
        import pandas
        import sklearn
        import torch
    except Exception as exc:
        print("X environment import failed:", exc)
        raise

    # -------------------------
    # 9. device info
    # -------------------------
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    print("Environment setup complete")