import os
import random
import subprocess
import sys
from pathlib import Path
import importlib
import numpy as np
import torch
import random


def _ensure_model_repo_paths(base_dir="/content"):
    base_path = Path(base_dir)
    itransformer_dir = base_path / "iTransformer"
    samformer_dir = base_path / "samformer"

    # Keep iTransformer paths before samformer to avoid picking the wrong top-level "utils" package.
    preferred_paths = [
        itransformer_dir,
        itransformer_dir / "iTransformer",
        samformer_dir,
    ]

    for repo_path in preferred_paths:
        repo_str = str(repo_path)
        while repo_str in sys.path:
            sys.path.remove(repo_str)

    # Insert in reverse so final order matches preferred_paths.
    for repo_path in reversed(preferred_paths):
        if repo_path.exists():
            sys.path.insert(0, str(repo_path))

    return itransformer_dir, samformer_dir


def _pip_install(package_spec: str):
    print("-> Installing missing dependency:", package_spec)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", package_spec],
        check=True,
    )


def _purge_conflicting_utils_cache(itransformer_dir: Path):
    # A third-party module named "utils" can get cached and break iTransformer imports.
    for name in ["utils", "utils.masking"]:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = str(getattr(mod, "__file__", "") or "")
        if str(itransformer_dir) not in mod_file:
            del sys.modules[name]


def import_itransformer_model():
    itransformer_dir, _ = _ensure_model_repo_paths()
    _purge_conflicting_utils_cache(itransformer_dir)
    importlib.invalidate_caches()

    try:
        from model.iTransformer import Model

        return Model
    except ModuleNotFoundError as exc:
        # iTransformer depends on reformer_pytorch via `reformer-pytorch` package.
        if exc.name == "reformer_pytorch":
            _pip_install("reformer-pytorch>=1.4.4")
            importlib.invalidate_caches()
            from model.iTransformer import Model

            return Model

        raise ImportError(
            "iTransformer import failed. Missing module: " + str(exc.name)
        ) from exc


def import_samformer_class():
    # Import path can vary by install mode and package layout.
    candidates = [
        "samformer_pytorch.samformer.samformer",
        "samformer.samformer.samformer",
        "samformer.samformer",
    ]

    _ensure_model_repo_paths()
    importlib.invalidate_caches()

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "SAMFormer"):
                return module.SAMFormer
        except Exception:
            continue

    raise ImportError(
        "SAMFormer import failed. Re-run Cell 2 (Setup), then retry this cell."
    )


def _run_cmd(cmd):
    print("->", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _restart_kernel_if_available(reason):
    print("->", reason)
    try:
        from IPython.display import display, Markdown

        display(Markdown("**Restarting kernel to finalize binary package changes...**"))
        get_ipython().kernel.do_shutdown(restart=True)
        return True
    except Exception:
        print(
            "X Could not auto-restart kernel. Please restart runtime manually and re-run from Cell 2."
        )
        return False


def _repair_scientific_stack():
    # Force reinstall to clear any ABI-mismatched wheels left in the environment.
    _run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "numpy==1.26.4",
            "pandas==2.2.2",
            "scipy==1.13.1",
            "scikit-learn==1.5.0",
        ]
    )


def setup_models(base_dir="/content", seed=42):
    """
    Unified setup for iTransformer and SAMFormer in a single Colab env.
    Designed for Python 3.12-compatible package versions.
    """
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # 1) Core tooling
    _run_cmd(
        [sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"]
    )

    # 2) Python 3.12-safe scientific + DL stack
    _run_cmd(
        [
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
            "reformer-pytorch>=1.4.4",
        ]
    )

    # 3) PyTorch (CUDA 12.1 wheels; works in Colab GPU runtime)
    _run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )

    # 4) Clone repos if needed
    itransformer_dir = base_path / "iTransformer"
    samformer_dir = base_path / "samformer"

    if not itransformer_dir.exists():
        _run_cmd(
            [
                "git",
                "clone",
                "https://github.com/thuml/iTransformer.git",
                str(itransformer_dir),
            ]
        )
    else:
        print("-> iTransformer already present, skipping clone")

    if not samformer_dir.exists():
        _run_cmd(
            [
                "git",
                "clone",
                "https://github.com/romilbert/samformer.git",
                str(samformer_dir),
            ]
        )
    else:
        print("-> samformer already present, skipping clone")

    # 5) Install SAMFormer package without forcing its old pinned deps.
    # Try editable first; if it fails, try non-editable local install.
    install_samformer_ok = False
    install_attempts = [
        [sys.executable, "-m", "pip", "install", "-e", str(samformer_dir), "--no-deps"],
        [sys.executable, "-m", "pip", "install", str(samformer_dir), "--no-deps"],
    ]

    for attempt in install_attempts:
        try:
            _run_cmd(attempt)
            install_samformer_ok = True
            break
        except subprocess.CalledProcessError as exc:
            print("X SAMFormer install attempt failed with return code", exc.returncode)

    if not install_samformer_ok:
        print("X SAMFormer installation failed; continuing with repo-on-sys.path mode.")
        print("-> Imports may still work directly from the cloned repository.")

    # 6) Add repos to import path for notebook runtime
    for repo_path in [str(itransformer_dir), str(samformer_dir)]:
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    # 7) Reproducibility + runtime info with ABI smoke test
    try:
        import pandas as pd
        import scipy
        import sklearn
    except Exception as exc:
        print("X Scientific stack import failed:", repr(exc))
        print("-> Attempting forced reinstall of binary scientific packages.")
        _repair_scientific_stack()
        _restart_kernel_if_available(
            "Binary packages were reinstalled after an ABI mismatch."
        )
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    print("Environment setup complete")
