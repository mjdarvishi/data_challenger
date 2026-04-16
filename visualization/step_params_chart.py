from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_step_params_chart(step: int, params: dict, output_dir: str | Path = "output") -> Path:
    """
    Save one chart per step with the generator parameters (b0, b1, b2).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    b0 = np.asarray(params["b0"]).reshape(-1)
    b1 = np.asarray(params["b1"]).reshape(-1)
    b2 = np.asarray(params["b2"]).reshape(-1)
    hours = np.arange(b0.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, values, label, color in (
        (axes[0], b0, "b0", "tab:blue"),
        (axes[1], b1, "b1", "tab:orange"),
        (axes[2], b2, "b2", "tab:green"),
    ):
        ax.plot(hours, values, color=color, linewidth=1.6)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_title(f"Generator Params - Step {step}")
    axes[-1].set_xlabel("Hour of Week")

    fig.tight_layout()
    file_path = output_path / f"step_{step:03d}_params.png"
    fig.savefig(file_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return file_path