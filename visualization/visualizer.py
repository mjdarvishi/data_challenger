from training.pipline_tracker import PipelineTracker
import matplotlib.pyplot as plt


def plot_loss(tracker: PipelineTracker):
    steps = [r.step for r in tracker.history]
    train_loss = [r.model_loss for r in tracker.history]
    test_loss = [r.generator_loss for r in tracker.history]

    plt.figure()
    plt.plot(steps, train_loss, label="Train Loss")
    plt.plot(steps, test_loss, label="Test Loss")

    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.show()


def plot_b0_heatmap(tracker: PipelineTracker):
    import numpy as np

    matrix = []

    for r in tracker.history:
        matrix.append(r.params["b0"])

    matrix = np.array(matrix)

    plt.figure()
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.xlabel("Hour of Week")
    plt.ylabel("Step")
    plt.title("b0 Evolution Heatmap")
    plt.show()


import pandas as pd


def print_mse_table(tracker: PipelineTracker, last_n: int | None = None):
    """
    Prints a clean table of model vs generator (test) MSE over steps.
    """

    history = tracker.history

    if last_n is not None:
        history = history[-last_n:]

    data = []

    for r in history:
        data.append(
            {
                "step": r.step,
                "train_mse": r.model_losses,
                "test_mse": r.generator_loss,
            }
        )

    df = pd.DataFrame(data)

    # nicer display
    print("\n===== MSE TABLE =====\n")
    print(df.to_string(index=False))


def print_grid_results(tracker: PipelineTracker):
    df = tracker.grid_search_history[-1]["results"]

    df = df.sort_values("mse")

    print("\n===== GRID SEARCH RESULTS =====\n")
    print(df.to_string(index=False))

    print("\nBEST CONFIG:")
    print(df.iloc[0].to_dict())


def plot_ground_truth_vs_prediction(tracker, step: int = -1, n: int = 200):
    """
    Plot ground truth vs model prediction for a given pipeline step.

    Args:
        tracker: PipelineTracker instance
        step: which step to visualize (-1 = last step)
        n: number of samples to plot
    """

    record = tracker.history[step]

    y_true = record.targets
    y_pred = record.predictions

    if y_true is None or y_pred is None:
        raise ValueError("Predictions or targets not found in tracker.")

    # convert to numpy safely
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten()

    y_true = y_true[:n]
    y_pred = y_pred[:n]

    plt.figure(figsize=(12, 5))

    plt.plot(y_true, label="Ground Truth (Y)", linewidth=2)
    plt.plot(y_pred, label="Prediction (Ŷ)", linewidth=2, alpha=0.8)

    plt.title(f"Ground Truth vs Prediction | Step {record.step}")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.show()
