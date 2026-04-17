from flask import Flask, render_template
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "responsive": True,
}


def _load_data(path: str = "output/dashboard_data.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _to_numpy(values):
    if values is None:
        return np.array([])
    arr = np.asarray(values)
    if arr.size == 0:
        return np.array([])
    return arr.astype(float)


def _safe_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _feature_series(step, key):
    return np.asarray([p[key] for p in step.get("data", [])], dtype=float)


def _global_time_series(step):
    values = [p.get("global_time", idx) for idx, p in enumerate(step.get("data", []))]
    return np.asarray(values, dtype=float)


def _build_params_exact_chart(data):
    if not data:
        fig = go.Figure()
        fig.update_layout(title="Exact Parameter Values per Step", height=500)
        return fig

    rows = len(data)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Epoch {step_data.get('step', idx)}" for idx, step_data in enumerate(data)],
        vertical_spacing=0.04,
    )

    for row, step_data in enumerate(data, start=1):
        params = step_data.get("params", {})
        b0 = _to_numpy(params.get("b0"))
        b1 = _to_numpy(params.get("b1"))
        b2 = _to_numpy(params.get("b2"))
        if b0.size == 0 and b1.size == 0 and b2.size == 0:
            continue

        x0 = np.arange(len(b0))
        x1 = np.arange(len(b1))
        x2 = np.arange(len(b2))

        fig.add_trace(
            go.Scatter(x=x0, y=b0, mode="lines", name="b0", legendgroup="b0", showlegend=(row == 1)),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x1, y=b1, mode="lines", name="b1", legendgroup="b1", showlegend=(row == 1)),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x2, y=b2, mode="lines", name="b2", legendgroup="b2", showlegend=(row == 1)),
            row=row,
            col=1,
        )

        fig.update_yaxes(title_text=f"Step {step_data.get('step', row - 1)}", row=row, col=1)

    fig.update_xaxes(title_text="Parameter index", row=rows, col=1)
    fig.update_layout(
        title="Exact Parameter Values per Step",
        height=max(250 * rows, 520),
        hovermode="x unified",
    )
    return fig


def _build_params_heatmap_chart(data):
    if not data:
        fig = go.Figure()
        fig.update_layout(title="Generated Parameters Heatmaps", height=500)
        return fig

    param_rows = []
    epoch_labels = []
    for step_data in data:
        params = step_data.get("params", {})
        b0 = _to_numpy(params.get("b0"))
        b1 = _to_numpy(params.get("b1"))
        b2 = _to_numpy(params.get("b2"))
        if b0.size == 0 and b1.size == 0 and b2.size == 0:
            continue

        combined = np.concatenate([b0.reshape(-1), b1.reshape(-1), b2.reshape(-1)])
        param_rows.append(combined)
        epoch_labels.append(step_data.get("step", len(epoch_labels)))

    if not param_rows:
        fig = go.Figure()
        fig.update_layout(title="Generated Parameters Heatmaps", height=500)
        return fig

    min_len = min(len(row) for row in param_rows)
    param_matrix = np.asarray([row[:min_len] for row in param_rows], dtype=float)
    param_delta = np.diff(param_matrix, axis=0, prepend=param_matrix[0:1])

    z_abs_max = max(float(np.max(np.abs(param_matrix))), 1e-8)
    z_delta_max = max(float(np.max(np.abs(param_delta))), 1e-8)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Params Absolute Value per Epoch", "Params Delta vs Previous Epoch"),
    )

    fig.add_trace(
        go.Heatmap(
            z=param_matrix,
            colorscale="RdBu",
            zmin=-z_abs_max,
            zmax=z_abs_max,
            colorbar=dict(title="param"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=param_delta,
            colorscale="RdBu",
            zmin=-z_delta_max,
            zmax=z_delta_max,
            colorbar=dict(title="delta"),
        ),
        row=1,
        col=2,
    )

    tickvals = np.arange(len(epoch_labels))
    ticktext = [str(s) for s in epoch_labels]
    fig.update_yaxes(title_text="Epoch index", row=1, col=1, tickmode="array", tickvals=tickvals, ticktext=ticktext)
    fig.update_yaxes(title_text="Epoch index", row=1, col=2, tickmode="array", tickvals=tickvals, ticktext=ticktext)
    fig.update_xaxes(title_text="Parameter index", row=1, col=1)
    fig.update_xaxes(title_text="Parameter index", row=1, col=2)
    fig.update_layout(title="Generated Parameters Heatmaps", height=540)
    return fig


def _build_params_delta_line_chart(data):
    if not data:
        fig = go.Figure()
        fig.update_layout(title="Parameter Changes vs Previous Epoch", height=500)
        return fig

    rows = len(data)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Epoch {step_data.get('step', idx)} delta" for idx, step_data in enumerate(data)],
        vertical_spacing=0.04,
    )

    prev_b0 = None
    prev_b1 = None
    prev_b2 = None

    for row, step_data in enumerate(data, start=1):
        params = step_data.get("params", {})
        b0 = _to_numpy(params.get("b0"))
        b1 = _to_numpy(params.get("b1"))
        b2 = _to_numpy(params.get("b2"))

        if b0.size == 0 and b1.size == 0 and b2.size == 0:
            continue

        if prev_b0 is None:
            d0 = np.zeros_like(b0)
        else:
            n0 = min(len(b0), len(prev_b0))
            d0 = b0[:n0] - prev_b0[:n0]
        if prev_b1 is None:
            d1 = np.zeros_like(b1)
        else:
            n1 = min(len(b1), len(prev_b1))
            d1 = b1[:n1] - prev_b1[:n1]
        if prev_b2 is None:
            d2 = np.zeros_like(b2)
        else:
            n2 = min(len(b2), len(prev_b2))
            d2 = b2[:n2] - prev_b2[:n2]

        if d0.size:
            fig.add_trace(
                go.Scatter(x=np.arange(len(d0)), y=d0, mode="lines", name="b0 delta", legendgroup="b0delta", showlegend=(row == 1)),
                row=row,
                col=1,
            )
        if d1.size:
            fig.add_trace(
                go.Scatter(x=np.arange(len(d1)), y=d1, mode="lines", name="b1 delta", legendgroup="b1delta", showlegend=(row == 1)),
                row=row,
                col=1,
            )
        if d2.size:
            fig.add_trace(
                go.Scatter(x=np.arange(len(d2)), y=d2, mode="lines", name="b2 delta", legendgroup="b2delta", showlegend=(row == 1)),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text=f"Step {step_data.get('step', row - 1)}", row=row, col=1)
        prev_b0, prev_b1, prev_b2 = b0, b1, b2

    fig.update_xaxes(title_text="Parameter index", row=rows, col=1)
    fig.update_layout(
        title="Parameter Changes vs Previous Epoch",
        height=max(250 * rows, 520),
        hovermode="x unified",
    )
    return fig


def _build_prediction_chart(last):
    targets = np.squeeze(_to_numpy(last.get("targets")))
    predictions = np.squeeze(_to_numpy(last.get("predictions")))

    if targets.size == 0 or predictions.size == 0:
        fig = go.Figure()
        fig.update_layout(title="Prediction Diagnostics (Last Epoch)", height=520)
        return fig

    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    n_samples = min(targets.shape[0], predictions.shape[0])
    n_horizons = min(targets.shape[1], predictions.shape[1])
    targets = targets[:n_samples, :n_horizons]
    predictions = predictions[:n_samples, :n_horizons]

    fig = make_subplots(
        rows=n_horizons,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Forecast step {i + 1}" for i in range(n_horizons)],
        vertical_spacing=0.03,
    )

    x = np.arange(n_samples)
    for horizon in range(n_horizons):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=targets[:, horizon],
                mode="lines",
                name=f"target step {horizon + 1}",
                legendgroup="target",
                showlegend=(horizon == 0),
            ),
            row=horizon + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=predictions[:, horizon],
                mode="lines",
                name=f"pred step {horizon + 1}",
                legendgroup="pred",
                showlegend=(horizon == 0),
            ),
            row=horizon + 1,
            col=1,
        )
        fig.update_yaxes(title_text="Scaled y", row=horizon + 1, col=1)

    fig.update_xaxes(title_text="Sample index", row=n_horizons, col=1)
    fig.update_layout(
        title="Prediction vs Ground Truth - All Forecast Steps",
        height=max(260 * n_horizons, 520),
        hovermode="x unified",
    )
    return fig


def _build_features_chart(last):
    x1 = _feature_series(last, "x1")
    x2 = _feature_series(last, "x2")
    t = _global_time_series(last)
    if len(t) != len(x1):
        t = np.arange(len(x1))

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("x1 - Full Time Range", "x2 - Full Time Range"),
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(x=t, y=x1, mode="lines", name="x1"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=x2, mode="lines", name="x2"),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Global time", row=2, col=1)
    fig.update_yaxes(title_text="x1", row=1, col=1)
    fig.update_yaxes(title_text="x2", row=2, col=1)
    fig.update_layout(title="X Features - Full Range", height=620, hovermode="x unified")
    return fig


def _build_y_history_chart(data):
    y_matrix = []
    for d in data:
        y = _feature_series(d, "y")
        if len(y) > 0:
            y_matrix.append(y)

    if not y_matrix:
        fig = go.Figure()
        fig.update_layout(title="Generated Y History", height=500)
        return fig

    min_len = min(len(row) for row in y_matrix)
    y_matrix = np.asarray([row[:min_len] for row in y_matrix], dtype=float)
    y_delta = np.diff(y_matrix, axis=0, prepend=y_matrix[0:1])

    z_abs_max = max(float(np.max(np.abs(y_matrix))), 1e-8)
    z_delta_max = max(float(np.max(np.abs(y_delta))), 1e-8)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Y Absolute Value per Epoch", "Y Delta vs Previous Epoch"),
    )
    fig.add_trace(
        go.Heatmap(z=y_matrix, colorscale="RdBu", zmin=-z_abs_max, zmax=z_abs_max, colorbar=dict(title="Y")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=y_delta,
            colorscale="RdBu",
            zmin=-z_delta_max,
            zmax=z_delta_max,
            colorbar=dict(title="delta Y"),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Time index", row=1, col=1)
    fig.update_xaxes(title_text="Time index", row=1, col=2)
    fig.update_yaxes(title_text="Epoch index", row=1, col=1)
    fig.update_yaxes(title_text="Epoch index", row=1, col=2)
    fig.update_layout(title="Generated Y Evolution", height=540)
    return fig


@app.route("/")
def index():
    data = _load_data()

    if not data:
        empty = go.Figure()
        empty.update_layout(title="No dashboard data found")
        empty_html = empty.to_html(full_html=False, config=PLOTLY_CONFIG)
        return render_template(
            "index.html",
            plot_params_exact=empty_html,
            plot_params_heat=empty_html,
            plot_params_delta=empty_html,
            plot_pred=empty_html,
            plot_x=empty_html,
            plot_y=empty_html,
        )

    fig_params_exact = _build_params_exact_chart(data)
    fig_params_heat = _build_params_heatmap_chart(data)
    fig_params_delta = _build_params_delta_line_chart(data)
    last = data[-1]
    fig_pred = _build_prediction_chart(last)
    fig_x = _build_features_chart(last)
    fig_y = _build_y_history_chart(data)

    return render_template(
        "index.html",
        plot_params_exact=fig_params_exact.to_html(full_html=False, config=PLOTLY_CONFIG),
        plot_params_heat=fig_params_heat.to_html(full_html=False, config=PLOTLY_CONFIG),
        plot_params_delta=fig_params_delta.to_html(full_html=False, config=PLOTLY_CONFIG),
        plot_pred=fig_pred.to_html(full_html=False, config=PLOTLY_CONFIG),
        plot_x=fig_x.to_html(full_html=False, config=PLOTLY_CONFIG),
        plot_y=fig_y.to_html(full_html=False, config=PLOTLY_CONFIG),
    )


if __name__ == "__main__":
    app.run(debug=True)