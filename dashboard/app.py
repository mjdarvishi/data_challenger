from flask import Flask, jsonify, render_template, request
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)


_DATA_CACHE = {}
_SECTION_HTML_CACHE = {}


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "responsive": True,
}


def _load_data(path: str = "output/new/dashboard_data.json"):
    try:
        with open(path, "r") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return payload.get("records", []), payload.get("grid_search_history", []), payload.get("config", {})
            return payload, [], {}
    except FileNotFoundError:
        return [], [], {}


def _load_data_cached(path: str = "output/new/dashboard_data.json"):
    abs_path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        return [], [], {}, abs_path, None

    cached = _DATA_CACHE.get(abs_path)
    if cached and cached.get("mtime") == mtime:
        return cached["data"], cached["grid_search_history"], cached["config"], abs_path, mtime

    data, grid_search_history, config_dict = _load_data(abs_path)
    _DATA_CACHE[abs_path] = {
        "mtime": mtime,
        "data": data,
        "grid_search_history": grid_search_history,
        "config": config_dict,
    }
    return data, grid_search_history, config_dict, abs_path, mtime


def _get_data_for_source(source_name: str):
    data_path = os.path.join("output/new/", source_name)
    data, grid_search_history, config_dict, abs_path, mtime = _load_data_cached(data_path)
    return {
        "data": data,
        "grid_search_history": grid_search_history,
        "config": config_dict,
        "cache_key": (abs_path, mtime),
    }


def _fig_to_html(fig):
    return fig.to_html(full_html=False, config=PLOTLY_CONFIG)


def _empty_state(message: str):
    return f'<div class="empty-state">{message}</div>'


def _list_output_sources(output_dir: str = "output/new"):
    try:
        names = [
            name
            for name in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, name)) and name.lower().endswith(".json")
        ]
    except FileNotFoundError:
        return []

    names.sort(key=str.lower)
    return names


def _resolve_selected_source(available_sources, selected_source):
    if not available_sources:
        return "dashboard_data.json"
    if selected_source in available_sources:
        return selected_source
    if "dashboard_data.json" in available_sources:
        return "dashboard_data.json"
    return available_sources[0]


def _to_numpy(values):
    if values is None:
        return np.array([])
    arr = np.asarray(values)
    if arr.size == 0:
        return np.array([])
    return arr.astype(float)


def _inverse_y(values, step_data):
    arr = _to_numpy(values)
    if arr.size == 0:
        return arr
    y_mean = float(step_data.get("Y_mean", 0.0))
    y_std = float(step_data.get("Y_std", 1.0))
    return arr * y_std + y_mean


def _safe_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _format_params(params):
    if not isinstance(params, dict):
        return "{}"
    return json.dumps(params, sort_keys=True)


def _format_scalar(value):
    try:
        fval = float(value)
        if fval.is_integer():
            return str(int(fval))
        text = f"{fval:.4f}".rstrip("0").rstrip(".")
        return text
    except (TypeError, ValueError):
        return str(value)


def _format_params_inline(params):
    if not isinstance(params, dict) or not params:
        return "{}"
    parts = [f"{key}={_format_scalar(value)}" for key, value in sorted(params.items())]
    return ", ".join(parts)


def _html_table(df: pd.DataFrame, empty_message: str):
    if df.empty:
        return f'<div class="empty-state">{empty_message}</div>'
    return df.to_html(index=False, classes="dashboard-table", border=0, escape=True)


def _build_config_table(config_dict):
    """
    Build an HTML table from the config dictionary loaded from JSON.
    Displays config as vertical rows (Key | Value) instead of horizontal columns.
    """
    if not config_dict:
        return '<div class="empty-state">No config values found.</div>'

    rows = [{"Key": key, "Value": _format_scalar(value)} for key, value in sorted(config_dict.items())]
    df = pd.DataFrame(rows)
    return _html_table(df, "No config values found.")


def _feature_series(step, key):
    return np.asarray([p[key] for p in step.get("data", [])], dtype=float)


def _extract_feature_matrix(step):
    points = step.get("data", [])
    if not points:
        return np.empty((0, 0), dtype=float), []

    x_values_rows = []
    for p in points:
        values = p.get("x_values")
        if isinstance(values, list):
            x_values_rows.append([float(v) for v in values])

    if x_values_rows:
        min_len = min(len(row) for row in x_values_rows)
        if min_len == 0:
            return np.empty((len(x_values_rows), 0), dtype=float), []
        matrix = np.asarray([row[:min_len] for row in x_values_rows], dtype=float)
        names = [f"x{i + 1}" for i in range(min_len)]
        return matrix, names

    # Backward compatibility with old payloads that only have x1/x2 scalars.
    fallback_names = []
    for name in ("x1", "x2"):
        if any(p.get(name) is not None for p in points):
            fallback_names.append(name)

    if not fallback_names:
        return np.empty((0, 0), dtype=float), []

    cols = []
    for name in fallback_names:
        col = [float(p[name]) for p in points if p.get(name) is not None]
        cols.append(np.asarray(col, dtype=float))

    min_len = min(len(col) for col in cols)
    matrix = np.column_stack([col[:min_len] for col in cols])
    return matrix, fallback_names


def _param_name_sort_key(name):
    if name == "b0":
        return (0, 0)
    if isinstance(name, str) and name.startswith("b") and name[1:].isdigit():
        return (1, int(name[1:]))
    return (2, str(name))


def _extract_param_vectors(step_data):
    params = step_data.get("params", {})
    series = []

    b0 = _to_numpy(params.get("b0"))
    if b0.size:
        series.append(("b0", b0.reshape(-1)))

    b_matrix = _to_numpy(params.get("b"))
    if b_matrix.size:
        if b_matrix.ndim == 1:
            b_matrix = b_matrix.reshape(1, -1)
        for i in range(b_matrix.shape[0]):
            series.append((f"b{i + 1}", np.asarray(b_matrix[i], dtype=float).reshape(-1)))
        return series

    # Backward compatibility with payloads that only expose b1/b2.
    fallback_names = []
    for key in params.keys():
        if key == "b0":
            continue
        if isinstance(key, str) and key.startswith("b"):
            fallback_names.append(key)

    for key in sorted(fallback_names, key=_param_name_sort_key):
        values = _to_numpy(params.get(key))
        if values.size:
            series.append((key, values.reshape(-1)))

    return series


def _global_time_series(step):
    values = [p.get("global_time", idx) for idx, p in enumerate(step.get("data", []))]
    return np.asarray(values, dtype=float)


def _build_grid_search_table_and_chart(grid_search_history):
    if not grid_search_history:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Grid Search MSE Trend", height=420)
        empty_table = (
            '<div class="empty-state">No grid search data found in the exported dashboard payload. '
            'Run <strong>main.py</strong> again to regenerate output/new/dashboard_data.json.</div>'
        )
        return empty_table, empty_fig

    run = grid_search_history[0]
    results = run.get("results", []) if isinstance(run, dict) else []
    df = pd.DataFrame(results)
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Grid Search MSE Trend", height=420)
        return _html_table(df, "No grid search results found."), empty_fig

    if "mse" not in df.columns:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Grid Search MSE Trend", height=420)
        return _html_table(df, "No grid search MSE column found."), empty_fig

    param_columns = [col for col in df.columns if col != "mse"]
    display_df = df.copy()

    ordered_columns = [*param_columns]
    if "mse" in display_df.columns:
        ordered_columns.append("mse")
    display_df = display_df[ordered_columns]

    for col in param_columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(
                lambda value: str(int(value)) if float(value).is_integer() else f"{float(value):.4f}"
            )

    display_df = display_df.rename(columns={"mse": "MSE"})
    display_df["MSE"] = display_df["MSE"].astype(float).map(lambda value: f"{value:.4f}")

    best_index = int(df["mse"].astype(float).idxmax())
    best_params = run.get("best_params", {}) if isinstance(run, dict) else {}
    best_score = float(run.get("best_score", df["mse"].min())) if isinstance(run, dict) else float(df["mse"].min())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(df)),
            y=df["mse"].astype(float),
            mode="lines+markers",
            name="MSE",
            text=[_format_params(row.to_dict()) for _, row in df[param_columns].iterrows()] if param_columns else None,
            hovertemplate="Test %{x}<br>MSE %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_index],
            y=[best_score],
            mode="markers+text",
            name="Best",
            text=["best"],
            textposition="top center",
            marker=dict(size=12, color="#ef4444"),
            hovertemplate="Best MSE %{y:.4f}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Grid Search MSE Trend",
        xaxis_title="Test index",
        yaxis_title="MSE",
        height=420,
        hovermode="x unified",
    )

    summary = (
        f'<div class="summary-line"><strong>Best params:</strong> {_format_params_inline(best_params)} '
        f'| <strong>Best MSE:</strong> {best_score:.4f}</div>'
    )
    return summary + _html_table(display_df, "No grid search results found."), fig
def _build_quality_panel(data):
    if not data:
        return '<div class="empty-state">No epoch data to evaluate.</div>'

    rows = []
    for step_data in data:
        train = step_data.get("train_eval_mse")
        val   = step_data.get("val_eval_mse")
        test  = step_data.get("test_eval_mse")
        if None in (train, val, test):
            continue
        rows.append({
            "step":  int(step_data.get("step", len(rows))),
            "train": float(train),
            "val":   float(val),
            "test":  float(test),
        })

    if not rows:
        return '<div class="empty-state">MSE values missing in epoch data.</div>'

    best_idx  = min(range(len(rows)), key=lambda i: rows[i]["val"])
    first_val = rows[0]["val"]
    best_val  = rows[best_idx]["val"]
    improvement = (first_val - best_val) / first_val * 100 if first_val else 0
    early_stop = best_idx < len(rows) - 1

    def badge(cls, label):
        return f'<span class="quality-badge quality-{cls}">{label}</span>'

    def score(r):
        overfit_gap = r["val"] - r["train"]
        gen_gap     = r["test"] - r["val"]
        overfit_cls = "good" if overfit_gap < 0.05 else ("warn" if overfit_gap < 0.12 else "bad")
        gen_cls     = "good" if gen_gap     < 0.03 else ("warn" if gen_gap     < 0.06 else "bad")
        return overfit_gap, gen_gap, overfit_cls, gen_cls

    css = """
    <style>
    .quality-badge{display:inline-block;font-size:11px;font-weight:500;padding:2px 7px;border-radius:6px}
    .quality-good{background:#dcfce7;color:#166534}
    .quality-warn{background:#fef9c3;color:#854d0e}
    .quality-bad {background:#fee2e2;color:#991b1b}
    .quality-neutral{background:#f3f4f6;color:#374151;border:0.5px solid #d1d5db}
    .quality-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:1.25rem}
    .quality-card{background:#f9fafb;border-radius:8px;padding:12px 14px}
    .quality-card p{margin:0}
    .quality-card .qlabel{font-size:12px;color:#6b7280}
    .quality-card .qval{font-size:20px;font-weight:500}
    .quality-card .qsub{font-size:11px;color:#9ca3af;margin-top:3px}
    .quality-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:1rem}
    .quality-table th{font-size:12px;font-weight:500;color:#6b7280;text-align:left;padding:6px 8px;border-bottom:1px solid #e5e7eb}
    .quality-table td{padding:7px 8px;border-bottom:1px solid #f3f4f6}
    .quality-table tr.best-epoch td{background:#f0fdf4}
    .quality-hint{font-size:11px;color:#9ca3af;margin-top:8px}
    </style>
    """

    early_badge = badge("warn", f"val rising after epoch {rows[best_idx]['step']}") if early_stop else badge("good", "still improving")
    
    cards = f"""
    <div class="quality-cards">
      <div class="quality-card">
        <p class="qlabel">Best val MSE</p>
        <p class="qval">{best_val:.4f}</p>
        <p class="qsub">epoch {rows[best_idx]['step']}</p>
      </div>
      <div class="quality-card">
        <p class="qlabel">Val improvement</p>
        <p class="qval">{improvement:.1f}%</p>
        <p class="qsub">epoch 0 → best</p>
      </div>
      <div class="quality-card">
        <p class="qlabel">Best test MSE</p>
        <p class="qval">{rows[best_idx]['test']:.4f}</p>
        <p class="qsub">at best val epoch</p>
      </div>
      <div class="quality-card">
        <p class="qlabel">Early stopping signal</p>
        <p class="qval" style="font-size:13px;padding-top:6px">{early_badge}</p>
        <p class="qsub">&nbsp;</p>
      </div>
    </div>
    """

    header = """
    <table class="quality-table">
    <thead><tr>
      <th>Epoch</th><th>Train MSE</th><th>Val MSE</th><th>Test MSE</th>
      <th title="val - train">Overfit gap</th><th title="test - val">Gen. gap</th><th>Signal</th>
    </tr></thead><tbody>
    """

    body = ""
    for i, r in enumerate(rows):
        og, gg, oc, gc = score(r)
        is_best = (i == best_idx)
        row_cls = ' class="best-epoch"' if is_best else ""
        star    = " ★" if is_best else ""
        og_str  = f"+{og:.3f}" if og >= 0 else f"{og:.3f}"
        gg_str  = f"+{gg:.3f}" if gg >= 0 else f"{gg:.3f}"

        if oc == "good" and gc == "good":
            sig = badge("good", "healthy")
        elif oc == "bad":
            sig = badge("bad", "overfitting")
        elif gc == "bad":
            sig = badge("bad", "poor gen.")
        else:
            sig = badge("warn", "watch")

        body += f"""<tr{row_cls}>
          <td><strong>{r['step']}{star}</strong></td>
          <td>{r['train']:.4f}</td>
          <td>{r['val']:.4f}</td>
          <td>{r['test']:.4f}</td>
          <td>{badge(oc, og_str)}</td>
          <td>{badge(gc, gg_str)}</td>
          <td>{sig}</td>
        </tr>"""

    hint = '<p class="quality-hint">Overfit gap = val − train &nbsp;|&nbsp; Gen. gap = test − val &nbsp;|&nbsp; ★ = best val epoch &nbsp;|&nbsp; thresholds: overfit &lt;0.05 good, &lt;0.12 warn; gen &lt;0.03 good, &lt;0.06 warn</p>'

    return css + cards + header + body + "</tbody></table>" + hint

def _build_epoch_summary_table_and_chart(data):
    rows = []
    for step_data in data:
        train_mse = step_data.get("train_eval_mse", 0.0)
        val_mse = step_data.get("val_eval_mse", 0.0)
        test_mse = step_data.get("test_eval_mse", 0.0)

        rows.append(
            {
                "Epoch": step_data.get("step", len(rows)),
                "Total time (s)": float(step_data.get("execution_time", 0.0)),
                "Forecast time (s)": float(step_data.get("forecast_time", 0.0)),
                "Generator time (s)": float(step_data.get("generator_time", 0.0)),
                "Train MSE": float(train_mse) if train_mse is not None else 0.0,
                "Val MSE": float(val_mse) if val_mse is not None else 0.0,
                "Test MSE": float(test_mse) if test_mse is not None else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Epoch Trend", height=420)
        return _html_table(df, "No epoch data found."), empty_fig

    display_df = df.copy()
    for column in ["Total time (s)", "Forecast time (s)", "Generator time (s)"]:
        display_df[column] = display_df[column].map(lambda value: f"{float(value):.2f}s")
    for column in ["Train MSE", "Val MSE", "Test MSE"]:
        display_df[column] = display_df[column].map(lambda value: f"{float(value):.4f}")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Time Breakdown by Epoch", "Loss and Accuracy Trend"),
        vertical_spacing=0.12,
    )

    epochs = df["Epoch"].astype(int)
    fig.add_trace(go.Scatter(x=epochs, y=df["Total time (s)"], mode="lines+markers", name="total"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=df["Forecast time (s)"], mode="lines+markers", name="forecast"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=df["Generator time (s)"], mode="lines+markers", name="generator"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=df["Train MSE"], mode="lines+markers", name="train MSE"), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=df["Val MSE"], mode="lines+markers", name="val MSE"), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=df["Test MSE"], mode="lines+markers", name="test MSE"), row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Seconds", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_layout(title="Epoch Trend", height=700, hovermode="x unified")
    return _html_table(display_df, "No epoch data found."), fig


def _build_params_exact_chart(data):
    if not data:
        fig = go.Figure()
        fig.update_layout(title="Exact Parameter Values per Step", height=500)
        return fig

    rows = len(data)
    vertical_spacing = min(0.04, 0.18 / max(rows - 1, 1))
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Epoch {step_data.get('step', idx)}" for idx, step_data in enumerate(data)],
        vertical_spacing=vertical_spacing,
    )

    for row, step_data in enumerate(data, start=1):
        series = _extract_param_vectors(step_data)
        if not series:
            continue

        for name, values in series:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(values)),
                    y=values,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=(row == 1),
                ),
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
        series = _extract_param_vectors(step_data)
        if not series:
            continue

        combined = np.concatenate([values.reshape(-1) for _, values in series])
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
        fig.update_layout(title="Parameter Values vs Previous Epoch", height=500)
        return fig

    rows = len(data)
    vertical_spacing = min(0.04, 0.18 / max(rows - 1, 1))
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Epoch {step_data.get('step', idx)} vs previous" for idx, step_data in enumerate(data)],
        vertical_spacing=vertical_spacing,
    )

    previous = {}

    for row, step_data in enumerate(data, start=1):
        series = _extract_param_vectors(step_data)
        if not series:
            continue

        current = {}
        for name, values in series:
            prev_values = previous.get(name)

            if prev_values is None:
                prev_plot = np.zeros_like(values)
                curr_plot = values
            else:
                n = min(len(values), len(prev_values))
                curr_plot = values[:n]
                prev_plot = prev_values[:n]

            current[name] = values
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(curr_plot)),
                    y=curr_plot,
                    mode="lines",
                    name=f"{name} current",
                    legendgroup=f"{name}_curr",
                    showlegend=(row == 1),
                    line=dict(width=2),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(prev_plot)),
                    y=prev_plot,
                    mode="lines",
                    name=f"{name} previous",
                    legendgroup=f"{name}_prev",
                    showlegend=(row == 1),
                    line=dict(width=1, dash="dash"),
                    opacity=0.7,
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text=f"Step {step_data.get('step', row - 1)}", row=row, col=1)
        previous = current

    fig.update_xaxes(title_text="Parameter index", row=rows, col=1)
    fig.update_layout(
        title="Parameter Values vs Previous Epoch",
        height=max(250 * rows, 520),
        hovermode="x unified",
    )
    return fig


def _build_prediction_chart(last):
    targets = np.squeeze(_inverse_y(last.get("targets"), last))
    predictions = np.squeeze(_inverse_y(last.get("predictions"), last))

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
        fig.update_yaxes(title_text="Original y", row=horizon + 1, col=1)

    fig.update_xaxes(title_text="Sample index", row=n_horizons, col=1)
    fig.update_layout(
        title="Prediction vs Ground Truth - All Forecast Steps (Original Scale)",
        height=max(260 * n_horizons, 520),
        hovermode="x unified",
    )
    return fig


def _build_prediction_history_chart(data):
    if not data:
        fig = go.Figure()
        fig.update_layout(title="Prediction vs Ground Truth over Epochs", height=500)
        return fig

    rows = len(data)
    subplot_titles = []
    for idx, step_data in enumerate(data):
        label_epoch = step_data.get("step", idx)
        mse_test = step_data.get("test_eval_mse")
        if mse_test is None:
            subplot_titles.append(f"Epoch {label_epoch}")
        else:
            subplot_titles.append(f"Epoch {label_epoch} | test MSE={float(mse_test):.4f}")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=min(0.04, 0.18 / max(rows - 1, 1)),
    )

    for row, step_data in enumerate(data, start=1):
        targets = np.squeeze(_to_numpy(step_data.get("targets")))
        predictions = np.squeeze(_to_numpy(step_data.get("predictions")))

        if targets.size == 0 or predictions.size == 0:
            continue

        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        n_samples = min(targets.shape[0], predictions.shape[0])
        n_horizons = min(targets.shape[1], predictions.shape[1])
        targets = targets[:n_samples, :n_horizons]
        predictions = predictions[:n_samples, :n_horizons]

        # Group by horizon so each contiguous area represents one prediction step.
        targets_plot = np.concatenate([targets[:, h] for h in range(n_horizons)])
        predictions_plot = np.concatenate([predictions[:, h] for h in range(n_horizons)])

        x = np.arange(targets_plot.shape[0])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=targets_plot,
                mode="lines",
                name="target",
                legendgroup="target",
                showlegend=(row == 1),
                line=dict(color="purple", width=2),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=predictions_plot,
                mode="lines",
                name="prediction",
                legendgroup="prediction",
                showlegend=(row == 1),
                line=dict(color="orange", width=2),
            ),
            row=row,
            col=1,
        )

        # Add shaded bands so users can see which region corresponds to each prediction step.
        for h in range(n_horizons):
            x0 = h * n_samples
            x1 = (h + 1) * n_samples
            band_color = "rgba(99, 102, 241, 0.05)" if h % 2 == 0 else "rgba(14, 165, 233, 0.05)"
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=band_color,
                line_width=0,
                row=row,
                col=1,
                annotation_text=f"Pred {h + 1}",
                annotation_position="top left",
            )

        fig.update_yaxes(title_text=f"Epoch {step_data.get('step', row - 1)}", row=row, col=1)

    fig.update_xaxes(title_text="Grouped index by prediction step", row=rows, col=1)
    fig.update_layout(
        title="Prediction vs Ground Truth over Epochs (Normalized, All Horizons, Grouped by Step)",
        height=max(250 * rows, 520),
        hovermode="x unified",
    )
    return fig


def _build_features_chart(last):
    matrix, names = _extract_feature_matrix(last)
    if matrix.size == 0 or not names:
        fig = go.Figure()
        fig.update_layout(title="X Features - Full Range", height=500)
        return fig

    t = _global_time_series(last)
    if t.size:
        t = t - np.min(t)
    if len(t) != len(matrix):
        t = np.arange(len(matrix))

    n_features = matrix.shape[1]
    vertical_spacing = min(0.03, 0.18 / max(n_features - 1, 1))

    fig = make_subplots(
        rows=n_features,
        cols=1,
        subplot_titles=[f"{name} - Full Time Range" for name in names],
        vertical_spacing=vertical_spacing,
    )

    for i, name in enumerate(names, start=1):
        fig.add_trace(
            go.Scatter(x=t, y=matrix[:, i - 1], mode="lines", name=name, showlegend=(i == 1)),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=name, row=i, col=1)

    fig.update_xaxes(title_text="Time index", row=n_features, col=1)
    fig.update_layout(
        title="X Features - Full Range",
        height=max(220 * n_features, 520),
        hovermode="x unified",
    )
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


def _build_y_exact_chart(data):
    if not data:
        fig = go.Figure()
        fig.update_layout(title="Generated Y over Each Epoch", height=500)
        return fig

    rows = len(data)
    vertical_spacing = min(0.04, 0.18 / max(rows - 1, 1))
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Epoch {step_data.get('step', idx)}" for idx, step_data in enumerate(data)],
        vertical_spacing=vertical_spacing,
    )

    for row, step_data in enumerate(data, start=1):
        y = _feature_series(step_data, "y")
        if y.size == 0:
            continue

        # Use a per-step local timeline so each epoch starts at 0 and ends at len(y)-1.
        x = np.arange(len(y), dtype=float)

        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name="generated y", showlegend=(row == 1)),
            row=row,
            col=1,
        )
        fig.update_xaxes(range=[0, max(len(y) - 1, 0)], row=row, col=1)
        fig.update_yaxes(title_text=f"Step {step_data.get('step', row - 1)}", row=row, col=1)

    fig.update_xaxes(title_text="Time index", row=rows, col=1)
    fig.update_layout(
        title="Generated Y over Each Epoch",
        height=max(250 * rows, 520),
        hovermode="x unified",
    )
    return fig


def _build_loss_trend_charts(data):
    if not data:
        empty_forecaster = go.Figure()
        empty_forecaster.update_layout(title="Forecaster Loss Trend over Epochs", height=420)
        empty_generator = go.Figure()
        empty_generator.update_layout(title="Generator Loss Trend over Epochs", height=420)
        return empty_forecaster, empty_generator

    def _normalize_step_loss_map(loss_obj):
        """Return a normalized {int_step_idx: float_loss} map."""
        normalized = {}
        if isinstance(loss_obj, dict):
            for raw_step, raw_loss in loss_obj.items():
                try:
                    step_idx = int(raw_step)
                    normalized[step_idx] = float(raw_loss)
                except (TypeError, ValueError):
                    continue
            return normalized

        if isinstance(loss_obj, (list, tuple)):
            for step_idx, raw_loss in enumerate(loss_obj):
                try:
                    normalized[int(step_idx)] = float(raw_loss)
                except (TypeError, ValueError):
                    continue
        return normalized

    outer_steps = [int(step_data.get("step", idx)) for idx, step_data in enumerate(data)]
    model_maps = [_normalize_step_loss_map(step_data.get("model_losses", {})) for step_data in data]
    generator_maps = [_normalize_step_loss_map(step_data.get("generator_loss", {})) for step_data in data]

    model_epoch_ids = sorted({k for loss_map in model_maps for k in loss_map.keys()})
    generator_epoch_ids = sorted({k for loss_map in generator_maps for k in loss_map.keys()})

    fig_forecaster = go.Figure()
    for outer_step, loss_map in zip(outer_steps, model_maps):
        y_values = [loss_map.get(epoch_idx, np.nan) for epoch_idx in model_epoch_ids]
        fig_forecaster.add_trace(
            go.Scatter(
                x=model_epoch_ids,
                y=y_values,
                mode="lines+markers",
                name=f"step {outer_step}",
                line=dict(width=2),
                marker=dict(size=6),
                connectgaps=False,
            )
        )

    fig_forecaster.update_layout(
        title="Forecaster Loss Trend over Epochs (Per Step)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=420,
        hovermode="x unified",
    )

    fig_generator = go.Figure()
    for outer_step, loss_map in zip(outer_steps, generator_maps):
        y_values = [loss_map.get(epoch_idx, np.nan) for epoch_idx in generator_epoch_ids]
        fig_generator.add_trace(
            go.Scatter(
                x=generator_epoch_ids,
                y=y_values,
                mode="lines+markers",
                name=f"step {outer_step}",
                line=dict(width=2),
                marker=dict(size=6),
                connectgaps=False,
            )
        )

    fig_generator.update_layout(
        title="Generator Loss Trend over Epochs (Per Step)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=420,
        hovermode="x unified",
    )

    return fig_forecaster, fig_generator


def _build_epoch_prediction_timelines(data):
    """Create a grid of 10 timeline charts (2x5), one for each epoch, showing first step prediction."""
    if not data:
        fig = go.Figure()
        fig.update_layout(title="Epoch Prediction Timelines", height=400)
        return fig

    n_epochs = len(data)
    n_cols = 5
    n_rows = (n_epochs + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Epoch {step_data.get('step', idx)}/{n_epochs - 1}" for idx, step_data in enumerate(data)],
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    for idx, step_data in enumerate(data):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        targets = np.squeeze(_inverse_y(step_data.get("targets"), step_data))
        predictions = np.squeeze(_inverse_y(step_data.get("predictions"), step_data))

        if targets.size == 0 or predictions.size == 0:
            continue

        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        n_samples = min(targets.shape[0], predictions.shape[0])
        n_horizons = min(targets.shape[1], predictions.shape[1])
        targets = targets[:n_samples, :n_horizons]
        predictions = predictions[:n_samples, :n_horizons]

        # Only plot first step (horizon 0)
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_samples),
                y=targets[:, 0],
                mode="lines",
                name="target",
                legendgroup="target",
                showlegend=(idx == 0),
                line=dict(color="blue", width=2),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_samples),
                y=predictions[:, 0],
                mode="lines",
                name="prediction",
                legendgroup="prediction",
                showlegend=(idx == 0),
                line=dict(color="orange", width=2),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Sample", row=row, col=col, tickfont=dict(size=10))
        fig.update_yaxes(title_text="Y", row=row, col=col, tickfont=dict(size=10))

    fig.update_layout(
        title="First Step Prediction Timeline - All Epochs (Original Scale)",
        height=max(280 * n_rows, 520),
        hovermode="x unified",
    )
    return fig
def _compute_prediction_metrics(targets_raw, predictions_raw, step_data):
    """Compute quality metrics from raw prediction arrays."""
    import numpy as np

    t = np.squeeze(_inverse_y(targets_raw, step_data))
    p = np.squeeze(_inverse_y(predictions_raw, step_data))

    if t.size == 0 or p.size == 0:
        return None

    # flatten multi-horizon to first step only for diagnostics
    if t.ndim > 1: t = t[:, 0]
    if p.ndim > 1: p = p[:, 0]

    n = min(len(t), len(p))
    t, p = t[:n], p[:n]

    err   = p - t
    mse   = float(np.mean(err ** 2))
    mae   = float(np.mean(np.abs(err)))
    bias  = float(np.mean(err))
    rmse  = float(np.sqrt(mse))

    # directional accuracy
    t_diff = np.diff(t)
    p_diff = np.diff(p)
    dir_acc = float(np.mean((t_diff * p_diff) > 0) * 100) if len(t_diff) > 0 else 0.0

    # R²
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    # relative RMSE (as % of target std)
    t_std = float(np.std(t))
    rel_rmse = (rmse / t_std * 100) if t_std > 1e-10 else 100.0

    return dict(mse=mse, mae=mae, bias=bias, rmse=rmse,
                dir_acc=dir_acc, r2=r2, rel_rmse=rel_rmse)


def _grade_metrics(m):
    pts = 0
    # Generator wins by making RMSE large relative to target variance
    if m["rel_rmse"] > 80:   pts += 3
    elif m["rel_rmse"] > 50: pts += 2
    elif m["rel_rmse"] > 20: pts += 1

    # Low R² = forecaster predictions are uncorrelated with targets = generator winning
    if m["r2"] < 0.1:    pts += 3
    elif m["r2"] < 0.4:  pts += 2
    elif m["r2"] < 0.7:  pts += 1

    # Directional accuracy low = forecaster can't track direction = good for generator
    if m["dir_acc"] < 45:    pts += 2
    elif m["dir_acc"] < 55:  pts += 1

    # Bias is still bad regardless — a biased generator is predictable
    if abs(m["bias"]) < 0.1:  pts += 2
    elif abs(m["bias"]) < 0.3: pts += 1

    if pts >= 8: return "quality-good", "generator winning"
    if pts >= 5: return "quality-warn", "contested"
    return "quality-bad", "forecaster winning"


def _build_prediction_quality_panel(data):
    if not data:
        return '<div class="empty-state">No prediction data available.</div>'

    css = """<style>
    .quality-badge{display:inline-block;font-size:11px;font-weight:500;padding:2px 8px;border-radius:6px;margin-right:4px}
    .quality-good{background:#dcfce7;color:#166534}
    .quality-warn{background:#fef9c3;color:#854d0e}
    .quality-bad{background:#fee2e2;color:#991b1b}
    .pq-epoch{border:0.5px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin-bottom:12px}
    .pq-header{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
    .pq-epoch-title{font-size:14px;font-weight:500}
    .pq-epoch-mse{font-size:13px;color:#6b7280}
    .pq-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-bottom:10px}
    .pq-metric{background:#f9fafb;border-radius:8px;padding:8px 10px}
    .pq-metric .ml{font-size:11px;color:#6b7280;margin:0 0 2px}
    .pq-metric .mv{font-size:15px;font-weight:500;margin:0}
    .pq-metric .ms{font-size:10px;color:#9ca3af;margin:2px 0 0}
    .pq-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
    .pq-bar-label{font-size:11px;color:#6b7280;width:90px;flex-shrink:0}
    .pq-bar-track{flex:1;height:6px;border-radius:3px;background:#e5e7eb;overflow:hidden}
    .pq-bar-fill{height:100%;border-radius:3px}
    .pq-bar-val{font-size:11px;color:#6b7280;width:40px;text-align:right;flex-shrink:0}
    .pq-verdict{font-size:12px;color:#6b7280;margin-top:8px;padding-top:8px;border-top:0.5px solid #f3f4f6}
    .pq-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:1.25rem}
    .pq-sum-card{background:#f9fafb;border-radius:8px;padding:12px}
    .pq-sum-card .sl{font-size:12px;color:#6b7280;margin:0 0 4px}
    .pq-sum-card .sv{font-size:20px;font-weight:500;margin:0}
    .pq-sum-card .ss{font-size:11px;color:#9ca3af;margin:3px 0 0}
    .pq-section-title{font-size:12px;font-weight:500;color:#6b7280;text-transform:uppercase;letter-spacing:.04em;margin:1.25rem 0 .75rem}
    </style>"""

    def badge(cls, label):
        return f'<span class="quality-badge {cls}">{label}</span>'

    def bar(label, pct, color):
        pct = max(2, min(100, pct))
        return (f'<div class="pq-bar-row">'
                f'<span class="pq-bar-label">{label}</span>'
                f'<div class="pq-bar-track"><div class="pq-bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
                f'<span class="pq-bar-val">{pct:.0f}%</span></div>')

    bar_color = {"quality-good": "#16a34a", "quality-warn": "#ca8a04", "quality-bad": "#dc2626"}

    all_metrics = []
    for step_data in data:
        m = _compute_prediction_metrics(
            step_data.get("predictions"),
            step_data.get("targets"),
            step_data,
        )
        if m:
            m["step"] = step_data.get("step", len(all_metrics))
            all_metrics.append(m)

    if not all_metrics:
        return '<div class="empty-state">No prediction arrays found in epoch data.</div>'

    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["mse"])
    best_mse = all_metrics[best_idx]["mse"]
    first_mse = all_metrics[0]["mse"]
    improvement = (best_mse - first_mse) / first_mse * 100  # higher is better
    early_stop = best_idx < len(all_metrics) - 1  # val started falling after peak
    best_dir = max(m["dir_acc"] for m in all_metrics)

    summary = f"""<div class="pq-summary">
      <div class="pq-sum-card"><p class="sl">Best test MSE</p><p class="sv">{best_mse:.4f}</p><p class="ss">epoch {all_metrics[best_idx]['step']}</p></div>
      <div class="pq-sum-card"><p class="sl">MSE improvement</p><p class="sv">{improvement:.1f}%</p><p class="ss">epoch 0 → best</p></div>
      <div class="pq-sum-card"><p class="sl">Best dir. accuracy</p><p class="sv">{best_dir:.0f}%</p><p class="ss">across all epochs</p></div>
      <div class="pq-sum-card"><p class="sl">Overfitting after ep.</p><p class="sv">{all_metrics[best_idx]['step'] if early_stop else '—'}</p><p class="ss">{'MSE rising' if early_stop else 'still improving'}</p></div>
    </div>"""

    parts = [css, summary, '<p class="pq-section-title">Per-epoch breakdown</p>']

    for i, m in enumerate(all_metrics):
        cls, label = _grade_metrics(m)
        is_best = (i == best_idx)
        star = " ★ best" if is_best else ""
        color = bar_color[cls]

        bias_dir = ("over-predicting" if m["bias"] > 0.05
                    else "under-predicting" if m["bias"] < -0.05
                    else "unbiased")
        bias_cls = ("quality-good" if abs(m["bias"]) < 0.1
                    else "quality-warn" if abs(m["bias"]) < 0.3
                    else "quality-bad")

        dir_cls  = "quality-good" if m["dir_acc"] > 65 else ("quality-warn" if m["dir_acc"] > 50 else "quality-bad")
        r2_cls   = "quality-good" if m["r2"] > 0.7 else ("quality-warn" if m["r2"] > 0.4 else "quality-bad")
        rel_cls  = "quality-good" if m["rel_rmse"] < 20 else ("quality-warn" if m["rel_rmse"] < 50 else "quality-bad")

        r2_pct   = max(0, min(100, m["r2"] * 100))
        rel_pct  = max(0, 100 - m["rel_rmse"])

        if cls == "quality-good":
            verdict = "Generator is successfully fooling the forecaster — high error, low correlation, poor directional tracking."
        elif cls == "quality-warn":
            parts_v = []
            if m["dir_acc"] > 60: parts_v.append("Forecaster still tracks direction reasonably well.")
            if m["r2"] > 0.5: parts_v.append("R² still moderate — forecaster partially correlated with targets.")
            if abs(m["bias"]) > 0.2: parts_v.append(f"Generator bias ({bias_dir}) makes it predictable.")
            verdict = "Contested. " + " ".join(parts_v) if parts_v else "Neither side has a clear edge."
        else:
            parts_v = []
            if m["rel_rmse"] < 20: parts_v.append("RMSE is low — forecaster handling generated data well.")
            if m["r2"] > 0.7: parts_v.append("High R² — predictions closely follow targets.")
            if abs(m["bias"]) > 0.3: parts_v.append(f"Generator is biased ({bias_dir}) and easy to predict.")
            verdict = "Forecaster winning. " + " ".join(parts_v) if parts_v else "Forecaster is handling the generated data well — generator needs improvement."

        bias_str = f"+{m['bias']:.4f}" if m["bias"] > 0 else f"{m['bias']:.4f}"

        parts.append(f"""<div class="pq-epoch">
          <div class="pq-header">
            <span class="pq-epoch-title">Epoch {m['step']}{star}</span>
            <span class="pq-epoch-mse">MSE {m['mse']:.4f}</span>
            {badge(cls, label)}
            {badge(bias_cls, bias_dir)}
          </div>
          <div class="pq-metrics">
            <div class="pq-metric"><p class="ml">RMSE</p><p class="mv">{m['rmse']:.4f}</p><p class="ms">error scale</p></div>
            <div class="pq-metric"><p class="ml">R²</p><p class="mv">{m['r2']:.3f}</p><p class="ms">variance explained</p></div>
            <div class="pq-metric"><p class="ml">Dir. accuracy</p><p class="mv">{m['dir_acc']:.1f}%</p><p class="ms">up/down correct</p></div>
            <div class="pq-metric"><p class="ml">Rel. RMSE</p><p class="mv">{m['rel_rmse']:.1f}%</p><p class="ms">% of target std</p></div>
            <div class="pq-metric"><p class="ml">Bias</p><p class="mv">{bias_str}</p><p class="ms">mean error</p></div>
          </div>
          {bar('R² score', r2_pct, bar_color[r2_cls])}
          {bar('Dir. accuracy', m['dir_acc'], bar_color[dir_cls])}
          {bar('RMSE quality', rel_pct, bar_color[rel_cls])}
          <p class="pq-verdict">{verdict}</p>
        </div>""")

    return "\n".join(parts)

def _render_section_html(section_id: str, data_bundle: dict):
    data = data_bundle["data"]
    grid_search_history = data_bundle["grid_search_history"]
    config_dict = data_bundle["config"]

    if section_id == "config":
        return _build_config_table(config_dict)

    if section_id == "grid_search":
        table_html, fig = _build_grid_search_table_and_chart(grid_search_history)
        return table_html + _fig_to_html(fig)

    if section_id == "epoch_summary":
        table_html, fig = _build_epoch_summary_table_and_chart(data)
        quality_html = _build_quality_panel(data)
        return quality_html + table_html + _fig_to_html(fig)
    if section_id == "params_exact":
        return _fig_to_html(_build_params_exact_chart(data))

    if section_id == "params_heat":
        return _fig_to_html(_build_params_heatmap_chart(data))

    if section_id == "params_delta":
        return _fig_to_html(_build_params_delta_line_chart(data))

    if section_id == "pred_history":
        quality_html = _build_prediction_quality_panel(data) 
        return quality_html + _fig_to_html(_build_prediction_history_chart(data))

    if section_id == "x_features":
        if not data:
            return _fig_to_html(_build_features_chart({}))
        return _fig_to_html(_build_features_chart(data[-1]))

    if section_id == "y_generated":
        return _fig_to_html(_build_y_history_chart(data)) + _fig_to_html(_build_y_exact_chart(data))

    if section_id == "loss_trends":
        fig_forecaster_loss, fig_generator_loss = _build_loss_trend_charts(data)
        return _fig_to_html(fig_forecaster_loss) + _fig_to_html(fig_generator_loss)

    return _empty_state("Unknown section requested.")


def _get_section_html(source_name: str, section_id: str):
    data_bundle = _get_data_for_source(source_name)
    cache_key = (*data_bundle["cache_key"], section_id)
    cached_html = _SECTION_HTML_CACHE.get(cache_key)
    if cached_html is not None:
        return cached_html

    html = _render_section_html(section_id, data_bundle)

    # Keep cache bounded to avoid unbounded memory growth.
    if len(_SECTION_HTML_CACHE) > 128:
        _SECTION_HTML_CACHE.clear()
    _SECTION_HTML_CACHE[cache_key] = html
    return html


@app.route("/")
def index():
    available_sources = _list_output_sources("output/new/")
    requested_source = request.args.get("source", "")
    active_source = _resolve_selected_source(available_sources, requested_source)
    return render_template(
        "index.html",
        available_sources=available_sources,
        active_source=active_source,
    )


@app.get("/api/section")
def api_section():
    available_sources = _list_output_sources("output/new/")
    requested_source = request.args.get("source", "")
    active_source = _resolve_selected_source(available_sources, requested_source)
    section_id = request.args.get("section", "")

    allowed_sections = {
        "config",
        "grid_search",
        "epoch_summary",
        "params_exact",
        "params_heat",
        "params_delta",
        "pred_history",
        "x_features",
        "y_generated",
        "loss_trends",
    }

    if section_id not in allowed_sections:
        return jsonify({"ok": False, "error": "Invalid section id."}), 400

    html = _get_section_html(active_source, section_id)
    return jsonify({"ok": True, "section": section_id, "source": active_source, "html": html})


if __name__ == "__main__":
    app.run(debug=True)