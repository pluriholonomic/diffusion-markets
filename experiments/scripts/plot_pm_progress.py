from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path
    pred_col: str


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _load_metrics(run_dir: Path) -> Dict:
    """
    Most runs write metrics.json as {"metrics": <payload>, ...}.
    We return the <payload>.
    """
    j = _read_json(run_dir / "metrics.json")
    if isinstance(j, dict) and "metrics" in j and isinstance(j["metrics"], dict):
        return j["metrics"]
    return j


def _infer_pred_col(run_dir: Path) -> str:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = _read_json(cfg_path)
        if isinstance(cfg, dict) and "pred_col" in cfg and cfg["pred_col"]:
            return str(cfg["pred_col"])
        # older configs nest under eval
        if isinstance(cfg, dict) and isinstance(cfg.get("eval"), dict) and cfg["eval"].get("pred_col"):
            return str(cfg["eval"]["pred_col"])
    # fallback: common names
    return "pred_prob"


def _load_predictions_df(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions.parquet in run_dir={run_dir}")
    return pd.read_parquet(p)


def calibration_bins(
    *,
    q: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Returns per-bin calibration summary:
      bin_lo, bin_hi, n, q_mean, y_mean, abs_gap
    """
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if q.shape[0] != y.shape[0]:
        raise ValueError("q and y must have same length")

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.digitize(q, edges, right=True) - 1
    idx = np.clip(idx, 0, int(n_bins) - 1)

    rows: List[Dict[str, object]] = []
    n = int(len(q))
    for b in range(int(n_bins)):
        m = idx == b
        if not np.any(m):
            continue
        qb = float(np.mean(q[m]))
        yb = float(np.mean(y[m]))
        rows.append(
            {
                "bin": int(b),
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": int(np.sum(m)),
                "q_mean": qb,
                "y_mean": yb,
                "abs_gap": abs(yb - qb),
                "mass": float(np.sum(m) / n) if n > 0 else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    return out


def plot_reliability(
    *,
    runs: List[RunSpec],
    out_path: Path,
    n_bins: int = 20,
    y_col: str = "y",
    clip_eps: float = 1e-6,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.6, 5.2))
    ax = plt.gca()

    ax.plot([0, 1], [0, 1], color="black", lw=1, alpha=0.6, label="ideal y=x")

    for rs in runs:
        df = _load_predictions_df(rs.run_dir)
        if y_col not in df.columns:
            raise ValueError(f"Missing y_col={y_col!r} in {rs.run_dir}")
        if rs.pred_col not in df.columns:
            raise ValueError(f"Missing pred_col={rs.pred_col!r} in {rs.run_dir}")

        y = df[y_col].to_numpy()
        q = df[rs.pred_col].to_numpy()
        q = np.clip(q.astype(np.float64), float(clip_eps), 1.0 - float(clip_eps))

        tab = calibration_bins(q=q, y=y, n_bins=int(n_bins))
        # point size ~ sqrt(n) for visibility
        s = np.sqrt(tab["n"].to_numpy(dtype=np.float64))
        s = 12.0 * (s / (np.max(s) if np.max(s) > 0 else 1.0))
        ax.scatter(tab["q_mean"], tab["y_mean"], s=s, alpha=0.85, label=rs.label)
        ax.plot(tab["q_mean"], tab["y_mean"], lw=1.5, alpha=0.7)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability in bin")
    ax.set_ylabel("Empirical frequency (mean y) in bin")
    ax.set_title(f"Reliability diagram (n_bins={n_bins})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_approachability_app_err(
    *,
    runs: List[RunSpec],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.2, 3.6))
    ax = plt.gca()

    for rs in runs:
        m = _load_metrics(rs.run_dir)
        app = m.get("approachability")
        if not isinstance(app, dict):
            raise ValueError(f"No approachability block in metrics for {rs.run_dir}")
        curve = app.get("curve")
        if not isinstance(curve, dict):
            raise ValueError(f"Missing approachability.curve in {rs.run_dir}")
        t = np.asarray(curve.get("t", []), dtype=np.float64)
        err = np.asarray(curve.get("app_err", []), dtype=np.float64)
        if t.size == 0 or err.size == 0:
            raise ValueError(f"Empty approachability curve for {rs.run_dir}")

        final = app.get("final", {}) if isinstance(app.get("final"), dict) else {}
        final_err = final.get("app_err", None)
        lbl = rs.label
        if final_err is not None:
            try:
                lbl = f"{lbl} (final={float(final_err):.3f})"
            except Exception:
                pass
        ax.plot(t, err, lw=2, label=lbl)

    ax.set_xlabel("T (forecasts processed)")
    ax.set_ylabel("AppErr(T)")
    ax.set_title("Blackwell approachability: group×bin payoff box distance")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def topic_summary_table(
    *,
    df: pd.DataFrame,
    pred_col: str,
    label: str,
    group_col: str = "topic",
    y_col: str = "y",
    n_bins: int = 20,
    clip_eps: float = 1e-6,
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col={group_col!r} in dataframe.")
    if y_col not in df.columns:
        raise ValueError(f"Missing y_col={y_col!r} in dataframe.")
    if pred_col not in df.columns:
        raise ValueError(f"Missing pred_col={pred_col!r} in dataframe.")

    g = df[group_col].astype(str).fillna("NA")
    y = df[y_col].astype(float)
    q = df[pred_col].astype(float).clip(lower=float(clip_eps), upper=1.0 - float(clip_eps))

    rows: List[Dict[str, object]] = []
    for key, ix in g.groupby(g).groups.items():
        # ix is index labels; convert to positional indices for iloc
        sub = df.loc[ix]
        yk = sub[y_col].to_numpy(dtype=np.float64)
        qk = sub[pred_col].to_numpy(dtype=np.float64)
        qk = np.clip(qk, float(clip_eps), 1.0 - float(clip_eps))
        tab = calibration_bins(q=qk, y=yk, n_bins=int(n_bins))
        ece_k = float(np.sum(tab["mass"].to_numpy(dtype=np.float64) * tab["abs_gap"].to_numpy(dtype=np.float64)))

        bias = float(np.mean(yk - qk))
        rows.append(
            {
                "model": label,
                group_col: str(key),
                "n": int(len(sub)),
                "q_mean": float(np.mean(qk)),
                "y_mean": float(np.mean(yk)),
                "bias": bias,
                "abs_bias": abs(bias),
                "ece": ece_k,
            }
        )

    out = pd.DataFrame(rows).sort_values([group_col, "model"]).reset_index(drop=True)
    return out


def plot_topic_bias_bars(
    *,
    runs: List[RunSpec],
    out_path: Path,
    group_col: str = "topic",
    y_col: str = "y",
    n_bins: int = 20,
    clip_eps: float = 1e-6,
) -> None:
    import matplotlib.pyplot as plt

    tabs = []
    for rs in runs:
        df = _load_predictions_df(rs.run_dir)
        tabs.append(
            topic_summary_table(
                df=df,
                pred_col=rs.pred_col,
                label=rs.label,
                group_col=group_col,
                y_col=y_col,
                n_bins=int(n_bins),
                clip_eps=float(clip_eps),
            )
        )
    full = pd.concat(tabs, axis=0, ignore_index=True)

    topics = sorted(full[group_col].astype(str).unique().tolist())
    models = [rs.label for rs in runs]

    # bar positions
    x = np.arange(len(topics), dtype=np.float64)
    width = 0.8 / max(1, len(models))

    plt.figure(figsize=(7.2, 3.8))
    ax = plt.gca()

    for i, m in enumerate(models):
        sub = full[full["model"] == m].set_index(group_col)
        vals = np.array([float(sub.loc[t, "abs_bias"]) if t in sub.index else np.nan for t in topics], dtype=np.float64)
        ax.bar(x + (i - (len(models) - 1) / 2.0) * width, vals, width=width, label=m, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=0)
    ax.set_ylabel("abs calibration bias |E[y - q] |")
    ax.set_title("Group calibration bias by topic")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topic_reliability_grid(
    *,
    runs: List[RunSpec],
    out_path: Path,
    group_col: str = "topic",
    y_col: str = "y",
    n_bins: int = 20,
    clip_eps: float = 1e-6,
) -> None:
    import matplotlib.pyplot as plt

    # Determine topics from the first run (assume common across runs).
    df0 = _load_predictions_df(runs[0].run_dir)
    if group_col not in df0.columns:
        raise ValueError(f"Missing group_col={group_col!r} in {runs[0].run_dir}")
    topics = sorted(df0[group_col].astype(str).fillna("NA").unique().tolist())
    if len(topics) == 0:
        raise ValueError("No topics found.")

    ncols = len(topics)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, topic in zip(axes, topics):
        ax.plot([0, 1], [0, 1], color="black", lw=1, alpha=0.5)
        for rs in runs:
            df = _load_predictions_df(rs.run_dir)
            if group_col not in df.columns:
                raise ValueError(f"Missing group_col={group_col!r} in {rs.run_dir}")
            if y_col not in df.columns:
                raise ValueError(f"Missing y_col={y_col!r} in {rs.run_dir}")
            if rs.pred_col not in df.columns:
                raise ValueError(f"Missing pred_col={rs.pred_col!r} in {rs.run_dir}")

            sub = df[df[group_col].astype(str).fillna("NA") == topic]
            if len(sub) == 0:
                continue
            y = sub[y_col].to_numpy(dtype=np.float64)
            q = sub[rs.pred_col].to_numpy(dtype=np.float64)
            q = np.clip(q, float(clip_eps), 1.0 - float(clip_eps))

            tab = calibration_bins(q=q, y=y, n_bins=int(n_bins))
            ax.plot(tab["q_mean"], tab["y_mean"], lw=1.8, alpha=0.85, label=rs.label)
            ax.scatter(tab["q_mean"], tab["y_mean"], s=18, alpha=0.85)

        ax.set_title(f"{group_col}={topic}")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Empirical frequency (mean y) in bin")
    for ax in axes:
        ax.set_xlabel("Mean predicted probability in bin")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

    # single legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=True)
    fig.suptitle(f"Reliability by {group_col} (n_bins={n_bins})", y=1.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-run", type=str, required=True, help="Run dir for market baseline pm_eval")
    ap.add_argument("--ar-run", type=str, required=True, help="Run dir for AR pm_eval")
    ap.add_argument("--diff-run", type=str, required=True, help="Run dir for diffusion pm_eval")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for plots")
    ap.add_argument("--n-bins", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    market = Path(args.market_run)
    ar = Path(args.ar_run)
    diff = Path(args.diff_run)

    runs = [
        RunSpec(label="market", run_dir=market, pred_col=_infer_pred_col(market)),
        RunSpec(label="AR (Qwen3)", run_dir=ar, pred_col=_infer_pred_col(ar)),
        RunSpec(label="diffusion", run_dir=diff, pred_col=_infer_pred_col(diff)),
    ]

    plot_approachability_app_err(
        runs=runs,
        out_path=out_dir / "pm_app_err_comparison.png",
    )
    plot_reliability(
        runs=runs,
        out_path=out_dir / "pm_reliability.png",
        n_bins=int(args.n_bins),
    )

    # Topic-group calibration plots
    plot_topic_reliability_grid(
        runs=runs,
        out_path=out_dir / "pm_topic_reliability.png",
        group_col="topic",
        n_bins=int(args.n_bins),
    )
    plot_topic_bias_bars(
        runs=runs,
        out_path=out_dir / "pm_topic_abs_bias.png",
        group_col="topic",
        n_bins=int(args.n_bins),
    )

    # Write per-topic table for easy inspection
    try:
        tables = []
        for rs in runs:
            df = _load_predictions_df(rs.run_dir)
            tables.append(topic_summary_table(df=df, pred_col=rs.pred_col, label=rs.label, n_bins=int(args.n_bins)))
        pd.concat(tables, axis=0, ignore_index=True).to_csv(out_dir / "pm_topic_table.csv", index=False)
    except Exception as e:
        (out_dir / "pm_topic_table_error.txt").write_text(f"{e}\n")

    # Also write a small text summary for quick scanning.
    lines = []
    for rs in runs:
        m = _load_metrics(rs.run_dir)
        app = m.get("approachability", {}) if isinstance(m.get("approachability"), dict) else {}
        final = app.get("final", {}) if isinstance(app.get("final"), dict) else {}
        lines.append(f"{rs.label}\n  run_dir: {rs.run_dir}\n  pred_col: {rs.pred_col}\n  brier: {m.get('brier')}\n  logloss: {m.get('logloss')}\n  ece: {m.get('ece')}\n  app_final: {final.get('app_err')}\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pm_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()


