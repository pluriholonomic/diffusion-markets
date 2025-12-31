from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forecastbench.benchmarks.parity import ParityMarket, ParitySpec, sample_parity_dataset
from forecastbench.benchmarks.subcubes import (
    parity_S_equals_J_diagnostic,
    worst_group_abs_cond_mean_over_assignments,
)
from forecastbench.metrics import (
    best_bounded_trader_profit,
    brier_loss,
    expected_calibration_error,
    log_loss,
    squared_calibration_error,
)
from forecastbench.models import ARCoTPredictor, ARCoTSpec
from forecastbench.runner import RunArtifacts


def _print_summary_table(rows: list[dict], title: str) -> None:
    cols = ["model", "brier", "log", "sce", "ece", "arb(B=1)"]
    widths = {c: len(c) for c in cols}
    for r in rows:
        widths["model"] = max(widths["model"], len(r["name"]))
        widths["brier"] = max(widths["brier"], len(f'{r["brier"]:.6f}'))
        widths["log"] = max(widths["log"], len(f'{r["logloss"]:.6f}'))
        widths["sce"] = max(widths["sce"], len(f'{r["sce"]:.6f}'))
        widths["ece"] = max(widths["ece"], len(f'{r["ece"]:.6f}'))
        widths["arb(B=1)"] = max(widths["arb(B=1)"], len(f'{r["arb_profit"]:.6f}'))

    def fmt_row(model: str, brier: str, log: str, sce: str, ece: str, arb: str) -> str:
        return (
            f"{model:<{widths['model']}}  "
            f"{brier:>{widths['brier']}}  "
            f"{log:>{widths['log']}}  "
            f"{sce:>{widths['sce']}}  "
            f"{ece:>{widths['ece']}}  "
            f"{arb:>{widths['arb(B=1)']}}"
        )

    print(title)
    print(fmt_row(*cols))
    print("-" * (sum(widths.values()) + 10))
    for r in rows:
        print(
            fmt_row(
                r["name"],
                f'{r["brier"]:.6f}',
                f'{r["logloss"]:.6f}',
                f'{r["sce"]:.6f}',
                f'{r["ece"]:.6f}',
                f'{r["arb_profit"]:.6f}',
            )
        )


def cmd_parity(args: argparse.Namespace) -> None:
    """
    Synthetic parity markets:
      f(z) = 1/2 + (alpha/2) chi_S(z)

    Compares:
    - oracle (truth)
    - constant 0.5 (optimal L-junta when L<k for parity)
    - analytic diffusion: T_rho f
    - L-query-oracle: succeeds iff L>=k (for sanity)
    """
    spec = ParitySpec(d=args.d, k=args.k, alpha=args.alpha, seed=args.seed)
    data = sample_parity_dataset(spec, n=args.n)
    z = data["z"]
    p_true = data["p_true"]
    y = data["y"]

    mkt = ParityMarket.create(spec)
    S = tuple(int(i) for i in data["S"].tolist())

    q_oracle = p_true
    q_const = np.full_like(p_true, 0.5, dtype=np.float32)
    q_diff = mkt.diffusion_analytic(z, rho=args.rho).astype(np.float32)
    q_L_query_oracle = q_oracle if args.L >= args.k else q_const

    models = [
        ("oracle", q_oracle),
        ("const_0.5", q_const),
        (f"diff_analytic_rho={args.rho}", q_diff),
        (f"L_query_oracle(L={args.L})", q_L_query_oracle),
    ]

    llm_meta = None
    if args.llm_model is not None:
        try:
            spec_llm = ARCoTSpec(
                model_name_or_path=args.llm_model,
                device=args.llm_device,
                dtype=args.llm_dtype,
                device_map=args.llm_device_map,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                max_new_tokens=args.llm_max_new_tokens,
                include_cot=not args.llm_no_cot,
            )
            pred = ARCoTPredictor(spec_llm)
            n_text = min(len(z), int(args.llm_max_examples))
            texts = [mkt.encode_text(z[i], style=args.text_style) for i in range(n_text)]
            q_llm, llm_meta = pred.predict_proba(
                texts,
                L=args.L,
                K=args.llm_K,
                seed=args.seed,
                max_examples=None,
                aggregate=args.llm_agg,
            )
            # Align lengths (evaluate on same subset)
            models.append((f"llm({args.llm_model})_K={args.llm_K}_L={args.L}", q_llm))
            # truncate evaluation arrays for LLM row
        except Exception as e:
            print(f"[warn] LLM runner failed ({args.llm_model}): {e}")

    rows: list[dict] = []
    for name, q in models:
        if q.shape[0] != p_true.shape[0]:
            # LLM rows may be evaluated on a subset for speed
            p_eval = p_true[: q.shape[0]]
            y_eval = y[: q.shape[0]]
        else:
            p_eval = p_true
            y_eval = y
        rows.append(
            {
                "name": name,
                "brier": brier_loss(q, y_eval),
                "logloss": log_loss(q, y_eval),
                "sce": squared_calibration_error(q, p_eval),
                "ece": expected_calibration_error(q, y_eval, n_bins=args.bins),
                "arb_profit": best_bounded_trader_profit(
                    p_eval, q, B=1.0, transaction_cost=args.transaction_cost
                ),
                "n_eval": int(len(q)),
            }
        )

    _print_summary_table(rows, title="Parity benchmark")

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "parity",
            "spec": asdict(spec),
            "S": S,
            "n": int(args.n),
            "rho": float(args.rho),
            "L": int(args.L),
            "bins": int(args.bins),
            "transaction_cost": float(args.transaction_cost),
            "llm": {
                "model": args.llm_model,
                "K": args.llm_K,
                "max_examples": args.llm_max_examples,
                "device": args.llm_device,
                "dtype": args.llm_dtype,
                "device_map": args.llm_device_map,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p,
                "max_new_tokens": args.llm_max_new_tokens,
                "no_cot": bool(args.llm_no_cot),
                "text_style": args.text_style,
                "agg": args.llm_agg,
            }
            if args.llm_model is not None
            else None,
        },
    )
    arts.write_json("metrics.json", {"models": rows, "llm_meta": llm_meta})

    # Small plot: SCE bars (this is the paper-aligned quantity for parity).
    fig = plt.figure(figsize=(8, 3.2))
    names = [r["name"] for r in rows]
    sces = [r["sce"] for r in rows]
    plt.bar(names, sces)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("SCE = E[(q - f)^2]")
    plt.title("Parity: squared calibration error")
    plt.tight_layout()
    arts.savefig(fig, "sce_bar.png")
    plt.close(fig)

    # Optional: synthetic Blackwell approachability diagnostics on parity.
    # We re-use the same group×bin payoff family implementation used for Polymarket.
    if getattr(args, "blackwell", False):
        try:
            import pandas as pd

            from forecastbench.benchmarks.polymarket_eval import evaluate_group_bin_approachability
            from forecastbench.benchmarks.subcubes import keys_for_J

            bw_group = str(getattr(args, "bw_group", "chi_S"))
            curve_every = int(getattr(args, "bw_curve_every", 1000))
            topk = int(getattr(args, "bw_topk", 10))
            clip_eps = float(getattr(args, "bw_clip_eps", 1e-6))

            if getattr(args, "bw_eps", None) is not None:
                eps = float(args.bw_eps)
            else:
                # Fee margin consistent with pm_eval: eps = c/B (here B=1).
                eps = float(args.transaction_cost)

            bw_rows = []
            for (name, q), r in zip(models, rows):
                n = int(r["n_eval"])
                z_eval = z[:n]
                y_eval = y[:n]
                q_eval = q[:n].astype(np.float32, copy=False)

                if bw_group == "chi_S":
                    g = mkt.chi(z_eval).astype(np.int8)
                    df_bw = pd.DataFrame({"y": y_eval.astype(np.int8), "pred_prob": q_eval, "chi_S": g})
                    group_cols = ["chi_S"]
                elif bw_group == "subcube_S":
                    # Groups are assignments a∈{-1,+1}^S encoded as keys in [0,2^k).
                    g = keys_for_J(z_eval, mkt.S).astype(np.int64)
                    df_bw = pd.DataFrame({"y": y_eval.astype(np.int8), "pred_prob": q_eval, "S_key": g})
                    group_cols = ["S_key"]
                elif bw_group == "none":
                    df_bw = pd.DataFrame({"y": y_eval.astype(np.int8), "pred_prob": q_eval})
                    group_cols = None
                else:
                    raise ValueError(f"Unknown bw_group={bw_group!r} (expected chi_S|subcube_S|none)")

                app = evaluate_group_bin_approachability(
                    df_bw,
                    pred_col="pred_prob",
                    y_col="y",
                    group_cols=group_cols,
                    n_bins=int(args.bins),
                    eps=float(eps),
                    time_col=None,
                    curve_every=int(curve_every),
                    topk=int(topk),
                    clip_eps=float(clip_eps),
                )
                bw_rows.append(
                    {
                        "name": name,
                        "group": bw_group,
                        "eps": float(app["eps"]),
                        "final_app_err": float(app["final"]["app_err"]),
                        "n_groups": int(app["n_groups"]),
                        "n_bins": int(app["n_bins"]),
                        "top_violations": app["final"]["top_violations"][: min(5, len(app["final"]["top_violations"]))],
                        "curve": app["curve"],
                    }
                )

            arts.write_json(
                "blackwell.json",
                {
                    "family": "group_bin_indicator",
                    "bw_group": bw_group,
                    "bins": int(args.bins),
                    "curve_every": int(curve_every),
                    "eps": float(eps),
                    "rows": bw_rows,
                },
            )

            # Plot the approachability curves.
            fig = plt.figure(figsize=(7.5, 3.6))
            for br in bw_rows:
                t = np.asarray(br["curve"]["t"], dtype=np.int64)
                e = np.asarray(br["curve"]["app_err"], dtype=np.float64)
                plt.plot(t, e, label=f'{br["name"]} (final {br["final_app_err"]:.4g})')
            plt.xlabel("t")
            plt.ylabel("AppErr(t)")
            plt.title(f"Synthetic Blackwell approachability (parity): group={bw_group}, bins={int(args.bins)}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            arts.savefig(fig, "blackwell_app_err.png")
            plt.close(fig)
        except Exception as e:
            arts.write_text("blackwell_error.txt", f"{e}\n")

    print(f"Artifacts: {arts.run_dir}")


def cmd_groupstress(args: argparse.Namespace) -> None:
    """
    Small-group stress test over subcubes G_{J,a} with |J|=k.

    For parity, the proof in main.tex relies on the fact that
      E[chi_S(Z) | Z_J=a] = 0 unless S ⊆ J.
    In the case |S|=|J|=k, this becomes the key step:
      only J == S can have non-zero conditional bias.
    """
    if args.exact and args.d > 18:
        raise SystemExit("Refusing --exact for d>18 (2^d is too large).")

    spec = ParitySpec(d=args.d, k=args.k, alpha=args.alpha, seed=args.seed)
    mkt = ParityMarket.create(spec)

    if args.exact:
        # enumerate full cube in {-1,+1}^d
        grid = np.array(list(np.ndindex(*(2,) * args.d)), dtype=np.int8)
        z = (2 * grid - 1).astype(np.int8)
        p_true = mkt.p_true(z).astype(np.float32)
        # y not needed for group-calibration of (f-q); still sample for completeness
        rng = np.random.default_rng(args.seed + 123)
        y = rng.binomial(1, p_true).astype(np.int8)
    else:
        data = sample_parity_dataset(spec, n=args.n)
        z = data["z"]
        p_true = data["p_true"]
        y = data["y"]

    S = mkt.S
    q_diff = mkt.diffusion_analytic(z, rho=args.rho).astype(np.float32)
    residual = (p_true - q_diff).astype(np.float64)

    # Empirical diagnostic over J subsets of size k:
    diag = parity_S_equals_J_diagnostic(z=z, residual=residual, S=S, max_J=args.max_J)

    # Pull out the entry for J == S (if it was enumerated)
    S_sorted = tuple(sorted(int(i) for i in S))
    row_S = next((r for r in diag if tuple(sorted(r.J)) == S_sorted), None)
    est_S_enum = row_S.max_abs_cond_mean if row_S is not None else None

    # Always compute J==S directly (even if --max-J truncates enumeration)
    est_S_direct, _ = worst_group_abs_cond_mean_over_assignments(
        z=z, residual=residual, J=S_sorted
    )

    theory = 0.5 * args.alpha * (1.0 - (args.rho**args.k))
    # For the “AR L<k” parity counterexample, worst-group is >= alpha/2 (constant bound).
    ar_lb = 0.5 * args.alpha if args.L < args.k else 0.0

    print("Parity subcube diagnostic (top J by max |E[residual | Z_J=a]|)")
    for i, r in enumerate(diag[: min(10, len(diag))]):
        print(f"{i+1:>2d}. J={r.J}  max_abs_cond_mean={r.max_abs_cond_mean:.6g}")

    print(f"J==S direct max abs conditional mean ≈ {est_S_direct:.6g} (theory {theory:.6g})")
    if row_S is not None:
        print(f"J==S also appeared in enumeration with ≈ {row_S.max_abs_cond_mean:.6g}")
    else:
        print("J==S not enumerated (this is expected if --max-J truncates the J list).")

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "groupstress",
            "spec": asdict(spec),
            "S": tuple(int(i) for i in S),
            "n": int(len(z)),
            "rho": float(args.rho),
            "L": int(args.L),
            "max_J": int(args.max_J) if args.max_J is not None else None,
            "exact": bool(args.exact),
        },
    )
    arts.write_json(
        "metrics.json",
        {
            "theory": {
                "diffusion_worst_group": theory,
                "ar_worst_group_lb_if_L<k": ar_lb,
            },
            "estimate": {
                "J_eq_S_max_abs_cond_mean_direct": est_S_direct,
                "J_eq_S_max_abs_cond_mean_enumerated": est_S_enum,
            },
            "top_J": [
                {"J": list(r.J), "max_abs_cond_mean": r.max_abs_cond_mean, "argmax_a_key": r.argmax_a_key}
                for r in diag[: min(200, len(diag))]
            ],
        },
    )

    # Simple plot: top few J conditional means
    fig = plt.figure(figsize=(8, 3.2))
    top = diag[: min(20, len(diag))]
    xs = np.arange(len(top))
    ys = [r.max_abs_cond_mean for r in top]
    labels = ["S" if tuple(sorted(r.J)) == S_sorted else "" for r in top]
    plt.bar(xs, ys)
    for x, yv, lab in zip(xs, ys, labels):
        if lab:
            plt.text(x, yv, lab, ha="center", va="bottom")
    plt.ylabel("max_a |E[residual | Z_J=a]|")
    plt.title("Parity diagnostic: only J==S should stand out")
    plt.tight_layout()
    arts.savefig(fig, "topJ_condmean.png")
    plt.close(fig)

    print(f"Artifacts: {arts.run_dir}")


def cmd_intrinsic_post(args: argparse.Namespace) -> None:
    """
    Synthetic control for Section "Intrinsic robustness vs post-processing" in main.tex.

    We compare four variants on parity data:
      - AR-intrinsic: an AR-like predictor (either HF LLM, or the L-query oracle surrogate)
      - Diffusion-intrinsic: analytic T_rho f on parity
      - AR+post: apply the same post-processing wrapper to AR forecasts
      - Diffusion+post: apply the same post-processing wrapper to diffusion forecasts

    The post-processing wrapper implemented here is intentionally minimal:
      group/bin histogram calibration on the subcube partition defined by the *true* parity set S.
    This keeps the synthetic control light while still exposing the group-frequency/sample tax:
      there are 2^k rare groups, so post-processing needs ~2^k samples to learn them well.
    """
    from forecastbench.postprocess import fit_group_bin_postprocessor

    spec = ParitySpec(d=args.d, k=args.k, alpha=args.alpha, seed=args.seed)
    n_total = int(args.n_train) + int(args.n_test)
    data = sample_parity_dataset(spec, n=n_total)
    z = data["z"]
    p_true = data["p_true"]
    y = data["y"]

    # Keep S explicit (the wrapper groups on the true S partition).
    mkt = ParityMarket.create(spec)
    S = tuple(int(i) for i in mkt.S)
    S_sorted = tuple(sorted(S))

    # Base forecasters
    q_diff = mkt.diffusion_analytic(z, rho=args.rho).astype(np.float32)

    llm_meta = None
    if args.llm_model is not None:
        spec_llm = ARCoTSpec(
            model_name_or_path=args.llm_model,
            device=args.llm_device,
            dtype=args.llm_dtype,
            device_map=args.llm_device_map,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            max_new_tokens=args.llm_max_new_tokens,
            include_cot=not args.llm_no_cot,
        )
        pred = ARCoTPredictor(spec_llm)
        texts = [mkt.encode_text(z[i], style=args.text_style) for i in range(len(z))]
        q_ar, llm_meta = pred.predict_proba(texts, L=args.L, K=args.llm_K, seed=args.seed, aggregate=args.llm_agg)
    else:
        # A stylized AR surrogate aligned with the paper's query model:
        # succeeds iff L>=k; else defaults to the best L-junta baseline (constant 0.5).
        q_ar = p_true if args.L >= args.k else np.full_like(p_true, 0.5, dtype=np.float32)

    # Train/test split
    n_tr = int(args.n_train)
    n_te = int(args.n_test)
    if n_tr <= 0 or n_te <= 0 or (n_tr + n_te) > len(z):
        raise SystemExit("Invalid train/test sizes.")

    sl_tr = slice(0, n_tr)
    sl_te = slice(n_tr, n_tr + n_te)

    z_tr, z_te = z[sl_tr], z[sl_te]
    y_tr, y_te = y[sl_tr], y[sl_te]
    p_te = p_true[sl_te]

    q_ar_tr, q_ar_te = q_ar[sl_tr], q_ar[sl_te]
    q_diff_tr, q_diff_te = q_diff[sl_tr], q_diff[sl_te]

    # Fit the *same* wrapper on the same stream (z, q, y), separately per base forecaster.
    post_ar = fit_group_bin_postprocessor(
        z=z_tr,
        q=q_ar_tr,
        y=y_tr,
        J=S_sorted,
        n_bins=args.post_bins,
        prior_strength=args.post_prior,
        clip_eps=args.post_clip_eps,
    )
    post_diff = fit_group_bin_postprocessor(
        z=z_tr,
        q=q_diff_tr,
        y=y_tr,
        J=S_sorted,
        n_bins=args.post_bins,
        prior_strength=args.post_prior,
        clip_eps=args.post_clip_eps,
    )

    q_ar_post = post_ar.predict(z_te, q_ar_te)
    q_diff_post = post_diff.predict(z_te, q_diff_te)

    def _row(name: str, q: np.ndarray) -> dict:
        residual = (p_te - q).astype(np.float64)
        gcal_S, argmax_key = worst_group_abs_cond_mean_over_assignments(z=z_te, residual=residual, J=S_sorted)
        return {
            "name": name,
            "brier": brier_loss(q, y_te),
            "logloss": log_loss(q, y_te),
            "sce": squared_calibration_error(q, p_te),
            "ece": expected_calibration_error(q, y_te, n_bins=args.bins),
            "gcal_S": float(gcal_S),
            "gcal_S_argmax_key": int(argmax_key),
            "arb_profit": best_bounded_trader_profit(
                p_te, q, B=1.0, transaction_cost=args.transaction_cost
            ),
            "n_eval": int(len(q)),
        }

    rows = [
        _row("AR-intrinsic", q_ar_te),
        _row("Diffusion-intrinsic", q_diff_te),
        _row("AR+post", q_ar_post),
        _row("Diffusion+post", q_diff_post),
    ]

    # Print a compact table with the extra worst-group metric.
    cols = ["model", "brier", "log", "sce", "ece", "gcal_S", "arb(B=1)"]
    widths = {c: len(c) for c in cols}
    for r in rows:
        widths["model"] = max(widths["model"], len(r["name"]))
        widths["brier"] = max(widths["brier"], len(f'{r["brier"]:.6f}'))
        widths["log"] = max(widths["log"], len(f'{r["logloss"]:.6f}'))
        widths["sce"] = max(widths["sce"], len(f'{r["sce"]:.6f}'))
        widths["ece"] = max(widths["ece"], len(f'{r["ece"]:.6f}'))
        widths["gcal_S"] = max(widths["gcal_S"], len(f'{r["gcal_S"]:.6f}'))
        widths["arb(B=1)"] = max(widths["arb(B=1)"], len(f'{r["arb_profit"]:.6f}'))

    def fmt_row(model: str, brier: str, log: str, sce: str, ece: str, gcal_s: str, arb: str) -> str:
        return (
            f"{model:<{widths['model']}}  "
            f"{brier:>{widths['brier']}}  "
            f"{log:>{widths['log']}}  "
            f"{sce:>{widths['sce']}}  "
            f"{ece:>{widths['ece']}}  "
            f"{gcal_s:>{widths['gcal_S']}}  "
            f"{arb:>{widths['arb(B=1)']}}"
        )

    print("Intrinsic robustness vs post-processing (parity)")
    print(fmt_row(*cols))
    print("-" * (sum(widths.values()) + 14))
    for r in rows:
        print(
            fmt_row(
                r["name"],
                f'{r["brier"]:.6f}',
                f'{r["logloss"]:.6f}',
                f'{r["sce"]:.6f}',
                f'{r["ece"]:.6f}',
                f'{r["gcal_S"]:.6f}',
                f'{r["arb_profit"]:.6f}',
            )
        )

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "intrinsic_post",
            "spec": asdict(spec),
            "S": list(S),
            "rho": float(args.rho),
            "L": int(args.L),
            "train": {"n_train": int(n_tr)},
            "test": {"n_test": int(n_te), "bins": int(args.bins), "transaction_cost": float(args.transaction_cost)},
            "post": {
                "group_J": list(S_sorted),
                "post_bins": int(args.post_bins),
                "post_prior": float(args.post_prior),
                "post_clip_eps": float(args.post_clip_eps),
            },
            "llm": {
                "model": args.llm_model,
                "K": int(args.llm_K),
                "device": args.llm_device,
                "dtype": args.llm_dtype,
                "device_map": args.llm_device_map,
                "temperature": float(args.llm_temperature),
                "top_p": float(args.llm_top_p),
                "max_new_tokens": int(args.llm_max_new_tokens),
                "no_cot": bool(args.llm_no_cot),
                "text_style": args.text_style,
                "agg": args.llm_agg,
            }
            if args.llm_model is not None
            else None,
        },
    )
    arts.write_json("metrics.json", {"models": rows, "llm_meta": llm_meta})
    print(f"Artifacts: {arts.run_dir}")


def cmd_polymarket(args: argparse.Namespace) -> None:
    raise SystemExit("Deprecated: use pm_eval / pm_build_* subcommands instead.")


def cmd_pm_eval(args: argparse.Namespace) -> None:
    from forecastbench.benchmarks.polymarket_eval import (
        evaluate_group_bin_approachability,
        evaluate_polymarket_dataset,
        repair_group_bin_at_resolution,
    )
    from forecastbench.data import load_dataset
    from forecastbench.data.derived_groups import add_derived_group_cols

    df = load_dataset(args.dataset_path)
    if args.max_examples is not None:
        df = df.head(int(args.max_examples)).copy()

    # Build text input
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing text col {c!r} in dataset; available={list(df.columns)}")
    infos = ["\n".join(str(row[c]) for c in text_cols if row[c] is not None) for _, row in df.iterrows()]

    pred_col = args.pred_col
    if args.llm_model is not None:
        spec_llm = ARCoTSpec(
            model_name_or_path=args.llm_model,
            device=args.llm_device,
            dtype=args.llm_dtype,
            device_map=args.llm_device_map,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            max_new_tokens=args.llm_max_new_tokens,
            include_cot=not args.llm_no_cot,
        )
        pred = ARCoTPredictor(spec_llm)
        probs, llm_meta = pred.predict_proba(
            infos, L=args.L, K=args.K, seed=args.seed, aggregate=args.llm_agg
        )
        df[pred_col] = probs
    else:
        llm_meta = None
        if pred_col not in df.columns:
            raise SystemExit(f"No --llm-model provided and pred_col {pred_col!r} not in dataset.")

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()] if args.group_cols else None

    # Derived group columns (volume/time-to-close buckets) if requested.
    requested_group_cols: list[str] = []
    if group_cols:
        requested_group_cols += group_cols
    if args.app_group_cols:
        requested_group_cols += [c.strip() for c in args.app_group_cols.split(",") if c.strip()]
    if args.repair_group_cols:
        requested_group_cols += [c.strip() for c in args.repair_group_cols.split(",") if c.strip()]
    if requested_group_cols:
        df, _created_cols = add_derived_group_cols(df, requested=requested_group_cols)

    metrics = evaluate_polymarket_dataset(
        df,
        pred_col=pred_col,
        bins=args.bins,
        transaction_cost=args.transaction_cost,
        B=args.B,
        trading_mode=args.trading_mode,
        group_cols=group_cols,
    )

    # Optional: Blackwell approachability diagnostics for group×bin calibration payoffs.
    app = None
    app_repair = None
    if args.approachability:
        app_group_cols = None
        if args.app_group_cols:
            app_group_cols = [c.strip() for c in args.app_group_cols.split(",") if c.strip()]
        else:
            app_group_cols = group_cols

        if args.app_eps is not None:
            eps = float(args.app_eps)
        else:
            # Fee margin model consistent with the paper: non-exploitable box half-width.
            B = float(args.B)
            eps = float(args.transaction_cost) / B if B > 0 else float(args.transaction_cost)

        app = evaluate_group_bin_approachability(
            df,
            pred_col=pred_col,
            group_cols=app_group_cols,
            n_bins=int(args.app_bins),
            eps=float(eps),
            time_col=args.app_time_col,
            curve_every=int(args.app_curve_every),
            topk=int(args.app_topk),
            clip_eps=float(args.app_clip_eps),
        )

    # Optional: repair-at-resolution (batch feedback) as an online post-processing wrapper.
    repair_block = None
    pred_col_repair = None
    if args.repair_at_resolution:
        if args.repair_group_cols:
            repair_group_cols = [c.strip() for c in args.repair_group_cols.split(",") if c.strip()]
        else:
            repair_group_cols = group_cols

        pred_col_repair = f"{pred_col}_repair"
        pred_repair, repair_meta = repair_group_bin_at_resolution(
            df,
            pred_col=pred_col,
            group_cols=repair_group_cols,
            n_bins=int(args.repair_bins),
            prior_strength=float(args.repair_prior_strength),
            clip_eps=float(args.repair_clip_eps),
            forecast_time_col=args.repair_forecast_time_col,
            event_time_col=args.repair_event_time_col,
        )
        df[pred_col_repair] = pred_repair

        metrics_repair = evaluate_polymarket_dataset(
            df,
            pred_col=pred_col_repair,
            bins=args.bins,
            transaction_cost=args.transaction_cost,
            B=args.B,
            trading_mode=args.trading_mode,
            group_cols=group_cols,
        )

        if app is not None:
            app_repair = evaluate_group_bin_approachability(
                df,
                pred_col=pred_col_repair,
                group_cols=app_group_cols,
                n_bins=int(args.app_bins),
                eps=float(app["eps"]),
                time_col=args.app_time_col,
                curve_every=int(args.app_curve_every),
                topk=int(args.app_topk),
                clip_eps=float(args.app_clip_eps),
            )

        repair_block = {
            "pred_col": pred_col_repair,
            "meta": repair_meta,
            "metrics": metrics_repair,
            "approachability": app_repair,
        }

    out_payload = metrics
    if app is not None:
        out_payload = {**out_payload, "approachability": app}
    if repair_block is not None:
        out_payload = {**out_payload, "repair_at_resolution": repair_block}
    print(json.dumps(out_payload, indent=2, sort_keys=True))

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.write_json(
        "config.json",
        {
            "benchmark": "polymarket_eval",
            "dataset_path": str(args.dataset_path),
            "max_examples": int(args.max_examples) if args.max_examples is not None else None,
            "text_cols": text_cols,
            "pred_col": pred_col,
            "llm": {
                "model": args.llm_model,
                "L": int(args.L),
                "K": int(args.K),
                "device": args.llm_device,
                "dtype": args.llm_dtype,
                "device_map": args.llm_device_map,
                "temperature": float(args.llm_temperature),
                "top_p": float(args.llm_top_p),
                "max_new_tokens": int(args.llm_max_new_tokens),
                "no_cot": bool(args.llm_no_cot),
                "agg": args.llm_agg,
            }
            if args.llm_model is not None
            else None,
            "eval": {
                "bins": int(args.bins),
                "transaction_cost": float(args.transaction_cost),
                "B": float(args.B),
                "trading_mode": args.trading_mode,
                "group_cols": group_cols,
            },
            "approachability": None
            if not args.approachability
            else {
                "enabled": True,
                "group_cols": app_group_cols,
                "n_bins": int(args.app_bins),
                "eps": float(args.app_eps) if args.app_eps is not None else None,
                "time_col": args.app_time_col,
                "curve_every": int(args.app_curve_every),
                "topk": int(args.app_topk),
                "clip_eps": float(args.app_clip_eps),
            },
            "repair_at_resolution": None
            if not args.repair_at_resolution
            else {
                "enabled": True,
                "group_cols": repair_group_cols,
                "n_bins": int(args.repair_bins),
                "prior_strength": float(args.repair_prior_strength),
                "forecast_time_col": args.repair_forecast_time_col,
                "event_time_col": args.repair_event_time_col,
                "clip_eps": float(args.repair_clip_eps),
                "out_pred_col": pred_col_repair,
            },
        },
    )
    arts.write_json("metrics.json", {"metrics": out_payload, "llm_meta": llm_meta})
    # Save evaluated dataframe with predictions for downstream analysis
    try:
        df.to_parquet(arts.run_dir / "predictions.parquet", index=False)
    except Exception as e:
        arts.write_text("predictions_error.txt", f"Failed to write predictions.parquet: {e}\n")

    # Plot approachability curve (if requested)
    if app is not None:
        try:
            t = np.asarray(app["curve"]["t"], dtype=np.float64)
            err = np.asarray(app["curve"]["app_err"], dtype=np.float64)
            fig = plt.figure(figsize=(7.2, 3.2))
            plt.plot(t, err, lw=2, label="forward-only")
            if app_repair is not None:
                t2 = np.asarray(app_repair["curve"]["t"], dtype=np.float64)
                err2 = np.asarray(app_repair["curve"]["app_err"], dtype=np.float64)
                plt.plot(t2, err2, lw=2, label="repair-at-resolution")
            plt.xlabel("T (forecasts processed)")
            plt.ylabel("AppErr(T)")
            plt.title("Blackwell approachability: group×bin payoff box distance")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            arts.savefig(fig, "approachability_app_err.png")
            plt.close(fig)
        except Exception as e:
            arts.write_text("approachability_plot_error.txt", f"{e}\n")
    print(f"Artifacts: {arts.run_dir}")


def cmd_pm_eval_v2(args: argparse.Namespace) -> None:
    """
    Evaluate Polymarket dataset with hierarchical constraints (v2).
    
    This version includes:
    - Multicalibration: E[Y-p | group, bin(p)] = 0 for (topic, volume) groups
    - Frechet constraints: Cross-market bounds via category bundling
    - Bootstrap CIs on approachability rate
    - Per-sample hybrid correction analysis
    """
    from forecastbench.benchmarks.polymarket_eval import evaluate_polymarket_dataset
    from forecastbench.data import load_dataset
    from forecastbench.data.derived_groups import add_derived_group_cols
    from forecastbench.metrics.hierarchical_constraints import (
        HierarchicalConstraintSet,
        compute_arbitrage_bound_hierarchical,
    )
    from forecastbench.benchmarks.hybrid_analysis import run_hybrid_analysis
    from forecastbench.runner import RunArtifacts
    
    df = load_dataset(args.dataset_path)
    if args.max_examples is not None:
        df = df.head(int(args.max_examples)).copy()
    
    # Add derived group columns
    multicalib_cols = [c.strip() for c in args.multicalib_groups.split(",") if c.strip()]
    df, created_cols = add_derived_group_cols(df, multicalib_cols)
    print(f"[pm_eval_v2] Created derived columns: {created_cols}")
    
    # Extract predictions and outcomes
    pred_col = args.pred_col
    y_col = args.y_col
    market_prob_col = args.market_prob_col
    
    if pred_col not in df.columns:
        raise SystemExit(f"Missing pred_col {pred_col!r} in dataset")
    if y_col not in df.columns:
        raise SystemExit(f"Missing y_col {y_col!r} in dataset")
    
    p_model = df[pred_col].values.astype(np.float64)
    y = df[y_col].values.astype(np.float64)
    
    # Market prices (if available)
    q_market = None
    if market_prob_col and market_prob_col in df.columns:
        q_market = df[market_prob_col].values.astype(np.float64)
    
    # Standard metrics
    metrics = {
        "n": len(df),
        "brier": float(brier_loss(p_model, y)),
        "ece": float(expected_calibration_error(p_model, y, n_bins=args.bins)),
    }
    
    # Hierarchical constraint analysis
    if args.hierarchical_constraints:
        print(f"[pm_eval_v2] Running hierarchical constraint analysis...")
        
        constraint_set = HierarchicalConstraintSet.create(
            group_cols=multicalib_cols,
            n_bins=args.bins,
            bundle_col=args.frechet_bundle_col,
            bundle_size=args.bundle_size,
        )
        
        constraint_metrics = constraint_set.update(
            p=p_model, y=y, df=df, seed=args.seed
        )
        metrics["hierarchical_constraints"] = constraint_metrics
        
        # Arbitrage bound (if market prices available)
        if q_market is not None:
            arb_bound = compute_arbitrage_bound_hierarchical(
                q_market=q_market,
                p_model=p_model,
                y=y,
                df=df,
                group_cols=multicalib_cols,
                n_bins=args.bins,
                bundle_col=args.frechet_bundle_col,
                bundle_size=args.bundle_size,
                seed=args.seed,
            )
            metrics["arbitrage_bound"] = arb_bound
    
    # Approachability rate with bootstrap CI
    if args.approachability_rate:
        print(f"[pm_eval_v2] Computing approachability rate with bootstrap CI...")
        
        from forecastbench.metrics.multiscale_approachability import BlackwellConstraintTracker
        
        tracker = BlackwellConstraintTracker(n_groups=len(set(multicalib_cols)), n_bins=args.bins)
        
        # Process in streaming fashion
        groups = np.zeros(len(df), dtype=np.int64)
        for i, col in enumerate(multicalib_cols):
            if col in df.columns:
                groups += (df[col].factorize()[0] * (i + 1))
        
        tracker.update(p_model, y, groups, log_every=100)
        
        rate_info = tracker.compute_approachability_rate_with_ci(
            n_bootstrap=args.bootstrap_n, seed=args.seed
        )
        metrics["approachability_rate"] = rate_info
    
    # Hybrid correction analysis (if AR predictions available)
    ar_pred_col = args.ar_pred_col
    if ar_pred_col and ar_pred_col in df.columns:
        print(f"[pm_eval_v2] Running hybrid correction analysis...")
        
        p_ar = df[ar_pred_col].values.astype(np.float64)
        hybrid_analysis = run_hybrid_analysis(
            p_ar=p_ar,
            p_hybrid=p_model,
            y=y,
        )
        metrics["hybrid_analysis"] = hybrid_analysis
    
    # Save results
    arts = RunArtifacts.create(
        run_name=args.run_name,
        spec={"cmd": "pm_eval_v2", "args": vars(args)},
    )
    arts.write_json("metrics.json", metrics)
    df.to_parquet(arts.run_dir / "predictions.parquet", index=False)
    
    # Print summary
    print(f"\n[pm_eval_v2] Results:")
    print(f"  Brier: {metrics['brier']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    if "hierarchical_constraints" in metrics:
        hc = metrics["hierarchical_constraints"]
        print(f"  Distance to C_t: {hc['distance_to_C']:.4f}")
        print(f"    - Multicalib: {hc['d_multicalib']:.4f}")
        print(f"    - Frechet: {hc['d_frechet']:.4f}")
    if "approachability_rate" in metrics:
        ar = metrics["approachability_rate"]
        print(f"  Approachability rate: {ar.get('rate', 'N/A'):.3f} (95% CI: [{ar.get('rate_ci_lo', 'N/A'):.3f}, {ar.get('rate_ci_hi', 'N/A'):.3f}])")
        print(f"    Consistent with Blackwell 1/sqrt(T): {ar.get('consistent_with_theory', False)}")
    if "arbitrage_bound" in metrics:
        ab = metrics["arbitrage_bound"]
        print(f"  Arbitrage capture rate: {ab['arbitrage_capture_rate']*100:.1f}%")
        print(f"  Sharpe ratio: {ab['sharpe_ratio']:.2f}")
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_multimarket_arb(args: argparse.Namespace) -> None:
    """
    Run multi-market arbitrage analysis with Frechet constraints.
    
    This command:
    1. Groups markets by category into bundles
    2. Computes Frechet constraint violations within bundles
    3. Measures distance to constraint set and correlation with profit
    """
    from forecastbench.data import load_dataset
    from forecastbench.data.bundles import make_group_bundles
    from forecastbench.metrics.hierarchical_constraints import (
        FrechetConstraintTracker,
        compute_arbitrage_bound_hierarchical,
    )
    from forecastbench.runner import RunArtifacts
    
    df = load_dataset(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    
    df = df.reset_index(drop=True)
    
    # Check required columns
    if args.bundle_col not in df.columns:
        raise SystemExit(f"Missing bundle column {args.bundle_col!r}")
    if args.pred_col not in df.columns:
        raise SystemExit(f"Missing pred column {args.pred_col!r}")
    if args.y_col not in df.columns:
        raise SystemExit(f"Missing y column {args.y_col!r}")
    
    p = df[args.pred_col].values.astype(np.float64)
    y = df[args.y_col].values.astype(np.float64)
    
    # Create bundles
    bundle_idx, mask = make_group_bundles(
        df,
        group_col=args.bundle_col,
        bundle_size=args.bundle_size,
        seed=args.seed,
        drop_last=True,
    )
    
    print(f"[multimarket_arb] Created {len(bundle_idx)} bundles of size {args.bundle_size}")
    print(f"[multimarket_arb] Using constraint type: {args.constraint_type}")
    
    # Track Frechet constraints
    tracker = FrechetConstraintTracker(
        bundle_col=args.bundle_col,
        bundle_size=args.bundle_size,
        constraint_type=args.constraint_type,
    )
    
    tracker.update_from_bundles(bundle_idx, mask, p, log_every=10)
    
    metrics = tracker._compute_metrics()
    
    # Market prices for arbitrage analysis
    q_market = None
    if args.market_prob_col and args.market_prob_col in df.columns:
        q_market = df[args.market_prob_col].values.astype(np.float64)
        
        arb_bound = compute_arbitrage_bound_hierarchical(
            q_market=q_market,
            p_model=p,
            y=y,
            df=df,
            group_cols=[],
            bundle_col=args.bundle_col,
            bundle_size=args.bundle_size,
            seed=args.seed,
        )
        metrics["arbitrage_bound"] = arb_bound
    
    # Save results
    arts = RunArtifacts.create(
        run_name=args.run_name,
        spec={"cmd": "multimarket_arb", "args": vars(args)},
    )
    arts.write_json("metrics.json", metrics)
    
    print(f"\n[multimarket_arb] Results:")
    print(f"  Bundles analyzed: {metrics['n_bundles']}")
    print(f"  Max Frechet violation: {metrics['max_violation']:.4f}")
    print(f"  Mean Frechet violation: {metrics['mean_violation']:.4f}")
    print(f"  Fraction violated: {metrics['frac_violated']*100:.1f}%")
    
    if "arbitrage_bound" in metrics:
        ab = metrics["arbitrage_bound"]
        print(f"  Arbitrage capture rate: {ab['arbitrage_capture_rate']*100:.1f}%")
        print(f"  Sharpe ratio: {ab['sharpe_ratio']:.2f}")
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_pm_build_polydata(args: argparse.Namespace) -> None:
    """
    Convert a PolyData Explorer JSON download into the minimal forecastbench schema.
    """
    from forecastbench.data import save_dataset
    from forecastbench.data.polydata_explorer import coerce_to_forecast_dataset, load_polydata_json

    df_raw = load_polydata_json(args.json_path)
    df = coerce_to_forecast_dataset(
        df_raw,
        id_col=args.id_col,
        question_col=args.question_col,
        y_col=args.y_col,
        market_prob_col=args.market_prob_col,
    )
    save_dataset(df, args.out, fmt="parquet" if str(args.out).endswith(".parquet") else "jsonl")
    print(f"Wrote dataset: {args.out}  (n={len(df)})")


def cmd_pm_build_subgraph(args: argparse.Namespace) -> None:
    """
    Run a GraphQL query against a Polymarket subgraph endpoint and extract a record list.
    """
    from forecastbench.data import save_dataset
    from forecastbench.data.subgraph import SubgraphClient

    query_text = Path(args.query_file).read_text()
    variables = json.loads(Path(args.variables_json).read_text()) if args.variables_json else None
    client = SubgraphClient(endpoint=args.endpoint)
    data = client.query(query_text, variables=variables)

    # record_path is dot-separated, e.g. "markets" or "data.markets"
    obj = data
    for part in args.record_path.split("."):
        obj = obj[part]
    if not isinstance(obj, list):
        raise SystemExit(f"record_path did not resolve to a list. Got: {type(obj)}")

    import pandas as pd

    df_raw = pd.DataFrame(obj)
    # Best-effort mapping to required schema columns
    df = pd.DataFrame(
        {
            "id": df_raw[args.id_col].astype(str),
            "question": df_raw[args.question_col].astype(str),
            "y": df_raw[args.y_col].astype(int),
        }
    )
    if args.market_prob_col and args.market_prob_col in df_raw.columns:
        df["market_prob"] = df_raw[args.market_prob_col].astype(float)

    save_dataset(df, args.out, fmt="parquet" if str(args.out).endswith(".parquet") else "jsonl")
    print(f"Wrote dataset: {args.out}  (n={len(df)})")


def cmd_pm_download_gamma(args: argparse.Namespace) -> None:
    """
    Download Polymarket market metadata from the public Gamma API to JSONL.
    """
    from forecastbench.data.gamma_api import download_gamma_markets

    out = download_gamma_markets(
        out_dir=args.out_dir,
        page_size=args.page_size,
        sleep_s=args.sleep_s,
        max_pages=args.max_pages,
        start_offset=args.start_offset,
        resume=not args.no_resume,
    )
    print(f"Wrote: {out}")


def cmd_pm_build_gamma(args: argparse.Namespace) -> None:
    """
    Build an evaluation dataset (Parquet) from a downloaded Gamma markets dump.

    This is the simplest fully-offline route to a Polymarket-style dataset for pm_eval.
    """
    from forecastbench.data.gamma_build import GammaFilter, build_gamma_yesno_dataset

    flt = GammaFilter(
        only_binary=True,
        only_yes_no=not args.allow_non_yesno,
        require_closed=not args.allow_open,
        require_sum_to_one=True,
        require_extreme=not args.allow_non_extreme,
        extreme_high=float(args.extreme_high),
        extreme_low=float(args.extreme_low),
        min_volume=float(args.min_volume),
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
    )
    meta = build_gamma_yesno_dataset(
        input_path=args.input,
        out_parquet=args.out,
        flt=flt,
        chunk_rows=int(args.chunk_rows),
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


def cmd_pm_enrich_clob(args: argparse.Namespace) -> None:
    """
    Enrich a Gamma-built dataset with a forecast-time market price from the Polymarket CLOB.

    This is the key step to make evaluation/trading more comparable to
    [arXiv:2505.17989](https://arxiv.org/abs/2505.17989): we need a market price
    from *before resolution*, not the final 0/1 settlement price.
    """
    from forecastbench.data.enrich_clob import EnrichConfig, enrich_clob_file

    cfg = EnrichConfig(
        fidelity=str(args.fidelity),
        earliest_timestamp=int(args.earliest_timestamp),
        sleep_s=float(args.sleep_s),
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
        resume=not args.no_resume,
    )
    meta = enrich_clob_file(in_path=args.input, out_path=args.out, cfg=cfg)
    print(json.dumps(meta, indent=2, sort_keys=True))


def cmd_pm_download_clob_history(args: argparse.Namespace) -> None:
    """
    Download high-frequency Polymarket CLOB price history for YES tokens.

    This is useful when you want price snapshots at many times (e.g. 24h before close),
    not just the first price at market creation.
    """
    from forecastbench.data.clob_history import ClobHistoryConfig, download_clob_history_file

    cfg = ClobHistoryConfig(
        fidelity=str(args.fidelity),
        earliest_timestamp=int(args.earliest_timestamp) if args.earliest_timestamp is not None else None,
        sleep_s=float(args.sleep_s),
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
        resume=not args.no_resume,
    )
    meta = download_clob_history_file(
        in_path=args.input,
        out_dir=args.out_dir,
        cfg=cfg,
        token_col=args.token_col,
        created_at_col=args.created_at_col,
        id_col=args.id_col,
        slug_col=args.slug_col,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


def cmd_pm_build_horizon_prices(args: argparse.Namespace) -> None:
    """
    Build a pm_eval-ready dataset where market_prob is the CLOB price at a fixed horizon before close.
    """
    from forecastbench.data.horizon_build import HorizonBuildConfig, build_horizon_price_dataset

    cfg = HorizonBuildConfig(
        horizon_s=int(args.horizon_s),
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
        min_volume=float(args.min_volume),
        require_history=not args.allow_missing_history,
        require_price=not args.allow_missing_price,
    )
    meta = build_horizon_price_dataset(
        in_gamma_parquet=args.input,
        clob_history_dir=args.clob_history_dir,
        out_parquet=args.out,
        cfg=cfg,
        token_col=args.token_col,
        closed_time_col=args.closed_time_col,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


def cmd_pm_build_criterion_prices(args: argparse.Namespace) -> None:
    """
    Build a pm_eval-ready dataset where market_prob is derived from full CLOB history using
    a Themis/brier.fyi-style criterion (e.g. midpoint, time-average, before-close-days-30).
    """
    from forecastbench.data.criterion_build import CriterionBuildConfig, build_criterion_price_dataset

    cfg = CriterionBuildConfig(
        criterion=str(args.criterion),
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
        min_volume=float(args.min_volume),
        require_history=not args.allow_missing_history,
        require_price=not args.allow_missing_price,
    )
    meta = build_criterion_price_dataset(
        in_gamma_parquet=args.input,
        clob_history_dir=args.clob_history_dir,
        out_parquet=args.out,
        cfg=cfg,
        token_col=args.token_col,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


def cmd_pm_enrich_news_gdelt(args: argparse.Namespace) -> None:
    """
    Enrich a pm_eval-style dataset with news headlines from the GDELT DOC API.

    This is intended to approximate the "question + relevant headlines" modality used in RLVR forecasting work.
    """
    from forecastbench.data.news_gdelt import GdeltDocConfig, enrich_with_gdelt_news

    import pandas as pd

    df = pd.read_parquet(args.input)
    cfg = GdeltDocConfig(
        sleep_s=float(args.sleep_s),
        max_articles=int(args.max_articles),
        window_days=int(args.window_days),
        sort=str(args.sort),
    )
    df2, meta = enrich_with_gdelt_news(
        df=df,
        query_col=str(args.query_col),
        time_col=str(args.time_col),
        out_text_col=str(args.out_text_col),
        out_json_col=str(args.out_json_col),
        out_n_col=str(args.out_n_col),
        cache_dir=args.cache_dir,
        cfg=cfg,
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
        resume=not args.no_resume,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(out_path, index=False)
    print(json.dumps({**meta, "in": str(args.input), "out": str(args.out)}, indent=2, sort_keys=True))


def cmd_pm_difftrain(args: argparse.Namespace) -> None:
    """
    Train a conditional logit-diffusion forecaster on a Polymarket-style dataset (evaluation-only baseline).

    This is meant as a **diffusion-side analogue** to AR evaluation in [arXiv:2505.17989](https://arxiv.org/abs/2505.17989):
    - the task is: (question + context) -> probability of YES
    - compute knob is diffusion steps (T, sample_steps) and Monte Carlo samples (mc)
    """
    from forecastbench.benchmarks.polymarket_eval import evaluate_polymarket_dataset
    from forecastbench.data import load_dataset, make_group_bundles
    from forecastbench.data.derived_groups import add_derived_group_cols
    from forecastbench.models.diffusion_core import DiffusionSchedule
    from forecastbench.models.diffusion_bundle import BundleLogitDiffusionForecaster, BundleLogitDiffusionSpec
    from forecastbench.models.diffusion_logit import LogitDiffusionForecaster, LogitDiffusionSpec
    from forecastbench.models.text_embedder import HFTextEmbedder, HFTextEmbedderSpec
    from forecastbench.utils.logits import logit_to_prob, prob_to_logit

    df = load_dataset(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    df = df.reset_index(drop=True)

    # Build texts
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing text col {c!r} in dataset; available={list(df.columns)}")
    texts = ["\n".join(str(row[c]) for c in text_cols if row[c] is not None) for _, row in df.iterrows()]

    y = df["y"].to_numpy().astype(np.int64)

    # Embed text
    emb_spec = HFTextEmbedderSpec(
        model_name_or_path=args.embed_model,
        device=args.embed_device,
        dtype=args.embed_dtype,
        trust_remote_code=not bool(getattr(args, "embed_no_trust_remote_code", False)),
        device_map=getattr(args, "embed_device_map", None),
        max_length=int(args.embed_max_length),
        normalize=not args.embed_no_normalize,
    )
    embedder = HFTextEmbedder(emb_spec)

    X_all = embedder.encode(texts, batch_size=int(args.embed_batch_size))

    bundle_col = getattr(args, "bundle_col", None)
    bundle_size = int(getattr(args, "bundle_size", 1) or 1)

    if bundle_col and bundle_size > 1:
        # If bundle_col is derived (e.g. topic), create it on the fly.
        if bundle_col not in df.columns:
            df, _created = add_derived_group_cols(df, requested=[str(bundle_col)])
        if bundle_col not in df.columns:
            raise SystemExit(f"Missing bundle_col {bundle_col!r} in dataset; available={list(df.columns)}")

        bundle_idx, bundle_mask = make_group_bundles(
            df,
            group_col=str(bundle_col),
            bundle_size=int(bundle_size),
            seed=int(args.seed),
            drop_last=bool(getattr(args, "bundle_drop_last", False)),
        )
        if len(bundle_idx) < 2:
            raise SystemExit(
                f"Not enough bundles to split/train (got {len(bundle_idx)}). "
                "Try lowering --bundle-size or disabling --bundle-drop-last."
            )

        # Bundle-level split
        rng = np.random.default_rng(int(args.seed))
        b_order = np.arange(len(bundle_idx))
        rng.shuffle(b_order)
        n_train_b = int(len(bundle_idx) * float(args.train_frac))
        n_train_b = max(1, min(n_train_b, len(bundle_idx) - 1))
        tr_b = b_order[:n_train_b]
        te_b = b_order[n_train_b:]

        # Build cond/y arrays at bundle level
        B = int(bundle_size)
        D = int(X_all.shape[1])
        cond = np.zeros((len(bundle_idx), B, D), dtype=np.float32)
        y_b = np.zeros((len(bundle_idx), B), dtype=np.int64)

        for bi in range(len(bundle_idx)):
            row_ix = bundle_idx[bi]
            m = bundle_mask[bi]
            if not np.any(m):
                continue
            valid_rows = row_ix[m]
            cond[bi, m, :] = X_all[valid_rows]
            y_b[bi, m] = y[valid_rows]

        # Label smoothing to avoid infinite logits
        eps = float(args.label_eps)
        p_tr = y_b[tr_b].astype(np.float32) * (1.0 - 2.0 * eps) + eps
        x0_tr = prob_to_logit(p_tr, eps=1e-9).astype(np.float32)

        sched = DiffusionSchedule(
            T=int(args.T), beta_start=float(args.beta_start), beta_end=float(args.beta_end)
        )
        bspec = BundleLogitDiffusionSpec(
            bundle_size=B,
            embed_dim=D,
            model_dim=int(args.hidden_dim),
            time_dim=int(args.time_dim),
            depth=int(args.depth),
            n_heads=int(getattr(args, "bundle_heads", 4)),
            dropout=float(getattr(args, "bundle_dropout", 0.0)),
            schedule=sched,
        )
        model = BundleLogitDiffusionForecaster(bspec, device=args.device)

        train_meta = model.train_mse_eps(
            x0=x0_tr,
            cond=cond[tr_b],
            mask=bundle_mask[tr_b],
            steps=int(args.train_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            log_every=int(args.log_every),
        )

        # Predict: Monte Carlo over diffusion samples (bundle-level)
        mc = int(args.mc)
        probs_mc = []
        for m in range(mc):
            logits = model.sample_x(
                cond=cond[te_b],
                mask=bundle_mask[te_b],
                n_steps=int(args.sample_steps),
                seed=int(args.seed) + 10_000 + m,
                eta=float(args.eta),
            )
            probs = logit_to_prob(logits).astype(np.float32)  # (n_te_bundles, B)
            probs_mc.append(probs)

        stack = np.stack(probs_mc, axis=0)  # (mc, n_te_bundles, B)
        if args.agg == "median":
            pred_b = np.median(stack, axis=0).astype(np.float32)
        else:
            pred_b = np.mean(stack, axis=0).astype(np.float32)

        # Unbundle back to per-row predictions for evaluation / artifacts
        te_bundle_idx = bundle_idx[te_b]
        te_bundle_mask = bundle_mask[te_b]
        te_rows = te_bundle_idx[te_bundle_mask]  # row-major flatten
        te_pred = pred_b[te_bundle_mask].astype(np.float32)

        df_te = df.iloc[te_rows].copy()
        df_te[args.pred_col] = te_pred

        bundle_meta = {
            "enabled": True,
            "bundle_col": str(bundle_col),
            "bundle_size": int(bundle_size),
            "bundle_drop_last": bool(getattr(args, "bundle_drop_last", False)),
            "n_bundles": int(len(bundle_idx)),
            "n_train_bundles": int(len(tr_b)),
            "n_test_bundles": int(len(te_b)),
        }
    else:
        # Row-level split (original baseline)
        rng = np.random.default_rng(int(args.seed))
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_train = int(len(df) * float(args.train_frac))
        n_train = max(1, min(n_train, len(df) - 1))
        tr_idx = idx[:n_train]
        te_idx = idx[n_train:]

        # Label smoothing to avoid infinite logits
        eps = float(args.label_eps)
        p_tr = y[tr_idx].astype(np.float32) * (1.0 - 2.0 * eps) + eps
        x0 = prob_to_logit(p_tr, eps=1e-9).reshape(-1, 1).astype(np.float32)

        sched = DiffusionSchedule(
            T=int(args.T), beta_start=float(args.beta_start), beta_end=float(args.beta_end)
        )
        model_spec = LogitDiffusionSpec(
            out_dim=1,
            cond_dim=int(X_all.shape[1]),
            time_dim=int(args.time_dim),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            schedule=sched,
        )
        model = LogitDiffusionForecaster(model_spec, device=args.device)

        train_meta = model.train_mse_eps(
            x0=x0,
            cond=X_all[tr_idx],
            steps=int(args.train_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            log_every=int(args.log_every),
        )

        # Predict: Monte Carlo over diffusion samples
        mc = int(args.mc)
        probs_mc = []
        for m in range(mc):
            logits = model.sample_x(
                cond=X_all[te_idx],
                n_steps=int(args.sample_steps),
                seed=int(args.seed) + 10_000 + m,
                eta=float(args.eta),
            )
            probs = logit_to_prob(logits).reshape(-1).astype(np.float32)
            probs_mc.append(probs)
        stack = np.stack(probs_mc, axis=0)  # (mc, n)
        if args.agg == "median":
            pred = np.median(stack, axis=0).astype(np.float32)
        else:
            pred = np.mean(stack, axis=0).astype(np.float32)

        df_te = df.iloc[te_idx].copy()
        df_te[args.pred_col] = pred

        bundle_meta = {"enabled": False}

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()] if args.group_cols else None
    metrics = evaluate_polymarket_dataset(
        df_te,
        pred_col=args.pred_col,
        bins=int(args.bins),
        transaction_cost=float(args.transaction_cost),
        B=float(args.B),
        trading_mode=args.trading_mode,
        group_cols=group_cols,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.write_json(
        "config.json",
        {
            "benchmark": "pm_difftrain",
            "dataset_path": str(args.dataset_path),
            "max_rows": int(args.max_rows) if args.max_rows is not None else None,
            "text_cols": text_cols,
            "split": {"train_frac": float(args.train_frac), "seed": int(args.seed)},
            "bundling": bundle_meta,
            "embed": {
                "model": args.embed_model,
                "device": args.embed_device,
                "dtype": args.embed_dtype,
                "trust_remote_code": not bool(getattr(args, "embed_no_trust_remote_code", False)),
                "device_map": getattr(args, "embed_device_map", None),
                "max_length": int(args.embed_max_length),
                "batch_size": int(args.embed_batch_size),
                "normalize": not args.embed_no_normalize,
            },
            "diffusion": {
                "device": args.device,
                "label_eps": float(args.label_eps),
                "train_steps": int(args.train_steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "T": int(args.T),
                "beta_start": float(args.beta_start),
                "beta_end": float(args.beta_end),
                "time_dim": int(args.time_dim),
                "hidden_dim": int(args.hidden_dim),
                "depth": int(args.depth),
                "sample_steps": int(args.sample_steps),
                "eta": float(args.eta),
                "mc": int(args.mc),
                "agg": args.agg,
            },
            "eval": {
                "pred_col": args.pred_col,
                "bins": int(args.bins),
                "transaction_cost": float(args.transaction_cost),
                "B": float(args.B),
                "trading_mode": args.trading_mode,
                "group_cols": group_cols,
            },
        },
    )
    arts.write_json("metrics.json", {"train_meta": train_meta, "metrics": metrics})
    try:
        df_te.to_parquet(arts.run_dir / "predictions.parquet", index=False)
    except Exception as e:
        arts.write_text("predictions_error.txt", f"Failed to write predictions.parquet: {e}\n")
    model.save(str(arts.run_dir / "model.pt"))
    print(f"Artifacts: {arts.run_dir}")


def cmd_pm_diff_sample(args: argparse.Namespace) -> None:
    """
    Inference-only sampling from a trained diffusion model on a Polymarket-style dataset.

    This is designed for *compute sweeps* (varying inference-time sample_steps and mc)
    without retraining a diffusion model.
    """
    from forecastbench.benchmarks.polymarket_eval import (
        evaluate_group_bin_approachability,
        evaluate_polymarket_dataset,
        repair_group_bin_at_resolution,
    )
    from forecastbench.data import load_dataset, make_group_bundles
    from forecastbench.data.derived_groups import add_derived_group_cols
    from forecastbench.models.diffusion_core import ContinuousDiffusionForecaster
    from forecastbench.models.diffusion_bundle import BundleLogitDiffusionForecaster
    from forecastbench.models.text_embedder import HFTextEmbedder, HFTextEmbedderSpec
    from forecastbench.utils.logits import logit_to_prob

    df = load_dataset(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    df = df.reset_index(drop=True)

    # Build texts
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing text col {c!r} in dataset; available={list(df.columns)}")
    texts = ["\n".join(str(row[c]) for c in text_cols if row[c] is not None) for _, row in df.iterrows()]

    # Embed text (must match the diffusion model's cond_dim)
    emb_spec = HFTextEmbedderSpec(
        model_name_or_path=args.embed_model,
        device=args.embed_device,
        dtype=args.embed_dtype,
        trust_remote_code=not bool(getattr(args, "embed_no_trust_remote_code", False)),
        device_map=getattr(args, "embed_device_map", None),
        max_length=int(args.embed_max_length),
        normalize=not args.embed_no_normalize,
    )
    embedder = HFTextEmbedder(emb_spec)
    X = embedder.encode(texts, batch_size=int(args.embed_batch_size))

    # Load model (support both scalar diffusion and bundle diffusion checkpoints)
    kind = None
    try:
        import torch

        try:
            payload = torch.load(str(args.model_path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(args.model_path), map_location="cpu", weights_only=False)
        kind = payload.get("kind")
    except Exception:
        kind = None

    pred_col = str(args.pred_col)

    if kind == "bundle_logit_diffusion":
        model = BundleLogitDiffusionForecaster.load(str(args.model_path), device=str(args.device))
        B = int(model.spec.bundle_size)
        bundle_col = getattr(args, "bundle_col", None)
        if not bundle_col:
            raise SystemExit(
                "Model is a bundle diffusion checkpoint but no --bundle-col was provided "
                "(e.g. --bundle-col category)."
            )
        bundle_size_arg = int(getattr(args, "bundle_size", B) or B)
        if bundle_size_arg != B:
            raise SystemExit(f"--bundle-size={bundle_size_arg} must match model bundle_size={B}")

        # If bundle_col is derived (e.g. topic), create it on the fly.
        if str(bundle_col) not in df.columns:
            df, _created = add_derived_group_cols(df, requested=[str(bundle_col)])

        bundle_idx, bundle_mask = make_group_bundles(
            df,
            group_col=str(bundle_col),
            bundle_size=B,
            seed=int(args.seed),
            drop_last=bool(getattr(args, "bundle_drop_last", False)),
        )
        if len(bundle_idx) == 0:
            raise SystemExit("No bundles were formed; check --bundle-col/--bundle-size.")

        # Build bundle cond
        D = int(X.shape[1])
        cond = np.zeros((len(bundle_idx), B, D), dtype=np.float32)
        for bi in range(len(bundle_idx)):
            row_ix = bundle_idx[bi]
            m = bundle_mask[bi]
            if not np.any(m):
                continue
            valid_rows = row_ix[m]
            cond[bi, m, :] = X[valid_rows]

        # Predict: Monte Carlo over diffusion samples (bundle-level), then unbundle to rows.
        mc = int(args.mc)
        probs_mc = []
        for m in range(mc):
            logits = model.sample_x(
                cond=cond,
                mask=bundle_mask,
                n_steps=int(args.sample_steps),
                seed=int(args.seed) + 10_000 + m,
                eta=float(args.eta),
            )
            probs_mc.append(logit_to_prob(logits).astype(np.float32))
        stack = np.stack(probs_mc, axis=0)  # (mc, n_bundles, B)
        if args.agg == "median":
            pred_b = np.median(stack, axis=0).astype(np.float32)
        else:
            pred_b = np.mean(stack, axis=0).astype(np.float32)

        # Unbundle back to per-row predictions.
        pred_rows = np.full((len(df),), np.nan, dtype=np.float32)
        rows = bundle_idx[bundle_mask]
        vals = pred_b[bundle_mask]
        pred_rows[rows] = vals

        # If we dropped tail bundles, some rows may be NaN; evaluate on rows with predictions.
        keep = np.isfinite(pred_rows)
        df = df.loc[keep].copy()
        df[pred_col] = pred_rows[keep]
    else:
        model = ContinuousDiffusionForecaster.load(str(args.model_path), device=str(args.device))

        # Predict: Monte Carlo over diffusion samples
        mc = int(args.mc)
        probs_mc = []
        for m in range(mc):
            logits = model.sample_x(
                cond=X,
                n_steps=int(args.sample_steps),
                seed=int(args.seed) + 10_000 + m,
                eta=float(args.eta),
            )
            probs = logit_to_prob(logits).reshape(-1).astype(np.float32)
            probs_mc.append(probs)
        stack = np.stack(probs_mc, axis=0)  # (mc, n)
        if args.agg == "median":
            pred = np.median(stack, axis=0).astype(np.float32)
        else:
            pred = np.mean(stack, axis=0).astype(np.float32)

        df[pred_col] = pred

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()] if args.group_cols else None

    # Derived group columns (volume/time-to-close buckets) if requested.
    requested_group_cols: list[str] = []
    if group_cols:
        requested_group_cols += group_cols
    if args.app_group_cols:
        requested_group_cols += [c.strip() for c in args.app_group_cols.split(",") if c.strip()]
    if args.repair_group_cols:
        requested_group_cols += [c.strip() for c in args.repair_group_cols.split(",") if c.strip()]
    if requested_group_cols:
        df, _created_cols = add_derived_group_cols(df, requested=requested_group_cols)

    metrics = evaluate_polymarket_dataset(
        df,
        pred_col=pred_col,
        bins=int(args.bins),
        transaction_cost=float(args.transaction_cost),
        B=float(args.B),
        trading_mode=args.trading_mode,
        group_cols=group_cols,
    )

    # Optional: Blackwell approachability diagnostics
    app = None
    app_repair = None
    if args.approachability:
        app_group_cols = None
        if args.app_group_cols:
            app_group_cols = [c.strip() for c in args.app_group_cols.split(",") if c.strip()]
        else:
            app_group_cols = group_cols

        if args.app_eps is not None:
            eps = float(args.app_eps)
        else:
            B = float(args.B)
            eps = float(args.transaction_cost) / B if B > 0 else float(args.transaction_cost)

        app = evaluate_group_bin_approachability(
            df,
            pred_col=pred_col,
            group_cols=app_group_cols,
            n_bins=int(args.app_bins),
            eps=float(eps),
            time_col=args.app_time_col,
            curve_every=int(args.app_curve_every),
            topk=int(args.app_topk),
            clip_eps=float(args.app_clip_eps),
        )

    # Optional: repair-at-resolution
    repair_block = None
    pred_col_repair = None
    if args.repair_at_resolution:
        if args.repair_group_cols:
            repair_group_cols = [c.strip() for c in args.repair_group_cols.split(",") if c.strip()]
        else:
            repair_group_cols = group_cols

        pred_col_repair = f"{pred_col}_repair"
        pred_repair, repair_meta = repair_group_bin_at_resolution(
            df,
            pred_col=pred_col,
            group_cols=repair_group_cols,
            n_bins=int(args.repair_bins),
            prior_strength=float(args.repair_prior_strength),
            clip_eps=float(args.repair_clip_eps),
            forecast_time_col=args.repair_forecast_time_col,
            event_time_col=args.repair_event_time_col,
        )
        df[pred_col_repair] = pred_repair

        metrics_repair = evaluate_polymarket_dataset(
            df,
            pred_col=pred_col_repair,
            bins=int(args.bins),
            transaction_cost=float(args.transaction_cost),
            B=float(args.B),
            trading_mode=args.trading_mode,
            group_cols=group_cols,
        )

        if app is not None:
            app_repair = evaluate_group_bin_approachability(
                df,
                pred_col=pred_col_repair,
                group_cols=app_group_cols,
                n_bins=int(args.app_bins),
                eps=float(app["eps"]),
                time_col=args.app_time_col,
                curve_every=int(args.app_curve_every),
                topk=int(args.app_topk),
                clip_eps=float(args.app_clip_eps),
            )

        repair_block = {
            "pred_col": pred_col_repair,
            "meta": repair_meta,
            "metrics": metrics_repair,
            "approachability": app_repair,
        }

    out_payload = metrics
    if app is not None:
        out_payload = {**out_payload, "approachability": app}
    if repair_block is not None:
        out_payload = {**out_payload, "repair_at_resolution": repair_block}

    print(json.dumps(out_payload, indent=2, sort_keys=True))

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "pm_diff_sample",
            "dataset_path": str(args.dataset_path),
            "max_rows": int(args.max_rows) if args.max_rows is not None else None,
            "text_cols": text_cols,
            "pred_col": pred_col,
            "model_path": str(args.model_path),
            "embed": {
                "model": args.embed_model,
                "device": args.embed_device,
                "dtype": args.embed_dtype,
                "max_length": int(args.embed_max_length),
                "batch_size": int(args.embed_batch_size),
                "normalize": not args.embed_no_normalize,
            },
            "diffusion_infer": {
                "device": str(args.device),
                "sample_steps": int(args.sample_steps),
                "eta": float(args.eta),
                "mc": int(args.mc),
                "agg": args.agg,
                "seed": int(args.seed),
            },
            "eval": {
                "bins": int(args.bins),
                "transaction_cost": float(args.transaction_cost),
                "B": float(args.B),
                "trading_mode": args.trading_mode,
                "group_cols": group_cols,
            },
            "approachability": None
            if not args.approachability
            else {
                "enabled": True,
                "group_cols": app_group_cols,
                "n_bins": int(args.app_bins),
                "eps": float(args.app_eps) if args.app_eps is not None else None,
                "time_col": args.app_time_col,
                "curve_every": int(args.app_curve_every),
                "topk": int(args.app_topk),
                "clip_eps": float(args.app_clip_eps),
            },
            "repair_at_resolution": None
            if not args.repair_at_resolution
            else {
                "enabled": True,
                "group_cols": repair_group_cols,
                "n_bins": int(args.repair_bins),
                "prior_strength": float(args.repair_prior_strength),
                "forecast_time_col": args.repair_forecast_time_col,
                "event_time_col": args.repair_event_time_col,
                "clip_eps": float(args.repair_clip_eps),
                "out_pred_col": pred_col_repair,
            },
        },
    )
    arts.write_json("metrics.json", {"metrics": out_payload})
    try:
        df.to_parquet(arts.run_dir / "predictions.parquet", index=False)
    except Exception as e:
        arts.write_text("predictions_error.txt", f"Failed to write predictions.parquet: {e}\n")

    if app is not None:
        try:
            t = np.asarray(app["curve"]["t"], dtype=np.float64)
            err = np.asarray(app["curve"]["app_err"], dtype=np.float64)
            fig = plt.figure(figsize=(7.2, 3.2))
            plt.plot(t, err, lw=2, label="forward-only")
            if app_repair is not None:
                t2 = np.asarray(app_repair["curve"]["t"], dtype=np.float64)
                err2 = np.asarray(app_repair["curve"]["app_err"], dtype=np.float64)
                plt.plot(t2, err2, lw=2, label="repair-at-resolution")
            plt.xlabel("T (forecasts processed)")
            plt.ylabel("AppErr(T)")
            plt.title("Blackwell approachability: group×bin payoff box distance")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            arts.savefig(fig, "approachability_app_err.png")
            plt.close(fig)
        except Exception as e:
            arts.write_text("approachability_plot_error.txt", f"{e}\n")

    print(f"Artifacts: {arts.run_dir}")


def cmd_pm_learnedCt_arb(args: argparse.Namespace) -> None:
    """
    Evaluate a *bundle diffusion* model by constructing a learned feasible set C_t from MC samples
    and estimating statistical arbitrage via online learners (Hedge + optional neural witness).

    This is designed to complement pm_diff_sample:
      - pm_diff_sample: scalar evaluation/trading + approachability + repair
      - pm_learnedCt_arb: bundle-level learned-C_t diagnostics + online arb estimates
    """
    import json
    from pathlib import Path

    import numpy as np

    from forecastbench.benchmarks.learned_constraints_arb import LearnedCtArbConfig, run_learnedCt_online_arb
    from forecastbench.benchmarks.polymarket_eval import evaluate_polymarket_dataset, repair_group_bin_at_resolution
    from forecastbench.data import load_dataset, make_group_bundles
    from forecastbench.data.derived_groups import add_derived_group_cols
    from forecastbench.models.diffusion_bundle import BundleLogitDiffusionForecaster
    from forecastbench.models.text_embedder import HFTextEmbedder, HFTextEmbedderSpec
    from forecastbench.runner.artifacts import RunArtifacts
    from forecastbench.utils.logits import logit_to_prob

    df = load_dataset(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    df = df.reset_index(drop=True)

    if "y" not in df.columns:
        raise SystemExit('Dataset missing "y" labels.')
    if "market_prob" not in df.columns:
        raise SystemExit(
            'Dataset missing "market_prob". Build it with pm_enrich_clob / pm_build_horizon_prices / pm_build_criterion_prices.'
        )

    # Build texts
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing text col {c!r} in dataset; available={list(df.columns)}")
    texts = ["\n".join(str(row[c]) for c in text_cols if row[c] is not None) for _, row in df.iterrows()]

    # Embed text
    emb_spec = HFTextEmbedderSpec(
        model_name_or_path=args.embed_model,
        device=args.embed_device,
        dtype=args.embed_dtype,
        trust_remote_code=not bool(getattr(args, "embed_no_trust_remote_code", False)),
        device_map=getattr(args, "embed_device_map", None),
        max_length=int(args.embed_max_length),
        normalize=not args.embed_no_normalize,
    )
    embedder = HFTextEmbedder(emb_spec)
    X = embedder.encode(texts, batch_size=int(args.embed_batch_size))

    # Load bundle diffusion model
    model = BundleLogitDiffusionForecaster.load(str(args.model_path), device=str(args.device))
    B_bundle = int(model.spec.bundle_size)

    bundle_col = getattr(args, "bundle_col", None)
    if not bundle_col:
        raise SystemExit("--bundle-col is required for pm_learnedCt_arb (e.g. --bundle-col topic).")

    # If bundle_col is derived (e.g. topic), create it on the fly.
    if str(bundle_col) not in df.columns:
        df, _created = add_derived_group_cols(df, requested=[str(bundle_col)])
    if str(bundle_col) not in df.columns:
        raise SystemExit(f"Missing bundle_col {bundle_col!r} in dataset; available={list(df.columns)}")

    bundle_size_arg = int(getattr(args, "bundle_size", B_bundle) or B_bundle)
    if bundle_size_arg != B_bundle:
        raise SystemExit(f"--bundle-size={bundle_size_arg} must match model bundle_size={B_bundle}")

    bundle_idx, bundle_mask = make_group_bundles(
        df,
        group_col=str(bundle_col),
        bundle_size=int(B_bundle),
        seed=int(args.seed),
        drop_last=bool(getattr(args, "bundle_drop_last", False)),
    )
    if len(bundle_idx) == 0:
        raise SystemExit("No bundles were formed; check --bundle-col/--bundle-size.")

    # For learned-C_t, prefer full bundles (fixed k). If padding exists, drop incomplete bundles.
    full = np.all(bundle_mask, axis=1)
    if not np.all(full):
        bundle_idx = bundle_idx[full]
        bundle_mask = bundle_mask[full]
        if len(bundle_idx) == 0:
            raise SystemExit("After dropping incomplete bundles, no full bundles remained. Try --bundle-drop-last.")

    n_bundles = int(len(bundle_idx))
    D = int(X.shape[1])

    # Bundle-level arrays: cond, labels, market prices, and a time key for ordering.
    cond = np.zeros((n_bundles, B_bundle, D), dtype=np.float32)
    y_rows = df["y"].to_numpy().astype(np.float32)
    q_rows = df["market_prob"].to_numpy().astype(np.float32)

    # Use an existing unix-seconds column if present; else fall back to index order.
    if "market_prob_ts" in df.columns:
        ts_row = df["market_prob_ts"].to_numpy(dtype=np.float64, copy=False)
    elif "market_prob_target_ts" in df.columns:
        ts_row = df["market_prob_target_ts"].to_numpy(dtype=np.float64, copy=False)
    else:
        ts_row = np.arange(len(df), dtype=np.float64)

    y_b = np.zeros((n_bundles, B_bundle), dtype=np.float32)
    q_b = np.zeros((n_bundles, B_bundle), dtype=np.float32)
    ts_b = np.full((n_bundles,), np.nan, dtype=np.float64)

    for bi in range(n_bundles):
        row_ix = bundle_idx[bi]
        m = bundle_mask[bi]
        if not np.any(m):
            continue
        valid_rows = row_ix[m]
        cond[bi, m, :] = X[valid_rows]
        y_b[bi, m] = y_rows[valid_rows]
        q_b[bi, m] = q_rows[valid_rows]
        try:
            ts_b[bi] = float(np.nanmin(ts_row[valid_rows]))
        except Exception:
            ts_b[bi] = float("nan")

    # Sample diffusion MC to build learned C_t per bundle
    mc = int(args.mc)
    probs_mc = []
    for m_i in range(mc):
        logits = model.sample_x(
            cond=cond,
            mask=bundle_mask,
            n_steps=int(args.sample_steps),
            seed=int(args.seed) + 10_000 + m_i,
            eta=float(args.eta),
        )
        probs_mc.append(logit_to_prob(logits).astype(np.float32))
    stack = np.stack(probs_mc, axis=0)  # (mc, n_bundles, B)
    if str(args.agg).lower() == "median":
        pred_b = np.median(stack, axis=0).astype(np.float32)
    else:
        pred_b = np.mean(stack, axis=0).astype(np.float32)

    arb_cfg = LearnedCtArbConfig(
        B_trade=float(args.B),
        transaction_cost=float(args.transaction_cost),
        hedge_eta=(None if args.arb_hedge_eta is None else float(args.arb_hedge_eta)),
        witness_hidden=int(args.arb_witness_hidden),
        witness_depth=int(args.arb_witness_depth),
        witness_lr=float(args.arb_witness_lr),
        witness_weight_decay=float(args.arb_witness_weight_decay),
        witness_grad_clip=float(args.arb_witness_grad_clip),
        seed=int(args.seed),
        max_steps=(None if args.arb_max_steps is None else int(args.arb_max_steps)),
    )
    learned_arb = run_learnedCt_online_arb(
        mc_samples=stack,
        q=q_b,
        y=y_b,
        cfg=arb_cfg,
        p_hat=pred_b,
        sort_key=ts_b,
        enable_neural=not bool(args.arb_no_neural),
    )

    # Unbundle point predictions to per-row predictions for standard scalar metrics + repair baseline.
    pred_col = str(args.pred_col)
    pred_rows = np.full((len(df),), np.nan, dtype=np.float32)
    rows = bundle_idx[bundle_mask]
    vals = pred_b[bundle_mask]
    pred_rows[rows] = vals
    keep = np.isfinite(pred_rows)
    df2 = df.loc[keep].copy()
    df2[pred_col] = pred_rows[keep]

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()] if args.group_cols else None
    if group_cols:
        df2, _created_cols = add_derived_group_cols(df2, requested=group_cols)

    metrics = evaluate_polymarket_dataset(
        df2,
        pred_col=pred_col,
        bins=int(args.bins),
        transaction_cost=float(args.transaction_cost),
        B=float(args.B),
        trading_mode=args.trading_mode,
        group_cols=group_cols,
    )

    # Optional: repair-at-resolution (scalar postprocessing) baseline
    repair_block = None
    if args.repair_at_resolution:
        if args.repair_group_cols:
            repair_group_cols = [c.strip() for c in args.repair_group_cols.split(",") if c.strip()]
        else:
            repair_group_cols = group_cols

        pred_col_repair = f"{pred_col}_repair"
        pred_repair, repair_meta = repair_group_bin_at_resolution(
            df2,
            pred_col=pred_col,
            group_cols=repair_group_cols,
            n_bins=int(args.repair_bins),
            prior_strength=float(args.repair_prior_strength),
            clip_eps=float(args.repair_clip_eps),
            forecast_time_col=args.repair_forecast_time_col,
            event_time_col=args.repair_event_time_col,
        )
        df2[pred_col_repair] = pred_repair
        metrics_repair = evaluate_polymarket_dataset(
            df2,
            pred_col=pred_col_repair,
            bins=int(args.bins),
            transaction_cost=float(args.transaction_cost),
            B=float(args.B),
            trading_mode=args.trading_mode,
            group_cols=group_cols,
        )
        repair_block = {"pred_col": pred_col_repair, "meta": repair_meta, "metrics": metrics_repair}

    out_payload = {"metrics": metrics, "learnedCt_arb": learned_arb, "repair_at_resolution": repair_block}
    print(json.dumps(out_payload, indent=2, sort_keys=True))

    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "pm_learnedCt_arb",
            "dataset_path": str(args.dataset_path),
            "model_path": str(args.model_path),
            "max_rows": int(args.max_rows) if args.max_rows is not None else None,
            "text_cols": text_cols,
            "pred_col": pred_col,
            "bundle": {
                "bundle_col": str(bundle_col),
                "bundle_size": int(B_bundle),
                "bundle_drop_last": bool(getattr(args, "bundle_drop_last", False)),
                "n_bundles": int(n_bundles),
            },
            "embed": {
                "model": args.embed_model,
                "device": args.embed_device,
                "dtype": args.embed_dtype,
                "device_map": getattr(args, "embed_device_map", None),
                "max_length": int(args.embed_max_length),
                "batch_size": int(args.embed_batch_size),
                "normalize": not args.embed_no_normalize,
            },
            "diffusion_infer": {
                "device": args.device,
                "sample_steps": int(args.sample_steps),
                "eta": float(args.eta),
                "mc": int(args.mc),
                "agg": str(args.agg),
            },
            "eval": {
                "bins": int(args.bins),
                "transaction_cost": float(args.transaction_cost),
                "B": float(args.B),
                "trading_mode": str(args.trading_mode),
                "group_cols": group_cols,
            },
            "arb": {
                "B_trade": float(args.B),
                "transaction_cost": float(args.transaction_cost),
                "hedge_eta": None if args.arb_hedge_eta is None else float(args.arb_hedge_eta),
                "no_neural": bool(args.arb_no_neural),
                "witness_hidden": int(args.arb_witness_hidden),
                "witness_depth": int(args.arb_witness_depth),
                "witness_lr": float(args.arb_witness_lr),
                "witness_weight_decay": float(args.arb_witness_weight_decay),
                "witness_grad_clip": float(args.arb_witness_grad_clip),
                "max_steps": None if args.arb_max_steps is None else int(args.arb_max_steps),
            },
            "repair_at_resolution": None
            if not args.repair_at_resolution
            else {
                "enabled": True,
                "group_cols": repair_group_cols,
                "n_bins": int(args.repair_bins),
                "prior_strength": float(args.repair_prior_strength),
                "forecast_time_col": args.repair_forecast_time_col,
                "event_time_col": args.repair_event_time_col,
                "clip_eps": float(args.repair_clip_eps),
            },
        },
    )
    arts.write_json("metrics.json", out_payload)
    try:
        df2.to_parquet(Path(arts.run_dir) / "predictions.parquet", index=False)
    except Exception as e:
        arts.write_text("predictions_error.txt", f"Failed to write predictions.parquet: {e}\n")
    print(f"Artifacts: {arts.run_dir}")


def cmd_pm_rlvr_train(args: argparse.Namespace) -> None:
    """
    RLVR-style outcome-based training loop for an AR model on a Polymarket-style dataset.

    This implements a lightweight REINFORCE + KL-to-reference loop with a hybrid reward:
      alpha*logscore + beta*PnL - kl_coef*KL.

    Note: This is compute-intensive; intended for the remote H200 box (LoRA-first).
    """
    import json

    import numpy as np

    from forecastbench.data import load_dataset
    from forecastbench.runner.artifacts import RunArtifacts
    from forecastbench.train.rlvr import RLVRHybridRewardSpec, RLVRTrainSpec, train_rlvr

    df = load_dataset(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    df = df.reset_index(drop=True)

    y_col = str(getattr(args, "y_col", "y"))
    q_col = str(getattr(args, "market_prob_col", "market_prob"))
    if y_col not in df.columns:
        raise SystemExit(f"Missing y_col={y_col!r} in dataset.")
    if q_col not in df.columns:
        raise SystemExit(f"Missing market_prob_col={q_col!r} in dataset.")

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing text col {c!r} in dataset; available={list(df.columns)}")
    infos = ["\n".join(str(row[c]) for c in text_cols if row[c] is not None) for _, row in df.iterrows()]

    y = df[y_col].to_numpy().astype(np.float64)
    q = df[q_col].to_numpy().astype(np.float64)
    # basic hygiene
    keep = np.isfinite(y) & np.isfinite(q)
    infos = [infos[i] for i in np.nonzero(keep)[0].tolist()]
    y = y[keep]
    q = q[keep]

    reward = RLVRHybridRewardSpec(
        alpha_logscore=float(args.alpha_logscore),
        beta_pnl=float(args.beta_pnl),
        B=float(args.B),
        transaction_cost=float(args.transaction_cost),
        trading_mode=str(args.trading_mode),
    )
    spec = RLVRTrainSpec(
        model_name_or_path=str(args.model_name_or_path),
        device=str(args.device),
        dtype=str(args.dtype),
        device_map=(None if args.device_map is None else str(args.device_map)),
        trust_remote_code=not bool(getattr(args, "no_trust_remote_code", False)),
        load_in_4bit=not bool(getattr(args, "no_4bit", False)),
        bnb_4bit_compute_dtype=str(args.bnb_4bit_compute_dtype),
        use_lora=not bool(getattr(args, "no_lora", False)),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        include_cot=not bool(getattr(args, "no_cot", False)),
        cot_max_steps=int(args.cot_max_steps),
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
        kl_coef=float(args.kl_coef),
        reward_clip=float(args.reward_clip),
        baseline_ema=float(args.baseline_ema),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        reward=reward,
    )

    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "pm_rlvr_train",
            "dataset_path": str(args.dataset_path),
            "max_rows": int(args.max_rows) if args.max_rows is not None else None,
            "text_cols": text_cols,
            "y_col": y_col,
            "market_prob_col": q_col,
            "spec": asdict(spec),
        },
    )

    out = train_rlvr(infos=infos, y=y, q=q, spec=spec, out_dir=arts.run_dir / "rlvr")
    arts.write_json("metrics.json", out)
    print(json.dumps({"run_dir": str(arts.run_dir), "train": out}, indent=2, sort_keys=True))
    print(f"Artifacts: {arts.run_dir}")


def cmd_pm_grpo_train(args: argparse.Namespace) -> None:
    """
    GRPO/Dr.GRPO/ReMax training for AR models on Polymarket data.
    
    Implements the Turtel et al. (2025) training approach:
    - Modified GRPO (Dr. GRPO): A_i = r_i - μ (no std normalization)
    - ReMax: Select best sample, subtract EMA baseline
    - Guard-rails: token limits, gibberish filter, early stopping, Brier-weighted gradients
    
    Reference: https://arxiv.org/abs/2505.17989
    """
    import json
    import numpy as np
    from dataclasses import asdict
    
    from forecastbench.data import load_dataset
    from forecastbench.runner.artifacts import RunArtifacts
    from forecastbench.train.grpo import GRPORewardSpec, GRPOTrainSpec, train_grpo
    
    df = load_dataset(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    df = df.reset_index(drop=True)
    
    y_col = str(getattr(args, "y_col", "y"))
    q_col = str(getattr(args, "market_prob_col", "market_prob"))
    if y_col not in df.columns:
        raise SystemExit(f"Missing y_col={y_col!r} in dataset.")
    if q_col not in df.columns:
        raise SystemExit(f"Missing market_prob_col={q_col!r} in dataset.")
    
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing text col {c!r} in dataset; available={list(df.columns)}")
    infos = ["\n".join(str(row[c]) for c in text_cols if row[c] is not None) for _, row in df.iterrows()]
    
    y = df[y_col].to_numpy().astype(np.float64)
    q = df[q_col].to_numpy().astype(np.float64)
    keep = np.isfinite(y) & np.isfinite(q)
    infos = [infos[i] for i in np.nonzero(keep)[0].tolist()]
    y = y[keep]
    q = q[keep]
    
    reward = GRPORewardSpec(
        alpha_logscore=float(args.alpha_logscore),
        beta_pnl=float(args.beta_pnl),
        B=float(args.B),
        transaction_cost=float(args.transaction_cost),
        trading_mode=str(args.trading_mode),
    )
    
    spec = GRPOTrainSpec(
        model_name_or_path=str(args.model_name_or_path),
        device=str(args.device),
        dtype=str(args.dtype),
        device_map=(None if args.device_map is None else str(args.device_map)),
        trust_remote_code=not bool(getattr(args, "no_trust_remote_code", False)),
        load_in_4bit=not bool(getattr(args, "no_4bit", False)),
        bnb_4bit_compute_dtype=str(args.bnb_4bit_compute_dtype),
        use_lora=not bool(getattr(args, "no_lora", False)),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        include_cot=not bool(getattr(args, "no_cot", False)),
        cot_max_steps=int(args.cot_max_steps),
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        K=int(args.K),  # GRPO group size
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
        kl_coef=float(args.kl_coef),
        dr_grpo=bool(args.dr_grpo),  # Turtel-style modified GRPO
        normalize_advantages=not bool(args.dr_grpo),
        reward_clip=float(args.reward_clip),
        max_response_tokens=int(args.max_response_tokens),
        gibberish_filter=not bool(getattr(args, "no_gibberish_filter", False)),
        early_stop_patience=int(args.early_stop_patience),
        brier_weighted_grads=bool(args.brier_weighted_grads),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        reward=reward,
    )
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "pm_grpo_train",
            "dataset_path": str(args.dataset_path),
            "max_rows": int(args.max_rows) if args.max_rows is not None else None,
            "text_cols": text_cols,
            "y_col": y_col,
            "market_prob_col": q_col,
            "spec": asdict(spec),
        },
    )
    
    out = train_grpo(infos=infos, y=y, q=q, spec=spec, out_dir=arts.run_dir / "grpo")
    arts.write_json("metrics.json", out)
    print(json.dumps({"run_dir": str(arts.run_dir), "train": out}, indent=2, sort_keys=True))
    print(f"Artifacts: {arts.run_dir}")


def cmd_pm_rlvr_eval(args: argparse.Namespace) -> None:
    """
    Evaluate a LoRA-adapted AR model (RLVR-trained) on a Polymarket-style dataset.

    Implements the paper-aligned inference knobs:
      - K self-consistency samples
      - median aggregation

    Outputs standard proper metrics + trading (bounded-PnL + Kelly ROI) and optional approachability/repair.
    """
    import json

    import numpy as np

    from forecastbench.benchmarks.polymarket_eval import evaluate_group_bin_approachability, evaluate_polymarket_dataset
    from forecastbench.benchmarks.polymarket_eval import repair_group_bin_at_resolution
    from forecastbench.data import load_dataset
    from forecastbench.data.derived_groups import add_derived_group_cols
    from forecastbench.models.rlvr_ar import ARLoRAPredictor, ARLoRAPredictorSpec
    from forecastbench.runner.artifacts import RunArtifacts

    df = load_dataset(args.dataset_path)
    if args.max_examples is not None:
        df = df.head(int(args.max_examples)).copy()
    df = df.reset_index(drop=True)

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing text col {c!r} in dataset; available={list(df.columns)}")
    infos = ["\n".join(str(row[c]) for c in text_cols if row[c] is not None) for _, row in df.iterrows()]

    pred_col = str(args.pred_col)

    spec = ARLoRAPredictorSpec(
        base_model_name_or_path=str(args.base_model),
        adapter_path=str(args.adapter_path),
        device=str(args.device),
        dtype=str(args.dtype),
        device_map=(None if args.device_map is None else str(args.device_map)),
        trust_remote_code=not bool(args.no_trust_remote_code),
        load_in_4bit=not bool(args.no_4bit),
        bnb_4bit_compute_dtype=str(args.bnb_4bit_compute_dtype),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        include_cot=not bool(args.no_cot),
        cot_max_steps=int(args.L),
        aggregate=str(args.agg),
    )
    pred = ARLoRAPredictor(spec)
    probs, llm_meta = pred.predict_proba(infos, K=int(args.K), seed=int(args.seed), aggregate=str(args.agg))
    df[pred_col] = probs.astype(np.float32)

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()] if args.group_cols else None

    requested_group_cols: list[str] = []
    if group_cols:
        requested_group_cols += group_cols
    if args.app_group_cols:
        requested_group_cols += [c.strip() for c in args.app_group_cols.split(",") if c.strip()]
    if args.repair_group_cols:
        requested_group_cols += [c.strip() for c in args.repair_group_cols.split(",") if c.strip()]
    if requested_group_cols:
        df, _created_cols = add_derived_group_cols(df, requested=requested_group_cols)

    metrics = evaluate_polymarket_dataset(
        df,
        pred_col=pred_col,
        bins=int(args.bins),
        transaction_cost=float(args.transaction_cost),
        B=float(args.B),
        trading_mode=str(args.trading_mode),
        group_cols=group_cols,
    )

    app = None
    app_repair = None
    if args.approachability:
        app_group_cols = (
            [c.strip() for c in args.app_group_cols.split(",") if c.strip()] if args.app_group_cols else group_cols
        )
        if args.app_eps is not None:
            eps = float(args.app_eps)
        else:
            B = float(args.B)
            eps = float(args.transaction_cost) / B if B > 0 else float(args.transaction_cost)
        app = evaluate_group_bin_approachability(
            df,
            pred_col=pred_col,
            group_cols=app_group_cols,
            n_bins=int(args.app_bins),
            eps=float(eps),
            time_col=args.app_time_col,
            curve_every=int(args.app_curve_every),
            topk=int(args.app_topk),
            clip_eps=float(args.app_clip_eps),
        )

    repair_block = None
    pred_col_repair = None
    if args.repair_at_resolution:
        repair_group_cols = (
            [c.strip() for c in args.repair_group_cols.split(",") if c.strip()]
            if args.repair_group_cols
            else group_cols
        )
        pred_col_repair = f"{pred_col}_repair"
        pred_repair, repair_meta = repair_group_bin_at_resolution(
            df,
            pred_col=pred_col,
            group_cols=repair_group_cols,
            n_bins=int(args.repair_bins),
            prior_strength=float(args.repair_prior_strength),
            clip_eps=float(args.repair_clip_eps),
            forecast_time_col=args.repair_forecast_time_col,
            event_time_col=args.repair_event_time_col,
        )
        df[pred_col_repair] = pred_repair
        metrics_repair = evaluate_polymarket_dataset(
            df,
            pred_col=pred_col_repair,
            bins=int(args.bins),
            transaction_cost=float(args.transaction_cost),
            B=float(args.B),
            trading_mode=str(args.trading_mode),
            group_cols=group_cols,
        )
        if app is not None:
            app_repair = evaluate_group_bin_approachability(
                df,
                pred_col=pred_col_repair,
                group_cols=app_group_cols,
                n_bins=int(args.app_bins),
                eps=float(app["eps"]),
                time_col=args.app_time_col,
                curve_every=int(args.app_curve_every),
                topk=int(args.app_topk),
                clip_eps=float(args.app_clip_eps),
            )
        repair_block = {"pred_col": pred_col_repair, "meta": repair_meta, "metrics": metrics_repair, "approachability": app_repair}

    out_payload = metrics
    if app is not None:
        out_payload = {**out_payload, "approachability": app}
    if repair_block is not None:
        out_payload = {**out_payload, "repair_at_resolution": repair_block}
    print(json.dumps(out_payload, indent=2, sort_keys=True))

    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    arts.write_json(
        "config.json",
        {
            "benchmark": "pm_rlvr_eval",
            "dataset_path": str(args.dataset_path),
            "max_examples": int(args.max_examples) if args.max_examples is not None else None,
            "text_cols": text_cols,
            "pred_col": pred_col,
            "base_model": str(args.base_model),
            "adapter_path": str(args.adapter_path),
            "infer": {
                "K": int(args.K),
                "L": int(args.L),
                "agg": str(args.agg),
                "device": str(args.device),
                "dtype": str(args.dtype),
                "device_map": args.device_map,
                "no_4bit": bool(args.no_4bit),
                "bnb_4bit_compute_dtype": str(args.bnb_4bit_compute_dtype),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_new_tokens": int(args.max_new_tokens),
                "no_cot": bool(args.no_cot),
            },
            "eval": {
                "bins": int(args.bins),
                "transaction_cost": float(args.transaction_cost),
                "B": float(args.B),
                "trading_mode": str(args.trading_mode),
                "group_cols": group_cols,
            },
            "approachability": None if app is None else {"enabled": True},
            "repair_at_resolution": None if repair_block is None else {"enabled": True, "out_pred_col": pred_col_repair},
            "llm_meta": llm_meta,
        },
    )
    arts.write_json("metrics.json", {"metrics": out_payload, "llm_meta": llm_meta})
    try:
        df.to_parquet(arts.run_dir / "predictions.parquet", index=False)
    except Exception as e:
        arts.write_text("predictions_error.txt", f"Failed to write predictions.parquet: {e}\n")
    print(f"Artifacts: {arts.run_dir}")


def cmd_pm_compare(args: argparse.Namespace) -> None:
    """
    Compare multiple Polymarket evaluation runs using their predictions.parquet, with bootstrap CIs.

    Models are provided as repeated --model specs:
      --model NAME,RUN_DIR[,PRED_COL]
    where PRED_COL is optional (otherwise inferred from config.json).
    """
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    from forecastbench.reporting.compare import ModelRunSpec, compare_polymarket_runs
    from forecastbench.runner.artifacts import RunArtifacts

    specs = []
    for raw in args.model or []:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if len(parts) < 2:
            raise SystemExit("--model must be NAME,RUN_DIR[,PRED_COL]")
        name = parts[0]
        run_dir = parts[1]
        pred_col = parts[2] if len(parts) >= 3 else None
        specs.append(ModelRunSpec(name=str(name), run_dir=Path(run_dir), pred_col=pred_col))

    res = compare_polymarket_runs(
        specs=specs,
        bins=int(args.bins),
        B=float(args.B),
        transaction_cost=float(args.transaction_cost),
        trading_mode=str(args.trading_mode),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        baseline=(None if args.baseline is None else str(args.baseline)),
    )

    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    arts.write_json("metrics.json", res)

    # Compact table (also written as CSV)
    rows = []
    for s in specs:
        name = str(s.name)
        pt = res["point"][name]
        ci = res["ci95"][name]
        rows.append(
            {
                "name": name,
                "n": int(res["n"]),
                "logloss": pt["logloss"],
                "logloss_lo": ci["logloss"]["lo"],
                "logloss_hi": ci["logloss"]["hi"],
                "brier": pt["brier"],
                "brier_lo": ci["brier"]["lo"],
                "brier_hi": ci["brier"]["hi"],
                "ece": pt["ece"],
                "ece_lo": ci["ece"]["lo"],
                "ece_hi": ci["ece"]["hi"],
                "pnl_per_event": pt["pnl_per_event"],
                "pnl_lo": ci["pnl_per_event"]["lo"],
                "pnl_hi": ci["pnl_per_event"]["hi"],
                "kelly_roi": pt["kelly_roi"],
                "roi_lo": ci["kelly_roi"]["lo"],
                "roi_hi": ci["kelly_roi"]["hi"],
            }
        )
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(arts.run_dir / "summary.csv", index=False)
    except Exception as e:
        arts.write_text("summary_csv_error.txt", f"{e}\n")

    print(json.dumps({"summary": rows, "diffs_vs_baseline": res.get("diffs_vs_baseline")}, indent=2, sort_keys=True))

    # Plot: error bars for key metrics
    metrics = [
        ("logloss", "LogLoss (lower better)"),
        ("brier", "Brier (lower better)"),
        ("ece", "ECE (lower better)"),
        ("pnl_per_event", "PnL/event (higher better)"),
        ("kelly_roi", "Kelly ROI (higher better)"),
    ]
    names = [r["name"] for r in rows]
    x = np.arange(len(names), dtype=np.float64)

    fig = plt.figure(figsize=(12, 6))
    for j, (m, title) in enumerate(metrics, start=1):
        ax = plt.subplot(2, 3, j)
        vals = [r[m] for r in rows]
        if m == "pnl_per_event":
            lo = [r["pnl_lo"] for r in rows]
            hi = [r["pnl_hi"] for r in rows]
        elif m == "kelly_roi":
            lo = [r["roi_lo"] for r in rows]
            hi = [r["roi_hi"] for r in rows]
        else:
            lo = [r[f"{m}_lo"] for r in rows]
            hi = [r[f"{m}_hi"] for r in rows]
        yerr = np.vstack([np.asarray(vals) - np.asarray(lo), np.asarray(hi) - np.asarray(vals)])
        ax.bar(x, vals, yerr=yerr, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    arts.savefig(fig, "compare_bootstrap.png")
    plt.close(fig)

    print(f"Artifacts: {arts.run_dir}")


def cmd_difftrain(args: argparse.Namespace) -> None:
    """
    Train a tiny conditional logit-diffusion model on synthetic parity data.

    This is meant as a **local sanity check** that a learned diffusion model
    (text/feature -> simplex/logit) can be trained + evaluated end-to-end.
    """
    from forecastbench.models.diffusion_logit import DiffusionSchedule, LogitDiffusionForecaster, LogitDiffusionSpec

    spec = ParitySpec(d=args.d, k=args.k, alpha=args.alpha, seed=args.seed)
    data = sample_parity_dataset(spec, n=args.n_train + args.n_test)
    z = data["z"].astype(np.float32)
    p_true = data["p_true"].astype(np.float32)
    y = data["y"].astype(np.int8)

    # Train/test split
    z_tr = z[: args.n_train]
    p_tr = p_true[: args.n_train]
    z_te = z[args.n_train :]
    p_te = p_true[args.n_train :]
    y_te = y[args.n_train :]

    from forecastbench.utils.logits import prob_to_logit

    x0 = prob_to_logit(p_tr, eps=1e-6).reshape(-1, 1).astype(np.float32)

    sched = DiffusionSchedule(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end)
    model_spec = LogitDiffusionSpec(
        out_dim=1,
        cond_dim=args.d,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        schedule=sched,
    )

    forecaster = LogitDiffusionForecaster(model_spec, device=args.device)
    train_meta = forecaster.train_mse_eps(
        x0=x0,
        cond=z_tr,
        steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
    )

    # Evaluate on test set using diffusion sampling
    from forecastbench.utils.logits import logit_to_prob

    logits_te = forecaster.sample_x(
        cond=z_te, n_steps=args.sample_steps, seed=args.seed + 1, eta=args.eta
    )
    q_te = logit_to_prob(logits_te).reshape(-1).astype(np.float32)

    rows = [
        {
            "name": f"learned_logit_diff(T={args.T},sample_steps={args.sample_steps})",
            "brier": brier_loss(q_te, y_te),
            "logloss": log_loss(q_te, y_te),
            "sce": squared_calibration_error(q_te, p_te),
            "ece": expected_calibration_error(q_te, y_te, n_bins=args.bins),
            "arb_profit": best_bounded_trader_profit(p_te, q_te, B=1.0, transaction_cost=args.transaction_cost),
            "n_eval": int(len(q_te)),
        }
    ]

    _print_summary_table(rows, title="Learned logit-diffusion (parity)")

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.write_json(
        "config.json",
        {
            "benchmark": "difftrain_parity",
            "parity": asdict(spec),
            "train": {
                "n_train": int(args.n_train),
                "n_test": int(args.n_test),
                "train_steps": int(args.train_steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "T": int(args.T),
                "beta_start": float(args.beta_start),
                "beta_end": float(args.beta_end),
                "time_dim": int(args.time_dim),
                "hidden_dim": int(args.hidden_dim),
                "depth": int(args.depth),
                "device": args.device,
            },
            "sample": {"sample_steps": int(args.sample_steps), "eta": float(args.eta)},
        },
    )
    arts.write_json("metrics.json", {"train_meta": train_meta, "models": rows})
    forecaster.save(str(arts.run_dir / "model.pt"))
    print(f"Artifacts: {arts.run_dir}")


def cmd_difftrain_simplex(args: argparse.Namespace) -> None:
    """
    Train a tiny conditional diffusion model for simplex outputs using the ALR transform.

    Synthetic task: multi-outcome “parity logits” in ALR space, then p = alr^{-1}(u).
    """
    from forecastbench.benchmarks.simplex_parity import SimplexParitySpec, sample_simplex_parity_dataset
    from forecastbench.metrics import multiclass_brier_loss, multiclass_log_loss, multiclass_sce, top_label_ece
    from forecastbench.models.diffusion_core import DiffusionSchedule
    from forecastbench.models.diffusion_simplex import SimplexALRDiffusionForecaster, SimplexDiffusionSpec

    spec = SimplexParitySpec(
        d=args.d, k=args.k, n_outcomes=args.n_outcomes, alpha=args.alpha, seed=args.seed
    )
    data = sample_simplex_parity_dataset(spec, n=args.n_train + args.n_test)
    z = data["z"].astype(np.float32)
    p_true = data["p_true"].astype(np.float32)
    y = data["y"].astype(np.int64)

    z_tr = z[: args.n_train]
    p_tr = p_true[: args.n_train]
    z_te = z[args.n_train :]
    p_te = p_true[args.n_train :]
    y_te = y[args.n_train :]

    sched = DiffusionSchedule(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end)
    model_spec = SimplexDiffusionSpec(
        n_outcomes=args.n_outcomes,
        cond_dim=args.d,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        schedule=sched,
    )
    forecaster = SimplexALRDiffusionForecaster(model_spec, device=args.device)
    train_meta = forecaster.train_mse_eps(
        p0=p_tr,
        cond=z_tr,
        steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
    )

    p_hat = forecaster.predict_simplex_from_cond(
        z_te, n_steps=args.sample_steps, seed=args.seed + 1, eta=args.eta
    )

    metrics = {
        "brier": multiclass_brier_loss(p_hat, y_te),
        "logloss": multiclass_log_loss(p_hat, y_te),
        "sce": multiclass_sce(p_hat, p_te),
        "ece_top": top_label_ece(p_hat, y_te, n_bins=args.bins),
        "n_eval": int(len(y_te)),
    }

    # small text table
    print("Learned simplex diffusion (ALR)")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    arts = RunArtifacts.create(run_name=args.run_name)
    arts.write_json(
        "config.json",
        {
            "benchmark": "difftrain_simplex_parity",
            "spec": asdict(spec),
            "subsets": data.get("subsets"),
            "train": {
                "n_train": int(args.n_train),
                "n_test": int(args.n_test),
                "train_steps": int(args.train_steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "T": int(args.T),
                "beta_start": float(args.beta_start),
                "beta_end": float(args.beta_end),
                "time_dim": int(args.time_dim),
                "hidden_dim": int(args.hidden_dim),
                "depth": int(args.depth),
                "device": args.device,
            },
            "sample": {"sample_steps": int(args.sample_steps), "eta": float(args.eta)},
        },
    )
    arts.write_json("metrics.json", {"train_meta": train_meta, "metrics": metrics})
    forecaster.save(str(arts.run_dir / "model.pt"))
    print(f"Artifacts: {arts.run_dir}")


def cmd_latex(args: argparse.Namespace) -> None:
    from forecastbench.reporting import run_to_latex_table

    tex = run_to_latex_table(args.run_dir)
    if args.out is not None:
        Path(args.out).write_text(tex)
        print(f"Wrote: {args.out}")
    else:
        print(tex)


def cmd_logical_graph(args: argparse.Namespace) -> None:
    """
    Synthetic 'Family S2' experiment: scalable bundles with sparse implication graphs.
    """
    from forecastbench.benchmarks.logical_graphs import (
        LogicalGraphSpec,
        sample_truth_prices,
        summarize_logical_graph_arbitrage,
    )
    from forecastbench.benchmarks.multimarket import logits_from_prices
    from forecastbench.models.diffusion_bundle import BundleLogitDiffusionForecaster, BundleLogitDiffusionSpec
    from forecastbench.models.diffusion_core import DiffusionSchedule
    from forecastbench.runner import RunArtifacts
    from forecastbench.utils.logits import logit_to_prob
    import numpy as np

    # 1. Generate data
    spec = LogicalGraphSpec(d=int(args.d), m=int(args.m), structure=str(args.structure), seed=int(args.seed), noise=float(args.noise))
    
    n_train = int(args.n_train)
    n_test = int(args.n_test)
    n = n_train + n_test
    if n_train <= 0 or n_test <= 0:
        raise SystemExit("--n-train and --n-test must be positive")

    X, P = sample_truth_prices(spec, n=n)
    X_tr, X_te = X[:n_train], X[n_train:]
    P_tr, P_te = P[:n_train], P[n_train:]
    
    # 2. Prepare diffusion training data
    # We reuse make_bundle_cond but need to adapt it for m markets (it assumes 3).
    # make_bundle_cond is hardcoded for 3 markets. Let's make a local version or use generic embedding.
    # For now, let's just use X repeated as condition, plus a one-hot market ID.
    def _make_cond(X_in, m_markets):
        N, D = X_in.shape
        # cond shape: (N, m, D + m)
        one_hot = np.eye(m_markets, dtype=np.float32)
        out = np.zeros((N, m_markets, D + m_markets), dtype=np.float32)
        for i in range(m_markets):
            out[:, i, :D] = X_in
            out[:, i, D:] = one_hot[i]
        return out

    cond_tr = _make_cond(X_tr, spec.m)
    cond_te = _make_cond(X_te, spec.m)
    mask_tr = np.ones((n_train, spec.m), dtype=bool)
    mask_te = np.ones((n_test, spec.m), dtype=bool)

    x0_tr = logits_from_prices(P_tr, eps=float(args.label_eps)).astype(np.float32)

    # 3. Train diffusion
    sched = DiffusionSchedule(T=int(args.T), beta_start=float(args.beta_start), beta_end=float(args.beta_end))
    model_spec = BundleLogitDiffusionSpec(
        bundle_size=spec.m,
        embed_dim=int(cond_tr.shape[2]),
        model_dim=int(args.hidden_dim),
        time_dim=int(args.time_dim),
        depth=int(args.depth),
        n_heads=int(args.heads),
        dropout=float(args.dropout),
        schedule=sched,
    )
    model = BundleLogitDiffusionForecaster(model_spec, device=str(args.device))

    train_meta = model.train_mse_eps(
        x0=x0_tr,
        cond=cond_tr,
        mask=mask_tr,
        steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        log_every=int(args.log_every),
    )

    # 4. Inference and Evaluation
    steps_list = [int(s) for s in str(args.sample_steps).split(",") if s.strip()]
    mc = int(args.mc)
    results: list[dict] = []

    # Oracle
    oracle = summarize_logical_graph_arbitrage(
        pred=P_te,
        p_true=P_te,
        structure=str(spec.structure),
        curve_every=int(args.curve_every),
        include_box=not bool(args.no_box),
    )
    results.append({"name": "oracle", **oracle})

    # Diffusion sweep
    for n_steps in steps_list:
        probs_mc = []
        for m_i in range(mc):
            logits = model.sample_x(
                cond=cond_te,
                mask=mask_te,
                n_steps=int(n_steps),
                seed=int(args.seed) + 10000 + m_i,
                eta=float(args.eta),
            )
            probs_mc.append(logit_to_prob(logits).astype(np.float32))
        
        stack = np.stack(probs_mc, axis=0)
        if str(args.agg).lower() == "median":
            pred = np.median(stack, axis=0)
        else:
            pred = np.mean(stack, axis=0)
            
        summ = summarize_logical_graph_arbitrage(
            pred=pred,
            p_true=P_te,
            structure=str(spec.structure),
            curve_every=int(args.curve_every),
            include_box=not bool(args.no_box),
        )
        results.append({"name": f"diff(steps={n_steps},mc={mc})", **summ})

    # 5. Report
    print(f"Logical Graph ({spec.structure}, m={spec.m}) Summary:")
    for r in results:
        sa = r["static_arbitrage"]
        print(f" - {r['name']}: mse={r['mse']:.6g} AppErr={sa['final_app_err']:.6g} frac_viol={sa['frac_any_violated']:.3f}")
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.write_json("metrics.json", {"train_meta": train_meta, "results": results})
    print(f"Artifacts: {arts.run_dir}")


def cmd_cliff_fog(args: argparse.Namespace) -> None:
    """Cliff vs Fog: AR spectral cliff vs diffusion continuous recovery."""
    from forecastbench.benchmarks.cliff_fog import (
        CliffFogSpec,
        run_cliff_fog_experiment,
        compute_matched_comparison,
        plot_cliff_fog,
    )
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    L_values = tuple(range(2, args.k + 4, 2))
    rho_values = (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
    K_values = (1, 4, 16) if args.ar_width else (1,)
    
    spec = CliffFogSpec(
        d=args.d,
        k=args.k,
        alpha=args.alpha,
        n_samples=args.n,
        L_values=L_values,
        rho_values=rho_values,
        K_values=K_values,
        seed=args.seed,
    )
    
    print(f"Running cliff vs fog experiment: d={args.d}, k={args.k}, n={args.n}")
    results = run_cliff_fog_experiment(spec)
    
    # Matched comparison
    matched_df = compute_matched_comparison(spec)
    results["matched_comparison"] = matched_df.to_dict(orient="records")
    
    # Write config and results
    arts.write_json("config.json", {
        "benchmark": "cliff_fog",
        "d": args.d,
        "k": args.k,
        "alpha": args.alpha,
        "n": args.n,
        "seed": args.seed,
        "ar_width": args.ar_width,
    })
    arts.write_json("metrics.json", results)
    
    # Plots
    plot_cliff_fog(results, str(arts.run_dir / "plots"))
    
    # Summary
    ar_summary = [r for r in results["ar_results"] if r["K"] == 1]
    diff_summary = results["diffusion_results"]
    
    print(f"\nAR Results (K=1):")
    for r in ar_summary:
        print(f"  L={r['L']:2d}: SCE={r['SCE']:.6f}, theory_lb={r['theory_SCE_lower_bound']:.6f}")
    
    print(f"\nDiffusion Results:")
    for r in diff_summary:
        print(f"  ρ={r['rho']:.2f}: SCE={r['SCE']:.6f}, theory={r['theory_SCE']:.6f}")
    
    print(f"\nMatched Comparison (compute budget):")
    for row in matched_df.to_dict(orient="records")[:5]:
        print(f"  budget={row['compute']:2d}: AR_SCE={row['ar_SCE']:.6f}, Diff_SCE={row['diff_SCE']:.6f}, diff_wins={row['diff_wins_sce']}")
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_group_robustness(args: argparse.Namespace) -> None:
    """Group robustness: Prop 8-9 experiments on small subgroups."""
    from forecastbench.benchmarks.group_robustness import (
        GroupRobustnessSpec,
        group_robustness_separation,
        exponential_group_scaling,
        rho_sweep_for_group_robustness,
        plot_group_robustness,
        plot_exponential_scaling,
    )
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    print(f"Running group robustness experiment: d={args.d}, k={args.k}, n={args.n}")
    
    spec = GroupRobustnessSpec(
        d=args.d,
        k=args.k,
        alpha=args.alpha,
        n_samples=args.n,
        rho=args.rho,
        L_ar=args.L_ar if args.L_ar else args.k - 1,
        delta=args.delta,
        seed=args.seed,
    )
    
    result = group_robustness_separation(spec)
    
    arts.write_json("config.json", {
        "benchmark": "group_robustness",
        "d": args.d,
        "k": args.k,
        "alpha": args.alpha,
        "n": args.n,
        "rho": args.rho,
        "L_ar": spec.L_ar,
        "delta": args.delta,
        "seed": args.seed,
    })
    arts.write_json("metrics.json", result)
    
    plots_dir = str(arts.run_dir / "plots")
    plot_group_robustness(result, plots_dir)
    
    # Optionally run scaling if requested
    if args.scaling:
        k_values = tuple(range(4, args.k + 2, 2))
        scaling_df = exponential_group_scaling(
            d=args.d,
            k_values=k_values,
            alpha=args.alpha,
            n_samples=args.n,
            rho=args.rho,
            seed=args.seed,
        )
        arts.write_json("scaling.json", scaling_df.to_dict(orient="records"))
        plot_exponential_scaling(scaling_df, plots_dir)
    
    # Summary
    print(f"\nGroup Robustness Results (k={args.k}, group_size=2^{{-{args.k}}}):")
    print(f"  AR (L={spec.L_ar}): GCal={result['ar']['empirical_gcal']:.6f}, theory_lb={result['ar']['theory_lower_bound']:.6f}")
    print(f"  Diffusion (ρ={args.rho}): GCal={result['diffusion']['empirical_gcal']:.6f}, theory={result['diffusion']['theory_prediction']:.6f}")
    print(f"  Separation: {result['separation']:.6f}, diffusion_wins={result['diffusion_wins']}")
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_approachability_suite(args: argparse.Namespace) -> None:
    """Multiscale approachability: constraint ladder experiments."""
    from forecastbench.metrics.multiscale_approachability import (
        multiscale_constraint_evaluation,
        approachability_dynamics,
        ApproachabilityDynamicsSpec,
        plot_constraint_ladder,
        plot_approachability_dynamics,
    )
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    degrees = tuple(range(1, args.max_degree + 1))
    rho_schedule = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    
    print(f"Running approachability suite: d={args.d}, degrees=1..{args.max_degree}, n={args.n}")
    
    # Multiscale constraint evaluation
    multiscale_results = multiscale_constraint_evaluation(
        d=args.d,
        degrees=degrees,
        rho_schedule=rho_schedule,
        n_samples=args.n,
        n_constraints_per_degree=args.n_constraints,
        seed=args.seed,
    )
    
    plots_dir = str(arts.run_dir / "plots")
    plot_constraint_ladder(multiscale_results, plots_dir)
    
    # Dynamics over time
    dynamics_spec = ApproachabilityDynamicsSpec(
        d=args.d,
        k=args.max_degree // 2,
        alpha=0.8,
        T=args.T_dynamics,
        rho=args.rho,
        seed=args.seed,
    )
    dynamics_results = approachability_dynamics(dynamics_spec)
    plot_approachability_dynamics(dynamics_results, plots_dir)
    
    arts.write_json("config.json", {
        "benchmark": "approachability_suite",
        "d": args.d,
        "max_degree": args.max_degree,
        "n": args.n,
        "n_constraints": args.n_constraints,
        "rho": args.rho,
        "T_dynamics": args.T_dynamics,
        "seed": args.seed,
    })
    arts.write_json("metrics.json", {
        "multiscale": multiscale_results,
        "dynamics": dynamics_results,
    })
    
    print(f"\nMultiscale results (summary):")
    for rho, deg_results in list(multiscale_results["results"].items())[:3]:
        print(f"  ρ={rho}:", end="")
        for s, v in list(deg_results.items())[:4]:
            print(f" deg{s}={v['max_violation']:.4f}", end="")
        print()
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_swap_regret(args: argparse.Namespace) -> None:
    """Swap regret vs external regret comparison."""
    from forecastbench.metrics.swap_regret import run_swap_regret_experiment
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    print(f"Running swap regret experiment: d={args.d}, k={args.k}, n={args.n}")
    
    results = run_swap_regret_experiment(
        d=args.d,
        k=args.k,
        alpha=args.alpha,
        n_samples=args.n,
        L_ar=args.L_ar,
        rho_diff=args.rho,
        output_dir=str(arts.run_dir / "plots"),
        seed=args.seed,
    )
    
    arts.write_json("config.json", {
        "benchmark": "swap_regret",
        "d": args.d,
        "k": args.k,
        "alpha": args.alpha,
        "n": args.n,
        "L_ar": args.L_ar,
        "rho": args.rho,
        "seed": args.seed,
    })
    arts.write_json("metrics.json", results)
    
    print(f"\nSwap vs External Regret:")
    print(f"  AR: external={results['ar']['external_regret']:.4f}, swap={results['ar']['swap_regret']:.4f}, gap={results['ar']['gap']:.4f}")
    print(f"  Diffusion: external={results['diffusion']['external_regret']:.4f}, swap={results['diffusion']['swap_regret']:.4f}, gap={results['diffusion']['gap']:.4f}")
    print(f"  Comparison: diff_wins_external={results['comparison']['diff_wins_external']}, diff_wins_swap={results['comparison']['diff_wins_swap']}")
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_synth_headlines(args: argparse.Namespace) -> None:
    """Run synthetic headlines benchmark for Type I/II error analysis."""
    from forecastbench.benchmarks.synthetic_headlines import (
        SyntheticHeadlineSpec,
        run_synthetic_headlines_benchmark,
    )
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    spec = SyntheticHeadlineSpec(
        n_samples=args.n_samples,
        n_test=args.n_test,
        structure=args.structure,
        n_factors=args.n_factors,
        noise=args.noise,
        seed=args.seed,
    )
    
    print(f"Running synthetic headlines benchmark: structure={spec.structure}")
    print(f"  n_samples={spec.n_samples}, n_test={spec.n_test}, noise={spec.noise}")
    
    results = run_synthetic_headlines_benchmark(spec)
    
    # Print sample headlines
    print("\nSample headlines:")
    for h in results.get("headlines_sample", [])[:3]:
        print(f"  - {h}")
    
    # Print baseline results
    print("\nBaseline results:")
    print(f"  {'Model':<15} {'MSE to Truth':<12} {'Correlation':<12} {'Excess Brier':<12}")
    for name, metrics in results["baselines"].items():
        corr = metrics.get("correlation_with_truth", 0)
        print(f"  {name:<15} {metrics['mse_to_truth']:.4f}       {corr:.3f}        {metrics['excess_brier']:.4f}")
    
    # Save results
    arts.write_json("config.json", {
        "benchmark": "synth_headlines",
        **results["spec"],
    })
    
    # Save test data for model evaluation
    test_data = results.pop("test_data", {})
    arts.write_json("test_data.json", test_data)
    arts.write_json("metrics.json", results)
    
    print(f"\nArtifacts: {arts.run_dir}")
    print(f"Test data saved to: {arts.run_dir}/test_data.json")
    print("Use this data to evaluate AR/Diffusion models and compare results.")


def cmd_synth_market(args: argparse.Namespace) -> None:
    """Run synthetic prediction market benchmark."""
    from forecastbench.benchmarks.synth_market_compare import (
        SynthMarketSpec,
        run_synth_market_benchmark,
        create_comparison_table,
    )
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    spec = SynthMarketSpec(
        d=args.d,
        m=args.m,
        n_train=args.n_train,
        n_test=args.n_test,
        structure=args.structure,
        n_factors=args.n_factors,
        factor_strength=args.factor_strength,
        noise=args.noise,
        seed=args.seed,
    )
    
    print(f"Running synthetic market benchmark: {spec.structure}")
    print(f"  d={spec.d}, m={spec.m}, n_train={spec.n_train}, n_test={spec.n_test}")
    
    results = run_synth_market_benchmark(spec)
    
    # Print comparison table
    table = create_comparison_table(results)
    print("\n" + table)
    
    # Save results
    arts.write_json("config.json", {
        "benchmark": "synth_market",
        **results["spec"],
    })
    arts.write_json("metrics.json", results)
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_pm_turtel_headlines(args: argparse.Namespace) -> None:
    """
    Enrich Polymarket data with Turtel-style headlines.
    
    Implements the headline enrichment approach from:
    Turtel et al. (2025) "Outcome-based RL to Predict the Future"
    
    Features:
    - Random prediction date sampling between market open/close
    - Headlines fetched BEFORE prediction date (no temporal leakage)
    - Optional LLM verification for leaked future information
    - Formatted prompts matching Turtel's structure
    """
    import json
    from pathlib import Path
    
    import pandas as pd
    
    from forecastbench.data.turtel_headlines import (
        TurtelHeadlineSpec,
        enrich_with_turtel_headlines,
    )
    
    df = pd.read_parquet(args.input)
    print(f"[turtel_headlines] Loaded {len(df)} rows from {args.input}")
    
    spec = TurtelHeadlineSpec(
        news_source=args.news_source,
        sample_prediction_date=args.sample_prediction_date,
        window_days=args.window_days,
        max_articles=args.max_articles,
        verify_no_leakage=args.verify_no_leakage,
        leakage_model=args.leakage_model,
        open_date_col=args.open_date_col,
        close_date_col=args.close_date_col,
        resolution_date_col=args.resolution_date_col,
        cache_dir=args.cache_dir,
        fuzzy_cache=getattr(args, 'fuzzy_cache', False),
        seed=args.seed,
    )
    
    df_enriched, meta = enrich_with_turtel_headlines(
        df=df,
        spec=spec,
        question_col=args.question_col,
        max_rows=args.max_rows,
        verbose=True,
    )
    
    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(out_path, index=False)
    
    print(f"[turtel_headlines] Saved to {out_path}")
    print(f"[turtel_headlines] Headlines coverage: {meta['headlines_coverage']:.1%}")
    print(json.dumps(meta["stats"], indent=2))


def cmd_grpo_train(args: argparse.Namespace) -> None:
    """
    GRPO/ReMax/RLCR training for AR forecasting models.
    
    Implements training approaches from:
    - Turtel et al. (2025) "Outcome-based RL to Predict the Future" (arXiv:2505.17989)
    - Damani et al. (2025) "Beyond Binary Rewards: Training LMs to Reason About Uncertainty" (arXiv:2507.16806)
    
    Algorithm options:
    - grpo: Standard GRPO with σ normalization
    - dr_grpo: Dr. GRPO (no σ normalization) - Turtel et al.
    - remax: ReMax with learned baseline (BEST in Turtel paper)
    
    Reward options:
    - turtel_brier: Pure Brier score R = -(p - y)²
    - rlcr: RLCR R = correctness + Brier (Damani et al., best for calibration)
    - kelly: Kelly criterion for log-wealth growth
    - blackwell_aware: Brier + Blackwell constraint penalty
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    import numpy as np
    import pandas as pd
    
    from forecastbench.train.grpo import (
        GRPORewardSpec,
        GRPOTrainSpec,
        train_grpo,
    )
    
    # Load data
    data_path = Path(args.data_path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Required columns: info (text), outcome (0/1), market_price (optional)
    if "info" not in df.columns:
        # Try to construct from other columns
        if "question" in df.columns and "headlines" in df.columns:
            df["info"] = df["question"] + "\n\n" + df["headlines"].fillna("")
        elif "question" in df.columns:
            df["info"] = df["question"]
        else:
            raise ValueError("Data must have 'info' column or 'question'+'headlines' columns")
    
    if "outcome" not in df.columns:
        if "resolved" in df.columns:
            df["outcome"] = df["resolved"].astype(int)
        elif "y" in df.columns:
            df["outcome"] = df["y"].astype(int)
        else:
            raise ValueError("Data must have 'outcome', 'resolved', or 'y' column")
    
    if "market_price" not in df.columns:
        if "market_prob" in df.columns:
            df["market_price"] = df["market_prob"]
        else:
            df["market_price"] = 0.5  # Default to uninformative market
    
    infos = df["info"].tolist()
    y = df["outcome"].values.astype(np.float64)
    q = df["market_price"].values.astype(np.float64)
    
    print(f"[grpo_train] Loaded {len(infos)} samples from {data_path}")
    
    # Build reward spec
    reward_spec = GRPORewardSpec(
        mode=args.reward_mode,
        rlcr_alpha=args.rlcr_alpha,
        rlcr_beta=args.rlcr_beta,
        rlcr_gamma=args.rlcr_gamma,
        rlcr_use_group_calibration=args.rlcr_use_groups,
    )
    
    # Build training spec
    train_spec = GRPOTrainSpec(
        model_name_or_path=args.model,
        algorithm=args.algorithm,
        K=args.K,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_coef=args.kl_coef,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gibberish_filter=args.gibberish_filter,
        max_input_chars=args.max_input_chars,
        early_stop_patience=args.early_stop,
        seed=args.seed,
        reward=reward_spec,
    )
    
    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{ts}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[grpo_train] Output directory: {out_dir}")
    print(f"[grpo_train] Algorithm: {args.algorithm}")
    print(f"[grpo_train] Reward mode: {args.reward_mode}")
    print(f"[grpo_train] Model: {args.model}")
    
    # Save config
    config = {
        "data_path": str(data_path),
        "n_samples": len(infos),
        "algorithm": args.algorithm,
        "reward_mode": args.reward_mode,
        "model": args.model,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "K": args.K,
        "lr": args.lr,
        "kl_coef": args.kl_coef,
        "rlcr_alpha": args.rlcr_alpha,
        "rlcr_beta": args.rlcr_beta,
        "rlcr_gamma": args.rlcr_gamma,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    # Train
    results = train_grpo(
        infos=infos,
        y=y,
        q=q,
        spec=train_spec,
        out_dir=out_dir,
    )
    
    # Save results
    (out_dir / "results.json").write_text(json.dumps({
        "n": results["n"],
        "steps": results["steps"],
        "best_step": results["best_step"],
        "best_brier": results["best_brier"],
        "early_stopped": results["early_stopped"],
    }, indent=2))
    
    print(f"[grpo_train] Training complete!")
    print(f"[grpo_train] Best Brier: {results['best_brier']:.4f} at step {results['best_step']}")
    print(f"[grpo_train] Saved to: {out_dir}")


# =============================================================================
# MEAN-REVERSION STAT ARB COMMANDS
# =============================================================================


def cmd_analyze_group_calibration(args: argparse.Namespace) -> None:
    """
    Analyze group-wise calibration regimes (mean-reversion vs momentum).
    
    For each group, computes rolling calibration statistics and classifies
    the regime as mean-reverting (low |E[Y-q|g]|) or momentum (persistent bias).
    """
    import pandas as pd
    from forecastbench.strategies.regime_detector import (
        GroupCalibrationTracker,
        RegimeDetectorConfig,
        compute_group_calibration_summary,
        detect_regime_changes,
    )
    from forecastbench.runner import RunArtifacts
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    # Load data
    df = pd.read_parquet(args.dataset_path)
    print(f"[analyze_group_calibration] Loaded {len(df)} rows")
    
    # Extract arrays (handle categorical columns)
    groups = df[args.group_col].astype(str).fillna("unknown").values
    prices = df[args.price_col].fillna(0.5).astype(float).values
    outcomes = df[args.outcome_col].fillna(0).astype(float).values
    
    # Configure regime detector
    cfg = RegimeDetectorConfig(
        ema_alpha=2.0 / (args.window + 1),  # Approximate EMA for window
        mean_revert_threshold=args.mean_revert_threshold,
        momentum_threshold=args.momentum_threshold,
        min_observations=10,
    )
    
    # Compute summary
    summary = compute_group_calibration_summary(groups, prices, outcomes, cfg=cfg)
    
    # Detect regime changes
    changes = detect_regime_changes(groups, prices, outcomes, cfg=cfg)
    
    # Compute aggregate stats
    regime_counts = {"mean_revert": 0, "momentum": 0, "neutral": 0}
    for g, stats in summary.items():
        regime = stats.get("regime", "neutral")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    # Report
    print(f"\n[analyze_group_calibration] Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} groups")
    
    print(f"\n[analyze_group_calibration] Top groups by |calibration error|:")
    sorted_groups = sorted(summary.items(), key=lambda x: abs(x[1].get("bias", 0)), reverse=True)
    for g, stats in sorted_groups[:10]:
        print(f"  {g}: bias={stats['bias']:.4f}, regime={stats['regime']}, n={stats['n_observations']}")
    
    print(f"\n[analyze_group_calibration] {len(changes)} regime changes detected")
    
    # Save results
    arts.write_json("group_calibration_summary.json", summary)
    arts.write_json("regime_changes.json", changes[:100])  # First 100 changes
    arts.write_json("regime_distribution.json", regime_counts)
    
    print(f"\n[analyze_group_calibration] Saved to: {arts.run_dir}")


def cmd_mean_reversion_backtest(args: argparse.Namespace) -> None:
    """
    Run walk-forward backtest for group mean-reversion strategy.
    
    This implements:
    1. Rolling calibration estimation (no lookahead)
    2. Regime detection per group
    3. Position construction per rebalance period
    4. PnL tracking with full attribution
    """
    import pandas as pd
    from forecastbench.strategies import (
        GroupMeanReversionConfig,
        RegimeDetectorConfig,
        BasketBuilderConfig,
        run_group_mean_reversion_backtest,
        run_walk_forward_backtest,
    )
    from forecastbench.runner import RunArtifacts
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    # Load data
    df = pd.read_parquet(args.dataset_path)
    print(f"[mean_reversion_backtest] Loaded {len(df)} rows")
    
    # Parse position methods
    if args.position_method == "all":
        methods = ("calibration", "dollar_neutral", "frechet")
    else:
        methods = tuple(m.strip() for m in args.position_method.split(","))
    
    # Configure
    regime_cfg = RegimeDetectorConfig(
        mean_revert_threshold=0.05,
        momentum_threshold=0.15,
        min_observations=10,
    )
    
    basket_cfg = BasketBuilderConfig(
        kelly_fraction=args.kelly_fraction,
        max_position_size=args.max_position,
        min_edge=args.min_edge,
    )
    
    cfg = GroupMeanReversionConfig(
        regime_cfg=regime_cfg,
        basket_cfg=basket_cfg,
        position_methods=methods,
        transaction_cost=args.transaction_cost,
        bootstrap_samples=args.bootstrap_n,
    )
    
    # Run backtest
    if args.train_frac > 0 and args.n_folds > 1:
        print(f"[mean_reversion_backtest] Running walk-forward with {args.n_folds} folds...")
        result = run_walk_forward_backtest(
            df,
            model_forecast_col=args.model_col,
            market_price_col=args.market_col,
            group_col=args.group_col,
            outcome_col=args.outcome_col,
            market_id_col=args.market_id_col,
            time_col=args.time_col,
            cfg=cfg,
            train_frac=args.train_frac,
            n_folds=args.n_folds,
            verbose=args.verbose,
        )
    else:
        print(f"[mean_reversion_backtest] Running single-pass backtest...")
        result = run_group_mean_reversion_backtest(
            df,
            model_forecast_col=args.model_col,
            market_price_col=args.market_col,
            group_col=args.group_col,
            outcome_col=args.outcome_col,
            market_id_col=args.market_id_col,
            time_col=args.time_col,
            cfg=cfg,
            verbose=args.verbose,
        )
    
    # Report
    print(f"\n[mean_reversion_backtest] Results:")
    print(f"  Total PnL: {result.total_pnl:.4f}")
    print(f"  ROI: {result.roi:.2%}")
    print(f"  Sharpe: {result.sharpe:.2f}")
    print(f"  Sharpe 95% CI: [{result.sharpe_ci[0]:.2f}, {result.sharpe_ci[1]:.2f}]")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  # Trades: {result.n_trades}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    
    print(f"\n[mean_reversion_backtest] PnL by group:")
    for g, pnl in sorted(result.pnl_by_group.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {g}: {pnl:.4f}")
    
    print(f"\n[mean_reversion_backtest] PnL by regime:")
    for r, pnl in result.pnl_by_regime.items():
        print(f"  {r}: {pnl:.4f}")
    
    print(f"\n[mean_reversion_backtest] PnL by method:")
    for m, pnl in result.pnl_by_method.items():
        print(f"  {m}: {pnl:.4f}")
    
    # Save results
    arts.write_json("backtest_results.json", result.to_dict())
    
    # Save equity curve if available
    if len(result.equity_curve) > 0:
        import numpy as np
        np.save(arts.run_dir / "equity_curve.npy", result.equity_curve)
    
    print(f"\n[mean_reversion_backtest] Saved to: {arts.run_dir}")


def cmd_compare_arb_strategies(args: argparse.Namespace) -> None:
    """
    Compare multiple stat arb strategies on the same dataset.
    
    Runs each strategy independently and computes bootstrap CIs for comparison.
    """
    import pandas as pd
    from forecastbench.strategies import (
        GroupMeanReversionConfig,
        RegimeDetectorConfig,
        BasketBuilderConfig,
        run_group_mean_reversion_backtest,
        compare_strategies,
    )
    from forecastbench.runner import RunArtifacts
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    # Load data
    df = pd.read_parquet(args.dataset_path)
    print(f"[compare_arb_strategies] Loaded {len(df)} rows")
    
    # Parse strategies
    strategies = [s.strip() for s in args.strategies.split(",")]
    print(f"[compare_arb_strategies] Comparing strategies: {strategies}")
    
    # Run comparison
    results = compare_strategies(
        df,
        model_forecast_col=args.model_col,
        market_price_col=args.market_col,
        group_col=args.group_col,
        outcome_col=args.outcome_col,
        strategies=strategies,
        bootstrap_n=args.bootstrap_n,
    )
    
    # Report
    print(f"\n[compare_arb_strategies] Results:")
    print(f"{'Strategy':<20} {'ROI':>10} {'Sharpe':>10} {'Win Rate':>10} {'# Trades':>10}")
    print("-" * 60)
    
    best_strategy = None
    best_sharpe = -float("inf")
    
    for name, result in results.items():
        print(f"{name:<20} {result.roi:>10.2%} {result.sharpe:>10.2f} {result.win_rate:>10.2%} {result.n_trades:>10}")
        if result.sharpe > best_sharpe:
            best_sharpe = result.sharpe
            best_strategy = name
    
    print("-" * 60)
    print(f"\nBest strategy: {best_strategy} (Sharpe = {best_sharpe:.2f})")
    
    # Save results
    comparison = {
        name: result.to_dict()
        for name, result in results.items()
    }
    comparison["_best_strategy"] = best_strategy
    
    arts.write_json("strategy_comparison.json", comparison)
    print(f"\n[compare_arb_strategies] Saved to: {arts.run_dir}")


def cmd_pm_hybrid_train(args: argparse.Namespace) -> None:
    """Train AR+Diffusion hybrid: AR for reasoning, diffusion for calibration."""
    import pandas as pd
    from forecastbench.models.ar_diffusion_hybrid import (
        ARDiffusionHybridForecaster,
        ARDiffusionHybridSpec,
        DiffusionRefinementHead,
    )
    from forecastbench.models.text_embedder import HFTextEmbedder, HFTextEmbedderSpec
    from forecastbench.models.ar_cot import ARCoTPredictor, ARCoTSpec
    from forecastbench.metrics import brier_loss, expected_calibration_error, log_loss
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    # Load data
    df = pd.read_parquet(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows))
    
    print(f"Loaded {len(df)} rows from {args.dataset_path}")
    
    # Split
    n = len(df)
    n_train = int(n * args.train_frac)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    text_cols = [c.strip() for c in str(args.text_cols).split(",")]
    
    def get_texts(df_sub):
        return [" ".join(str(row[c]) for c in text_cols if c in row and pd.notna(row[c])) 
                for _, row in df_sub.iterrows()]
    
    train_texts = get_texts(df_train)
    test_texts = get_texts(df_test)
    
    y_train = df_train[args.y_col].values.astype(np.float32)
    y_test = df_test[args.y_col].values.astype(np.float32)
    
    market_train = df_train[args.market_prob_col].values.astype(np.float32) if args.market_prob_col in df_train.columns else None
    market_test = df_test[args.market_prob_col].values.astype(np.float32) if args.market_prob_col in df_test.columns else None
    
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")
    
    # Step 1: Get AR predictions on training set
    print("\n=== Step 1: AR Predictions ===")
    ar_spec = ARCoTSpec(
        model_name_or_path=args.ar_model,
        device=args.ar_device,
        device_map=args.ar_device_map,
        temperature=args.ar_temperature,
        max_new_tokens=args.ar_max_new_tokens,
        include_cot=not args.ar_no_cot,
    )
    ar_predictor = ARCoTPredictor(ar_spec)
    
    q_ar_train, ar_meta_train = ar_predictor.predict_proba(train_texts, K=args.ar_K, seed=args.seed)
    q_ar_test, ar_meta_test = ar_predictor.predict_proba(test_texts, K=args.ar_K, seed=args.seed + 1)
    
    print(f"AR train: mean={np.mean(q_ar_train):.3f}, std={np.std(q_ar_train):.3f}")
    print(f"AR test: mean={np.mean(q_ar_test):.3f}, std={np.std(q_ar_test):.3f}")
    
    # Step 2: Get text embeddings
    print("\n=== Step 2: Text Embeddings ===")
    embed_spec = HFTextEmbedderSpec(
        model_name_or_path=args.embed_model,
        device=args.embed_device,
        dtype=args.embed_dtype,
        device_map=args.embed_device_map,
        max_length=args.embed_max_length,
    )
    embedder = HFTextEmbedder(embed_spec)
    
    embed_train = embedder.encode(train_texts, batch_size=args.embed_batch_size)
    embed_test = embedder.encode(test_texts, batch_size=args.embed_batch_size)
    
    print(f"Embeddings: train={embed_train.shape}, test={embed_test.shape}")
    
    # Step 3: Train diffusion refinement head
    print("\n=== Step 3: Train Diffusion Refinement ===")
    diff_head = DiffusionRefinementHead(
        cond_dim=embed_train.shape[1],
        hidden_dim=args.diff_hidden_dim,
        depth=args.diff_depth,
        T=args.diff_T,
        device=args.ar_device,
    )
    
    train_meta = diff_head.train_loop(
        q_ar=q_ar_train,
        cond=embed_train,
        p_true=y_train,
        steps=args.diff_train_steps,
        batch_size=args.diff_batch_size,
        lr=args.diff_lr,
        seed=args.seed,
    )
    
    # Step 4: Evaluate on test set
    print("\n=== Step 4: Evaluation ===")
    q_refined_test = diff_head.sample(
        q_ar=q_ar_test,
        cond=embed_test,
        n_samples=args.diff_samples,
        seed=args.seed + 2,
    )
    
    # Compute metrics for AR alone
    ar_brier = brier_loss(q_ar_test, y_test)
    ar_ece = expected_calibration_error(q_ar_test, y_test, n_bins=args.bins)
    ar_logloss = log_loss(q_ar_test, y_test)
    
    # Compute metrics for hybrid (AR + diffusion)
    hybrid_brier = brier_loss(q_refined_test, y_test)
    hybrid_ece = expected_calibration_error(q_refined_test, y_test, n_bins=args.bins)
    hybrid_logloss = log_loss(q_refined_test, y_test)
    
    # Market baseline if available
    if market_test is not None:
        market_brier = brier_loss(market_test, y_test)
        market_ece = expected_calibration_error(market_test, y_test, n_bins=args.bins)
        market_logloss = log_loss(market_test, y_test)
    else:
        market_brier = market_ece = market_logloss = None
    
    # Trading metrics (PNL, ROI, Sharpe)
    from forecastbench.benchmarks.polymarket_eval import realized_trading_pnl
    from forecastbench.metrics.trading_sim import simulate_kelly_roi, KellySimConfig
    
    tc = args.transaction_cost
    B = args.B
    
    def compute_trading_metrics(pred_prob, market_prob, y, name="model"):
        """Compute PNL, ROI, Sharpe for a model's predictions."""
        # Simple PNL (sign-based trading)
        pnl = realized_trading_pnl(
            y=y, market_prob=market_prob, pred_prob=pred_prob,
            B=B, transaction_cost=tc, mode="sign"
        )
        # Kelly ROI
        kelly_cfg = KellySimConfig(initial_bankroll=1.0, fee=tc, scale=0.25, frac_cap=0.1)
        kelly_result = simulate_kelly_roi(p=pred_prob, q=market_prob, y=y, cfg=kelly_cfg, return_curve=True)
        roi = kelly_result["roi"]
        
        # Sharpe ratio from per-trade returns
        positions = B * np.sign(pred_prob - market_prob)
        returns = positions * (y - market_prob) - tc * np.abs(positions)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(len(returns)))
        
        return {"pnl": float(pnl), "roi": float(roi), "sharpe": float(sharpe)}
    
    if market_test is not None:
        ar_trading = compute_trading_metrics(q_ar_test, market_test, y_test, "AR")
        hybrid_trading = compute_trading_metrics(q_refined_test, market_test, y_test, "Hybrid")
    else:
        ar_trading = {"pnl": 0.0, "roi": 0.0, "sharpe": 0.0}
        hybrid_trading = {"pnl": 0.0, "roi": 0.0, "sharpe": 0.0}
    
    print("\n=== Results ===")
    print(f"{'Model':<20} {'Brier':>10} {'ECE':>10} {'LogLoss':>10} {'PNL':>10} {'ROI%':>10} {'Sharpe':>10}")
    print("-" * 82)
    print(f"{'AR (Qwen)':<20} {ar_brier:>10.4f} {ar_ece:>10.4f} {ar_logloss:>10.4f} {ar_trading['pnl']:>10.4f} {ar_trading['roi']*100:>10.2f} {ar_trading['sharpe']:>10.2f}")
    print(f"{'Hybrid (AR+Diff)':<20} {hybrid_brier:>10.4f} {hybrid_ece:>10.4f} {hybrid_logloss:>10.4f} {hybrid_trading['pnl']:>10.4f} {hybrid_trading['roi']*100:>10.2f} {hybrid_trading['sharpe']:>10.2f}")
    if market_brier is not None:
        print(f"{'Market':<20} {market_brier:>10.4f} {market_ece:>10.4f} {market_logloss:>10.4f} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    # Improvement
    print(f"\nDiffusion improves AR Brier by: {(ar_brier - hybrid_brier) / ar_brier * 100:.1f}%")
    print(f"Diffusion improves AR ECE by: {(ar_ece - hybrid_ece) / ar_ece * 100:.1f}%")
    if ar_trading['pnl'] != 0:
        print(f"Diffusion improves AR PNL by: {(hybrid_trading['pnl'] - ar_trading['pnl']) / abs(ar_trading['pnl']) * 100:.1f}%")
        print(f"Diffusion improves AR ROI by: {(hybrid_trading['roi'] - ar_trading['roi']) * 100:.2f} pp")
    
    # Save results
    results = {
        "n_train": len(df_train),
        "n_test": len(df_test),
        "ar": {"brier": float(ar_brier), "ece": float(ar_ece), "logloss": float(ar_logloss), **ar_trading},
        "hybrid": {"brier": float(hybrid_brier), "ece": float(hybrid_ece), "logloss": float(hybrid_logloss), **hybrid_trading},
        "train_meta": train_meta,
        "improvement": {
            "brier_pct": float((ar_brier - hybrid_brier) / ar_brier * 100),
            "ece_pct": float((ar_ece - hybrid_ece) / ar_ece * 100),
            "pnl_diff": float(hybrid_trading['pnl'] - ar_trading['pnl']),
            "roi_diff_pp": float((hybrid_trading['roi'] - ar_trading['roi']) * 100),
        },
    }
    if market_brier is not None:
        results["market"] = {"brier": float(market_brier), "ece": float(market_ece), "logloss": float(market_logloss)}
    
    arts.write_json("config.json", {
        "benchmark": "pm_hybrid_train",
        "dataset_path": str(args.dataset_path),
        "max_rows": args.max_rows,
        "ar_model": args.ar_model,
        "ar_K": args.ar_K,
        "embed_model": args.embed_model,
        "diff_hidden_dim": args.diff_hidden_dim,
        "diff_depth": args.diff_depth,
        "diff_T": args.diff_T,
        "diff_train_steps": args.diff_train_steps,
    })
    arts.write_json("metrics.json", results)
    
    print(f"\nArtifacts: {arts.run_dir}")


def cmd_turtel_compare(args: argparse.Namespace) -> None:
    """Compare with Turtel et al. (2025) RLVR paper."""
    import pandas as pd
    from forecastbench.benchmarks.turtel_comparison import (
        run_turtel_comparison,
        create_turtel_comparison_table,
    )
    
    arts = RunArtifacts.create(run_name=str(args.run_name))
    arts.maybe_write_env()
    
    # Load dataset
    df = pd.read_parquet(args.dataset_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows))
    
    print(f"Running Turtel comparison on {len(df)} samples")
    
    group_cols = None
    if args.group_cols:
        group_cols = [c.strip() for c in str(args.group_cols).split(",")]
    
    results = run_turtel_comparison(
        df=df,
        pred_col=args.pred_col,
        y_col=args.y_col,
        market_prob_col=args.market_prob_col,
        group_cols=group_cols,
        output_dir=str(arts.run_dir / "plots"),
        turtel_brier=args.turtel_brier,
        turtel_ece=args.turtel_ece,
        turtel_roi=args.turtel_roi,
    )
    
    arts.write_json("config.json", {
        "benchmark": "turtel_comparison",
        "dataset_path": str(args.dataset_path),
        "max_rows": args.max_rows,
        "pred_col": args.pred_col,
        "y_col": args.y_col,
        "market_prob_col": args.market_prob_col,
        "group_cols": group_cols,
        "turtel_brier": args.turtel_brier,
        "turtel_ece": args.turtel_ece,
        "turtel_roi": args.turtel_roi,
    })
    arts.write_json("metrics.json", results)
    
    print(f"\nComparison Summary:")
    print(f"  Our Brier: {results['our_metrics']['brier']:.4f}")
    print(f"  Our ECE: {results['our_metrics']['ece']:.4f}")
    if "trading" in results["our_metrics"]:
        print(f"  Our ROI: {results['our_metrics']['trading']['roi']:.2%}")
    print(f"  Turtel ROI (reported): {args.turtel_roi:.2%}")
    
    print(f"\nArtifacts: {arts.run_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forecastbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_parity = sub.add_parser("parity", help="Synthetic parity benchmark.")
    p_parity.add_argument("--d", type=int, default=16)
    p_parity.add_argument("--k", type=int, default=8)
    p_parity.add_argument("--alpha", type=float, default=0.8)
    p_parity.add_argument("--n", type=int, default=50_000)
    p_parity.add_argument("--seed", type=int, default=0)
    p_parity.add_argument("--rho", type=float, default=0.95)
    p_parity.add_argument("--L", type=int, default=4)
    p_parity.add_argument("--bins", type=int, default=20)
    p_parity.add_argument("--transaction-cost", type=float, default=0.0)
    p_parity.add_argument("--run-name", type=str, default="parity")
    # Optional: synthetic Blackwell approachability diagnostics (AppErr_t curves).
    p_parity.add_argument("--blackwell", action="store_true", help="Compute synthetic Blackwell curves.")
    p_parity.add_argument(
        "--bw-group",
        type=str,
        default="chi_S",
        help="Grouping function for payoff family: chi_S|subcube_S|none",
    )
    p_parity.add_argument("--bw-eps", type=float, default=None, help="Box half-width eps (defaults to transaction-cost).")
    p_parity.add_argument("--bw-curve-every", type=int, default=1000)
    p_parity.add_argument("--bw-topk", type=int, default=10)
    p_parity.add_argument("--bw-clip-eps", type=float, default=1e-6)
    # Optional: run an actual HF LLM pricer on a subset (for local evidence / GPU runs).
    p_parity.add_argument("--llm-model", type=str, default=None, help="HF model name/path (e.g. Qwen/DeepSeek).")
    p_parity.add_argument("--llm-K", type=int, default=1, help="Self-consistency width K for LLM.")
    p_parity.add_argument("--llm-max-examples", type=int, default=200, help="Max examples for LLM eval.")
    p_parity.add_argument("--llm-device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p_parity.add_argument("--llm-dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    p_parity.add_argument("--llm-device-map", type=str, default=None, help='e.g. "auto" to shard across GPUs')
    p_parity.add_argument("--llm-temperature", type=float, default=0.7)
    p_parity.add_argument("--llm-top-p", type=float, default=0.95)
    p_parity.add_argument("--llm-max-new-tokens", type=int, default=256)
    p_parity.add_argument("--llm-no-cot", action="store_true", help="Disable requesting CoT.")
    p_parity.add_argument("--llm-agg", type=str, default="mean", help="mean|median aggregation over K samples")
    p_parity.add_argument("--text-style", type=str, default="natural", help="plain|natural")
    p_parity.set_defaults(func=cmd_parity)

    p_group = sub.add_parser("groupstress", help="Subcube small-group diagnostic for parity.")
    p_group.add_argument("--d", type=int, default=14)
    p_group.add_argument("--k", type=int, default=7)
    p_group.add_argument("--alpha", type=float, default=0.8)
    p_group.add_argument("--n", type=int, default=200_000)
    p_group.add_argument("--seed", type=int, default=0)
    p_group.add_argument("--rho", type=float, default=0.9)
    p_group.add_argument("--L", type=int, default=3)
    p_group.add_argument("--max-J", dest="max_J", type=int, default=None)
    p_group.add_argument("--exact", action="store_true")
    p_group.add_argument("--run-name", type=str, default="groupstress")
    p_group.set_defaults(func=cmd_groupstress)

    p_ip = sub.add_parser(
        "intrinsic_post",
        help="Synthetic intrinsic-vs-post-processing control on parity (Section sec:intrinsic-vs-post).",
    )
    p_ip.add_argument("--d", type=int, default=16)
    p_ip.add_argument("--k", type=int, default=8)
    p_ip.add_argument("--alpha", type=float, default=0.8)
    p_ip.add_argument("--seed", type=int, default=0)
    p_ip.add_argument("--rho", type=float, default=0.95, help="Diffusion compute knob (analytic T_rho f).")
    p_ip.add_argument("--L", type=int, default=4, help="AR depth knob / LLM prompt depth.")
    p_ip.add_argument("--n-train", dest="n_train", type=int, default=50_000)
    p_ip.add_argument("--n-test", dest="n_test", type=int, default=10_000)
    p_ip.add_argument("--bins", type=int, default=20)
    p_ip.add_argument("--transaction-cost", type=float, default=0.0)
    # Post-processing wrapper knobs
    p_ip.add_argument("--post-bins", dest="post_bins", type=int, default=20)
    p_ip.add_argument("--post-prior", dest="post_prior", type=float, default=5.0)
    p_ip.add_argument("--post-clip-eps", dest="post_clip_eps", type=float, default=1e-4)
    # Optional: run an actual HF LLM as "AR-intrinsic" (GPU recommended).
    p_ip.add_argument("--llm-model", type=str, default=None)
    p_ip.add_argument("--llm-K", type=int, default=1, help="Self-consistency width K for LLM.")
    p_ip.add_argument("--llm-device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p_ip.add_argument("--llm-dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    p_ip.add_argument("--llm-device-map", type=str, default=None, help='e.g. "auto" to shard across GPUs')
    p_ip.add_argument("--llm-temperature", type=float, default=0.7)
    p_ip.add_argument("--llm-top-p", type=float, default=0.95)
    p_ip.add_argument("--llm-max-new-tokens", type=int, default=256)
    p_ip.add_argument("--llm-no-cot", action="store_true")
    p_ip.add_argument("--llm-agg", type=str, default="mean", help="mean|median aggregation over K samples")
    p_ip.add_argument("--text-style", type=str, default="natural", help="plain|natural")
    p_ip.add_argument("--run-name", type=str, default="intrinsic_post")
    p_ip.set_defaults(func=cmd_intrinsic_post)

    # Real-data: Polymarket-style evaluation (no training loop).
    p_pm_eval = sub.add_parser("pm_eval", help="Evaluate a Polymarket-style dataset (optionally with an LLM).")
    p_pm_eval.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_pm_eval.add_argument("--run-name", type=str, default="pm_eval")
    p_pm_eval.add_argument("--max-examples", type=int, default=None)
    p_pm_eval.add_argument("--text-cols", type=str, default="question", help="Comma-separated columns to join.")
    p_pm_eval.add_argument("--pred-col", type=str, default="pred_prob")
    p_pm_eval.add_argument("--bins", type=int, default=20)
    p_pm_eval.add_argument("--transaction-cost", type=float, default=0.0)
    p_pm_eval.add_argument("--B", type=float, default=1.0)
    p_pm_eval.add_argument("--trading-mode", type=str, default="sign", help="sign|linear")
    p_pm_eval.add_argument("--group-cols", type=str, default=None, help="Comma-separated group columns (optional).")
    # Approachability diagnostics (optional)
    p_pm_eval.add_argument(
        "--approachability",
        action="store_true",
        help="Compute Blackwell approachability curves for group×bin calibration payoffs.",
    )
    p_pm_eval.add_argument(
        "--app-group-cols",
        type=str,
        default=None,
        help="Comma-separated group cols for approachability constraints (defaults to --group-cols).",
    )
    p_pm_eval.add_argument("--app-bins", type=int, default=20, help="Bins for group×bin constraint family.")
    p_pm_eval.add_argument(
        "--app-eps",
        type=float,
        default=None,
        help="Box half-width eps. If omitted, uses transaction_cost/B (fee margin model).",
    )
    p_pm_eval.add_argument(
        "--app-time-col",
        type=str,
        default=None,
        help='Column to order forecasts by (defaults to "market_prob_target_ts" then "createdAt" if present).',
    )
    p_pm_eval.add_argument("--app-curve-every", type=int, default=10, help="Sample curve every N steps.")
    p_pm_eval.add_argument("--app-topk", type=int, default=10, help="Report top-k violated constraints.")
    p_pm_eval.add_argument("--app-clip-eps", type=float, default=1e-6, help="Clipping epsilon for probabilities.")
    # Repair-at-resolution (optional)
    p_pm_eval.add_argument(
        "--repair-at-resolution",
        action="store_true",
        help="Simulate an online group×bin repair map updated only at market resolution times.",
    )
    p_pm_eval.add_argument(
        "--repair-group-cols",
        type=str,
        default=None,
        help="Comma-separated group cols for repair (defaults to --group-cols).",
    )
    p_pm_eval.add_argument("--repair-bins", type=int, default=20, help="Bins for the repair map.")
    p_pm_eval.add_argument("--repair-prior-strength", type=float, default=5.0, help="Beta prior strength for repair.")
    p_pm_eval.add_argument(
        "--repair-forecast-time-col",
        type=str,
        default=None,
        help='Forecast-time column (defaults to "market_prob_target_ts" then "createdAt" if present).',
    )
    p_pm_eval.add_argument(
        "--repair-event-time-col",
        type=str,
        default=None,
        help='Resolution-time column (defaults to "market_event_ts" then "endDate"/"closedTime" if present).',
    )
    p_pm_eval.add_argument("--repair-clip-eps", type=float, default=1e-6, help="Clipping epsilon for repair.")
    # LLM options (optional)
    p_pm_eval.add_argument("--llm-model", type=str, default=None)
    p_pm_eval.add_argument("--L", type=int, default=4)
    p_pm_eval.add_argument("--K", type=int, default=1)
    p_pm_eval.add_argument("--seed", type=int, default=0)
    p_pm_eval.add_argument("--llm-device", type=str, default="auto")
    p_pm_eval.add_argument("--llm-dtype", type=str, default="auto")
    p_pm_eval.add_argument("--llm-device-map", type=str, default=None, help='e.g. "auto" for multi-GPU sharding')
    p_pm_eval.add_argument("--llm-temperature", type=float, default=0.7)
    p_pm_eval.add_argument("--llm-top-p", type=float, default=0.95)
    p_pm_eval.add_argument("--llm-max-new-tokens", type=int, default=256)
    p_pm_eval.add_argument("--llm-no-cot", action="store_true")
    p_pm_eval.add_argument("--llm-agg", type=str, default="mean", help="mean|median aggregation over K samples")
    p_pm_eval.set_defaults(func=cmd_pm_eval)

    # ====== PM_EVAL_V2: Hierarchical Constraints ======
    p_pm_eval_v2 = sub.add_parser(
        "pm_eval_v2",
        help="Evaluate Polymarket dataset with hierarchical constraints (multicalibration + Frechet).",
    )
    p_pm_eval_v2.add_argument("--dataset-path", type=str, required=True)
    p_pm_eval_v2.add_argument("--run-name", type=str, default="pm_eval_v2")
    p_pm_eval_v2.add_argument("--max-examples", type=int, default=None)
    p_pm_eval_v2.add_argument("--pred-col", type=str, default="pred_prob", help="Prediction column")
    p_pm_eval_v2.add_argument("--y-col", type=str, default="y", help="Outcome column")
    p_pm_eval_v2.add_argument("--market-prob-col", type=str, default="market_prob", help="Market price column")
    p_pm_eval_v2.add_argument("--bins", type=int, default=10, help="Number of calibration bins")
    p_pm_eval_v2.add_argument("--seed", type=int, default=0)
    
    # Hierarchical constraints
    p_pm_eval_v2.add_argument(
        "--hierarchical-constraints",
        action="store_true",
        help="Compute hierarchical constraint violations (multicalib + Frechet)",
    )
    p_pm_eval_v2.add_argument(
        "--multicalib-groups",
        type=str,
        default="topic,volume_q5",
        help="Comma-separated columns for multicalibration groups",
    )
    p_pm_eval_v2.add_argument(
        "--frechet-bundle-col",
        type=str,
        default="category",
        help="Column for Frechet bundling (cross-market constraints)",
    )
    p_pm_eval_v2.add_argument("--bundle-size", type=int, default=3, help="Markets per Frechet bundle")
    
    # Approachability rate
    p_pm_eval_v2.add_argument(
        "--approachability-rate",
        action="store_true",
        help="Compute approachability rate with bootstrap CIs",
    )
    p_pm_eval_v2.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap resamples for rate CI")
    
    # Hybrid analysis
    p_pm_eval_v2.add_argument(
        "--ar-pred-col",
        type=str,
        default=None,
        help="AR-only prediction column for hybrid correction analysis",
    )
    p_pm_eval_v2.set_defaults(func=cmd_pm_eval_v2)

    # ====== MULTIMARKET_ARB: Frechet Arbitrage ======
    p_mm_arb = sub.add_parser(
        "multimarket_arb",
        help="Multi-market arbitrage analysis with Frechet constraints.",
    )
    p_mm_arb.add_argument("--dataset-path", type=str, required=True)
    p_mm_arb.add_argument("--run-name", type=str, default="multimarket_arb")
    p_mm_arb.add_argument("--max-rows", type=int, default=None)
    p_mm_arb.add_argument("--pred-col", type=str, default="pred_prob", help="Prediction column")
    p_mm_arb.add_argument("--y-col", type=str, default="y", help="Outcome column")
    p_mm_arb.add_argument("--market-prob-col", type=str, default="market_prob", help="Market price column")
    p_mm_arb.add_argument("--bundle-col", type=str, default="category", help="Column for bundling markets")
    p_mm_arb.add_argument("--bundle-size", type=int, default=3, help="Markets per bundle")
    p_mm_arb.add_argument(
        "--constraint-type",
        type=str,
        default="frechet",
        choices=["frechet", "implication", "mutual_exclusion"],
        help="Type of cross-market constraint",
    )
    p_mm_arb.add_argument("--seed", type=int, default=0)
    p_mm_arb.set_defaults(func=cmd_multimarket_arb)

    p_pm_bd = sub.add_parser("pm_build_polydata", help="Build dataset from a PolyData Explorer JSON download.")
    p_pm_bd.add_argument("--json-path", type=str, required=True)
    p_pm_bd.add_argument("--out", type=str, required=True)
    p_pm_bd.add_argument("--id-col", type=str, default="id")
    p_pm_bd.add_argument("--question-col", type=str, default="question")
    p_pm_bd.add_argument("--y-col", type=str, default="y")
    p_pm_bd.add_argument("--market-prob-col", type=str, default=None)
    p_pm_bd.set_defaults(func=cmd_pm_build_polydata)

    p_pm_sg = sub.add_parser("pm_build_subgraph", help="Build dataset from a Polymarket subgraph GraphQL query.")
    p_pm_sg.add_argument("--endpoint", type=str, required=True)
    p_pm_sg.add_argument("--query-file", type=str, required=True)
    p_pm_sg.add_argument("--record-path", type=str, default="markets", help="Dot-path to list in GraphQL response.")
    p_pm_sg.add_argument("--variables-json", type=str, default=None)
    p_pm_sg.add_argument("--out", type=str, required=True)
    p_pm_sg.add_argument("--id-col", type=str, default="id")
    p_pm_sg.add_argument("--question-col", type=str, default="question")
    p_pm_sg.add_argument("--y-col", type=str, default="y")
    p_pm_sg.add_argument("--market-prob-col", type=str, default=None)
    p_pm_sg.set_defaults(func=cmd_pm_build_subgraph)

    p_pm_dl = sub.add_parser(
        "pm_download_gamma",
        help="Download Polymarket markets via the public Gamma API to an append-only JSONL (with progress.json).",
    )
    p_pm_dl.add_argument("--out-dir", type=str, default="data/polymarket/gamma")
    p_pm_dl.add_argument("--page-size", type=int, default=1000)
    p_pm_dl.add_argument("--sleep-s", type=float, default=0.2)
    p_pm_dl.add_argument("--max-pages", type=int, default=None)
    p_pm_dl.add_argument("--start-offset", type=int, default=0)
    p_pm_dl.add_argument("--no-resume", action="store_true")
    p_pm_dl.set_defaults(func=cmd_pm_download_gamma)

    p_pm_bg = sub.add_parser(
        "pm_build_gamma",
        help="Build a Parquet evaluation dataset from a Gamma markets JSONL/JSONL.GZ dump (yes/no binary).",
    )
    p_pm_bg.add_argument("--input", type=str, required=True, help="Path to markets_raw.jsonl[.gz]")
    p_pm_bg.add_argument("--out", type=str, required=True, help="Output parquet path")
    p_pm_bg.add_argument("--min-volume", type=float, default=0.0)
    p_pm_bg.add_argument("--max-rows", type=int, default=None)
    p_pm_bg.add_argument("--chunk-rows", type=int, default=20000)
    p_pm_bg.add_argument("--allow-open", action="store_true", help="Include markets where closed=false")
    p_pm_bg.add_argument("--allow-non-extreme", action="store_true", help="Include markets without 0/1-like prices")
    p_pm_bg.add_argument("--allow-non-yesno", action="store_true", help="Include binary markets not labeled Yes/No")
    p_pm_bg.add_argument("--extreme-high", type=float, default=0.99)
    p_pm_bg.add_argument("--extreme-low", type=float, default=0.01)
    p_pm_bg.set_defaults(func=cmd_pm_build_gamma)

    p_pm_ec = sub.add_parser(
        "pm_enrich_clob",
        help="Enrich a Gamma-built dataset with a forecast-time market price from the Polymarket CLOB (needed for trading sim).",
    )
    p_pm_ec.add_argument("--input", type=str, required=True, help="Input parquet from pm_build_gamma")
    p_pm_ec.add_argument("--out", type=str, required=True, help="Output parquet with market_prob filled")
    p_pm_ec.add_argument("--fidelity", type=str, default="60", help="CLOB prices-history fidelity (e.g. 60, 1440)")
    p_pm_ec.add_argument("--earliest-timestamp", dest="earliest_timestamp", type=int, default=1704096000)
    p_pm_ec.add_argument("--sleep-s", dest="sleep_s", type=float, default=0.05)
    p_pm_ec.add_argument("--max-rows", type=int, default=None)
    p_pm_ec.add_argument("--no-resume", action="store_true")
    p_pm_ec.set_defaults(func=cmd_pm_enrich_clob)

    p_pm_ch = sub.add_parser(
        "pm_download_clob_history",
        help="Download full Polymarket CLOB price history for YES tokens (high-frequency time series).",
    )
    p_pm_ch.add_argument("--input", type=str, required=True, help="Input parquet (e.g. from pm_build_gamma)")
    p_pm_ch.add_argument("--out-dir", dest="out_dir", type=str, required=True, help="Output directory for per-token parquet files")
    p_pm_ch.add_argument("--fidelity", type=str, default="5", help="CLOB prices-history fidelity (e.g. 1, 5, 15, 60, 1440)")
    p_pm_ch.add_argument("--earliest-timestamp", dest="earliest_timestamp", type=int, default=1704096000)
    p_pm_ch.add_argument("--sleep-s", dest="sleep_s", type=float, default=0.05)
    p_pm_ch.add_argument("--max-rows", type=int, default=None)
    p_pm_ch.add_argument("--no-resume", action="store_true")
    p_pm_ch.add_argument("--token-col", dest="token_col", type=str, default="yes_token_id")
    p_pm_ch.add_argument("--created-at-col", dest="created_at_col", type=str, default="createdAt")
    p_pm_ch.add_argument("--id-col", dest="id_col", type=str, default="id")
    p_pm_ch.add_argument("--slug-col", dest="slug_col", type=str, default="slug")
    p_pm_ch.set_defaults(func=cmd_pm_download_clob_history)

    p_pm_hp = sub.add_parser(
        "pm_build_horizon_prices",
        help="Build a pm_eval dataset with market_prob taken at a fixed horizon before close, using downloaded CLOB histories.",
    )
    p_pm_hp.add_argument("--input", type=str, required=True, help="Input parquet from pm_build_gamma (resolved yes/no)")
    p_pm_hp.add_argument("--clob-history-dir", dest="clob_history_dir", type=str, required=True)
    p_pm_hp.add_argument("--out", type=str, required=True, help="Output parquet path")
    p_pm_hp.add_argument("--horizon-s", dest="horizon_s", type=int, required=True, help="Seconds before close to sample price")
    p_pm_hp.add_argument("--min-volume", dest="min_volume", type=float, default=0.0)
    p_pm_hp.add_argument("--max-rows", type=int, default=None)
    p_pm_hp.add_argument("--allow-missing-history", action="store_true")
    p_pm_hp.add_argument("--allow-missing-price", action="store_true")
    p_pm_hp.add_argument("--token-col", dest="token_col", type=str, default="yes_token_id")
    p_pm_hp.add_argument("--closed-time-col", dest="closed_time_col", type=str, default="closedTime")
    p_pm_hp.set_defaults(func=cmd_pm_build_horizon_prices)

    p_pm_cp = sub.add_parser(
        "pm_build_criterion_prices",
        help="Build a pm_eval dataset with market_prob computed from full CLOB history using a Themis/brier.fyi criterion (midpoint/time-average/before-close-...).",
    )
    p_pm_cp.add_argument("--input", type=str, required=True, help="Input parquet from pm_build_gamma (resolved yes/no)")
    p_pm_cp.add_argument("--clob-history-dir", dest="clob_history_dir", type=str, required=True)
    p_pm_cp.add_argument("--out", type=str, required=True, help="Output parquet path")
    p_pm_cp.add_argument(
        "--criterion",
        type=str,
        required=True,
        help="Criterion name (e.g. midpoint, time-average, before-close-hours-24, before-close-days-7, before-close-days-30).",
    )
    p_pm_cp.add_argument("--min-volume", dest="min_volume", type=float, default=0.0)
    p_pm_cp.add_argument("--max-rows", type=int, default=None)
    p_pm_cp.add_argument("--allow-missing-history", action="store_true")
    p_pm_cp.add_argument("--allow-missing-price", action="store_true")
    p_pm_cp.add_argument("--token-col", dest="token_col", type=str, default="yes_token_id")
    p_pm_cp.set_defaults(func=cmd_pm_build_criterion_prices)

    p_pm_news = sub.add_parser(
        "pm_enrich_news_gdelt",
        help="Enrich a dataset with retrieved news headlines from the (public) GDELT DOC API.",
    )
    p_pm_news.add_argument("--input", type=str, required=True)
    p_pm_news.add_argument("--out", type=str, required=True)
    p_pm_news.add_argument("--query-col", type=str, default="question")
    p_pm_news.add_argument("--time-col", type=str, default="market_prob_ts", help="Unix seconds column for retrieval end time")
    p_pm_news.add_argument("--window-days", type=int, default=7)
    p_pm_news.add_argument("--max-articles", type=int, default=10)
    p_pm_news.add_argument("--sleep-s", type=float, default=0.2)
    p_pm_news.add_argument("--sort", type=str, default="HybridRel")
    p_pm_news.add_argument("--cache-dir", type=str, default="data/polymarket/news_cache_gdelt")
    p_pm_news.add_argument("--max-rows", type=int, default=None)
    p_pm_news.add_argument("--no-resume", action="store_true")
    p_pm_news.add_argument("--out-text-col", type=str, default="news_text")
    p_pm_news.add_argument("--out-json-col", type=str, default="news_json")
    p_pm_news.add_argument("--out-n-col", type=str, default="news_n")
    p_pm_news.set_defaults(func=cmd_pm_enrich_news_gdelt)

    p_pm_dt = sub.add_parser(
        "pm_difftrain",
        help="Train a diffusion (conditional logit DDPM) forecaster on a Polymarket-style dataset and evaluate on a held-out split.",
    )
    p_pm_dt.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_pm_dt.add_argument("--run-name", type=str, default="pm_difftrain")
    p_pm_dt.add_argument("--max-rows", type=int, default=20000)
    p_pm_dt.add_argument("--text-cols", type=str, default="question,description")
    p_pm_dt.add_argument("--pred-col", type=str, default="pred_prob")
    p_pm_dt.add_argument("--train-frac", type=float, default=0.8)
    p_pm_dt.add_argument("--seed", type=int, default=0)
    p_pm_dt.add_argument("--group-cols", type=str, default=None)
    # Optional: bundle multiple markets into a joint prediction vector (e.g. bundle by category).
    p_pm_dt.add_argument("--bundle-col", type=str, default=None, help="e.g. category")
    p_pm_dt.add_argument("--bundle-size", type=int, default=1, help="Bundle size B (1 disables bundling)")
    p_pm_dt.add_argument("--bundle-drop-last", action="store_true", help="Drop incomplete bundles instead of padding")
    p_pm_dt.add_argument("--bundle-heads", type=int, default=4, help="Transformer heads for bundle denoiser")
    p_pm_dt.add_argument("--bundle-dropout", type=float, default=0.0, help="Transformer dropout for bundle denoiser")
    p_pm_dt.add_argument("--bins", type=int, default=20)
    p_pm_dt.add_argument("--transaction-cost", type=float, default=0.0)
    p_pm_dt.add_argument("--B", type=float, default=1.0)
    p_pm_dt.add_argument("--trading-mode", type=str, default="sign")
    # embedding
    p_pm_dt.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_pm_dt.add_argument("--embed-device", type=str, default="auto")
    p_pm_dt.add_argument("--embed-dtype", type=str, default="auto")
    p_pm_dt.add_argument("--embed-device-map", type=str, default=None, help='e.g. "auto" for large LLM encoders')
    p_pm_dt.add_argument("--embed-no-trust-remote-code", action="store_true")
    p_pm_dt.add_argument("--embed-max-length", type=int, default=256)
    p_pm_dt.add_argument("--embed-batch-size", type=int, default=64)
    p_pm_dt.add_argument("--embed-no-normalize", action="store_true")
    # diffusion training
    p_pm_dt.add_argument("--device", type=str, default="auto")
    p_pm_dt.add_argument("--label-eps", type=float, default=1e-3)
    p_pm_dt.add_argument("--train-steps", type=int, default=2000)
    p_pm_dt.add_argument("--batch-size", type=int, default=256)
    p_pm_dt.add_argument("--lr", type=float, default=2e-4)
    p_pm_dt.add_argument("--T", type=int, default=64)
    p_pm_dt.add_argument("--beta-start", type=float, default=1e-4)
    p_pm_dt.add_argument("--beta-end", type=float, default=2e-2)
    p_pm_dt.add_argument("--time-dim", type=int, default=64)
    p_pm_dt.add_argument("--hidden-dim", type=int, default=256)
    p_pm_dt.add_argument("--depth", type=int, default=3)
    # diffusion inference
    p_pm_dt.add_argument("--sample-steps", type=int, default=32)
    p_pm_dt.add_argument("--eta", type=float, default=0.0)
    p_pm_dt.add_argument("--mc", type=int, default=16)
    p_pm_dt.add_argument("--agg", type=str, default="mean", help="mean|median")
    p_pm_dt.add_argument("--log-every", type=int, default=200)
    p_pm_dt.set_defaults(func=cmd_pm_difftrain)

    p_pm_ds = sub.add_parser(
        "pm_diff_sample",
        help="Inference-only sampling from a trained diffusion model on a Polymarket-style dataset (for compute sweeps).",
    )
    p_pm_ds.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_pm_ds.add_argument("--model-path", dest="model_path", type=str, required=True)
    p_pm_ds.add_argument("--run-name", type=str, default="pm_diff_sample")
    p_pm_ds.add_argument("--max-rows", type=int, default=None)
    p_pm_ds.add_argument("--text-cols", type=str, default="question,description")
    p_pm_ds.add_argument("--pred-col", type=str, default="pred_prob")
    p_pm_ds.add_argument("--seed", type=int, default=0)
    p_pm_ds.add_argument("--group-cols", type=str, default=None)
    p_pm_ds.add_argument("--bins", type=int, default=20)
    p_pm_ds.add_argument("--transaction-cost", type=float, default=0.0)
    p_pm_ds.add_argument("--B", type=float, default=1.0)
    p_pm_ds.add_argument("--trading-mode", type=str, default="sign")
    # embedding
    p_pm_ds.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_pm_ds.add_argument("--embed-device", type=str, default="auto")
    p_pm_ds.add_argument("--embed-dtype", type=str, default="auto")
    p_pm_ds.add_argument("--embed-device-map", type=str, default=None, help='e.g. "auto" for large LLM encoders')
    p_pm_ds.add_argument("--embed-no-trust-remote-code", action="store_true")
    p_pm_ds.add_argument("--embed-max-length", type=int, default=256)
    p_pm_ds.add_argument("--embed-batch-size", type=int, default=64)
    p_pm_ds.add_argument("--embed-no-normalize", action="store_true")
    # Optional: bundling flags (required when sampling a bundle diffusion checkpoint)
    p_pm_ds.add_argument("--bundle-col", type=str, default=None, help="e.g. category")
    p_pm_ds.add_argument("--bundle-size", type=int, default=None, help="Must match bundle diffusion checkpoint")
    p_pm_ds.add_argument("--bundle-drop-last", action="store_true")
    # diffusion inference
    p_pm_ds.add_argument("--device", type=str, default="auto")
    p_pm_ds.add_argument("--sample-steps", type=int, default=32)
    p_pm_ds.add_argument("--eta", type=float, default=0.0)
    p_pm_ds.add_argument("--mc", type=int, default=16)
    p_pm_ds.add_argument("--agg", type=str, default="mean", help="mean|median")
    # Approachability (same flags as pm_eval)
    p_pm_ds.add_argument("--approachability", action="store_true")
    p_pm_ds.add_argument("--app-group-cols", type=str, default=None)
    p_pm_ds.add_argument("--app-bins", type=int, default=20)
    p_pm_ds.add_argument("--app-eps", type=float, default=None)
    p_pm_ds.add_argument("--app-time-col", type=str, default=None)
    p_pm_ds.add_argument("--app-curve-every", type=int, default=10)
    p_pm_ds.add_argument("--app-topk", type=int, default=10)
    p_pm_ds.add_argument("--app-clip-eps", type=float, default=1e-6)
    # Repair-at-resolution
    p_pm_ds.add_argument("--repair-at-resolution", action="store_true")
    p_pm_ds.add_argument("--repair-group-cols", type=str, default=None)
    p_pm_ds.add_argument("--repair-bins", type=int, default=20)
    p_pm_ds.add_argument("--repair-prior-strength", type=float, default=5.0)
    p_pm_ds.add_argument("--repair-forecast-time-col", type=str, default=None)
    p_pm_ds.add_argument("--repair-event-time-col", type=str, default=None)
    p_pm_ds.add_argument("--repair-clip-eps", type=float, default=1e-6)
    p_pm_ds.set_defaults(func=cmd_pm_diff_sample)

    p_pm_ct = sub.add_parser(
        "pm_learnedCt_arb",
        help="Evaluate a bundle diffusion model by learning C_t from MC samples and estimating statistical arbitrage (Hedge + optional neural witness).",
    )
    p_pm_ct.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_pm_ct.add_argument("--model-path", dest="model_path", type=str, required=True)
    p_pm_ct.add_argument("--run-name", type=str, default="pm_learnedCt_arb")
    p_pm_ct.add_argument("--max-rows", type=int, default=None)
    p_pm_ct.add_argument("--text-cols", type=str, default="question,description")
    p_pm_ct.add_argument("--pred-col", type=str, default="pred_prob")
    p_pm_ct.add_argument("--seed", type=int, default=0)
    p_pm_ct.add_argument("--group-cols", type=str, default=None)
    p_pm_ct.add_argument("--bins", type=int, default=20)
    p_pm_ct.add_argument("--transaction-cost", type=float, default=0.0)
    p_pm_ct.add_argument("--B", type=float, default=1.0)
    p_pm_ct.add_argument("--trading-mode", type=str, default="sign")
    # embedding
    p_pm_ct.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_pm_ct.add_argument("--embed-device", type=str, default="auto")
    p_pm_ct.add_argument("--embed-dtype", type=str, default="auto")
    p_pm_ct.add_argument("--embed-device-map", type=str, default=None)
    p_pm_ct.add_argument("--embed-no-trust-remote-code", action="store_true")
    p_pm_ct.add_argument("--embed-max-length", type=int, default=256)
    p_pm_ct.add_argument("--embed-batch-size", type=int, default=64)
    p_pm_ct.add_argument("--embed-no-normalize", action="store_true")
    # bundling (required)
    p_pm_ct.add_argument("--bundle-col", type=str, required=True, help="e.g. topic|category")
    p_pm_ct.add_argument("--bundle-size", type=int, default=None, help="Must match bundle diffusion checkpoint")
    p_pm_ct.add_argument("--bundle-drop-last", action="store_true")
    # diffusion inference
    p_pm_ct.add_argument("--device", type=str, default="auto")
    p_pm_ct.add_argument("--sample-steps", type=int, default=32)
    p_pm_ct.add_argument("--eta", type=float, default=0.0)
    p_pm_ct.add_argument("--mc", type=int, default=16)
    p_pm_ct.add_argument("--agg", type=str, default="mean", help="mean|median")
    # learned-C_t arb knobs
    p_pm_ct.add_argument("--arb-hedge-eta", type=float, default=None)
    p_pm_ct.add_argument("--arb-no-neural", action="store_true")
    p_pm_ct.add_argument("--arb-witness-hidden", type=int, default=128)
    p_pm_ct.add_argument("--arb-witness-depth", type=int, default=2)
    p_pm_ct.add_argument("--arb-witness-lr", type=float, default=1e-3)
    p_pm_ct.add_argument("--arb-witness-weight-decay", type=float, default=0.0)
    p_pm_ct.add_argument("--arb-witness-grad-clip", type=float, default=1.0)
    p_pm_ct.add_argument("--arb-max-steps", type=int, default=None)
    # repair-at-resolution (optional; scalar postprocessing baseline)
    p_pm_ct.add_argument("--repair-at-resolution", action="store_true")
    p_pm_ct.add_argument("--repair-group-cols", type=str, default=None)
    p_pm_ct.add_argument("--repair-bins", type=int, default=20)
    p_pm_ct.add_argument("--repair-prior-strength", type=float, default=5.0)
    p_pm_ct.add_argument("--repair-forecast-time-col", type=str, default=None)
    p_pm_ct.add_argument("--repair-event-time-col", type=str, default=None)
    p_pm_ct.add_argument("--repair-clip-eps", type=float, default=1e-6)
    p_pm_ct.set_defaults(func=cmd_pm_learnedCt_arb)

    p_pm_rl = sub.add_parser(
        "pm_rlvr_train",
        help="Train an AR model with an RLVR-style hybrid reward (logscore + PnL - KL) on a Polymarket-style dataset.",
    )
    p_pm_rl.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_pm_rl.add_argument("--run-name", type=str, default="pm_rlvr_train")
    p_pm_rl.add_argument("--max-rows", type=int, default=None)
    p_pm_rl.add_argument("--text-cols", type=str, default="question,description")
    p_pm_rl.add_argument("--y-col", dest="y_col", type=str, default="y")
    p_pm_rl.add_argument("--market-prob-col", dest="market_prob_col", type=str, default="market_prob")
    # Model
    p_pm_rl.add_argument("--model-name-or-path", dest="model_name_or_path", type=str, default="Qwen/Qwen3-14B")
    p_pm_rl.add_argument("--device", type=str, default="auto")
    p_pm_rl.add_argument("--dtype", type=str, default="auto")
    p_pm_rl.add_argument("--device-map", dest="device_map", type=str, default=None, help='e.g. "auto"')
    p_pm_rl.add_argument("--no-trust-remote-code", action="store_true")
    # QLoRA-ish knobs
    p_pm_rl.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading (more VRAM).")
    p_pm_rl.add_argument("--bnb-4bit-compute-dtype", type=str, default="bfloat16")
    p_pm_rl.add_argument("--no-lora", action="store_true", help="Disable LoRA (full finetune; usually infeasible).")
    p_pm_rl.add_argument("--lora-r", type=int, default=16)
    p_pm_rl.add_argument("--lora-alpha", type=int, default=32)
    p_pm_rl.add_argument("--lora-dropout", type=float, default=0.05)
    # Prompting / decoding
    p_pm_rl.add_argument("--no-cot", action="store_true")
    p_pm_rl.add_argument("--cot-max-steps", type=int, default=4)
    p_pm_rl.add_argument("--max-prompt-tokens", type=int, default=512)
    p_pm_rl.add_argument("--max-new-tokens", type=int, default=192)
    p_pm_rl.add_argument("--temperature", type=float, default=0.7)
    p_pm_rl.add_argument("--top-p", type=float, default=0.95)
    # Reward (hybrid)
    p_pm_rl.add_argument("--alpha-logscore", type=float, default=1.0)
    p_pm_rl.add_argument("--beta-pnl", type=float, default=0.1)
    p_pm_rl.add_argument("--B", type=float, default=1.0)
    p_pm_rl.add_argument("--transaction-cost", type=float, default=0.0)
    p_pm_rl.add_argument("--trading-mode", type=str, default="linear", help="sign|linear")
    # Training
    p_pm_rl.add_argument("--steps", type=int, default=200)
    p_pm_rl.add_argument("--batch-size", type=int, default=4)
    p_pm_rl.add_argument("--lr", type=float, default=1e-4)
    p_pm_rl.add_argument("--weight-decay", type=float, default=0.0)
    p_pm_rl.add_argument("--grad-clip", type=float, default=1.0)
    p_pm_rl.add_argument("--kl-coef", type=float, default=0.02)
    p_pm_rl.add_argument("--reward-clip", type=float, default=10.0)
    p_pm_rl.add_argument("--baseline-ema", type=float, default=0.95)
    p_pm_rl.add_argument("--log-every", type=int, default=10)
    p_pm_rl.add_argument("--save-every", type=int, default=50)
    p_pm_rl.add_argument("--seed", type=int, default=0)
    p_pm_rl.set_defaults(func=cmd_pm_rlvr_train)

    p_pm_re = sub.add_parser(
        "pm_rlvr_eval",
        help="Evaluate a LoRA-adapted AR model (RLVR-trained) with K-sampling + median aggregation on a Polymarket-style dataset.",
    )
    p_pm_re.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_pm_re.add_argument("--run-name", type=str, default="pm_rlvr_eval")
    p_pm_re.add_argument("--max-examples", type=int, default=None)
    p_pm_re.add_argument("--text-cols", type=str, default="question,description")
    p_pm_re.add_argument("--pred-col", type=str, default="pred_prob")
    p_pm_re.add_argument("--bins", type=int, default=20)
    p_pm_re.add_argument("--transaction-cost", type=float, default=0.0)
    p_pm_re.add_argument("--B", type=float, default=1.0)
    p_pm_re.add_argument("--trading-mode", type=str, default="sign")
    p_pm_re.add_argument("--group-cols", type=str, default=None)
    # RLVR model loading/inference
    p_pm_re.add_argument("--base-model", type=str, required=True)
    p_pm_re.add_argument("--adapter-path", type=str, required=True)
    p_pm_re.add_argument("--K", type=int, default=5)
    p_pm_re.add_argument("--L", type=int, default=4, help="CoT step budget knob (prompt).")
    p_pm_re.add_argument("--agg", type=str, default="median", help="mean|median over K samples")
    p_pm_re.add_argument("--seed", type=int, default=0)
    p_pm_re.add_argument("--device", type=str, default="auto")
    p_pm_re.add_argument("--dtype", type=str, default="auto")
    p_pm_re.add_argument("--device-map", type=str, default=None, help='e.g. "auto"')
    p_pm_re.add_argument("--no-trust-remote-code", action="store_true")
    p_pm_re.add_argument("--no-4bit", action="store_true")
    p_pm_re.add_argument("--bnb-4bit-compute-dtype", type=str, default="bfloat16")
    p_pm_re.add_argument("--temperature", type=float, default=0.7)
    p_pm_re.add_argument("--top-p", type=float, default=0.95)
    p_pm_re.add_argument("--max-new-tokens", type=int, default=256)
    p_pm_re.add_argument("--no-cot", action="store_true")
    # Optional: approachability + repair (same as pm_eval)
    p_pm_re.add_argument("--approachability", action="store_true")
    p_pm_re.add_argument("--app-group-cols", type=str, default=None)
    p_pm_re.add_argument("--app-bins", type=int, default=20)
    p_pm_re.add_argument("--app-eps", type=float, default=None)
    p_pm_re.add_argument("--app-time-col", type=str, default=None)
    p_pm_re.add_argument("--app-curve-every", type=int, default=10)
    p_pm_re.add_argument("--app-topk", type=int, default=10)
    p_pm_re.add_argument("--app-clip-eps", type=float, default=1e-6)
    p_pm_re.add_argument("--repair-at-resolution", action="store_true")
    p_pm_re.add_argument("--repair-group-cols", type=str, default=None)
    p_pm_re.add_argument("--repair-bins", type=int, default=20)
    p_pm_re.add_argument("--repair-prior-strength", type=float, default=5.0)
    p_pm_re.add_argument("--repair-forecast-time-col", type=str, default=None)
    p_pm_re.add_argument("--repair-event-time-col", type=str, default=None)
    p_pm_re.add_argument("--repair-clip-eps", type=float, default=1e-6)
    p_pm_re.set_defaults(func=cmd_pm_rlvr_eval)

    p_pm_cmp = sub.add_parser(
        "pm_compare",
        help="Compare multiple pm_eval-like runs using predictions.parquet and bootstrap CIs.",
    )
    p_pm_cmp.add_argument("--run-name", type=str, default="pm_compare")
    p_pm_cmp.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeated: NAME,RUN_DIR[,PRED_COL] where PRED_COL defaults to inferred config pred_col.",
    )
    p_pm_cmp.add_argument("--baseline", type=str, default=None, help="Optional baseline NAME for paired diff CIs.")
    p_pm_cmp.add_argument("--bins", type=int, default=20)
    p_pm_cmp.add_argument("--transaction-cost", type=float, default=0.0)
    p_pm_cmp.add_argument("--B", type=float, default=1.0)
    p_pm_cmp.add_argument("--trading-mode", type=str, default="sign")
    p_pm_cmp.add_argument("--n-boot", dest="n_boot", type=int, default=200)
    p_pm_cmp.add_argument("--seed", type=int, default=0)
    p_pm_cmp.set_defaults(func=cmd_pm_compare)

    p_dt = sub.add_parser("difftrain", help="Train a tiny logit-diffusion model on parity (local sanity).")
    p_dt.add_argument("--d", type=int, default=16)
    p_dt.add_argument("--k", type=int, default=8)
    p_dt.add_argument("--alpha", type=float, default=0.8)
    p_dt.add_argument("--seed", type=int, default=0)
    p_dt.add_argument("--n-train", dest="n_train", type=int, default=50_000)
    p_dt.add_argument("--n-test", dest="n_test", type=int, default=10_000)
    p_dt.add_argument("--train-steps", type=int, default=2000)
    p_dt.add_argument("--batch-size", type=int, default=256)
    p_dt.add_argument("--lr", type=float, default=2e-4)
    p_dt.add_argument("--T", type=int, default=64)
    p_dt.add_argument("--beta-start", type=float, default=1e-4)
    p_dt.add_argument("--beta-end", type=float, default=2e-2)
    p_dt.add_argument("--time-dim", type=int, default=64)
    p_dt.add_argument("--hidden-dim", type=int, default=256)
    p_dt.add_argument("--depth", type=int, default=3)
    p_dt.add_argument("--device", type=str, default="auto")
    p_dt.add_argument("--sample-steps", type=int, default=32)
    p_dt.add_argument("--eta", type=float, default=0.0)
    p_dt.add_argument("--bins", type=int, default=20)
    p_dt.add_argument("--transaction-cost", type=float, default=0.0)
    p_dt.add_argument("--log-every", type=int, default=200)
    p_dt.add_argument("--run-name", type=str, default="difftrain_parity")
    p_dt.set_defaults(func=cmd_difftrain)

    p_dts = sub.add_parser(
        "difftrain_simplex",
        help="Train a tiny ALR-simplex diffusion model on a synthetic multi-outcome parity task.",
    )
    p_dts.add_argument("--d", type=int, default=16)
    p_dts.add_argument("--k", type=int, default=8)
    p_dts.add_argument("--n-outcomes", dest="n_outcomes", type=int, default=4)
    p_dts.add_argument("--alpha", type=float, default=1.0, help="ALR-space amplitude (logit-like).")
    p_dts.add_argument("--seed", type=int, default=0)
    p_dts.add_argument("--n-train", dest="n_train", type=int, default=50_000)
    p_dts.add_argument("--n-test", dest="n_test", type=int, default=10_000)
    p_dts.add_argument("--train-steps", type=int, default=2000)
    p_dts.add_argument("--batch-size", type=int, default=256)
    p_dts.add_argument("--lr", type=float, default=2e-4)
    p_dts.add_argument("--T", type=int, default=64)
    p_dts.add_argument("--beta-start", type=float, default=1e-4)
    p_dts.add_argument("--beta-end", type=float, default=2e-2)
    p_dts.add_argument("--time-dim", type=int, default=64)
    p_dts.add_argument("--hidden-dim", type=int, default=256)
    p_dts.add_argument("--depth", type=int, default=3)
    p_dts.add_argument("--device", type=str, default="auto")
    p_dts.add_argument("--sample-steps", type=int, default=32)
    p_dts.add_argument("--eta", type=float, default=0.0)
    p_dts.add_argument("--bins", type=int, default=15)
    p_dts.add_argument("--log-every", type=int, default=200)
    p_dts.add_argument("--run-name", type=str, default="difftrain_simplex")
    p_dts.set_defaults(func=cmd_difftrain_simplex)

    p_lg = sub.add_parser(
        "logical_graph",
        help="Train a bundle diffusion model on a synthetic logical implication graph (Family S2).",
    )
    p_lg.add_argument("--d", type=int, default=16)
    p_lg.add_argument("--m", type=int, default=10, help="Number of events/nodes in the graph.")
    p_lg.add_argument("--structure", type=str, default="chain", help="Graph structure: 'chain' or 'tree' (tree not implemented).")
    p_lg.add_argument("--noise", type=float, default=1.0)
    p_lg.add_argument("--seed", type=int, default=0)
    p_lg.add_argument("--n-train", dest="n_train", type=int, default=50_000)
    p_lg.add_argument("--n-test", dest="n_test", type=int, default=10_000)
    p_lg.add_argument("--train-steps", type=int, default=2000)
    p_lg.add_argument("--batch-size", type=int, default=256)
    p_lg.add_argument("--lr", type=float, default=2e-4)
    p_lg.add_argument("--T", type=int, default=64)
    p_lg.add_argument("--beta-start", type=float, default=1e-4)
    p_lg.add_argument("--beta-end", type=float, default=2e-2)
    p_lg.add_argument("--time-dim", type=int, default=64)
    p_lg.add_argument("--hidden-dim", type=int, default=256)
    p_lg.add_argument("--depth", type=int, default=3)
    p_lg.add_argument("--heads", type=int, default=4)
    p_lg.add_argument("--dropout", type=float, default=0.0)
    p_lg.add_argument("--device", type=str, default="auto")
    p_lg.add_argument("--label-eps", type=float, default=1e-3)
    p_lg.add_argument("--sample-steps", type=str, default="32,64")
    p_lg.add_argument("--eta", type=float, default=0.0)
    p_lg.add_argument("--mc", type=int, default=16)
    p_lg.add_argument("--agg", type=str, default="median")
    p_lg.add_argument("--curve-every", type=int, default=200)
    p_lg.add_argument("--no-box", action="store_true")
    p_lg.add_argument("--log-every", type=int, default=200)
    p_lg.add_argument("--run-name", type=str, default="logical_graph")
    p_lg.set_defaults(func=cmd_logical_graph)

    # ====== NEW THEORY-ALIGNED BENCHMARKS ======

    p_cf = sub.add_parser(
        "cliff_fog",
        help="Cliff vs Fog: AR spectral cliff vs diffusion continuous recovery (Propositions 6-7).",
    )
    p_cf.add_argument("--d", type=int, default=20)
    p_cf.add_argument("--k", type=int, default=8)
    p_cf.add_argument("--alpha", type=float, default=0.8)
    p_cf.add_argument("--n", type=int, default=100000)
    p_cf.add_argument("--seed", type=int, default=0)
    p_cf.add_argument("--ar-width", action="store_true", help="Include K=4,16 for AR ablation")
    p_cf.add_argument("--run-name", type=str, default="cliff_fog")
    p_cf.set_defaults(func=cmd_cliff_fog)

    p_gr = sub.add_parser(
        "group_robustness",
        help="Group robustness: Prop 8-9 experiments on exponentially small subgroups.",
    )
    p_gr.add_argument("--d", type=int, default=24)
    p_gr.add_argument("--k", type=int, default=8)
    p_gr.add_argument("--alpha", type=float, default=0.8)
    p_gr.add_argument("--n", type=int, default=500000)
    p_gr.add_argument("--rho", type=float, default=0.95)
    p_gr.add_argument("--L-ar", dest="L_ar", type=int, default=None, help="AR depth; defaults to k-1")
    p_gr.add_argument("--delta", type=float, default=0.05, help="Confidence level for CI")
    p_gr.add_argument("--scaling", action="store_true", help="Run scaling experiment across k values")
    p_gr.add_argument("--seed", type=int, default=0)
    p_gr.add_argument("--run-name", type=str, default="group_robustness")
    p_gr.set_defaults(func=cmd_group_robustness)

    p_app = sub.add_parser(
        "approachability_suite",
        help="Multiscale approachability: constraint ladder experiments (§7.5).",
    )
    p_app.add_argument("--d", type=int, default=20)
    p_app.add_argument("--max-degree", type=int, default=8)
    p_app.add_argument("--n", type=int, default=50000)
    p_app.add_argument("--n-constraints", type=int, default=50)
    p_app.add_argument("--rho", type=float, default=0.95)
    p_app.add_argument("--T-dynamics", dest="T_dynamics", type=int, default=20000)
    p_app.add_argument("--seed", type=int, default=0)
    p_app.add_argument("--run-name", type=str, default="approachability_suite")
    p_app.set_defaults(func=cmd_approachability_suite)

    p_sr = sub.add_parser(
        "swap_regret",
        help="Swap regret vs external regret comparison (§9).",
    )
    p_sr.add_argument("--d", type=int, default=16)
    p_sr.add_argument("--k", type=int, default=8)
    p_sr.add_argument("--alpha", type=float, default=0.8)
    p_sr.add_argument("--n", type=int, default=50000)
    p_sr.add_argument("--L-ar", dest="L_ar", type=int, default=4)
    p_sr.add_argument("--rho", type=float, default=0.95)
    p_sr.add_argument("--seed", type=int, default=0)
    p_sr.add_argument("--run-name", type=str, default="swap_regret")
    p_sr.set_defaults(func=cmd_swap_regret)

    p_tc = sub.add_parser(
        "turtel_compare",
        help="Compare predictions against Turtel et al. (2025) RLVR paper baseline.",
    )
    p_tc.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_tc.add_argument("--pred-col", type=str, default="pred_prob")
    p_tc.add_argument("--y-col", dest="y_col", type=str, default="y")
    p_tc.add_argument("--market-prob-col", dest="market_prob_col", type=str, default="market_prob")
    p_tc.add_argument("--group-cols", type=str, default=None)
    p_tc.add_argument("--max-rows", type=int, default=None)
    p_tc.add_argument("--turtel-brier", type=float, default=None, help="Turtel reported Brier (if known)")
    p_tc.add_argument("--turtel-ece", type=float, default=None, help="Turtel reported ECE (if known)")
    p_tc.add_argument("--turtel-roi", type=float, default=0.10, help="Turtel reported ROI (default: 10%)")
    p_tc.add_argument("--run-name", type=str, default="turtel_compare")
    p_tc.set_defaults(func=cmd_turtel_compare)

    p_hybrid = sub.add_parser(
        "pm_hybrid_train",
        help="Train AR+Diffusion hybrid: AR for reasoning, diffusion for calibration/refinement.",
    )
    p_hybrid.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    p_hybrid.add_argument("--run-name", type=str, default="pm_hybrid_train")
    p_hybrid.add_argument("--max-rows", type=int, default=2000)
    p_hybrid.add_argument("--text-cols", type=str, default="question,description")
    p_hybrid.add_argument("--y-col", dest="y_col", type=str, default="y")
    p_hybrid.add_argument("--market-prob-col", dest="market_prob_col", type=str, default="market_prob")
    p_hybrid.add_argument("--train-frac", type=float, default=0.8)
    p_hybrid.add_argument("--seed", type=int, default=0)
    # AR settings
    p_hybrid.add_argument("--ar-model", type=str, default="Qwen/Qwen3-14B")
    p_hybrid.add_argument("--ar-K", dest="ar_K", type=int, default=5)
    p_hybrid.add_argument("--ar-max-new-tokens", type=int, default=256)
    p_hybrid.add_argument("--ar-temperature", type=float, default=0.7)
    p_hybrid.add_argument("--ar-no-cot", action="store_true")
    p_hybrid.add_argument("--ar-device", type=str, default="cuda")
    p_hybrid.add_argument("--ar-device-map", type=str, default="auto")
    # Embedding for diffusion conditioning
    p_hybrid.add_argument("--embed-model", type=str, default="Qwen/Qwen3-14B")
    p_hybrid.add_argument("--embed-device", type=str, default="cuda")
    p_hybrid.add_argument("--embed-dtype", type=str, default="bfloat16")
    p_hybrid.add_argument("--embed-device-map", type=str, default="auto")
    p_hybrid.add_argument("--embed-batch-size", type=int, default=4)
    p_hybrid.add_argument("--embed-max-length", type=int, default=512)
    # Diffusion refinement settings
    p_hybrid.add_argument("--diff-hidden-dim", type=int, default=256)
    p_hybrid.add_argument("--diff-depth", type=int, default=4)
    p_hybrid.add_argument("--diff-T", type=int, default=50)
    p_hybrid.add_argument("--diff-train-steps", type=int, default=2000)
    p_hybrid.add_argument("--diff-batch-size", type=int, default=256)
    p_hybrid.add_argument("--diff-lr", type=float, default=1e-4)
    p_hybrid.add_argument("--diff-samples", type=int, default=16)
    # Eval
    p_hybrid.add_argument("--bins", type=int, default=20)
    p_hybrid.add_argument("--transaction-cost", type=float, default=0.0)
    p_hybrid.add_argument("--B", type=float, default=1.0)
    p_hybrid.set_defaults(func=cmd_pm_hybrid_train)

    p_synth_market = sub.add_parser(
        "synth_market",
        help="Synthetic prediction market benchmark: AR vs Diffusion on correlated markets.",
    )
    p_synth_market.add_argument("--d", type=int, default=32, help="Context dimension")
    p_synth_market.add_argument("--m", type=int, default=8, help="Number of markets")
    p_synth_market.add_argument("--n-train", type=int, default=10000)
    p_synth_market.add_argument("--n-test", type=int, default=2000)
    p_synth_market.add_argument(
        "--structure", 
        type=str, 
        default="factor",
        choices=["independent", "factor", "chain", "frechet", "hierarchical"],
        help="Market correlation structure"
    )
    p_synth_market.add_argument("--n-factors", type=int, default=3, help="Number of factors (for factor model)")
    p_synth_market.add_argument("--factor-strength", type=float, default=0.7, help="Factor strength")
    p_synth_market.add_argument("--noise", type=float, default=0.3, help="Noise level")
    p_synth_market.add_argument("--seed", type=int, default=0)
    p_synth_market.add_argument("--run-name", type=str, default="synth_market")
    p_synth_market.set_defaults(func=cmd_synth_market)

    p_synth_headlines = sub.add_parser(
        "synth_headlines",
        help="Synthetic headlines benchmark: Type I/II error analysis for structure learning.",
    )
    p_synth_headlines.add_argument("--n-samples", type=int, default=2000)
    p_synth_headlines.add_argument("--n-test", type=int, default=500)
    p_synth_headlines.add_argument(
        "--structure",
        type=str,
        default="factor",
        choices=["factor", "independent", "adversarial"],
        help="factor=learnable structure, independent=no structure (Type I test), adversarial=contradicts surface"
    )
    p_synth_headlines.add_argument("--n-factors", type=int, default=3)
    p_synth_headlines.add_argument("--noise", type=float, default=0.2)
    p_synth_headlines.add_argument("--seed", type=int, default=0)
    p_synth_headlines.add_argument("--run-name", type=str, default="synth_headlines")
    p_synth_headlines.set_defaults(func=cmd_synth_headlines)

    # ====== TURTEL-COMPATIBLE HEADLINES ======
    
    p_turtel_headlines = sub.add_parser(
        "pm_turtel_headlines",
        help="Enrich Polymarket data with Turtel-style headlines (temporal controls, no leakage).",
    )
    p_turtel_headlines.add_argument("--input", type=str, required=True, help="Input parquet file")
    p_turtel_headlines.add_argument("--out", type=str, required=True, help="Output parquet file")
    p_turtel_headlines.add_argument("--question-col", type=str, default="question", help="Question column")
    p_turtel_headlines.add_argument("--news-source", type=str, default="gdelt", choices=["gdelt", "exa"],
                                    help="News source: gdelt (free) or exa (paid, higher quality)")
    p_turtel_headlines.add_argument("--sample-prediction-date", action="store_true", default=True,
                                    help="Sample prediction date between open/close (Turtel-style)")
    p_turtel_headlines.add_argument("--window-days", type=int, default=7, help="Days of news before prediction")
    p_turtel_headlines.add_argument("--max-articles", type=int, default=10, help="Max headlines per question")
    p_turtel_headlines.add_argument("--verify-no-leakage", action="store_true",
                                    help="Use LLM to check for temporal leakage (expensive)")
    p_turtel_headlines.add_argument("--leakage-model", type=str, default="gpt-4o-mini")
    p_turtel_headlines.add_argument("--open-date-col", type=str, default="createdAt")
    p_turtel_headlines.add_argument("--close-date-col", type=str, default="endDate")
    p_turtel_headlines.add_argument("--resolution-date-col", type=str, default="resolutionTime")
    p_turtel_headlines.add_argument("--cache-dir", type=str, default=None, help="Cache directory for headlines")
    p_turtel_headlines.add_argument("--fuzzy-cache", action="store_true",
                                    help="Match cache by question only (reuse cached headlines across datasets)")
    p_turtel_headlines.add_argument("--max-rows", type=int, default=None, help="Max rows to process")
    p_turtel_headlines.add_argument("--seed", type=int, default=0)
    p_turtel_headlines.set_defaults(func=cmd_pm_turtel_headlines)

    # ====== GRPO TRAINING ======
    
    p_grpo = sub.add_parser(
        "grpo_train",
        help="GRPO/ReMax/RLCR training for AR forecasting models (Turtel et al. 2025, Damani et al. 2025).",
    )
    p_grpo.add_argument("--data-path", type=str, required=True, help="Path to training data (CSV/Parquet)")
    p_grpo.add_argument("--out-dir", type=str, default="runs/grpo", help="Output directory for checkpoints")
    p_grpo.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="Model name")
    
    # Algorithm choice (Turtel et al. 2025)
    p_grpo.add_argument(
        "--algorithm",
        type=str,
        default="remax",
        choices=["grpo", "dr_grpo", "remax"],
        help="GRPO variant: grpo=standard, dr_grpo=no std norm (Turtel), remax=learned baseline (best)"
    )
    
    # Reward mode
    p_grpo.add_argument(
        "--reward-mode",
        type=str,
        default="rlcr",
        choices=["turtel_brier", "hybrid", "kelly", "blackwell_aware", "rlcr"],
        help="Reward: turtel_brier=pure Brier, rlcr=correctness+Brier (Damani), kelly=log wealth"
    )
    
    # RLCR params (Damani et al. 2025)
    p_grpo.add_argument("--rlcr-alpha", type=float, default=1.0, help="RLCR: weight on correctness")
    p_grpo.add_argument("--rlcr-beta", type=float, default=1.0, help="RLCR: weight on Brier score")
    p_grpo.add_argument("--rlcr-gamma", type=float, default=0.1, help="RLCR: weight on group calibration")
    p_grpo.add_argument("--rlcr-use-groups", action="store_true", help="Use group-conditional calibration")
    
    # Training params
    p_grpo.add_argument("--steps", type=int, default=1000, help="Training steps")
    p_grpo.add_argument("--batch-size", type=int, default=4, help="Batch size (inputs per step)")
    p_grpo.add_argument("--K", type=int, default=4, help="Samples per input (Turtel uses 4)")
    p_grpo.add_argument("--lr", type=float, default=2e-6, help="Learning rate (ReMax: 2e-6)")
    p_grpo.add_argument("--kl-coef", type=float, default=0.005, help="KL coefficient")
    
    # LoRA
    p_grpo.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p_grpo.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p_grpo.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune)")
    
    # Guard-rails (Turtel et al.)
    p_grpo.add_argument("--gibberish-filter", action="store_true", default=True, help="Filter gibberish outputs")
    p_grpo.add_argument("--max-input-chars", type=int, default=16000, help="Max input chars (Turtel: 16k)")
    p_grpo.add_argument("--early-stop", type=int, default=20, help="Early stop patience (0=disable)")
    
    # Misc
    p_grpo.add_argument("--seed", type=int, default=0)
    p_grpo.add_argument("--run-name", type=str, default="grpo_train")
    p_grpo.set_defaults(func=cmd_grpo_train)

    # ====== MEAN-REVERSION STAT ARB ======
    
    p_analyze_calib = sub.add_parser(
        "analyze_group_calibration",
        help="Analyze group-wise calibration regimes (mean-reversion vs momentum).",
    )
    p_analyze_calib.add_argument("--dataset-path", type=str, required=True, help="Path to dataset parquet")
    p_analyze_calib.add_argument("--group-col", type=str, default="category", help="Column for grouping")
    p_analyze_calib.add_argument("--price-col", type=str, default="market_prob", help="Market price column")
    p_analyze_calib.add_argument("--outcome-col", type=str, default="y", help="Outcome column")
    p_analyze_calib.add_argument("--window", type=int, default=50, help="Rolling window for calibration")
    p_analyze_calib.add_argument("--mean-revert-threshold", type=float, default=0.05, help="Threshold for mean-revert regime")
    p_analyze_calib.add_argument("--momentum-threshold", type=float, default=0.15, help="Threshold for momentum regime")
    p_analyze_calib.add_argument("--run-name", type=str, default="group_calibration_analysis")
    p_analyze_calib.set_defaults(func=cmd_analyze_group_calibration)
    
    p_mr_backtest = sub.add_parser(
        "mean_reversion_backtest",
        help="Run walk-forward backtest for group mean-reversion strategy.",
    )
    p_mr_backtest.add_argument("--dataset-path", type=str, required=True, help="Path to dataset parquet")
    p_mr_backtest.add_argument("--model-col", type=str, default="pred_prob", help="Model forecast column")
    p_mr_backtest.add_argument("--market-col", type=str, default="market_prob", help="Market price column")
    p_mr_backtest.add_argument("--group-col", type=str, default="category", help="Column for grouping")
    p_mr_backtest.add_argument("--outcome-col", type=str, default="y", help="Outcome column")
    p_mr_backtest.add_argument("--market-id-col", type=str, default="id", help="Market ID column")
    p_mr_backtest.add_argument("--time-col", type=str, default=None, help="Time column for ordering")
    p_mr_backtest.add_argument(
        "--position-method",
        type=str,
        default="all",
        help="Position method: calibration, dollar_neutral, frechet, or 'all'",
    )
    p_mr_backtest.add_argument("--kelly-fraction", type=float, default=0.25, help="Kelly fraction")
    p_mr_backtest.add_argument("--max-position", type=float, default=0.10, help="Max position size")
    p_mr_backtest.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge to trade")
    p_mr_backtest.add_argument("--transaction-cost", type=float, default=0.01, help="Transaction cost")
    p_mr_backtest.add_argument("--train-frac", type=float, default=0.5, help="Training fraction for walk-forward")
    p_mr_backtest.add_argument("--n-folds", type=int, default=5, help="Number of walk-forward folds")
    p_mr_backtest.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap samples for CI")
    p_mr_backtest.add_argument("--run-name", type=str, default="mean_reversion_backtest")
    p_mr_backtest.add_argument("--verbose", action="store_true", help="Print progress")
    p_mr_backtest.set_defaults(func=cmd_mean_reversion_backtest)
    
    p_compare_arb = sub.add_parser(
        "compare_arb_strategies",
        help="Compare multiple stat arb strategies on the same dataset.",
    )
    p_compare_arb.add_argument("--dataset-path", type=str, required=True, help="Path to dataset parquet")
    p_compare_arb.add_argument("--model-col", type=str, default="pred_prob", help="Model forecast column")
    p_compare_arb.add_argument("--market-col", type=str, default="market_prob", help="Market price column")
    p_compare_arb.add_argument("--group-col", type=str, default="category", help="Column for grouping")
    p_compare_arb.add_argument("--outcome-col", type=str, default="y", help="Outcome column")
    p_compare_arb.add_argument(
        "--strategies",
        type=str,
        default="calibration,dollar_neutral,frechet",
        help="Comma-separated list of strategies to compare",
    )
    p_compare_arb.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap samples for CI")
    p_compare_arb.add_argument("--run-name", type=str, default="compare_arb_strategies")
    p_compare_arb.set_defaults(func=cmd_compare_arb_strategies)

    # ====== END NEW BENCHMARKS ======

    p_ltx = sub.add_parser("latex", help="Export a run's metrics.json to a LaTeX table.")
    p_ltx.add_argument("--run-dir", type=str, required=True)
    p_ltx.add_argument("--out", type=str, default=None)
    p_ltx.set_defaults(func=cmd_latex)

    # ====== BACKTESTING ======
    # Import and add backtest subparser
    try:
        from backtest.cli import add_backtest_parser
        add_backtest_parser(sub)
    except ImportError:
        # Backtest module not available; skip
        pass

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


 