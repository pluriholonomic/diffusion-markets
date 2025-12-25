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
            payload = torch.load(str(args.model_path), map_location="cpu")
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

    p_ltx = sub.add_parser("latex", help="Export a run's metrics.json to a LaTeX table.")
    p_ltx.add_argument("--run-dir", type=str, required=True)
    p_ltx.add_argument("--out", type=str, default=None)
    p_ltx.set_defaults(func=cmd_latex)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


 