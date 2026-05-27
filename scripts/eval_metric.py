"""Evaluate any cam-derived metric against CDIP 201 Hs.

Loads motion.parquet (or any time-indexed df), computes a few rolling-window
statistics, aligns to buoy, reports correlations + a quick fit.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from camwaveheight import cdip, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--col", default="motion")
    ap.add_argument("--buoy", default="201")
    ap.add_argument("--start", default="2026-05-26")
    ap.add_argument("--end", default="2026-05-27")
    ap.add_argument("--window-sec", type=int, default=1020)
    ap.add_argument("--step-sec", type=int, default=300)
    ap.add_argument("--out", default="reports/eval_metric.png")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"loaded: {len(df):,} rows, {df.index.min()}..{df.index.max()}")
    s = df[args.col].copy().astype(float)
    print(f"{args.col} non-null: {s.notna().sum():,} ({s.notna().mean():.0%})")

    fu = stats._resample_uniform(s, fs_hz=5.0, max_gap_sec=10)

    rows = []
    t0, t1 = fu.index.min(), fu.index.max() - pd.Timedelta(seconds=args.window_sec)
    cur = t0
    while cur <= t1:
        win = fu.loc[cur : cur + pd.Timedelta(seconds=args.window_sec)]
        if win.notna().mean() > 0.7 and win.size > 64:
            wmean = win.mean()
            wstd = win.std()
            # Detrended std (high-frequency variability only)
            detr = win - win.rolling(int(5 * 60), center=True, min_periods=1).mean()
            rows.append({
                "t": cur + pd.Timedelta(seconds=args.window_sec / 2),
                "mean": wmean,
                "std": wstd,
                "cv": wstd / wmean if wmean else np.nan,
                "detrend_std": detr.std(),
                "p90": win.quantile(0.9),
                "p10": win.quantile(0.1),
                "iqr": win.quantile(0.75) - win.quantile(0.25),
            })
        cur += pd.Timedelta(seconds=args.step_sec)
    rd = pd.DataFrame(rows).set_index("t")
    rd.index = pd.to_datetime(rd.index, utc=True).as_unit("ns")
    print(f"rolling windows: {len(rd)}")

    buoy = cdip.fetch_cdip_params(args.buoy, args.start, args.end, cache_dir="data/cdip")
    buoy.index = pd.to_datetime(buoy.index, utc=True).as_unit("ns")

    paired = pd.merge_asof(
        buoy[["waveHs", "waveTp"]].sort_index(),
        rd.sort_index(),
        left_index=True, right_index=True,
        tolerance=pd.Timedelta("15min"),
        direction="nearest",
    ).dropna(subset=["waveHs", "mean"])

    print(f"\npaired samples: {len(paired)}")
    print("correlations with buoy waveHs:")
    for c in ["mean", "std", "cv", "detrend_std", "p90", "p10", "iqr"]:
        if c in paired.columns:
            r = paired["waveHs"].corr(paired[c])
            print(f"  {c}: r={r:.3f}")

    # Best linear fit (use the strongest-correlated metric)
    cands = {c: abs(paired["waveHs"].corr(paired[c])) for c in ["mean","std","cv","detrend_std","p90","iqr"] if c in paired.columns}
    best = max(cands, key=cands.get)
    print(f"\nbest metric: {best}  |r|={cands[best]:.3f}")

    x = paired[best].to_numpy()
    y = paired["waveHs"].to_numpy()
    n = len(x)
    n_train = int(n * 0.7)
    A = np.vstack([x[:n_train], np.ones(n_train)]).T
    (a, b), *_ = np.linalg.lstsq(A, y[:n_train], rcond=None)
    pred = a * x + b
    rmse_test = float(np.sqrt(np.mean((pred[n_train:] - y[n_train:]) ** 2)))
    print(f"fit: y = {a:.6f}*x + {b:.4f}")
    print(f"RMSE train: {np.sqrt(np.mean((pred[:n_train]-y[:n_train])**2)):.3f} m")
    print(f"RMSE test:  {rmse_test:.3f} m  (target ≤ 0.25 m)")
    print(f"r² test:    {1 - np.var(pred[n_train:]-y[n_train:])/np.var(y[n_train:]):.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(paired.index, paired["waveHs"], "o-", color="C0", label="CDIP 201")
    ax[0].plot(paired.index, pred, "o-", color="C3", label=f"cam ({best})")
    ax[0].axvline(paired.index[n_train], ls="--", color="k", alpha=0.4, label="train | test")
    ax[0].legend(); ax[0].set_ylabel("Hs (m)")
    ax[0].set_title(f"best metric: {best}; r={cands[best]:.2f}; RMSE_test={rmse_test:.3f}m")
    ax[1].scatter(paired["waveHs"][:n_train], pred[:n_train], label="train")
    ax[1].scatter(paired["waveHs"][n_train:], pred[n_train:], color="C3", label="test")
    lim = [paired["waveHs"].min()*0.9, paired["waveHs"].max()*1.1]
    ax[1].plot(lim, lim, "k--", alpha=0.4); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("CDIP Hs (m)"); ax[1].set_ylabel("cam Hs (m)"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
