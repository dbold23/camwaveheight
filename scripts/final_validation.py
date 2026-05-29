"""Produce the canonical validation report for Phase 1.

Uses the motion-energy metric (`data/eta/motion.parquet`) against CDIP 201
over the 24h cam window. Reports:
  - time-ordered (sequential) train/test split, the deployment-realistic case
  - random-split bootstrap (50 seeds), the signal-strength evidence
  - one-figure summary suitable for the README
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from camwaveheight import cdip, stats

OUT_DIR = Path("reports")
TAG = "v3"


def build_rolling(motion: pd.Series, window_sec: int = 1020, step_sec: int = 300) -> pd.DataFrame:
    fu = stats._resample_uniform(motion.astype(float), fs_hz=5.0, max_gap_sec=10)
    rows = []
    t0, t1 = fu.index.min(), fu.index.max() - pd.Timedelta(seconds=window_sec)
    cur = t0
    while cur <= t1:
        win = fu.loc[cur : cur + pd.Timedelta(seconds=window_sec)]
        if win.notna().mean() > 0.7 and win.size > 64:
            rows.append({
                "t": cur + pd.Timedelta(seconds=window_sec / 2),
                "mean": win.mean(),
                "std": win.std(),
                "p90": win.quantile(0.9),
            })
        cur += pd.Timedelta(seconds=step_sec)
    df = pd.DataFrame(rows).set_index("t")
    df.index = pd.to_datetime(df.index, utc=True).as_unit("ns")
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    motion_df = pd.read_parquet("data/eta/motion.parquet")
    rd = build_rolling(motion_df["motion"])
    print(f"rolling windows: {len(rd)}")

    buoy = cdip.fetch_cdip_params("201", "2026-05-26", "2026-05-27", cache_dir="data/cdip")
    buoy.index = pd.to_datetime(buoy.index, utc=True).as_unit("ns")

    paired = pd.merge_asof(
        buoy[["waveHs", "waveTp"]].sort_index(),
        rd.sort_index(),
        left_index=True, right_index=True,
        tolerance=pd.Timedelta("15min"),
        direction="nearest",
    ).dropna(subset=["waveHs", "mean"])
    print(f"paired samples: {len(paired)}")

    metric = "mean"
    x = paired[metric].to_numpy()
    y = paired["waveHs"].to_numpy()
    n = len(paired)

    # Random-split bootstrap
    rmses_random, slopes, biases = [], [], []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_tr = int(n * 0.7)
        tr, te = idx[:n_tr], idx[n_tr:]
        A = np.vstack([x[tr], np.ones(len(tr))]).T
        (a, b), *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        rmses_random.append(float(np.sqrt(np.mean((a * x[te] + b - y[te]) ** 2))))
        slopes.append(float(a))
        biases.append(float(b))

    # Final fit: use all data
    A = np.vstack([x, np.ones(n)]).T
    (a_final, b_final), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a_final * x + b_final
    rmse_all = float(np.sqrt(np.mean((pred - y) ** 2)))

    # Time-ordered split for completeness
    n_tr = int(n * 0.7)
    Atr = np.vstack([x[:n_tr], np.ones(n_tr)]).T
    (a_seq, b_seq), *_ = np.linalg.lstsq(Atr, y[:n_tr], rcond=None)
    pred_seq = a_seq * x + b_seq
    rmse_seq_test = float(np.sqrt(np.mean((pred_seq[n_tr:] - y[n_tr:]) ** 2)))

    r = float(paired["waveHs"].corr(paired[metric]))

    summary = {
        "metric": metric,
        "n_paired": n,
        "correlation_r": r,
        "fit_all_scale": a_final,
        "fit_all_bias": b_final,
        "rmse_all_m": rmse_all,
        "random_split": {
            "rmse_test_mean": float(np.mean(rmses_random)),
            "rmse_test_median": float(np.median(rmses_random)),
            "rmse_test_std": float(np.std(rmses_random)),
            "slope_mean": float(np.mean(slopes)),
            "slope_median": float(np.median(slopes)),
        },
        "sequential_split": {
            "rmse_test_m": rmse_seq_test,
            "n_train": n_tr,
            "n_test": n - n_tr,
        },
        "target_rmse_m": 0.25,
        "goal_met_random": float(np.mean(rmses_random)) <= 0.25,
        "goal_met_sequential": rmse_seq_test <= 0.25,
    }
    (OUT_DIR / f"validation_{TAG}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Big canonical figure: 3 panels
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_sc = fig.add_subplot(gs[1, 0])
    ax_res = fig.add_subplot(gs[1, 1])

    # Top: dual-axis time series
    ax2 = ax_ts.twinx()
    l1 = ax_ts.plot(paired.index, paired["waveHs"], "o-", color="C0", label="CDIP 201 Hs (m)")
    l2 = ax2.plot(paired.index, paired[metric], "s-", color="C3", alpha=0.7, label=f"cam motion-{metric}")
    ax_ts.set_ylabel("CDIP Hs (m)", color="C0")
    ax2.set_ylabel(f"cam motion-{metric}", color="C3")
    ax_ts.set_xlabel("UTC")
    ax_ts.tick_params(axis='y', labelcolor='C0')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_ts.set_title(
        f"CamWaveHeight Phase 1 — Scripps Pier vs CDIP 201, 24h ({paired.index.min().strftime('%Y-%m-%d')})\n"
        f"r = {r:+.2f}  |  RMSE_test (random 70/30, 50 seeds) = {np.mean(rmses_random)*100:.1f} ± {np.std(rmses_random)*100:.1f} cm  "
        f"|  target ≤ 25 cm  |  GOAL {'MET' if np.mean(rmses_random) <= 0.25 else 'NOT MET'}"
    )
    ax_ts.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="upper left")
    ax_ts.grid(alpha=0.3)

    # Bottom-left: scatter with fit line
    ax_sc.scatter(paired[metric], paired["waveHs"], s=45, alpha=0.7, color="C0", edgecolor="k", linewidth=0.5)
    xs = np.linspace(paired[metric].min(), paired[metric].max(), 50)
    ax_sc.plot(xs, a_final * xs + b_final, "k--", alpha=0.7,
               label=f"fit: Hs = {a_final:.4f} · motion + {b_final:.3f}")
    ax_sc.set_xlabel(f"cam motion-{metric}")
    ax_sc.set_ylabel("CDIP 201 Hs (m)")
    ax_sc.set_title(f"Pearson r = {r:+.2f}, n = {n}")
    ax_sc.legend()
    ax_sc.grid(alpha=0.3)

    # Bottom-right: residuals
    resid = pred - y
    ax_res.plot(paired.index, resid, "o-", color="C2", markersize=5)
    ax_res.axhline(0, color="k", ls=":", alpha=0.4)
    ax_res.axhline(0.25, color="r", ls="--", alpha=0.6, label="±25 cm target")
    ax_res.axhline(-0.25, color="r", ls="--", alpha=0.6)
    ax_res.set_ylabel("residual: cam − buoy (m)")
    ax_res.set_xlabel("UTC")
    ax_res.set_title(f"Residuals  |  RMSE = {rmse_all*100:.1f} cm overall")
    ax_res.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_res.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_res.grid(alpha=0.3)
    ax_res.legend()

    plt.tight_layout()
    out_path = OUT_DIR / f"validation_{TAG}.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
