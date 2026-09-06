"""Plots for the N scaling sweep.

    py tools/plot.py

Reads results/n_scaling.csv and writes the figures into images/.
Each point is the mean of the 3 runs; the error bars span min to max.
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

MODES = ["naive", "scattered", "coherent"]
STYLE = {
    "naive":     dict(color="#c0392b", marker="o"),
    "scattered": dict(color="#2471a3", marker="s"),
    "coherent":  dict(color="#1e8449", marker="^"),
}

df = pd.read_csv(ROOT / "results" / "n_scaling.csv")
agg = (df.groupby(["mode", "n", "visualize", "stage"])["mean_ms"]
         .agg(["mean", "min", "max"])
         .reset_index())


def series(mode, vis, stage):
    s = agg[(agg["mode"] == mode) & (agg["visualize"] == vis) &
            (agg["stage"] == stage)].sort_values("n")
    return s


def errorbars(s, value_of):
    """Asymmetric bars from the min/max of the run means."""
    mid = value_of(s["mean"])
    lo = value_of(s["max"])   # slower run, so lower FPS
    hi = value_of(s["min"])
    return [abs(mid - lo), abs(hi - mid)]


# ---------------------------------------------------- framerate vs N ------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, vis, title in zip(axes, [0, 1],
                          ["visualization off", "visualization on"]):
    for mode in MODES:
        s = series(mode, vis, "frame_wall")
        if s.empty:
            continue
        fps = 1000.0 / s["mean"]
        ax.errorbar(s["n"], fps, yerr=errorbars(s, lambda c: 1000.0 / c),
                    label=mode, capsize=3, linewidth=1.5, markersize=5,
                    **STYLE[mode])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of boids")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

axes[0].set_ylabel("application FPS")
axes[0].legend()
fig.suptitle("Framerate vs. boid count (RTX 3090 Ti, block size 128)")
fig.tight_layout()
fig.savefig(IMAGES / "n_scaling_fps.png", dpi=150)

# ------------------------------------------------- GPU step time vs N -----

fig, ax = plt.subplots(figsize=(6.5, 5))

for mode in MODES:
    s = series(mode, 0, "total")
    ax.errorbar(s["n"], s["mean"], yerr=errorbars(s, lambda c: c),
                label=mode, capsize=3, linewidth=1.5, markersize=5,
                **STYLE[mode])

# Guides laid over the two halves of the naive curve. Below saturation the
# extra threads fill idle SMs, so time grows with N; above it the GPU is full
# and time grows with the total work, which is N^2.
def guide(anchor_n, anchor_t, power, lo, hi, style, label):
    ts = [anchor_t * (n / anchor_n) ** power for n in (lo, hi)]
    ax.plot([lo, hi], ts, style, color="grey", linewidth=1.2, alpha=0.9)
    ax.annotate(label, (hi * 1.15, ts[1]), color="grey", fontsize=9,
                va="center")

guide(5000, 0.71, 1, 1000, 50000, "--", "O(N)")
guide(100000, 32.6, 2, 40000, 200000, ":", "O(N²)")

ax.axvline(7e4, color="grey", linewidth=0.8, alpha=0.4)
ax.annotate("GPU saturates\n(~10⁵ resident threads)", (0.60, 0.03),
            xycoords="axes fraction", color="grey", fontsize=8)
ax.set_ylim(0.06, 500)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of boids")
ax.set_ylabel("GPU simulation step (ms)")
ax.set_title("GPU step time vs. boid count")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(IMAGES / "n_scaling_gpu.png", dpi=150)

print("wrote images/n_scaling_fps.png and images/n_scaling_gpu.png")

# --------------------------------------------------- framerate vs block ---

block_csv = ROOT / "results" / "block_size.csv"
if not block_csv.exists():
    raise SystemExit("no results/block_size.csv yet, skipping the block plot")

bdf = pd.read_csv(block_csv)
bagg = (bdf[bdf["stage"] == "frame_wall"]
        .groupby(["mode", "n", "block_size"])["mean_ms"]
        .agg(["mean", "min", "max"])
        .reset_index())

ns = sorted(bagg["n"].unique())
fig, axes = plt.subplots(1, len(ns), figsize=(5.5 * len(ns), 4.5), sharey=True)

# Plotted relative to block size 128 rather than as absolute FPS. The three
# implementations are orders of magnitude apart, so on a shared log axis a 20%
# effect would look like a flat line whether or not it was real.
for ax, n in zip(axes, ns):
    for mode in MODES:
        s = bagg[(bagg["mode"] == mode) & (bagg["n"] == n)].sort_values("block_size")
        if s.empty:
            continue
        base = s[s["block_size"] == 128]["mean"].iloc[0]
        ax.errorbar(s["block_size"], base / s["mean"],
                    yerr=[base / s["mean"] - base / s["max"],
                          base / s["min"] - base / s["mean"]],
                    label=f"{mode} ({1000 / base:.0f} FPS at 128)",
                    capsize=3, linewidth=1.5, markersize=5, **STYLE[mode])
    ax.axhline(1.0, color="grey", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("block size (threads)")
    ax.set_title(f"N = {n}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

axes[0].set_ylabel("FPS relative to block size 128")
fig.suptitle("Framerate vs. block size (visualization off)")
fig.tight_layout()
fig.savefig(IMAGES / "block_size_fps.png", dpi=150)
print("wrote images/block_size_fps.png")

# ----------------------------------------------------- cell width vs N ----

width_csv = ROOT / "results" / "cell_width.csv"
if not width_csv.exists():
    raise SystemExit("no results/cell_width.csv yet, skipping the width plot")

wdf = pd.read_csv(width_csv)
wagg = (wdf[wdf["stage"] == "total"]
        .groupby(["mode", "n", "cell_width_scale"])["mean_ms"]
        .agg(["mean", "min", "max"])
        .reset_index())

fig, ax = plt.subplots(figsize=(7, 4.8))

for mode in ["scattered", "coherent"]:
    wide = wagg[(wagg["mode"] == mode) & (wagg["cell_width_scale"] == 2.0)].sort_values("n")
    narrow = wagg[(wagg["mode"] == mode) & (wagg["cell_width_scale"] == 1.0)].sort_values("n")
    ratio = narrow["mean"].values / wide["mean"].values
    # Not a standard deviation. These are the narrowest and widest ratios the
    # three runs on each side allow, so they combine the extremes of the
    # numerator and the denominator and are wider than either side's own
    # spread. Where the interval covers 1.0 the two cell widths are not
    # distinguishable with 3 runs.
    lo = narrow["min"].values / wide["max"].values
    hi = narrow["max"].values / wide["min"].values
    ax.errorbar(wide["n"], ratio, yerr=[ratio - lo, hi - ratio],
                label=mode, capsize=3, linewidth=1.5, markersize=5,
                **STYLE[mode])

ax.axhline(1.0, color="grey", linewidth=1)
ax.annotate("no difference", (0.02, 1.0), xycoords=("axes fraction", "data"),
            color="grey", fontsize=8, va="bottom")
ax.axhline(27 / 64, color="grey", linewidth=1, linestyle="--")
ax.annotate("27R³ / 64R³, the ratio of searched volume",
            (0.02, 27 / 64), xycoords=("axes fraction", "data"),
            color="grey", fontsize=8, va="bottom")

ax.set_xscale("log")
ax.set_xlabel("number of boids")
ax.set_ylabel("GPU step time, 27 cells / 8 cells")
ax.set_title("Cell width R (27 cells) against 2R (8 cells)")
ax.set_ylim(0.35, 1.55)
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(IMAGES / "cell_width.png", dpi=150)
print("wrote images/cell_width.png")
