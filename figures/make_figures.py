#!/usr/bin/env python3
"""R1 (second analysis, R1b) figures (PNG 300 dpi + TIFF 1,200 dpi(LZW) + EPS). Okabe–Ito colours, distinct markers, one axis per panel.
  Figure 1  flow of goals to analysed events
  Figure 2  reference vs primary (within-unit, pooled, out-of-sample controls) by window
  Figure 3  forest: competition-specific and stratum estimates (reference vs primary), anchor window
  Figure 4  time course: five-minute window moved in one-minute steps after the goal
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures/output"; FIG.mkdir(exist_ok=True, parents=True)
R = json.loads((ROOT / "docs/ijpas_r2.json").read_text()); TC = json.loads((ROOT / "docs/ijpas_r2_timecourse.json").read_text())
G = json.loads((ROOT / "docs/ijpas_r1_goals.json").read_text())
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#7f7f7f"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})
WINS = ["post5", "post10", "post15"]; WLAB = ["0–5", "5–10", "10–15"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tif", dpi=1200, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIG / f"{name}.eps", bbox_inches="tight")
    plt.close(fig); print("✓", name)


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 6.4)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-0.01, 1)
    tot = G["totals"]["goals"]
    def box(x, y, w, h, text, fc="#f2f2f2", fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.01", fc=fc, ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3, wrap=True)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))
    box(0.12, 0.925, 0.76, 0.06, f"{tot:,} goals in 2,404 matches, penalty shoot-outs excluded ({G['totals']['own_goals']} own goals, {G['totals']['goals_ET']} in extra time)\none scoring and one conceding event per goal", fs=6.6)
    arrow(0.5, 0.925, 0.5, 0.895); arrow(0.5, 0.895, 0.25, 0.875); arrow(0.5, 0.895, 0.75, 0.875)
    steps = [("no_window_data", "No event data in the window\n(goal too late in the period)"), ("other_goal_before_window_end", "Another goal (either team)\nbefore the end of the window"),
             ("dismissal_in_window", "Dismissal in the window"), ("cell_short_due_to_dismissal_control_exclusion", "Same-state control cell < 3 windows\nonly because dismissal windows\nwere removed from the controls"),
             ("cell_short_or_z_undefined", "Same-state control cell < 3 windows\n(state superseded or ended too soon)\nor z undefined")]
    for side, x0, fc, name in (("score", 0.02, "#e8f1fa", "Scoring team"), ("concede", 0.52, "#fdf0e6", "Conceding team")):
        f = R[f"{side}|flow"]; xc = x0 + 0.23
        box(x0, 0.82, 0.46, 0.05, f"{name}: {f['goals_total']:,} events", fc, fs=7.5)
        y = 0.80
        for key, lab in steps:
            vals = "  ·  ".join(f"{wl}: −{f[f'state|{w}'][key]:,}" for w, wl in zip(WINS, WLAB))
            box(x0, y - 0.105, 0.46, 0.095, lab + "\n" + vals, fc); arrow(xc, y, xc, y - 0.01); y -= 0.115
        vals = "  ·  ".join(f"{wl}: {f[f'state|{w}']['analysed']:,}" for w, wl in zip(WINS, WLAB))
        box(x0, y - 0.08, 0.46, 0.07, "Analysed (primary analysis)\n" + vals, "#ffffff", fs=7.5); arrow(xc, y, xc, y - 0.01)
        inc = f["included_vs_excluded_anchor"]["included"]; exc = f["included_vs_excluded_anchor"]["excluded_all"]
        box(x0, 0.01, 0.46, 0.115, f"5 to 10 min window: analysed vs other goals\n(n = {inc['n']:,} vs {exc['n']:,})\nminute of the goal {inc['minute_mean']} vs {exc['minute_mean']}\ngoals in the match {inc['goals_in_match_mean']} vs {exc['goals_in_match_mean']}\n"
            f"knockout {inc['share_knockout_pct']}% vs {exc['share_knockout_pct']}%; women {inc['share_women_pct']}% vs {exc['share_women_pct']}%", "#ffffff", fs=6.6)
    save(fig, "Figure_1")


def est(key, path):
    v = R[key]
    for p in path.split("."): v = v[p]
    return v


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    # R2: unit-cluster bootstrap intervals (cluster_unit / cluster_unit_pooled, every window), as the caption states
    series = [("Reference (pre-goal state), within-unit SD", "pre", "cluster_unit", OI["orange"], "o", True),
              ("Primary (same state), within-unit SD", "state", "cluster_unit", OI["blue"], "o", True),
              ("Primary, pooled SD", "state", "cluster_unit_pooled", OI["blue"], "s", False)]
    for ax, side, title in zip(axes, ("score", "concede"), ("Scoring team", "Conceding team")):
        x = np.arange(3)
        for k, (lab, key, den, col, mk, filled) in enumerate(series):
            d = [est(f"{side}|{key}|{w}", f"{den}.d") for w in WINS]; lo = [est(f"{side}|{key}|{w}", f"{den}.ci95")[0] for w in WINS]; hi = [est(f"{side}|{key}|{w}", f"{den}.ci95")[1] for w in WINS]
            off = (k - 1.5) * 0.17
            ax.errorbar(x + off, d, yerr=[np.array(d) - lo, np.array(hi) - d], fmt=mk, color=col, mfc=col if filled else "white", mec=col, ms=5.5, capsize=2, lw=1, label=lab)
        # LOO controls (anchor only)
        loo = R[f"{side}|standard"]["loo_controls"]["unit"]
        ax.errorbar([1 + 0.255], [loo["d"]], yerr=[[loo["d"] - loo["ci95"][0]], [loo["ci95"][1] - loo["d"]]], fmt="^", color=OI["green"], mfc="white", ms=6, capsize=2, lw=1, label="Primary, within-unit SD, controls scored out of sample (leave-one-out; anchor; window-resampled CI)")
        ax.axhspan(-0.15, 0.15, color="#000000", alpha=0.06, lw=0); ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([f"{w} min" for w in WLAB]); ax.set_title(title, fontsize=10); ax.set_xlabel("Window after the goal")
        ax.set_ylim(-0.75, 0.4); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    axes[0].set_ylabel("Cohen's d of net z (post-goal − control)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); save(fig, "Figure_2")


def fig3():
    comps = ["La Liga", "Premier League", "Serie A", "Ligue 1", "FIFA World Cup", "UEFA Euro", "Copa America", "African Cup of Nations", "FA Women's Super League", "Women's World Cup"]
    strata = [("Men, league", "men_league"), ("Men, tournament", "men_tournament"), ("Women, league", "women_league"), ("Women, tournament", "women_tournament"), ("All", None)]
    rows = [(c, c) for c in comps] + [None] + strata
    rows = [(c, c) for c in comps] + [None] + [("League (regular season)", "stage_league"), ("Group stage", "stage_group"), ("Knockout", "stage_knockout")] + [None] + strata
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 6.0), sharey=False)
    xmin, xmax = 0, 0
    for ax, side, title in zip(axes, ("score", "concede"), ("Scoring team", "Conceding team")):
        ys = []; labels = []
        y = 0
        for row in rows:
            y -= 1
            if row is None: ys.append(None); labels.append(""); continue
            lab, key = row
            if key is None:
                ref = R[f"{side}|pre|post10"]["unit"]; prim = R[f"{side}|state|post10"]["unit"]
            else:
                ref = R[f"{side}|strata"][key]["pre"]["unit"]; prim = R[f"{side}|strata"][key]["state"]["unit"]
            for est_, col, mk, off in ((ref, OI["orange"], "o", 0.18), (prim, OI["blue"], "o", -0.18)):
                if est_["d"] != est_["d"]: continue
                ax.errorbar([est_["d"]], [y + off], xerr=[[est_["d"] - est_["ci95"][0]], [est_["ci95"][1] - est_["d"]]], fmt=mk, color=col, ms=4.5, capsize=1.5, lw=0.9)
                xmin = min(xmin, est_["ci95"][0]); xmax = max(xmax, est_["ci95"][1])
            ys.append(y); labels.append(f"{lab} (n = {prim['n_event']:,})")
        ax.axvline(0, color="black", lw=0.6); ax.axvspan(-0.15, 0.15, color="#000000", alpha=0.06, lw=0)
        ax.set_yticks([yy for yy in ys if yy is not None]); ax.set_yticklabels([l for l, yy in zip(labels, ys) if yy is not None], fontsize=7.5)
        ax.set_title(title, fontsize=10); ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    for ax in axes:
        ax.set_xlim(np.floor(xmin * 4) / 4 - 0.05, np.ceil(xmax * 4) / 4 + 0.05)
    axes[0].errorbar([], [], fmt="o", color=OI["orange"], label="Reference (pre-goal state)"); axes[0].errorbar([], [], fmt="o", color=OI["blue"], label="Primary (same state)")
    fig.supxlabel("Cohen's d of net z (within-unit SD), 5–10 min window", fontsize=9, y=0.06)
    fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "Figure_3")


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    starts = list(range(0, 11))
    for ax, side, title in zip(axes, ("score", "concede"), ("Scoring team", "Conceding team")):
        for den, col, mk, lab, filled in (("unit", OI["blue"], "o", "Within-unit SD", True), ("pooled", OI["blue"], "s", "Pooled SD", False)):
            d = [TC[side][f"s{s}"][den]["d"] for s in starts]; lo = [TC[side][f"s{s}"][den]["ci95"][0] for s in starts]; hi = [TC[side][f"s{s}"][den]["ci95"][1] for s in starts]
            ax.plot(starts, d, color=col, lw=1, ls="-" if filled else "--"); ax.fill_between(starts, lo, hi, color=col, alpha=0.12 if filled else 0.08, lw=0)
            ax.plot(starts, d, mk, color=col, mfc=col if filled else "white", ms=4.5, label=lab)
        dc = [TC[side][f"s{s}"]["common"]["unit"]["d"] for s in starts]
        ax.plot(starts, dc, "^", color=OI["green"], mfc="white", ms=4.5, lw=0.8, ls=":", label="Within-unit SD, goals analysable in every window")
        ax.axhspan(-0.15, 0.15, color="#000000", alpha=0.06, lw=0); ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(starts); ax.set_title(title, fontsize=10); ax.set_ylim(-0.45, 0.4); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    axes[0].set_ylabel("Cohen's d of net z (post-goal − control)"); axes[1].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.supxlabel("Start of the five-minute window (min after the goal)", fontsize=9)
    fig.tight_layout(); save(fig, "Figure_4")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which: globals()[f"fig{w}"]()
