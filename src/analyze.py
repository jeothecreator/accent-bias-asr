"""
Phase 4: confidence intervals, gap statistics, charts, and the worst-clips list.

    python src/analyze.py

Reads results/results.csv (from score.py) and writes:
    results/wer_by_group_model.csv   pooled WER per group x model with 95% CIs
    results/gap_by_model.csv         best-vs-worst gap and tier ratio per model, with CIs
    results/group_vs_us.csv          each group's WER minus the US group's, with CIs
    results/worst_clips_to_tag.csv   the 50 worst clips for large-v3, to listen to and tag
    results/figures/*.png            the charts
    results/PHASE4_SUMMARY.md        all of the above in one readable page

Method
------
WER is pooled: total word errors / total reference words, so long and short
clips are weighted by their length rather than equally.

Confidence intervals come from a speaker-level (cluster) bootstrap: within each
accent group, resample speakers with replacement, keep all of each chosen
speaker's clips, recompute WER; repeat --n-boot times; the 95% CI is the 2.5th
to 97.5th percentile. Resampling speakers rather than clips matters because two
clips from the same person aren't independent. The same resample is used for
every model in an iteration, so model-vs-model comparisons are paired.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accent_groups as ag  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WHISPER_ORDER = ["tiny", "base", "small", "medium", "large-v3"]
MODEL_ORDER = WHISPER_ORDER + ["turbo", "wav2vec2"]
# Parameter counts (millions), from the Whisper paper / model cards
PARAMS_M = {"tiny": 39, "base": 74, "small": 244, "medium": 769, "large-v3": 1550,
            "turbo": 809, "wav2vec2": 317}

# Kachru's three circles: used for the "tier" statistics and for chart color
CIRCLE = {"us": "inner", "england": "inner", "canada": "inner", "australia_nz": "inner",
          "south_asia": "outer", "southern_africa": "outer", "southeast_asia": "outer",
          "l2_european": "expanding"}
INNER = [g for g, c in CIRCLE.items() if c == "inner"]
SHORT_NAME = {"us": "US", "england": "England", "canada": "Canada",
              "australia_nz": "Australia/NZ", "south_asia": "South Asia",
              "southern_africa": "Southern Africa", "southeast_asia": "SE Asia (PH/MY/SG)",
              "l2_european": "L2 (European L1)"}


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def load(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, keep_default_na=False, na_values=[""])
    r["errors"] = r["substitutions"] + r["deletions"] + r["insertions"]
    missing = set(MODEL_ORDER) - set(r["model"])
    models = [m for m in MODEL_ORDER if m not in missing]
    return r[r["model"].isin(models)], models


def bootstrap(r: pd.DataFrame, groups, models, n_boot: int, seed: int):
    """
    Returns (point, boot):
      point[g]  -> array (M,) pooled WER per model
      boot[g]   -> array (B, M) bootstrap pooled WER per model
    plus per-group error/word totals for tier statistics.
    """
    rng = np.random.default_rng(seed)
    point, boot, boot_err, boot_words = {}, {}, {}, {}
    for g in groups:
        sub = r[r["accent_group"] == g]
        err = sub.pivot_table(index="speaker", columns="model", values="errors", aggfunc="sum")[models]
        words = sub.pivot_table(index="speaker", columns="model", values="ref_words", aggfunc="sum")[models]
        E, N = err.to_numpy(float), words.to_numpy(float)          # (S, M)
        S = len(E)
        # counts[b, s] = how many times speaker s was drawn in resample b
        draws = rng.integers(0, S, size=(n_boot, S))
        counts = np.zeros((n_boot, S))
        np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1)
        be, bn = counts @ E, counts @ N                               # (B, M)
        point[g] = E.sum(0) / N.sum(0)
        boot[g], boot_err[g], boot_words[g] = be / bn, be, bn
    return point, boot, boot_err, boot_words


def ci(samples, axis=0):
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=axis)
    return lo, hi


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def table_wer(r, groups, models, point, boot):
    rows = []
    for g in groups:
        lo, hi = ci(boot[g])
        sub = r[r["accent_group"] == g]
        for j, m in enumerate(models):
            sm = sub[sub["model"] == m]
            rows.append({"model": m, "accent_group": g, "circle": CIRCLE[g],
                         "wer": 100 * point[g][j], "ci_low": 100 * lo[j], "ci_high": 100 * hi[j],
                         "clips": len(sm), "speakers": sm["speaker"].nunique(),
                         "ref_words": int(sm["ref_words"].sum())})
    return pd.DataFrame(rows).round(2)


def table_gap(groups, models, point, boot, boot_err, boot_words, tiers):
    P = np.stack([point[g] for g in groups])                    # (G, M)
    Bt = np.stack([boot[g] for g in groups])                    # (G, B, M)
    outer = [g for g in groups if g not in INNER]
    inner = [g for g in groups if g in INNER]

    def tier(gs, src):
        return sum(src[g] for g in gs)
    tier_in_b = tier(inner, boot_err) / tier(inner, boot_words)   # (B, M)
    tier_out_b = tier(outer, boot_err) / tier(outer, boot_words)

    rows = []
    for j, m in enumerate(models):
        best, worst = int(P[:, j].argmin()), int(P[:, j].argmax())
        gap_b = Bt[:, :, j].max(0) - Bt[:, :, j].min(0)
        ratio_b = Bt[:, :, j].max(0) / Bt[:, :, j].min(0)
        tr_b = tier_out_b[:, j] / tier_in_b[:, j]
        rows.append({
            "model": m, "params_m": PARAMS_M.get(m),
            "best_group": groups[best], "best_wer": 100 * P[best, j],
            "worst_group": groups[worst], "worst_wer": 100 * P[worst, j],
            "gap_pp": 100 * (P[worst, j] - P[best, j]),
            "gap_ci_low": 100 * ci(gap_b)[0], "gap_ci_high": 100 * ci(gap_b)[1],
            "ratio": P[worst, j] / P[best, j],
            "ratio_ci_low": ci(ratio_b)[0], "ratio_ci_high": ci(ratio_b)[1],
            "inner_wer": 100 * tiers["inner"][j], "other_wer": 100 * tiers["other"][j],
            "tier_ratio": tiers["other"][j] / tiers["inner"][j],
            "tier_ratio_ci_low": ci(tr_b)[0], "tier_ratio_ci_high": ci(tr_b)[1],
        })
    return pd.DataFrame(rows).round(3), tier_in_b, tier_out_b


def tier_points(r, groups, models):
    out = {}
    for name, gs in [("inner", [g for g in groups if g in INNER]),
                     ("other", [g for g in groups if g not in INNER])]:
        sub = r[r["accent_group"].isin(gs)].groupby("model")[["errors", "ref_words"]].sum()
        out[name] = (sub["errors"] / sub["ref_words"]).reindex(models).to_numpy()
    return out


def table_vs_us(groups, models, point, boot):
    rows = []
    for g in groups:
        if g == "us":
            continue
        diff_b = boot[g] - boot["us"]
        lo, hi = ci(diff_b)
        for j, m in enumerate(models):
            d = point[g][j] - point["us"][j]
            rows.append({"model": m, "accent_group": g, "diff_pp": 100 * d,
                         "ci_low": 100 * lo[j], "ci_high": 100 * hi[j],
                         "significant": bool(lo[j] > 0 or hi[j] < 0)})
    return pd.DataFrame(rows).round(2)


def worst_clips(r, n=50, model="large-v3"):
    m = r[r["model"] == model].copy()
    tiny = r[r["model"] == "tiny"].set_index("clip_id")["wer"]
    m["wer_tiny"] = m["clip_id"].map(tiny)
    m = m.sort_values(["wer", "errors"], ascending=False).head(n)
    out = pd.DataFrame({
        "rank": range(1, len(m) + 1),
        "clip_id": m["clip_id"], "accent_group": m["accent_group"],
        f"wer_{model}": m["wer"].round(2), "wer_tiny": m["wer_tiny"].round(2),
        "reference": m["reference"], f"prediction_{model}": m["prediction"].fillna(""),
        "audio": "data/audio_subset/" + m["clip_id"],
        "tag": "", "notes": "",
    })
    return out


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CIRCLE_COLOR = {"inner": "#2a78d6", "outer": "#eb6834", "expanding": "#1baf7a"}
CIRCLE_LABEL = {"inner": "Inner circle (US, England, Canada, Australia/NZ)",
                "outer": "Outer circle (South Asia, Southern Africa, SE Asia)",
                "expanding": "Expanding circle (L2, European first language)"}


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False,
    })


def bar_panel(ax, wer_df, model, groups, show_values=True):
    d = wer_df[wer_df["model"] == model].set_index("accent_group").loc[groups]
    y = np.arange(len(groups))[::-1]
    colors = [CIRCLE_COLOR[CIRCLE[g]] for g in groups]
    ax.barh(y, d["wer"], height=0.62, color=colors, zorder=2)
    ax.errorbar(d["wer"], y, xerr=[d["wer"] - d["ci_low"], d["ci_high"] - d["wer"]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
    ax.set_yticks(y, [SHORT_NAME[g] for g in groups])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, d["ci_high"].max() * 1.22)
    ax.tick_params(axis="y", length=0)
    if show_values:
        for yi, (w, hi) in zip(y, zip(d["wer"], d["ci_high"])):
            ax.text(hi + d["ci_high"].max() * 0.02, yi, f"{w:.1f}", va="center",
                    fontsize=8.5, color=INK2)


def circle_legend(fig, y=0.005, ncol=3):
    from matplotlib.patches import Patch
    handles = [Patch(color=CIRCLE_COLOR[c], label=CIRCLE_LABEL[c])
               for c in ["inner", "outer", "expanding"]]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8.5,
               bbox_to_anchor=(0.5, y), handlelength=1.2, columnspacing=1.5)


def fig1_single(wer_df, groups, model, out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bar_panel(ax, wer_df, model, groups)
    ax.set_xlabel("Word error rate (%)  ·  bars: 95% CI, speaker-level bootstrap")
    ax.set_title(f"Whisper {model}: word error rate by accent group", loc="left")
    circle_legend(fig, ncol=1, y=-0.02)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig1_all(wer_df, groups, models, out: Path):
    import matplotlib.pyplot as plt
    n = len(models)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.0 * rows), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i // cols][i % cols]
        bar_panel(ax, wer_df, m, groups, show_values=False)
        ax.set_title(f"{m}  ({PARAMS_M[m]:,}M params)", loc="left", fontsize=10)
        ax.set_xlabel("WER (%)", fontsize=8.5)
        if i % cols:
            ax.set_yticklabels([])
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Word error rate by accent group, every model (95% CIs; x-axes differ per panel)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    circle_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig2_gap(gap_df, out: Path):
    import matplotlib.pyplot as plt
    g = gap_df.set_index("model")
    w = [m for m in WHISPER_ORDER if m in g.index]
    x = g.loc[w, "params_m"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    color = CIRCLE_COLOR["inner"]
    panels = [
        (a1, "gap_pp", "gap_ci_low", "gap_ci_high", "Gap (percentage points)",
         "Absolute gap: worst group minus best group", "{:.1f} pp", None),
        (a2, "tier_ratio", "tier_ratio_ci_low", "tier_ratio_ci_high", "Ratio",
         "Relative gap: other 4 groups ÷ inner circle", "{:.2f}×", 1.0),
    ]
    for ax, col, lo, hi, ylabel, title, fmt, ref in panels:
        ax.fill_between(x, g.loc[w, lo], g.loc[w, hi], color=color, alpha=0.12, lw=0, zorder=1)
        ax.plot(x, g.loc[w, col], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(x, g.loc[w, col], s=42, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        for m, xi in zip(w, x):
            ax.annotate(m, (xi, g.loc[m, col]), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=INK2)
        for m, xi in [(w[0], x[0]), (w[-1], x[-1])]:
            ax.annotate(fmt.format(g.loc[m, col]), (xi, g.loc[m, col]), textcoords="offset points",
                        xytext=(9, -13), ha="left", fontsize=9, color=INK)
        if ref is not None:
            ax.axhline(ref, color=AXIS, lw=1, zorder=1)
            ax.annotate("1× = no gap", (30, ref), textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color=MUTED)
        ax.set_xscale("log")
        ax.set_xticks([39, 74, 244, 769, 1550], ["39M", "74M", "244M", "769M", "1.55B"])
        ax.minorticks_off()
        ax.set_xlabel("Whisper model size (parameters, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_ylim(0, g.loc[w, hi].max() * 1.12)
        ax.set_xlim(28, 2600)
    fig.suptitle("As Whisper gets bigger, the gap between accent groups shrinks in points but grows in proportion",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.text(0.01, -0.03, "Shaded band: 95% CI (speaker-level bootstrap). \"Other 4 groups\" = South Asia, "
             "Southern Africa, SE Asia, L2 (European L1). turbo and wav2vec 2.0 are in Figure 1b and the tables.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig3_lines(wer_df, groups, out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    w = [m for m in WHISPER_ORDER if m in set(wer_df["model"])]
    x = np.array([PARAMS_M[m] for m in w])
    ends = []
    for g in groups:
        d = wer_df[(wer_df["accent_group"] == g)].set_index("model").loc[w]
        c = CIRCLE_COLOR[CIRCLE[g]]
        ax.plot(x, d["wer"], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(x, d["wer"], s=30, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ends.append((d["wer"].iloc[-1], d["wer"].iloc[0], g))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{m}\n{PARAMS_M[m]:,}M" if PARAMS_M[m] < 1000 else f"{m}\n{PARAMS_M[m]/1000:.2f}B"
                      for m in w])
    yt = [1, 2, 3, 5, 10, 20, 30]
    ax.set_yticks(yt, [str(v) for v in yt])
    ax.minorticks_off()
    ax.set_xlim(30, 3400)
    ax.set_ylabel("Word error rate (%, log scale)")
    # End labels, de-overlapped with leader lines
    ends.sort()
    placed = []
    min_gap = 1.15  # multiplicative spacing between labels on the log axis
    for wer_end, _, g in ends:
        yl = wer_end if not placed else max(wer_end, placed[-1] * min_gap)
        placed.append(yl)
        ax.annotate(SHORT_NAME[g], xy=(x[-1], wer_end), xytext=(x[-1] * 1.28, yl),
                    fontsize=8.5, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=3))
    ax.set_title("WER falls with model size for every group, but the two bundles never converge",
                 loc="left", fontsize=10.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CIRCLE_COLOR[c], lw=2, marker="o", ms=5,
                      label=CIRCLE_LABEL[c]) for c in ["inner", "outer", "expanding"]]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, handlelength=1.8)
    fig.text(0.01, -0.01, "Log scale: equal vertical distance = equal ratio. The space between the blue "
             "bundle and the others doesn't shrink as models grow; it widens slightly.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary page
# --------------------------------------------------------------------------- #

def fmt_ci(v, lo, hi, unit=""):
    return f"{v:.1f}{unit} [{lo:.1f}, {hi:.1f}]"


def write_summary(path, wer_df, gap_df, vs_us, worst, tiers, tier_ratio_diff, models, n_boot):
    groups = [g for g in ag.GROUPS if g in set(wer_df["accent_group"])]
    L = ["# Phase 4 results summary\n",
         f"_Generated by `src/analyze.py`. 95% confidence intervals from a speaker-level bootstrap "
         f"({n_boot:,} resamples). WER is pooled (total errors / total reference words)._\n"]

    L.append("## WER (%) by accent group and model\n")
    piv = wer_df.assign(cell=[fmt_ci(w, lo, hi) for w, lo, hi in
                              zip(wer_df["wer"], wer_df["ci_low"], wer_df["ci_high"])])
    piv = piv.pivot(index="accent_group", columns="model", values="cell").loc[groups, models]
    piv.index = [SHORT_NAME[g] for g in piv.index]
    L.append(piv.to_markdown())
    L.append("")

    L.append("## Gap between groups, by model\n")
    gd = gap_df.copy()
    gd["best"] = [f"{SHORT_NAME[g]} ({w:.1f})" for g, w in zip(gd["best_group"], gd["best_wer"])]
    gd["worst"] = [f"{SHORT_NAME[g]} ({w:.1f})" for g, w in zip(gd["worst_group"], gd["worst_wer"])]
    gd["gap (pp)"] = [fmt_ci(a, b, c) for a, b, c in zip(gd["gap_pp"], gd["gap_ci_low"], gd["gap_ci_high"])]
    gd["worst ÷ best"] = [fmt_ci(a, b, c, "×") for a, b, c in zip(gd["ratio"], gd["ratio_ci_low"], gd["ratio_ci_high"])]
    gd["other ÷ inner"] = [fmt_ci(a, b, c, "×") for a, b, c in
                            zip(gd["tier_ratio"], gd["tier_ratio_ci_low"], gd["tier_ratio_ci_high"])]
    L.append(gd[["model", "params_m", "best", "worst", "gap (pp)", "worst ÷ best", "other ÷ inner"]]
             .to_markdown(index=False))
    L.append("\n_\"other ÷ inner\" = pooled WER of the four outer/expanding-circle groups divided by "
             "pooled WER of the four inner-circle groups. It's steadier than worst ÷ best, which "
             "depends on just two groups._\n")

    L.append("## Each group vs. the US group (difference in percentage points)\n")
    sig = vs_us.assign(cell=[fmt_ci(d, lo, hi) + (" *" if s else "") for d, lo, hi, s in
                             zip(vs_us["diff_pp"], vs_us["ci_low"], vs_us["ci_high"], vs_us["significant"])])
    sp = sig.pivot(index="accent_group", columns="model", values="cell")
    sp = sp.loc[[g for g in groups if g != "us"], models]
    sp.index = [SHORT_NAME[g] for g in sp.index]
    L.append(sp.to_markdown())
    L.append("\n`*` = the 95% CI excludes zero (a difference unlikely to be noise).\n")

    tiny, large = gap_df.set_index("model").loc["tiny"], gap_df.set_index("model").loc["large-v3"]
    t_in, t_out = tiers["inner"], tiers["other"]
    i_t, i_l = models.index("tiny"), models.index("large-v3")
    red_in = 100 * (1 - t_in[i_l] / t_in[i_t])
    red_out = 100 * (1 - t_out[i_l] / t_out[i_t])
    lo, hi = tier_ratio_diff
    L.append("## Draft key findings (check the numbers, then rewrite in your own words)\n")
    L.append(f"1. **Every model makes more errors on outer- and expanding-circle accents.** With tiny, "
             f"those four groups have {tiny['tier_ratio']:.1f}× the error rate of the inner-circle groups; "
             f"with large-v3 it is {large['tier_ratio']:.1f}×.")
    L.append(f"2. **Bigger models help everyone a lot.** From tiny to large-v3, pooled WER falls from "
             f"{100*t_in[i_t]:.1f}% to {100*t_in[i_l]:.1f}% for the inner circle ({red_in:.0f}% fewer errors) "
             f"and from {100*t_out[i_t]:.1f}% to {100*t_out[i_l]:.1f}% for the other groups ({red_out:.0f}% fewer).")
    change = large['tier_ratio'] - tiny['tier_ratio']
    if lo > 0:
        verdict = "the ratio between groups actually grows"
    elif hi < 0:
        verdict = "the ratio between groups shrinks"
    else:
        verdict = "the ratio between groups doesn't clearly change"
    L.append(f"3. **The gap shrinks in points but not in proportion.** The best-to-worst gap falls from "
             f"{tiny['gap_pp']:.1f} to {large['gap_pp']:.1f} percentage points, but {verdict}: "
             f"other ÷ inner goes from {tiny['tier_ratio']:.2f}× to {large['tier_ratio']:.2f}× "
             f"(change {change:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}]).")
    if "turbo" in gap_df["model"].values:
        tb = gap_df.set_index("model").loc["turbo"]
        L.append(f"4. **turbo, the faster pruned large-v3, gives back some of the gains, mostly on the "
                 f"groups with higher WER:** other ÷ inner {tb['tier_ratio']:.2f}× vs "
                 f"{large['tier_ratio']:.2f}× for large-v3.")
    if "wav2vec2" in gap_df["model"].values:
        wv = gap_df.set_index("model").loc["wav2vec2"]
        L.append(f"5. **The gap is not Whisper-specific.** wav2vec 2.0, trained only on audiobooks, shows "
                 f"other ÷ inner {wv['tier_ratio']:.2f}×.")
    L.append("")

    L.append("## Worst clips to listen to\n")
    L.append(f"`worst_clips_to_tag.csv` lists the {len(worst)} clips large-v3 got most wrong. Listen to "
             "each one (`data/audio_subset/<clip_id>`) and fill in `tag` with one or more of: "
             "`hallucination` (the model invented text unrelated to the audio, e.g. \"Transcription by "
             "CastingWords\"), `names` (proper nouns, rare words), `sounds` (specific vowels/consonants), `speed`, "
             "`noise` (background/mic quality), `label_error` (the reference text is wrong or the "
             "speaker read something else), `other`. By group:\n")
    L.append(worst["accent_group"].map(SHORT_NAME).value_counts().rename("clips").to_frame().to_markdown())
    L.append("")
    L.append("## Charts\n")
    for f in ["fig1_wer_by_group_large-v3.png", "fig1b_wer_by_group_all_models.png",
              "fig2_gap_vs_model_size.png", "fig3_wer_vs_model_size_by_group.png"]:
        L.append(f"![{f}](figures/{f})\n")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(PROJECT_ROOT / "results" / "results.csv"))
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-worst", type=int, default=50)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    r, models = load(Path(args.results))
    groups = [g for g in ag.GROUPS if g in set(r["accent_group"])]
    print(f"{len(r):,} rows, {len(models)} models, {len(groups)} groups; "
          f"bootstrapping {args.n_boot:,}x by speaker ...")

    point, boot, boot_err, boot_words = bootstrap(r, groups, models, args.n_boot, args.seed)
    wer_df = table_wer(r, groups, models, point, boot)
    tiers = tier_points(r, groups, models)
    gap_df, tin_b, tout_b = table_gap(groups, models, point, boot, boot_err, boot_words, tiers)
    vs_us = table_vs_us(groups, models, point, boot)
    i_t, i_l = models.index("tiny"), models.index("large-v3")
    tr_b = tout_b / tin_b
    tier_ratio_diff = ci(tr_b[:, i_l] - tr_b[:, i_t])
    worst = worst_clips(r, args.n_worst)

    wer_df.to_csv(out / "wer_by_group_model.csv", index=False)
    gap_df.to_csv(out / "gap_by_model.csv", index=False)
    vs_us.to_csv(out / "group_vs_us.csv", index=False)
    worst.to_csv(out / "worst_clips_to_tag.csv", index=False)

    style()
    fig1_single(wer_df, groups, "large-v3", out / "figures" / "fig1_wer_by_group_large-v3.png")
    fig1_all(wer_df, groups, models, out / "figures" / "fig1b_wer_by_group_all_models.png")
    fig2_gap(gap_df, out / "figures" / "fig2_gap_vs_model_size.png")
    fig3_lines(wer_df, groups, out / "figures" / "fig3_wer_vs_model_size_by_group.png")

    write_summary(out / "PHASE4_SUMMARY.md", wer_df, gap_df, vs_us, worst, tiers,
                  tier_ratio_diff, models, args.n_boot)

    print("\nGap by model:")
    print(gap_df[["model", "gap_pp", "gap_ci_low", "gap_ci_high", "ratio", "tier_ratio",
                  "tier_ratio_ci_low", "tier_ratio_ci_high"]].to_string(index=False))
    print(f"\nChange in other÷inner ratio, tiny -> large-v3: "
          f"{gap_df.set_index('model').loc['large-v3','tier_ratio'] - gap_df.set_index('model').loc['tiny','tier_ratio']:+.3f} "
          f"95% CI [{tier_ratio_diff[0]:+.3f}, {tier_ratio_diff[1]:+.3f}]")
    print(f"\nWrote tables, figures and {out / 'PHASE4_SUMMARY.md'}")


if __name__ == "__main__":
    main()
