import datetime
import os
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

os.chdir("/home/lea/Programs/drmed-git")
df_eval = pl.read_parquet(
    "data/exp-250327-masters/2025-05-28-simulations/parquet/"
    "2025-08-14-fit-quality.parquet"
)

def save_plot(filename, format: Literal["svg", "png"]):
    plot_file = (
        "data/exp-250327-masters/2025-05-28-simulations/jupyter/"
        f"{datetime.date.today()}-{filename}"
    )
    if format == "svg":
        plt.savefig(f"{plot_file}.pdf", bbox_inches="tight", dpi=300)
        os.system(f"pdf2svg {plot_file}.pdf {plot_file}.svg")
        os.system(f"rm {plot_file}.pdf")
    elif format == "png":
        plt.savefig(f"{plot_file}.jpg", bbox_inches="tight", dpi=300)

############# Plot time-series, correlation, and fit examples ##############

def tscorres(axd: dict, idx: int, rec_l: dict, rec_f: dict) -> None:
    colf = "tab:pink"
    axts = axd[f"ts{idx}"]
    axcn = axd[f"cor{idx}no"]
    axcy = axd[f"cor{idx}yes"]
    axrn = axd[f"res{idx}no"]
    axry = axd[f"res{idx}yes"]
    sns.lineplot(data=rec_f["trace"], label="artifact", color=colf, ax=axts)
    sns.lineplot(data=rec_l["trace"], label="no artifact", alpha=0.7, ax=axts)
    sns.lineplot(x=rec_l["tc"], y=rec_l["cor"], ax=axcn)
    sns.lineplot(x=rec_l["tc"], y=rec_l["fit"], ax=axcn, color="C2")
    sns.lineplot(x=rec_f["tc"], y=rec_f["cor"], ax=axcy, color=colf)
    sns.lineplot(x=rec_f["tc"], y=rec_f["fit"], ax=axcy, color="C2")
    sns.lineplot(x=rec_l["tc"], y=rec_l["residual"], ax=axrn)
    sns.lineplot(x=rec_f["tc"], y=rec_f["residual"], ax=axry, color=colf)

def prepare_grid(n: int) -> tuple[Figure, dict[str, Axes]]:
    gs_kw = dict(height_ratios=[1, 1, 0.5])
    g1 = ["ts1", "ts1"]
    g2 = ["cor1no", "cor1yes"]
    g3 = ["res1no", "res1yes"]
    for i in range(2, n + 1):
       g1 = g1 + [f"ts{i}"] * 2
       g2 = g2 + [f"cor{i}no", f"cor{i}yes"]
       g3 = g3 + [f"res{i}no", f"res{i}yes"]
    fig, axd = plt.subplot_mosaic(
        mosaic=[g1, g2, g3], figsize=(4*n, 6), gridspec_kw=gs_kw,
        layout="constrained"
    )
    for i in range(1, n + 1):
        axd[f"cor{i}no"].sharey(axd[f"cor{i}yes"])
        axd[f"res{i}no"].sharey(axd[f"res{i}yes"])
    plt.setp([axd[g] for g in g1], xlabel=r"macrotime [$ms$]",
             ylabel="intensity [a.u.]")
    plt.setp([axd[g].get_xticklabels() for g in g2], visible=False)
    plt.setp([axd[g].get_yticklabels() for g in g2 + g3 if "yes" in g],
             visible=False)
    plt.setp([axd[g] for g in g2 if "yes" in g], title="artifact", xscale="log")
    plt.setp([axd[g] for g in g2 if "no" in g], ylabel=r"$G(\tau)$", xscale="log",
             title="no artifact")
    plt.setp([axd[g] for g in g3], xlabel=r"lag time $\tau$ [$ms$]", xscale="log")
    plt.setp([axd[g] for g in g3 if "no" in g], ylabel="residual")
    return fig, axd

# Plot time-series with peak artifacts examples

def pa_tsfilter(peak_dmol: float, record_type: str) -> pl.Expr:
    return (pl.col("artifact").eq("peak_artifacts") &
            pl.col("clean_dmol").eq(1.) &
            pl.col("clean_nmol").eq(3000) &
            pl.col("peak_dmol").eq(peak_dmol) &
            pl.col("record_type").eq(record_type))

def pa_ts(idx: int, peak_dmol: float, record_type: str) -> dict:
    return (df_eval
            .filter(pa_tsfilter(peak_dmol, record_type))
            .row(idx, named=True))

fig, axd = prepare_grid(3)
tscorres(axd, 1, pa_ts(3, 0.01, "label_restoration"), pa_ts(3, 0.01, "feature"))
tscorres(axd, 2, pa_ts(2, 0.1, "label_restoration"), pa_ts(2, 0.1, "feature"))
tscorres(axd, 3, pa_ts(0, 1., "label_restoration"), pa_ts(0, 1., "feature"))
plt.setp(axd["ts1"], title="broad peak artifacts")
plt.setp(axd["ts2"], title="middle-sized peak artifacts")
plt.setp(axd["ts3"], title="steep peak artifacts")
save_plot("peak-artifacts-examples", format="png")

# Plot time-series with photobleaching examples

def pb_tsfilter(
        bleach_type: str, bleach_exp_scale: float, record_type: str
) -> pl.Expr:
    return (pl.col("artifact").eq("photobleaching") &
            pl.col("clean_dmol").eq(1.) &
            pl.col("clean_nmol").eq(3000) &
            pl.col("bleach_type").eq(bleach_type) &
            pl.col("bleach_exp_scale").eq(bleach_exp_scale) &
            pl.col("record_type").eq(record_type))

def pb_ts(
        idx: int, bleach_type: str, bleach_exp_scale: float, record_type: str
) -> dict:
    return (df_eval
            .filter(pb_tsfilter(bleach_type, bleach_exp_scale, record_type))
            .row(idx, named=True))

fig, axd = prepare_grid(4)
tscorres(axd, 1,
         pb_ts(0, "both", 0.01, "label_restoration"),
         pb_ts(0, "both", 0.01, "feature"))
tscorres(axd, 2,
         pb_ts(0, "both", 0.16, "label_restoration"),
         pb_ts(0, "both", 0.16, "feature"))
tscorres(axd, 3,
         pb_ts(0, "immobile", 0.08, "label_restoration"),
         pb_ts(0, "immobile", 0.08, "feature"))
tscorres(axd, 4,
         pb_ts(0, "mobile", 0.08, "label_restoration"),
         pb_ts(0, "mobile", 0.08, "feature"))
plt.setp(axd["ts1"], title=("steep photobleaching (scale = 0.01)\n"
                            "both immobile and mobile molecules"))
plt.setp(axd["ts2"], title=("shallow photobleaching (scale = 0.16)\n"
                            "both immobile and mobile molecules"))
plt.setp(axd["ts3"], title=("middle photobleaching (scale = 0.08)\n"
                            "only immobile molecules bleached"))
plt.setp(axd["ts4"], title=("middle photobleaching (scale = 0.08)\n"
                            "only mobile molecules bleached"))
save_plot("photobleaching-examples", format="png")

# Plot time-series with detector dropout

def dd_tsfilter(
        dropout_n: Literal["few", "middle", "many"], record_type: str
) -> pl.Expr:
    if dropout_n == "few":
        maxdrop_expr = pl.col("dropout_n") < 13
    elif dropout_n == "middle":
        maxdrop_expr = ((pl.col("dropout_n") >= 13) &
                        (pl.col("dropout_n") <= 25))
    elif dropout_n == "many":
        maxdrop_expr = pl.col("dropout_n") > 25
    return (pl.col("artifact").eq("detector_dropout") &
            pl.col("clean_dmol").eq(1.) &
            pl.col("clean_nmol").is_in([3000, 4000]) &
            maxdrop_expr & pl.col("record_type").eq(record_type))

def dd_ts(
        idx: int, dropout_n: Literal["few", "middle", "many"], record_type: str
) -> dict:
    return (df_eval
            .filter(dd_tsfilter(dropout_n, record_type))
            .row(idx, named=True))

fig, axd = prepare_grid(3)
tscorres(axd, 1,
         dd_ts(1, "few", "label_restoration"), dd_ts(1, "few", "feature"))
tscorres(axd, 2,
         dd_ts(0, "middle", "label_restoration"), dd_ts(0, "middle", "feature"))
tscorres(axd, 3,
         dd_ts(1, "many", "label_restoration"), dd_ts(1, "many", "feature"))
plt.setp(axd["ts1"], title="few dropouts, $n<13$")
plt.setp(axd["ts2"], title="middle amount of dropouts, $n=13...25$")
plt.setp(axd["ts3"], title="many dropouts, $n>25$")
save_plot("detector-dropout-examples", format="png")

########## Plot fit results and fit quality measure distributions ###########

def fitres(
        data, x: Literal["diffcoeff", "n", "redchi", "nrmse", "adjr2"],
        y: Literal["peak_dmol", "bleach_group", "dd_group"],
        col_lab: float | int, order: list, log_scale: bool, ax: Axes,
        legend: bool, cut: int,
) -> Axes:
    if x == "diffcoeff":
        xlab = "measured diffusion coeff. D [$\\frac{{\\mu m^2}}{{s}}$]"
    elif x == "n":
        xlab = "measured molecule number n"
    elif x == "redchi":
        xlab = "$\\chi^2_{\\nu}$"
    elif x == "nrmse":
        xlab = "NRMSE"
    elif x == "adjr2":
        xlab = "Adjusted R²"
    if col_lab in [0.1, 1., 10.]:
        title = f"simulated D = {col_lab} $\\frac{{\\mu m^2}}{{s}}$"
    elif col_lab in [125, 1000, 3000, 4000]:
        title = f"simulated n = {col_lab} (not physical)"
    else:
        raise ValueError("'col_lab' has to be 0.1, 1., 10., 125, 1000, 3000 or"
                         " 4000")
    ax = sns.violinplot(
        data, x=x, y=y, order=order, log_scale=log_scale, ax=ax, legend=legend,
        cut=cut, inner="quart", hue="record_type", density_norm="width",
        fill=False, split=True, palette=["tab:pink", "tab:blue"]
    )
    plt.setp(ax, title=title, xlabel=xlab, ylabel=None)
    return ax

def resfit_subplot(
        data, figsize: tuple[float, float],
        x: Literal["diffcoeff_and_n", "redchi", "nrmse", "adjr2"],
        y: Literal["peak_dmol", "bleach_group", "dd_group"],
        order: list[str], nmol_list: list[int], xlim: tuple[float, float],
        yticklabels: list[str],
):
    _, ax = plt.subplots(
        2, 3, figsize=figsize, layout="constrained", sharey=True
    )
    if x == "diffcoeff_and_n":
        x1 = "diffcoeff"
        x2 = "n"
        log_scale1 = True
        log_scale2 = False
        cut1 = 2
        cut2 = 0
        axl = ax[0, :]
    elif x == "redchi":
        log_scale1 = log_scale2 = True
        cut1 = cut2 = 2
        x1 = x2 = x
        axl = ax
    elif x == "nrmse":
        log_scale1 = log_scale2 = False
        cut1 = cut2 = 0
        x1 = x2 = x
        axl = ax
    elif x == "adjr2":
        log_scale1 = log_scale2 = False
        cut1 = cut2 = 0
        x1 = x2 = x
        axl = ax
    for i, dmol in enumerate([0.1, 1, 10]):
        fitres(
            data.loc[data["clean_dmol"] == dmol], x=x1, ax=ax[0, i], y=y,
            col_lab=dmol, log_scale=log_scale1, legend=False, cut=cut1,
            order=order
        )
    for i, nmol in enumerate(nmol_list):
        if nmol == min(nmol_list):
            legend = True
        else:
            legend = False
        fitres(
            data.loc[data["clean_nmol"] == nmol], x=x2, y=y, col_lab=nmol,
            log_scale=log_scale2, ax=ax[1, i], legend=legend, cut=cut2,
            order=order
        )
    plt.setp(axl, xlim=xlim)
    ax[0, 0].set_yticklabels(yticklabels)
    sns.move_legend(ax[1, 0], "lower center", bbox_to_anchor=(.5, -0.4), ncol=2,
                    title=None, frameon=True)
    for t, l in zip(ax[1, 0].axes.get_legend().texts,
                    ["with artifact", "without artifact"]):
        t.set_text(l)
    return ax

# Plot time-series with peak artifacts fit results and fit quality metrics

def pa_filter() -> pl.Expr:
    return ((pl.col("artifact") == "peak_artifacts") &
            (pl.col("clean_dmol").is_in([0.1, 1., 10.])) &
            (pl.col("clean_nmol").is_in([125, 1000, 3000])))
pa_pd = (
    df_eval.filter(pa_filter())
    .cast({"peak_dmol": pl.String, "peak_nmol": pl.String})
    .select("record_type", "diffcoeff", "n", "clean_dmol", "clean_nmol",
            "peak_dmol", "peak_nmol", "nrmse", "redchi", "adjr2")
    .to_pandas()
)
pa_yticklabels = [
    "broad peak artifacts\n$D_{sim}$ = 0.01 $\\frac{{\\mu m^2}}{{s}}$"
    "\n$n_{\\mathrm{clusters}} = 10$",
    "middle-sized peak artifacts\n$D_{sim}$ = 0.1 $\\frac{{\\mu m^2}}{{s}}$"
    "\n$n_{\\mathrm{clusters}} = 7$",
    "steep peak artifacts\n$D_{sim}$ = 1 $\\frac{{\\mu m^2}}{{s}}$"
    "\n$n_{\\mathrm{clusters}} = 3$",
]
pa_order = ["0.01", "0.1", "1.0"]
pa_nmol_list = [125, 1000, 3000]

resfit_subplot(
    pa_pd, figsize=(11, 6), x="diffcoeff_and_n", y="peak_dmol", order=pa_order,
    nmol_list=pa_nmol_list, xlim=(10e-5, 10e2), yticklabels=pa_yticklabels
)
save_plot("peak-artifacts-fit-distributions", "svg")
resfit_subplot(
    pa_pd, figsize=(11, 6), x="redchi", y="peak_dmol", order=pa_order,
    nmol_list=pa_nmol_list, xlim=(1e-5, 1e2), yticklabels=pa_yticklabels
)
save_plot("peak-artifacts-fit-quality-redchi", "svg")
resfit_subplot(
    pa_pd, figsize=(11, 6), x="nrmse", y="peak_dmol", order=pa_order,
    nmol_list=pa_nmol_list,  xlim=(0, 0.13), yticklabels=pa_yticklabels
)
save_plot("peak-artifacts-fit-quality-nrmse", "svg")
resfit_subplot(
    pa_pd, figsize=(11, 6), x="adjr2", y="peak_dmol", order=pa_order,
    nmol_list=pa_nmol_list, xlim=(0.8, 1), yticklabels=pa_yticklabels
)
save_plot("peak-artifacts-fit-quality-adjr2", "svg")

# Plot time-series with photobleaching fit results and fit quality metrics

def pb_filter() -> pl.Expr:
    return ((pl.col("artifact") == "photobleaching") &
            (pl.col("clean_dmol").is_in([0.1, 1., 10.])) &
            (pl.col("clean_nmol").is_in([125, 1000, 3000])))
pb_pd = (
    pl.concat([
        (df_eval
         .filter(pb_filter() & (pl.col("bleach_exp_scale") < 0.09) &
                 (pl.col("bleach_type") == "both"))
         .with_columns(bleach_group=pl.lit("steep_both"))),
        (df_eval
         .filter(pb_filter() & (pl.col("bleach_exp_scale") >= 0.09) &
                 (pl.col("bleach_type") == "both"))
         .with_columns(bleach_group=pl.lit("shallow_both"))),
        (df_eval
         .filter(pb_filter() & ((pl.col("bleach_exp_scale") >= 0.05) &
                                (pl.col("bleach_exp_scale") <= 0.12)) &
                 (pl.col("bleach_type") == "immobile"))
         .with_columns(bleach_group=pl.lit("middle_immob"))),
        (df_eval
         .filter(pb_filter() & ((pl.col("bleach_exp_scale") >= 0.05) &
                                (pl.col("bleach_exp_scale") <= 0.12)) &
                 (pl.col("bleach_type") == "mobile"))
         .with_columns(bleach_group=pl.lit("middle_mobile"))),
    ], how="vertical")
    .select("record_type", "diffcoeff", "n", "clean_dmol", "clean_nmol",
            "bleach_group", "nrmse", "redchi", "adjr2")
    .to_pandas()
)
pb_yticklabels = [
    "steep photobleaching\nscale = 0.01...0.08\nboth immobile and mobile",
    "shallow photobleaching\nscale = 0.09...0.16\nboth immobile and mobile",
    "middle photobleaching\nscale = 0.05...0.12\nonly immobile mol. bleached",
    "middle photobleaching\nscale = 0.05...0.12\nonly mobile mol. bleached",
]
pb_order = ["steep_both", "shallow_both", "middle_immob", "middle_mobile"]
pb_nmol_list = [125, 1000, 3000]
pb_group = "bleach_group"

resfit_subplot(
    pb_pd, figsize=(11, 7), x="diffcoeff_and_n", y=pb_group, order=pb_order,
    nmol_list=pb_nmol_list, xlim=(10e-5, 10e2), yticklabels=pb_yticklabels,
)
save_plot("photobleaching-fit-distributions", "svg")
resfit_subplot(
    pb_pd, figsize=(11, 7), x="redchi", y=pb_group, order=pb_order,
    nmol_list=pb_nmol_list, xlim=(1e-5, 1e2), yticklabels=pb_yticklabels,
)
save_plot("photobleaching-fit-quality-redchi", "svg")
resfit_subplot(
    pb_pd, figsize=(11, 7), x="nrmse", y=pb_group, order=pb_order,
    nmol_list=pb_nmol_list, xlim=(0, 0.11), yticklabels=pb_yticklabels,
)
save_plot("photobleaching-fit-quality-nrmse", "svg")
resfit_subplot(
    pb_pd, figsize=(11, 7), x="adjr2", y=pb_group, order=pb_order,
    nmol_list=pb_nmol_list, xlim=(0.6, 1), yticklabels=pb_yticklabels,
)
save_plot("photobleaching-fit-quality-adjr2", "svg")

# Plot time-series with detector dropout fit results and fit quality metrics

def dd_filter() -> pl.Expr:
    return ((pl.col("artifact") == "detector_dropout") &
            (pl.col("clean_dmol").is_in([0.1, 1., 10.])) &
            (pl.col("clean_nmol").is_in([1000, 4000])))
dd_pd = (
    pl.concat([
        (df_eval
         .filter(dd_filter() & (pl.col("dropout_n") < 13))
         .with_columns(dd_group=pl.lit("few"))),
        (df_eval
         .filter(dd_filter() & ((pl.col("dropout_n") >= 13) &
                                (pl.col("dropout_n") <= 25)))
         .with_columns(dd_group=pl.lit("middle"))),
        (df_eval
         .filter(dd_filter() & (pl.col("dropout_n") > 25))
         .with_columns(dd_group=pl.lit("many"))),
    ], how="vertical")
    .select("record_type", "diffcoeff", "n", "clean_dmol", "clean_nmol",
            "dd_group", "nrmse", "redchi", "adjr2")
    .to_pandas()
)
dd_yticklabels = [
    "detector dropout\n$n_{\\mathrm{dropouts}} < 13$",
    "detector dropout\n$n_{\\mathrm{dropouts}} = 13...25$",
    "detector dropout\n$n_{\\mathrm{dropouts}} > 25$",
]
dd_order = ["few", "middle", "many"]
dd_nmol_list = [1000, 4000]
dd_groupsize = [dd_pd.loc[(dd_pd["clean_dmol"] == d) &
                          (dd_pd["clean_nmol"] == n) &
                          (dd_pd["dd_group"] == p)
                 ].shape[0]
       for d in [0.1, 1.0, 10.0]
       for n in [1000, 4000]
       for p in dd_order]
dd_min = min(dd_groupsize)
def dd_filter2(
        df: pl.DataFrame, limit: int, group: str, clean1_ex: pl.Expr,
        clean2_ex: pl.Expr, dropout_ex: pl.Expr,
) -> pl.DataFrame:
    return (df
            .filter((pl.col("artifact") == "detector_dropout") & clean1_ex &
                    clean2_ex & dropout_ex)
            .limit(limit)
            .with_columns(dd_group=pl.lit(group))
            )
dd_pd = (
    pl.concat(
        [dd_filter2(df_eval, dd_min, "few", pl.col("clean_dmol") == d,
                    pl.col("clean_nmol") == n, pl.col("dropout_n") < 13)
         for d in [0.1, 1.0, 10.0] for n in [1000, 4000]] +
        [dd_filter2(df_eval, dd_min, "middle", pl.col("clean_dmol") == d,
                    pl.col("clean_nmol") == n,
                    (pl.col("dropout_n") >= 13) & (pl.col("dropout_n") <= 25))
         for d in [0.1, 1.0, 10.0] for n in [1000, 4000]] +
        [dd_filter2(df_eval, dd_min, "many", pl.col("clean_dmol") == d,
                    pl.col("clean_nmol") == n, pl.col("dropout_n") > 25)
         for d in [0.1, 1.0, 10.0] for n in [1000, 4000]]
        , how="vertical")
    .select("record_type", "diffcoeff", "n", "clean_dmol", "clean_nmol",
            "dd_group", "nrmse", "redchi", "adjr2")
    .to_pandas()
)

ax = resfit_subplot(
    dd_pd, figsize=(11, 6), x="diffcoeff_and_n", y="dd_group", order=dd_order,
    nmol_list=dd_nmol_list, xlim=(10e-5, 10e2), yticklabels=dd_yticklabels,
)
plt.delaxes(ax[1, 2])
save_plot("detector-dropout-fit-distributions", "svg")
ax = resfit_subplot(
    dd_pd, figsize=(11, 6), x="redchi", y="dd_group", order=dd_order,
    nmol_list=dd_nmol_list, xlim=(1e-5, 1e2), yticklabels=dd_yticklabels,
)
plt.delaxes(ax[1, 2])
save_plot("detector-dropout-fit-quality-redchi", "svg")
ax = resfit_subplot(
    dd_pd, figsize=(11, 6), x="nrmse", y="dd_group", order=dd_order,
    nmol_list=dd_nmol_list, xlim=(0, 0.10), yticklabels=dd_yticklabels,
)
plt.delaxes(ax[1, 2])
save_plot("detector-dropout-fit-quality-nrmse", "svg")
ax = resfit_subplot(
    dd_pd, figsize=(11, 6), x="adjr2", y="dd_group", order=dd_order,
    nmol_list=dd_nmol_list, xlim=(0.85, 1), yticklabels=dd_yticklabels,
)
plt.delaxes(ax[1, 2])
save_plot("detector-dropout-fit-quality-adjr2", "svg")

# plot overview of fit quality parameters

pd_eval = df_eval.select(
    "redchi", "nrmse", "adjr2", "artifact", "record_type"
).to_pandas()

f, ax = plt.subplots(1, 3, figsize=(9, 3), layout="constrained", sharey=True)
sns.violinplot(
    pd_eval, x="redchi", y="artifact", hue="record_type", density_norm="width",
    log_scale=True, fill=False, inner="quart", split=True, ax=ax[0],
    legend=True, palette=["tab:pink", "tab:blue"],
).set_yticklabels(["with and without\ndetector dropout",
                   "with and without\npeak artifacts",
                   "with and without\nphotobleaching"])
sns.violinplot(
    pd_eval, x="nrmse", y="artifact", hue="record_type", density_norm="width",
    log_scale=False, fill=False, inner="quart", split=True, ax=ax[1],
    legend=False, palette=["tab:pink", "tab:blue"],
)
sns.violinplot(
    pd_eval, x="adjr2", y="artifact", hue="record_type", density_norm="width",
    log_scale=False, fill=False, inner="quart", split=True, ax=ax[2],
    legend=False, palette=["tab:pink", "tab:blue"],
)
sns.move_legend(ax[0], "lower center", bbox_to_anchor=(.1, -0.42), ncol=2,
                title=None, frameon=True)
for t, l in zip(ax[0].axes.get_legend().texts,
                ["with artifact", "without artifact"]):
    t.set_text(l)
plt.setp(ax[0], xlabel="$\\chi^2_{\\nu}$", ylabel=None)
plt.setp(ax[1], xlabel="NRMSE")
plt.setp(ax[2], xlabel="Adjusted R²", xlim=(0.75, 1))
save_plot("fit-quality-overview", "svg")
