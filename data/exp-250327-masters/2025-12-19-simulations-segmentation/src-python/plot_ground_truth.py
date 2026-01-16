#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from datetime import datetime
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Literal

os.chdir("/home/alva/Programs/drmed-git")

inputdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"


def get_data(myfile: str) -> pl.DataFrame:
    out_file = myfile.split(".")
    out_first = out_file[0].split("-")[3:]
    out_first = "-".join(out_first)
    df = pl.concat(
        [
            pl.read_parquet(f"{inputdir}/{myfile}"),
            pl.read_parquet(
                f"{workdir}/parquet/2026-01-14-{out_first}-ground-truth.parquet"
            ).rename({"label_segmentation": "label_ground_truth"})
        ], how="align"
    )
    df = df.with_columns(
        artifact=pl.col("sim_params").struct.field("sim_artifact"),
        clean_dmol=(
            pl.col("sim_params").struct.field("clean_dmol")
            .cast(pl.Float64).round(3)
           ),
        clean_nmol=pl.col("sim_params").struct.field("clean_nmol"),
        peak_dmol=(
            pl.col("sim_params").struct.field("peak_dmol").cast(pl.Float64)
            .round(3)
           ),
        peak_nmol=pl.col("sim_params").struct.field("peak_nmol"),
        bleach_exp_scale=(
            pl.col("sim_params").struct.field("bleach_exp_scale")
            .cast(pl.Float64).round(3)
           ),
        bleach_type=pl.col("sim_params").struct.field("bleach_type"),
        dropout_n=pl.col("sim_params").struct.field("dropout_n"),
        dropout_maxdrop=(
            pl.col("sim_params").struct.field("dropout_maxdrop")
           ),
    )
    df = df.drop(["sim_params", "ts_params"])
    df = df.with_columns(
        (
            pl.when(pl.col.bleach_exp_scale > 0.05,
                    pl.col.bleach_exp_scale <= 0.10)
            .then(pl.lit("shallow"))
            .otherwise(pl.when(pl.col.bleach_exp_scale <= 0.05)
                       .then(pl.lit("steep")))
            .alias("bleach_exp_scale")),
        (
            pl.when(pl.col.dropout_n > 19)
            .then(pl.lit("many"))
            .otherwise(pl.when(pl.col.dropout_n <= 19)
                       .then(pl.lit("few")))
            .alias("dropout_n"))
    )
    return df


def get_record(
        df: pl.DataFrame, idx: int,
        group: Literal["dropout_n", "peak_dmol", "bleach_exp_scale"],
        subgroup: str
) -> dict:
    return (df
            .filter(pl.col(group).eq(subgroup) &
                    pl.col.clean_dmol.eq(1.) &
                    pl.col.clean_nmol.is_in([3000, 4000]))
            .with_columns(group=pl.lit(group), subgroup=pl.lit(subgroup))
            .row(idx, named=True))


def prepare_grid(n: int) -> tuple[Figure, dict[str, Axes]]:
    gs_kw = dict(height_ratios=[1, 1, 1])
    g1 = ["ts1",]
    g2 = ["lab1",]
    g3 = ["seg1",]
    for i in range(2, n + 1):
       g1 = g1 + [f"ts{i}",]
       g2 = g2 + [f"lab{i}",]
       g3 = g3 + [f"seg{i}",]
    fig, axd = plt.subplot_mosaic(
        mosaic=[g1, g2, g3], figsize=(5*n, 5), gridspec_kw=gs_kw,
        layout="constrained", sharex=True,
    )
    plt.setp([axd[g] for g in g1], ylabel=r"intensity [a.u.]")
    plt.setp([axd[g] for g in g2], ylabel=r"intensity [a.u.]")
    plt.setp([axd[g] for g in g3], xlabel=r"macrotime [$ms$]",
             ylabel=None, yticks=[])
    return fig, axd


def plot_ground_truth(axd: dict, idx: int, rec: dict) -> None:
    colp = "tab:pink"
    colg = "tab:green"
    axts = axd[f"ts{idx}"]
    axlab = axd[f"lab{idx}"]
    axseg = axd[f"seg{idx}"]
    p = np.array(rec["label_ground_truth"])
    x = np.arange(len(p))
    p_bool = max(rec["feature"]) * p
    p_invbool = max(rec["feature"]) * ~p
    if rec["group"] == "dropout_n":
        threshold_lab = "threshold: t < 0"
    elif rec["group"] in ["bleach_exp_scale", "peak_dmol"]:
        threshold_lab = "threshold: t > 0.01"
    else:
        threshold_lab = rec["group"]
    if rec["subgroup"] == "few":
        ts_title = "few dropouts, $n \\leq 19$"
    elif rec["subgroup"] == "many":
        ts_title = "many dropouts, $n > 19$"
    elif rec["subgroup"] == "shallow":
        ts_title = "shallow photobleaching, scale = 0.06...0.1"
    elif rec["subgroup"] == "steep":
        ts_title = "broad photobleaching, scale = 0.01...0.05"
    elif rec["subgroup"] == 0.01:
        ts_title = ("broad peak artifacts\n$D_{{sim}} = 0.01 "
                    "\\frac{{\\mu m^2}}{{s}}$, $n_{{clusters}}=10$")
    elif rec["subgroup"] == 0.1:
        ts_title = ("middle-sized peak artifacts\n$D_{{sim}} = 0.1 "
                    "\\frac{{\\mu m^2}}{{s}}$, $n_{{clusters}}=7$")
    elif rec["subgroup"] == 1.:
        ts_title = ("steep peak artifacts\n$D_{{sim}} = 1."
                    "\\frac{{\\mu m^2}}{{s}}$, $n_{{clusters}}=3$")
    else:
        ts_title = rec["subgroup"]

    # upper plot: feature with and without artifact
    axts.set_title(ts_title)
    sns.lineplot(data=rec["feature"], color=colp, ax=axts,
                 label="artifact")
    sns.lineplot(data=rec["label_restoration"], alpha=0.7, ax=axts,
                 label="no artifact")
    axts.legend(loc="lower right")
    # middle plot: artifact simulation trace and threshold
    axlab.set_title("artifact label from simulations")
    sns.lineplot(data=rec["label_segmentation"], color=colp, ax=axlab)
    axlab.axhline(0, ls="--", c="tab:grey", label=threshold_lab)
    axlab.legend(loc="lower right")
    # lower plot: ground truth segmentation vector
    axseg.set_title("ground truth segmentation vector")
    # sns.lineplot(data=rec["feature"], alpha=0.3, ax=axseg)
    sns.lineplot(data=p_bool, alpha=0.5, color=colp, ax=axseg)
    axseg.fill_between(x=x, y1=0, y2=p_bool, color=colp,
                       label="1 = artifact")
    axseg.fill_between(x=x, y1=0, y2=p_invbool, alpha=0.5, color=colg,
                       label="0 = no artifact")
    axseg.legend(loc="lower right")
