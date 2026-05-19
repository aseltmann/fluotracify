#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import skimage as ski
import seaborn as sns



from datetime import datetime
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Literal

os.chdir("/home/alva/Programs/drmed-git")

inputdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
segdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"
workdir = "data/exp-250327-masters/2026-03-12-defense/jupyter-python"


def get_data(myfile: str) -> pl.DataFrame:
    out_file = myfile.split(".")
    out_first = out_file[0].split("-")[3:]
    out_first = "-".join(out_first)
    df = pl.concat(
        [
            pl.read_parquet(f"{inputdir}/{myfile}"),
            pl.read_parquet(
                f"{segdir}/parquet/2026-01-14-{out_first}-ground-truth.parquet"
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


def ex_local_thresholding_gaussian():
    out_date = datetime.today().date()
    df = get_data("2025-05-28-peak-artifacts-training.parquet")
    trace = df["feature"][0][:30].to_numpy()
    thresh = ski.filters.threshold_local(trace, block_size=11)
    p_bool = max(trace) * (trace > thresh)
    p_invbool = max(trace) * ~(trace > thresh)
    fig, ax = plt.subplots()
    sns.lineplot(trace, ax=ax)
    sns.lineplot(thresh, ax=ax)
    ax.fill_between(x=np.arange(len(trace)), y1=0, y2=p_bool, alpha=0.5,
                    where=p_bool, color="tab:pink", label="below thr")
    ax.fill_between(x=np.arange(len(trace)), y1=0, y2=p_invbool, alpha=0.5,
                    where=p_invbool, color="tab:green", label="above thr")
    ax.set_axis_off()
    plt.legend()
    plt.savefig(f"{workdir}/{out_date}-local-thresholding-ex.png")
