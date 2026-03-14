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


def get_record(
        df: pl.DataFrame, idx: int,
        group: Literal["dropout_n", "peak_dmol", "bleach_exp_scale"],
        subgroup: str | float,
) -> dict:
    return (df
            .filter(pl.col(group).eq(subgroup) &
                    pl.col.clean_dmol.eq(1.) &
                    pl.col.clean_nmol.is_in([3000, 4000]))
            .with_columns(group=pl.lit(group), subgroup=pl.lit(subgroup))
            .row(idx, named=True))
