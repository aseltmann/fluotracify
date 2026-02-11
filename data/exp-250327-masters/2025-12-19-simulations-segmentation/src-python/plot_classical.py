#!/usr/bin/env python3

import os

import numpy as np
import polars as pl
import seaborn as sns

from collections.abc import Callable, Iterable
from datetime import datetime

os.chdir("/home/alva/Programs/drmed-git")

workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"


def get_data(myfile: str) -> tuple[pl.DataFrame, str]:
    out_date = datetime.today().date()
    out_file = myfile.split(".")
    out_file = out_file[0].split("-")[3:]
    out_file = "-".join(out_file)
    out_file = f"{out_date}-{out_file}"
    df = pl.read_parquet(f"{workdir}/parquet/{myfile}")
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
    return df, out_file

def pivot_data(df: pl.DataFrame, col_artifact: str) -> pl.DataFrame:
    df = (
        df
        .unpivot(index=["clean_dmol", "clean_nmol", col_artifact],
                 on=pl.selectors.matches(
                     "precision|recall|fbeta2|biniou|meaniou|overlap"
                    ))
        .with_columns(
            pl.col.variable
            .str.replace("t_", "t-")
            .str.replace("tm_", "tm-")
            .str.replace("chan_vese", "chan-vese")
            .str.replace("random_walker", "random-walker")
            .str.split_exact("_", 1)
            .struct.rename_fields(["method", "metric"])
           ).unnest("variable")
       )
    return df

def groupby_mean_std(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    df = (
        df
        .group_by(["method", "metric"], maintain_order=True)
        .agg(
            pl.col.value.drop_nans().mean().alias(f"mean {metric}"),
            pl.col.value.drop_nans().std().alias(f"std {metric}")
        )
        .filter(pl.col.metric.eq(f"{metric}"))
        .drop("metric")
    )
    return df
