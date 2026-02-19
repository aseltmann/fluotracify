#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from collections.abc import Callable, Iterable
from datetime import datetime

os.chdir("/home/alva/Programs/drmed-git")

workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"


def get_data(classical_file: str, unet_file: str) -> tuple[pl.DataFrame, str]:
    out_date = datetime.today().date()
    out_file = classical_file.split(".")
    out_file = out_file[0].split("-")[3:-1]
    out_file = "-".join(out_file)
    out_file = f"{out_date}-{out_file}"
    dfa = pl.read_parquet(f"{workdir}/parquet/{classical_file}")
    dfb = pl.read_parquet(f"{workdir}/parquet/{unet_file}")
    df = dfa.join(dfb, on=["uuid"], how="full")
    df = get_artifact_columns(df)
    df = compute_score(df)
    return df, out_file


def get_artifact_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
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


def compute_score(df: pl.DataFrame) -> pl.DataFrame:
    algos = [
        c.removesuffix("_seg")
        for c in df.select(pl.selectors.matches("_seg")).columns
    ]


    for a in algos:
        df = df.with_columns(
            (pl.col(f"{a}_fbeta2")
             .add(pl.col(f"{a}_meaniou"))
             .add(pl.col(f"{a}_overlap"))
             ).truediv(3).alias(f"{a}_score")
        )
    return df


def pivot_data(df: pl.DataFrame, col_artifact: str) -> pl.DataFrame:
    df = (
        df
        .unpivot(index=["clean_dmol", "clean_nmol", col_artifact],
                 on=pl.selectors.matches(
                     "precision|recall|fbeta2|biniou|meaniou|overlap|score"
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

def groupby_mean_std(df: pl.DataFrame, metrics: list[str]) -> pl.DataFrame:
    out = pl.DataFrame()
    for i, m in enumerate(metrics):
        if i < 1:
            how = "horizontal"
        else:
            how = "align"
        out = pl.concat([
            out,
            (
                df
                .group_by(["method", "metric"], maintain_order=True)
                .agg(
                    pl.col.value.drop_nans().mean().alias(f"mean {m}"),
                    pl.col.value.drop_nans().std().alias(f"std {m}")
                )
                .filter(pl.col.metric.eq(f"{m}"))
                .drop("metric")
            )
        ], how=how)
    return out


def score_violin(df: pl.DataFrame) -> None:

    order = (
        df
        .filter(pl.col.metric.eq("score"))
        .group_by("method").agg(pl.col.value.drop_nans().mean())
        .sort("value", descending=True)
       )["method"].to_list()

    _, ax = plt.subplots(1, 4, figsize=(14, 14), sharex=True, sharey=True)

    sns.violinplot(
        df.filter(pl.col.metric.eq("score")), x="value", y="method", cut=0,
        density_norm="width", order=order, ax=ax[0]
       ).set_title("score")
    sns.violinplot(
        df.filter(pl.col.metric.eq("fbeta2")), x="value", y="method", cut=0,
        density_norm="width", order=order, ax=ax[1]
       ).set_title("$F_2$")
    sns.violinplot(
        df.filter(pl.col.metric.eq("meaniou")), x="value", y="method", cut=0,
        density_norm="width", order=order, ax=ax[2]
       ).set_title("mean IOU")
    sns.violinplot(
        df.filter(pl.col.metric.eq("overlap")), x="value", y="method", cut=0,
        density_norm="width", order=order, ax=ax[3]
       ).set_title("OVL")
    for axis in ax:
        plt.setp(axis, xlabel="", ylabel="")
