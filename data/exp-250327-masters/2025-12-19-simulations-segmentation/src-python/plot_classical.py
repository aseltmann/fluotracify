#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
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

    _, ax = plt.subplots(1, 4, figsize=(12, 10), sharex=True, sharey=True)

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


def get_unet_params(
        df: pl.DataFrame, params_df: pl.DataFrame, return_full=False
) -> pl.DataFrame:
    algos = [
        c.removesuffix("_seg")
        for c in df.select(pl.selectors.matches("_seg")).columns
    ]
    out = pl.DataFrame()
    for a in algos:
        id_col = f"{a}_full-id"
        if df.get_column(id_col, default=None) is not None:
            run_id = df[id_col].unique()[0]
            run_pars = (
                params_df
                .filter(pl.col.run_id.eq(run_id))
                .select(
                    "exp_name", "parent_id", "run_id", "hp_pool_size",
                    "hp_lr_start", "hp_first_filters", "hp_lr_power", "hp_scaler",
                    "hp_batch_size", "hp_n_levels"
                )
            )
            run_score = (
                df.select(f"{a}_score")
                .drop_nans().mean()
                .rename({f"{a}_score": "score"})
            )
            out = pl.concat([
                out,
                pl.concat([run_pars, run_score], how="horizontal")
            ], how="vertical")
    out = pl.concat(
        [
            out.drop("score", "run_id").group_by("parent_id").first(),
            (
                out
                .group_by("parent_id").agg(pl.col("score").alias("sorting"))
            ),
            (
                out
                .with_columns(run_id=pl.col.run_id.str.head(5))
                .group_by("parent_id").agg(pl.struct(pl.col("run_id", "score"))
                                           .alias("scores"))

            ),

        ], how="align"
    )
    out = out.sort("sorting", descending=True).drop("sorting")
    if not return_full:
        out = out.drop("parent_id").with_row_index("parent_id")
    else:
        out = out.with_row_index("id")
    return out


def get_parent_id_dict(df: pl.DataFrame, params: pl.DataFrame) -> dict:
    out = get_unet_params(df, params, return_full=True)
    out = {p: newid for p, newid in zip(out["parent_id"], out["id"])}
    return out


def plot_loss_auc(
        df: pl.DataFrame, params_df: pl.DataFrame, metrics_df: pl.DataFrame,
        out_file: str | None = None
) -> None:
    fig, ax = plt.subplots(5, 4, figsize=(10, 10), sharex=True)

    p_dict = get_parent_id_dict(df, params_df)

    algos = [
        c.removesuffix("_seg")
        for c in df.select(pl.selectors.matches("_seg")).columns
        if c.removesuffix("_seg") not in [
                "t_isodata", "t_li", "t_mean", "t_min", "t_otsu", "t_triangle",
                "t_yen", "tm_local", "tm_niblack", "tm_bradley", "tm_sauvola",
                "chan_vese", "random_walker", "watershed"
        ]
    ]


    metrics = (
        metrics_df
        .filter(pl.col.run_id.str.contains_any(algos))
        .with_columns(pl.struct("parent_id").map_elements(
            lambda x: p_dict[x["parent_id"]], return_dtype=pl.Int64
        ))
        .sort("parent_id")
    )
    for i, p in enumerate(metrics["parent_id"].unique(maintain_order=True)):
        axl = ax[i // 2, i % 2 * 2]
        axr = ax[i // 2, i % 2 * 2 + 1]
        for r in metrics.filter(pl.col.parent_id.eq(p))["run_id"].unique():
            m = metrics.filter(pl.col.run_id.eq(r))
            t_loss = m.filter(pl.col.metric.eq("loss"))["metric_history"][0]
            v_loss = m.filter(pl.col.metric.eq("val_loss"))["metric_history"][0]
            t_auc = m.filter(pl.col.metric.eq("auc"))["metric_history"][0]
            v_auc = m.filter(pl.col.metric.eq("val_auc"))["metric_history"][0]
            sns.lineplot(t_loss, ax=axl, label=f"{r:.5}, train")
            sns.lineplot(v_loss, ax=axl, label=f"{r:.5}, val")
            sns.lineplot(t_auc, ax=axr)
            sns.lineplot(v_auc, ax=axr)
            plt.setp(axl, title=f"id {p} - loss")
            plt.setp(axr, title=f"id {p} - PR-AUC")

    plt.setp(ax[:, 0::2], yscale="log")
    plt.setp(ax[:, 1::2], ylim=[0, 1])
    plt.setp(ax.flatten(), xlabel="epoch")
    plt.setp(ax[:, 0], ylabel="loss or auc in a.u.")
    fig.tight_layout()
    fig.align_ylabels()
    if out_file:
        out_date = datetime.today().date()
        plt.savefig(f"{workdir}/jupyter-python/{out_date}-{out_file}.png")
    else:
        plt.show()
