#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import skimage as ski

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Literal

os.chdir("/home/alva/Programs/drmed-git")

inputdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"

def get_data(myfile: str) -> pl.DataFrame:
    out_file = myfile.split(".")
    out_first = out_file[0].split("-")[3:]
    out_first = "-".join(out_first)
    df = pl.read_parquet(f"{inputdir}/{myfile}")
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
    # df = df.with_columns(
    #     (
    #         pl.when(pl.col.bleach_exp_scale > 0.05,
    #                 pl.col.bleach_exp_scale <= 0.10)
    #         .then(pl.lit("shallow"))
    #         .otherwise(pl.when(pl.col.bleach_exp_scale <= 0.05)
    #                    .then(pl.lit("steep")))
    #         .alias("bleach_exp_scale")),
    #     (
    #         pl.when(pl.col.dropout_n > 19)
    #         .then(pl.lit("many"))
    #         .otherwise(pl.when(pl.col.dropout_n <= 19)
    #                    .then(pl.lit("few")))
    #         .alias("dropout_n"))
    # )
    return df


def threshold_bradley(trace: list) -> Callable:
    # https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_niblack
    # bradley is a special case of niblack which usese a scaling factor q
    # standard q: q=1. determined by trial and error that q=0.75 provides better
    # results (less false positives)
    q = 0.75
    return ski.filters.threshold_niblack(trace, k=0) * q


def threshold_sauvola(trace: list) -> Callable:
    # https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
    # sauvola has a scaling factor R which when not set is set to half of the
    # dtype range. For float32 values this is far too high. Set it to maximum
    # of the trace
    r = max(trace)
    return ski.filters.threshold_sauvola(trace, r=r)


def random_walker_quant(trace: Iterable) -> Callable:
    """
    random_walker needs seed markers (labels) - use lowest 5% and highest 5%
    intensity as two types of markers
    see
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.random_walker
    """
    trace = np.array(trace)
    labels = np.zeros_like(trace, dtype=int)
    labels[trace < np.quantile(trace, 0.05)] = 1
    labels[trace > np.quantile(trace, 0.95)] = 2
    return ski.segmentation.random_walker(trace, labels=labels, mode="bf")


def watershed_quant(trace: Iterable) -> Callable:
    """
    watershed needs markers which are used as "basins", so starting points for
    "flooding" the trace → use the lowest 5% intensity and highest 5% intensity
    as two types of markers
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed
    """
    trace = np.array(trace)
    markers = np.zeros_like(trace, dtype=int)
    markers[trace < np.quantile(trace, 0.05)] = 1
    markers[trace > np.quantile(trace, 0.95)] = 2
    return ski.segmentation.watershed(trace, markers=markers)


def seg_classical(
        trace: Iterable, algo_str: str, algorithm: Callable
) -> np.ndarray:
    trace = np.array(trace)

    if algo_str in ["chan_vese", "random_walker", "watershed"]:
        # segmentation algos expect 2D input
        trace = trace.reshape((trace.size, 1))
        out = algorithm(trace).astype(np.float32)
        if algo_str in ["random_walker", "watershed"]:
            out -= 1
    else:
        try:
            thresh = algorithm(trace)
            if algo_str == "tm_sauvola":
                out = np.greater(thresh, trace)
            else:
                out = np.greater(trace, thresh)
        except RuntimeError as e:
            if algo_str == "t_min":
                # if no minimum is found, the algorithm predicts no artifacts
                out = np.array(0).repeat(trace.shape)
            else:
                raise e
    out = out.reshape(trace.size)
    return out
