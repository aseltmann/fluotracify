#!/usr/bin/env python3

import os

import numpy as np
import polars as pl
import skimage as ski
import sklearn.metrics as skm

from collections.abc import Callable, Iterable
from datetime import datetime

os.chdir("/home/alva/Programs/drmed-git")

inputdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"

def get_data(myfile: str) -> tuple[pl.DataFrame, str]:
    out_file = myfile.split(".")
    out_first = out_file[0].split("-")[3:]
    out_first = "-".join(out_first)
    df = pl.concat(
        [pl.read_parquet(f"{inputdir}/{myfile}"),
         (pl.read_parquet(f"{workdir}/parquet/2026-01-14-{out_first}-ground-"
                          "truth.parquet")
          .rename({"label_segmentation": "label_ground_truth"}))
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
    df = df.drop(["label_restoration", "label_segmentation", "sim_params",
                  "ts_params"])
    df_clean = df.filter(pl.col.label_ground_truth.arr.sum().eq(0)).shape[0]
    print(f"Dropping {df_clean} traces without artifacts")
    df = df.filter(pl.col.label_ground_truth.arr.sum().ne(0))
    out_date = datetime.today().date()
    out_file = f"{out_date}-{out_first}-classical.{out_file[1]}"
    return df, out_file


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
    return ski.segmentation.random_walker(trace, labels=labels, mode="bf")  # type: ignore


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
    return ski.segmentation.watershed(trace, markers=markers)  # type: ignore


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

def jaccard(
        true: list, pred: list, precision: float, recall: float, average: str
) -> float:
    if np.isnan(precision) | np.isnan(recall):
        out = np.nan
    else:
        out = float(skm.jaccard_score(true, pred, average=average))
    return out


t_algos = {
    # algos returning a float
    "t_isodata": ski.filters.threshold_isodata,
    "t_li": ski.filters.threshold_li,
    "t_mean": ski.filters.threshold_mean,
    "t_min": ski.filters.threshold_minimum,
    "t_otsu": ski.filters.threshold_otsu,
    "t_triangle": ski.filters.threshold_triangle,
    "t_yen": ski.filters.threshold_yen,
    # algos returning a threshold mask
    "tm_local": ski.filters.threshold_local,
    "tm_niblack": ski.filters.threshold_niblack,
    "tm_bradley": threshold_bradley,
    "tm_sauvola": threshold_sauvola,
    # classical segmentation algorithms, outputting a mask
    "chan_vese": ski.segmentation.chan_vese,
    "random_walker": random_walker_quant,
    "watershed": watershed_quant,
}


for myfile in [
        "2025-05-28-detector-dropout-testing.parquet",
        "2025-05-28-peak-artifacts-testing.parquet",
        "2025-05-28-photobleaching-testing.parquet",
]:
    print(f"Perform and evaluate classical segmentation for {myfile} ...")
    df, out_file = get_data(myfile)
    for k, algo in t_algos.items():
        print(f"{k}")
        seg = f"{k}_seg"
        cm = f"{k}_cm"
        precision = f"{k}_precision"
        recall = f"{k}_recall"
        fbeta2 = f"{k}_fbeta2"
        biniou = f"{k}_biniou"
        meaniou = f"{k}_meaniou"
        overlap = f"{k}_overlap"
        df = df.with_columns(
            (pl.struct("feature").map_elements(
                lambda x: seg_classical(x["feature"], k, algo),
                return_dtype=pl.List(pl.Float32)
               )).alias(seg)
           )
        df = df.cast({seg: pl.Array(pl.Boolean, shape=(16384))})
        df = df.with_columns(
            (pl.struct("label_ground_truth", seg).map_elements(
                lambda x: skm.confusion_matrix(
                    x["label_ground_truth"], x[seg], labels=[0, 1],
                   ), return_dtype=pl.List(pl.Array(pl.Int64, shape=(2)))
               )).alias(cm),
            (pl.struct("label_ground_truth", seg).map_elements(
                lambda x: skm.precision_score(
                    x["label_ground_truth"], x[seg], zero_division=np.nan  # type: ignore
                   ), return_dtype=pl.Float32
               )).alias(precision),
            (pl.struct("label_ground_truth", seg).map_elements(
                lambda x: skm.recall_score(
                    x["label_ground_truth"], x[seg], zero_division=np.nan  # type: ignore
                   ), return_dtype=pl.Float32
               )).alias(recall),
            (pl.struct("label_ground_truth", seg).map_elements(
                lambda x: skm.fbeta_score(
                    x["label_ground_truth"], x[seg], beta=2,
                    zero_division=np.nan  # type: ignore
                   ), return_dtype=pl.Float32
               )).alias(fbeta2),
        )
        df = df.cast({cm: pl.Array(pl.Int64, shape=(2, 2))})
        df = df.with_columns(
            (pl.struct("label_ground_truth", seg, precision, recall)
             .map_elements(
                 lambda x: jaccard(x["label_ground_truth"], x[seg],
                                   x[precision], x[recall], average="binary"),
                 return_dtype=pl.Float32
               )).alias(biniou),
            (pl.struct("label_ground_truth", seg, precision, recall)
             .map_elements(
                 lambda x: jaccard(x["label_ground_truth"], x[seg],
                                   x[precision], x[recall], average="macro"),
                 return_dtype=pl.Float32
               )).alias(meaniou),
            # overlap coefficient see
            # https://en.wikipedia.org/wiki/Overlap_coefficient
            # from confusion matrix:
            # overlap coef = tp / min((tn + fp), (fn + tp))
            (pl.col(cm).arr.get(1).arr.get(1) /
             pl.min_horizontal(pl.col(cm).arr.get(0).arr.sum(),
                               pl.col(cm).arr.get(1).arr.sum())
             ).alias(overlap),
        )
    df = df.drop("feature")
    df.write_parquet(f"{workdir}/parquet/{out_file}")
