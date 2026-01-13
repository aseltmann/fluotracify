
#!/usr/bin/env python3

import os
import sys
import lmfit

import matplotlib.pyplot as plt
import multipletau
import numpy as np
import polars as pl
import seaborn as sns

from datetime import datetime
from pathlib import Path
from typing import Literal

from IPython.display import display

os.chdir("/home/alva/Programs/drmed-git")
FLUOTRACIFY_PATH = Path("/home/alva/Programs/drmed-git/src/")
sys.path.append(FLUOTRACIFY_PATH.as_posix())

from fluotracify import fcsdc

inputdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"

cor_method: Literal["multipletau"] = "multipletau"
multipletau_m = 16
multipletau_norm = True
multipletau_compress: Literal["average"] = "average"

fit_method: Literal["lmfit"] = "lmfit"
param_1sp = lmfit.Parameters()
param_1sp.add("offset", value=0.01, min=-0.5, max=1.5, vary=True)
param_1sp.add("gn0", value=1., min=-0.0001, max=3000., vary=True)
param_1sp.add("a1", value=1., min=0.0001, max=1., vary=False)
param_1sp.add("txy1", value=1., min=0.001, max=2000., vary=True)
param_1sp.add("alpha1", value=1., min=0.6, max=2., vary=False)
equation_dim: Literal["2D"] = "2D"
equation_dspecies: Literal[1] = 1


def segment_threshold(
        trace: pl.Series, artifact: Literal["peak_artifacts", "photobleaching",
                                            "detector_dropout"],
        threshold: float
) -> np.ndarray:
    if artifact in ["peak_artifacts", "photobleaching"]:
        seg = trace.to_numpy() > threshold
    elif artifact == "detector_dropout":
        seg = trace.to_numpy() < threshold
    else:
        raise ValueError(f"{artifact=} is not a valid artifact.")
    return seg


def cut_and_stitch(
        trace: list, seg: list
) -> np.ndarray:
    new = np.delete(trace, seg)
    # pad corrected trace to length of original trace, pad with NaN
    new = np.pad(new, (0, np.sum(seg)), constant_values=np.array(None))
    return new


def diffcoeff(tau: float, fwhm: float = 250) -> float:
    return (fwhm / 1000)**2 / (8 * np.log(2.0) * tau / 1000)


def nrmse(cor: np.ndarray, fit: np.ndarray) -> float:
    rmse = np.sqrt(np.mean(np.pow((cor - fit), 2), axis=0))
    nrmse = rmse / (np.max(cor, axis=0) - np.min(cor, axis=0))
    return nrmse


def adjr2(
        cor: np.ndarray, fit: np.ndarray, ndata: int, nvarys: int
) -> float:
    """see
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
    """
    r2 = 1 - (np.sum(np.pow((cor - fit), 2), axis=0) /
              np.sum(np.pow((cor - np.mean(cor, axis=0)), 2), axis=0))
    adjr2 = 1 - (1 - r2) * ((ndata - 1) / (ndata - nvarys - 1))
    return adjr2


def correlate_multipletau(
        df: dict, colname: str, outname: str, pad_max_length: int,
) -> pl.Series:
    trace = pl.Series(df[colname]).drop_nans().to_numpy()
    cor = fcsdc.FCSCor(
        uuid=df["uuid"], method=cor_method, multipletau_m=multipletau_m,
        multipletau_deltat=df["bin"], multipletau_norm=multipletau_norm,
        multipletau_compress=multipletau_compress,
    )
    try:
        cor.autocorrelate(trace)
        assert (cor.tc is not None) and (cor.g is not None)
        cor.tc = np.pad(cor.tc, (0, pad_max_length - len(cor.tc)),
                        constant_values=np.array(None))
        cor.g = np.pad(cor.g, (0, pad_max_length - len(cor.g)),
                       constant_values=np.array(None))
    except (ValueError, AssertionError, IndexError):
        cor.tc = np.tile(np.nan, pad_max_length)
        cor.g = np.tile(np.nan, pad_max_length)
    out = cor.to_polars().to_struct().struct.rename_fields([
        f"{outname}_tc", f"{outname}_g", f"{outname}_params"
    ])
    return out


def fcs_fit(
        df: dict, colname: str, outname: str, pad_max_length: int,
) -> pl.Series:
    tc = pl.Series(df[f"{colname}_tc"]).drop_nans().to_numpy()
    g = pl.Series(df[f"{colname}_g"]).drop_nans().to_numpy()

    fit = fcsdc.FCSFit(
        uuid=df["uuid"], method=fit_method, params=param_1sp,
        equation_dim=equation_dim, equation_dspecies=equation_dspecies,
    )
    try:
        fit.minimize(tc, g)
    except (TypeError, ValueError):
        fit.tc = np.tile(np.nan, pad_max_length)
        fit.g = np.tile(np.nan, pad_max_length)
        fit.residual = np.tile(np.nan, pad_max_length)
    if fit.result_minimizer is not None:
        assert (fit.tc is not None) and (fit.g is not None) and (
            fit.residual is not None)
        assert hasattr(fit.result_minimizer, "params")
        fit_nrmse = pl.DataFrame(
            [nrmse(g, fit.g)], schema={"nrmse": pl.Float32}
        )
        fit_adjr2 = pl.DataFrame(
            [adjr2(g, fit.g, fit.result_minimizer.ndata,
                   fit.result_minimizer.nvarys)],
            schema={"adjr2": pl.Float32}
        )
        fit_txy1 = pl.DataFrame(
            [fit.result_minimizer.params["txy1"].value],
            schema={"txy1": pl.Float32}
        )
        fit_diffcoeff = pl.DataFrame(
            [diffcoeff(fit.result_minimizer.params["txy1"].value)],
            schema={"diffcoeff": pl.Float32}
        )
        fit_n = pl.DataFrame(
            [1 / fit.result_minimizer.params["gn0"].value],
            schema={"n": pl.Float32}
        )
        fit.tc = np.pad(fit.tc, (0, pad_max_length - len(fit.tc)),
                        constant_values=np.array(None))
        fit.g = np.pad(fit.g, (0, pad_max_length - len(fit.g)),
                       constant_values=np.array(None))
        fit.residual = np.pad(fit.residual,
                              (0, pad_max_length - len(fit.residual)),
                              constant_values=np.array(None))
    else:
        fit_nrmse = pl.DataFrame([np.nan], schema={"nrmse": pl.Float32})
        fit_adjr2 = pl.DataFrame([np.nan], schema={"adjr2": pl.Float32})
        fit_txy1 = pl.DataFrame([np.nan], schema={"txy1": pl.Float32})
        fit_diffcoeff = pl.DataFrame([np.nan], schema={"diffcoeff": pl.Float32})
        fit_n = pl.DataFrame([np.nan], schema={"n": pl.Float32})
    out = fit.to_polars()
    out = out.drop(["uuid", "tc", "initial_params", "minimizer_params"])
    out = pl.concat([out, fit_nrmse, fit_adjr2, fit_txy1, fit_diffcoeff, fit_n],
                    how="horizontal")
    out = out.to_struct().struct.rename_fields([
        f"{outname}_g", f"{outname}_residual", f"{outname}_nrmse",
        f"{outname}_adjr2", f"{outname}_txy1", f"{outname}_diffcoeff",
        f"{outname}_n"
    ])
    return out


def polars_correlate_and_fit(
        df: pl.DataFrame, col_trace: str, col_cor: str, col_fit: str
) -> pl.DataFrame:
    df = df.with_columns(
        (pl.struct("uuid", col_trace, "bin").map_elements(
            lambda x: correlate_multipletau(
                x, col_trace, col_cor, pad_max_length
            ),
           )).alias(col_cor)
       )
    df = df.with_columns(pl.col(col_cor).list.first().struct.unnest())
    df = df.with_columns(
        (pl.struct("uuid", f"{col_cor}_tc", f"{col_cor}_g").map_elements(
            lambda x: fcs_fit(x, col_cor, col_fit, pad_max_length),
           )).alias(col_fit)
       )
    df = df.with_columns(pl.col(col_fit).list.first().struct.unnest())
    df = df.drop([col_cor, f"{col_cor}_params", col_fit])
    return df


for myfile in [
        "2025-05-28-peak-artifacts-training.parquet",
        "2025-05-28-photobleaching-training.parquet",
]:
    print(f"Computing segmentation thresholds for {myfile} ...")
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
        bin=pl.col("ts_params").struct.field("bin")
    )
    df = df.drop(["sim_params", "ts_params"])
    # df = df.head()
    # display(df)
    testcor = multipletau.autocorrelate(
        df["feature"][0], m=multipletau_m, deltat=df["bin"][0],
        normalize=multipletau_norm, compress=multipletau_compress,
       )
    pad_max_length = testcor[1:].shape[0]

    df = polars_correlate_and_fit(
        df, "feature", "corfeat", "fitfeat"
    )
    df = polars_correlate_and_fit(
        df, "label_restoration", "corclean", "fitclean"
    )

    for t in [0.0001, 0.001, 0.005, 0.01, 0.025, 0.04, 0.06, 0.1]:
        print(f"threshold {t}")
        t = round(t, 4)
        seg = f"seg{t}".replace(".", "p")
        new = f"new{t}".replace(".", "p")
        cor = f"cor{t}".replace(".", "p")
        fit = f"fit{t}".replace(".", "p")
        df = df.with_columns(
            (pl.col("label_segmentation").map_batches(
                lambda x: segment_threshold(
                    trace=x, artifact="peak_artifacts", threshold=t
                   )
               )).alias(seg),
           )
        df = df.with_columns(
            (pl.struct("feature", seg).map_elements(
                lambda x: cut_and_stitch(
                    x["feature"], x[seg]
                   ),
                return_dtype=pl.List(pl.Float32)
               )).alias(new),
           )
        df = df.cast({new: pl.Array(pl.Float32, shape=(16384))})
        df = df.with_columns(
            (pl.col(new).arr.len() - pl.col(new).arr.count_matches(np.nan)
             ).alias(f"{new}_len")
        )
        df = polars_correlate_and_fit(df, new, cor, fit)
        # for evaluating which threshold is best, drop the trace and correlation
        # data and only keep evaluation data
        df = df.drop([seg, new, f"{cor}_tc", f"{cor}_g", f"{fit}_g",
                      f"{fit}_residual"])

    # for evaluating which threshold is best, drop the trace and correlation
    # data and only keep evaluation data
    df = df.drop([
        "feature", "label_restoration", "label_segmentation", "corfeat_tc",
        "corfeat_g", "fitfeat_g", "fitfeat_residual", "corclean_tc",
        "corclean_g", "fitclean_g", "fitclean_residual"
    ])

    out_file = myfile.split(".")
    out_first = out_file[0].split("-")[3:]
    out_first = "-".join(out_first)
    out_date = datetime.today().date()
    out_file = f"{out_date}-{out_first}-segmentation-ground-truth.{out_file[1]}"
    df.write_parquet(f"{workdir}/parquet/{out_file}")

# load the correlation and fit results for different segmentation thresholds
# and plot them to determine a good threshold as a gold standard
def plot_ground_truth_segmentations(
        df: pl.DataFrame, sel: str, row: str, col: str, log_scale: bool,
        sharex: bool, cut: int, vline_idx: int, outname: str
) -> None:
    df_dmol = (
        df
        .filter(pl.col("clean_dmol").is_in([0.1, 1., 10.]) &
                pl.col("clean_nmol").is_in([125, 1000, 3000]))
        .select(pl.selectors.matches(sel))
        .unpivot(index=["clean_dmol", "clean_nmol", row])
    )
    g = sns.catplot(df_dmol, x="value", y="variable", hue="variable",
                    row=row, col=col, kind="violin",
                    log_scale=log_scale, sharex=sharex, cut=cut)

    for ax in g.axes.flatten():
        if vline_idx == 16384:
            ax.axvline(16384, 0, 1, color="red")
        else:
            med = ax.lines[vline_idx].get_xdata()
            ax.axvline(med, 0, 1, color="red")
    out_date = datetime.today().date()
    plt.savefig(f"{workdir}/jupyter-python/{out_date}-{outname}.png")


df = pl.concat(
    [pl.read_parquet(f"{workdir}/parquet/2026-01-09-peak-artifacts-training"
                     "-segmentation-ground-truth.parquet"),
     pl.read_parquet(f"{workdir}/parquet/2026-01-09-photobleaching-training"
                     "-segmentation-ground-truth.parquet")],
    how="vertical"
)

# peak artifacts, dmol
plot_ground_truth_segmentations(
    df, "(0p|clean|feat).*_diffcoeff|peak_dmol|clean_dmol|clean_nmol",
    "peak_dmol", "clean_dmol", log_scale=True, sharex=True, cut=0, vline_idx=5,
    outname="peak-artifacts-diffcoeff"
)
# peak artifacts, nmol
plot_ground_truth_segmentations(
    df, "(0p|clean|feat).*_n$|peak_dmol|clean_dmol|clean_nmol",
    "peak_dmol", "clean_nmol", log_scale=False, sharex=False, cut=0, vline_idx=5,
    outname="peak-artifacts-n"
)
# peak artifacts, nrmse
plot_ground_truth_segmentations(
    df, "(0p|clean|feat).*_nrmse|peak_dmol|clean_dmol|clean_nmol",
    "peak_dmol", "clean_dmol", log_scale=False, sharex=False, cut=0, vline_idx=5,
    outname="peak-artifacts-nrmse"
)
# peak artifacts, adjr2
plot_ground_truth_segmentations(
    df, "(0p|clean|feat).*_adjr2|peak_dmol|clean_dmol|clean_nmol",
    "peak_dmol", "clean_dmol", log_scale=False, sharex=False, cut=0, vline_idx=5,
    outname="peak-artifacts-adjr2"
)
# peak artifacts, trace length
plot_ground_truth_segmentations(
    df, "(0p)*_len|peak_dmol|clean_dmol|clean_nmol",
    "peak_dmol", "clean_dmol", log_scale=False, sharex=True, cut=0,
    vline_idx=16384, outname="peak-artifacts-trace-length"
)
