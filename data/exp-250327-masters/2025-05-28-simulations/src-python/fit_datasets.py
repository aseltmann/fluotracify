#!/usr/bin/env python3

import os
import sys
import lmfit

import numpy as np
import polars as pl

from datetime import datetime
from pathlib import Path

from IPython.display import display

os.chdir("/home/lea/Programs/drmed-git")
FLUOTRACIFY_PATH = Path("/home/lea/Programs/drmed-git/src/")
sys.path.append(FLUOTRACIFY_PATH.as_posix())

from fluotracify import fcsdc

workdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
fit_method = "lmfit"

param_1sp = lmfit.Parameters()
param_1sp.add("offset", value=0.01, min=-0.5, max=1.5, vary=True)
param_1sp.add("gn0", value=1., min=-0.0001, max=3000., vary=True)
param_1sp.add("a1", value=1., min=0.0001, max=1., vary=False)
param_1sp.add("txy1", value=1., min=0.001, max=2000., vary=True)
param_1sp.add("alpha1", value=1., min=0.6, max=2., vary=False)
equation_dim = "2D"
equation_dspecies = 1.

for myfile in [
        "2025-08-13-detector-dropout-testing-correlation.parquet",
        "2025-08-13-detector-dropout-training-correlation.parquet",
        "2025-08-13-detector-dropout-validation-correlation.parquet",
        "2025-08-13-peak-artifacts-testing-correlation.parquet",
        "2025-08-13-peak-artifacts-training-correlation.parquet",
        "2025-08-13-peak-artifacts-validation-correlation.parquet",
        "2025-08-13-photobleaching-testing-correlation.parquet",
        "2025-08-13-photobleaching-training-correlation.parquet",
        "2025-08-13-photobleaching-validation-correlation.parquet",
]:
    print(f"Fitting correlations in {myfile} ...")
    df = pl.read_parquet(f"{workdir}/{myfile}")
    fit_df = pl.DataFrame()
    for idx, row in enumerate(df.iter_slices(n_rows=1)):
        cor_record = {}
        for s in row.select(pl.selectors.matches(
                "^feature_g$|^label_restoration_g$|^label_segmentation_g$")):
            cor = row.select(
                pl.col("tc", s.name, "cor_params")).rename({s.name: "g"})
            cor_record[s.name.rstrip("_g")] = fcsdc.FCSCor.from_polars(cor)
        sim_params = fcsdc.FCSSimParams.from_polars(row["sim_params"])
        cor_ts = fcsdc.SimulatedFCSTimeSeriesCor(
            uuid=row["uuid"].item(), sim_params=sim_params, record=cor_record
        )
        fit_record = {}
        for k, rec in cor_ts.record.items():
            assert isinstance(rec.tc,  np.ndarray)
            assert isinstance(rec.g,  np.ndarray)
            fit = fcsdc.FCSFit(
                uuid=cor_ts.uuid, method=fit_method, params=param_1sp,
                equation_dim="2D", equation_dspecies=1
            )
            fit.minimize(rec.tc, rec.g)
            fit_record[k] = fit
        sim_fit = fcsdc.SimulatedFCSTimeSeriesFit(
            uuid=cor_ts.uuid, sim_params=sim_params, record=fit_record
        )
        fit_df = pl.concat([fit_df, sim_fit.to_polars()], how="vertical")
    out_file = myfile.split(".")
    out_first = out_file[0].split("-")[3:]
    out_first = "-".join(out_first).rstrip("-correlation")
    out_date = datetime.today().date()
    out_file = f"{out_date}-{out_first}-fit.{out_file[1]}"
    fit_df.write_parquet(f"{workdir}/{out_file}")

def redchi(
        cor: pl.Series, fit: pl.Series, ndata: pl.Series, nvarys: pl.Series
) -> np.ndarray:
    chisq = np.sum(
        np.pow((cor - fit).to_numpy(), 2) / np.abs(fit.to_numpy()),
        axis=1
    )
    redchi = chisq / (ndata.to_numpy() - nvarys.to_numpy())
    return redchi

def nrmse(cor: pl.Series, fit: pl.Series) -> np.ndarray:
    rmse = np.sqrt(np.mean(np.pow((cor - fit).to_numpy(), 2), axis=1))
    nrmse = rmse / (np.max(cor.to_numpy(), axis=1) -
                    np.min(cor.to_numpy(), axis=1))
    return nrmse

def adjr2(
        cor: pl.Series, fit: pl.Series, ndata: pl.Series, nvarys: pl.Series
) -> np.ndarray:
    """see
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
    """
    r2 = 1 - (np.sum(np.pow((cor - fit).to_numpy(), 2), axis=1) /
              np.sum(np.pow((cor - np.mean(cor.to_numpy(), axis=1)).to_numpy(),
                            2), axis=1))
    adjr2 = 1 - (1 - r2) * ((ndata.to_numpy() - 1) /
                            (ndata.to_numpy() - nvarys.to_numpy() - 1))
    return adjr2

def diffcoeff(tau: pl.Series, fwhm: float = 250) -> np.ndarray:
    return np.round((fwhm / 1000)**2 /
                    (8 * np.log(2.0) * tau.to_numpy() / 1000), 2)

df_eval = pl.DataFrame()
for myfile in [
        "2025-08-14-detector-dropout-training-fit.parquet",
        "2025-08-14-peak-artifacts-training-fit.parquet",
        "2025-08-14-photobleaching-training-fit.parquet",
]:
    df = pl.read_parquet(f"{workdir}/{myfile}")
    stem = myfile.lstrip("2025-08-14").rstrip("-fit.parquet")
    corfile = f"2025-08-13-{stem}-correlation.parquet"
    tsfile = f"2025-05-28-{stem}.parquet"
    df_cor = pl.read_parquet(f"{workdir}/{corfile}")
    df_ts = pl.read_parquet(f"{workdir}/{tsfile}")
    df_record = pl.DataFrame()
    for record in ["feature_g", "label_restoration_g"]:
        minimizer = f"{record.rstrip('_g')}_minimizer_params"
        residual = f"{record.rstrip('_g')}_residual"
        df_nrmse = pl.concat([
            df.select("uuid", "tc", "sim_params", minimizer, record, residual)
            .rename({minimizer: "minimizer", record: "fit",
                     residual: "residual"}),
            df_cor.select("uuid", record).rename({record: "cor"}),
            df_ts.select("uuid", record.strip("_g"), "ts_params")
            .rename({record.strip("_g"): "trace"})
        ], how="align")
        df_nrmse = df_nrmse.with_columns(
            nrmse=pl.struct("fit", "cor").map_batches(
                lambda x: nrmse(x.struct.field("cor"), x.struct.field("fit"))
            ),
            ndata=(
                pl.col("minimizer").struct.field("fit_stats")
                .struct.field("ndata")
            ),
            nvarys=(
                pl.col("minimizer").struct.field("fit_stats")
                .struct.field("nvarys")
            ),
            artifact=pl.col("sim_params").struct.field("sim_artifact"),
            record_type=pl.lit(f"{record.rstrip('_g')}"),
            tau=(
                pl.col("minimizer").struct.field("params")
                .struct.field("txy1").struct.field("value")
            ),
            n=(
                1 / pl.col("minimizer").struct.field("params")
                .struct.field("gn0").struct.field("value")
            ),
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
        df_nrmse = df_nrmse.with_columns(
            redchi=pl.struct("fit", "cor", "ndata", "nvarys").map_batches(
                lambda x: redchi(
                    x.struct.field("cor"), x.struct.field("fit"),
                    x.struct.field("ndata"), x.struct.field("nvarys")
                )
            ),
            adjr2=pl.struct("fit", "cor", "ndata", "nvarys").map_batches(
                lambda x: adjr2(
                    x.struct.field("cor"), x.struct.field("fit"),
                    x.struct.field("ndata"), x.struct.field("nvarys")
                )
            ),
            diffcoeff=pl.struct("tau").map_batches(
                lambda x: diffcoeff(x.struct.field("tau"))
            )
        )
        df_record = pl.concat([df_record, df_nrmse], how="vertical")
    df_eval = pl.concat([df_eval, df_record], how="vertical")

display(df_eval)
out_file = "2025-08-14-fit-quality.parquet"
df_eval.write_parquet(f"{workdir}/{out_file}")
