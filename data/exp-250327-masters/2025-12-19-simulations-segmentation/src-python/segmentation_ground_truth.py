#!/usr/bin/env python3

import os
import sys
import lmfit

import numpy as np
import polars as pl

from datetime import datetime
from pathlib import Path

from IPython.display import display

os.chdir("/home/alva/Programs/drmed-git")
FLUOTRACIFY_PATH = Path("src/")
sys.path.append(FLUOTRACIFY_PATH.as_posix())

from fluotracify import fcsdc

inputdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation/parquet"
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
        # "2025-05-28-detector-dropout-testing.parquet",
        # "2025-05-28-detector-dropout-training.parquet",
        # "2025-05-28-detector-dropout-validation.parquet",
        "2025-05-28-peak-artifacts-testing.parquet",
        # "2025-05-28-peak-artifacts-training.parquet",
        # "2025-05-28-peak-artifacts-validation.parquet",
        # "2025-05-28-photobleaching-testing.parquet",
        # "2025-05-28-photobleaching-training.parquet",
        # "2025-05-28-photobleaching-validation.parquet",
        # "2025-08-13-detector-dropout-testing-correlation.parquet",
        # "2025-08-13-detector-dropout-training-correlation.parquet",
        # "2025-08-13-detector-dropout-validation-correlation.parquet",
        # "2025-08-13-peak-artifacts-testing-correlation.parquet",
        # "2025-08-13-peak-artifacts-training-correlation.parquet",
        # "2025-08-13-peak-artifacts-validation-correlation.parquet",
        # "2025-08-13-photobleaching-testing-correlation.parquet",
        # "2025-08-13-photobleaching-training-correlation.parquet",
        # "2025-08-13-photobleaching-validation-correlation.parquet",
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
            artifact=pl.col("sim_params").struct.field("sim_artifact"),
            record_type=pl.lit(f"{record.rstrip('_g')}"),
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
        df_record = pl.concat([df_record, df_nrmse], how="vertical")
    df_eval = pl.concat([df_eval, df_record], how="vertical")
    print(f"Correlating traces in {myfile} ...")
    df = pl.read_parquet(
        f"data/exp-250327-masters/2025-05-28-simulations/parquet/{myfile}"
    )
    stop = df["sim_params"].struct.field("total_sim_time")
    step = df["sim_params"].struct.field("time_step")
    cor_df = pl.DataFrame()
    for idx, row in enumerate(df.iter_slices(n_rows=1)):
        record = {}
        for s in row.select(pl.selectors.matches(
                "^feature$|^label_restoration$|^label_segmentation$")):
            size = s.arr.len().item()
            scale = pl.DataFrame(
                {"scale": [np.arange(0, stop[idx], step[idx])]},
                schema={"scale": pl.Array(pl.Float32, shape=(size))}
            )
            ts = row.select(
                pl.col(s.name, "ts_params")).rename({s.name: "trace"})
            ts = pl.concat([ts, scale], how="horizontal")
            if s.name == "feature":
                record[s.name] = fcsdc.FCSTimeSeries.from_polars(ts)
            else:
                record[s.name] = fcsdc.FCSTimeSeriesLabel.from_polars(ts)
        sim_params = fcsdc.FCSSimParams.from_polars(row["sim_params"])
        sim_ts = fcsdc.SimulatedFCSTimeSeries(
            uuid=row["uuid"].item(), sim_params=sim_params, record=record
        )
        cor_record = {}
        for k, rec in sim_ts.record.items():
            if k in ["feature", "label_restoration"]:
                cor = fcsdc.FCSCor(
                    uuid=sim_ts.uuid, method=cor_method,
                    multipletau_m=multipletau_m,
                    multipletau_deltat=multipletau_deltat,
                    multipletau_norm=multipletau_norm,
                    multipletau_compress=multipletau_compress,
                )
                cor.autocorrelate(rec.trace)
                cor_record[k] = cor
        sim_cor = fcsdc.SimulatedFCSTimeSeriesCor(
            uuid=sim_ts.uuid, sim_params=sim_params, record=cor_record
        )
        cor_df = pl.concat([cor_df, sim_cor.to_polars()], how="vertical")
    out_file = myfile.split(".")
    out_first = out_file[0].split("-")[3:]
    out_date = datetime.today().date()
    out_file = f"{out_date}-{'-'.join(out_first)}-correlation.{out_file[1]}"
    cor_df.write_parquet(
        f"data/exp-250327-masters/2025-05-28-simulations/parquet/{out_file}"
    )
    print(f"Fitting correlations in {myfile} ...")
    df = pl.read_parquet(f"{inputdir}/{myfile}")
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
