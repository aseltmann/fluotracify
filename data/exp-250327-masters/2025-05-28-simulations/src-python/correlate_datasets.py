#!/usr/bin/env python3

import os
import sys

import numpy as np
import polars as pl

from datetime import datetime
from pathlib import Path

os.chdir("/home/lea/Programs/drmed-git")
FLUOTRACIFY_PATH = Path("/home/lea/Programs/drmed-git/src/")
sys.path.append(FLUOTRACIFY_PATH.as_posix())

from fluotracify import fcsdc


cor_method = "multipletau"
multipletau_m = 16
multipletau_deltat = 1.
multipletau_norm = True
multipletau_compress = "average"

for myfile in [
        "2025-05-28-detector-dropout-testing.parquet",
        "2025-05-28-detector-dropout-training.parquet",
        "2025-05-28-detector-dropout-validation.parquet",
        "2025-05-28-peak-artifacts-testing.parquet",
        "2025-05-28-peak-artifacts-training.parquet",
        "2025-05-28-peak-artifacts-validation.parquet",
        "2025-05-28-photobleaching-testing.parquet",
        "2025-05-28-photobleaching-training.parquet",
        "2025-05-28-photobleaching-validation.parquet",
]:
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
