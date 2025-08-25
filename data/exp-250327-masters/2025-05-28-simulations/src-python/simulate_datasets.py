#!/usr/bin/env python3

import datetime
import logging
import logging.config
import os
import sys

import numpy as np
import polars as pl

from pathlib import Path
from typing import Literal

os.chdir("/home/lea/Programs/drmed-git")
FLUOTRACIFY_PATH = Path("/home/lea/Programs/drmed-git/src/")
sys.path.append(FLUOTRACIFY_PATH.as_posix())

from fluotracify import fcsdc
from fluotracify.simulations import simulate_trace_with_artifact as stwa

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
    "loggers": {__name__: {"level": "DEBUG"}},
})
logging.basicConfig(format="%(asctime)s - sim - %(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

TOTAL_SIM_TIME = 16384
TIME_STEP = 1.
PSF_FWHM = 250
PSF_DISTANCE = 4000
BOX_WIDTH = 3000
BOX_HEIGHT = 3000
SIM_LABEL_FOR = "both"
CLEAN_TRAIN = {
    0.05: [50, 100, 125, 250, 500, 1000, 2000, 3000, 3500, 4000],
    0.075: [50, 500, 1000, 4000],
    0.1: [50, 125, 500, 1000, 3000, 4000],
    0.25: [50, 125, 3000, 4000],
    0.5: [50, 125, 500, 1000, 3000, 4000],
    1: [50, 125, 500, 1000, 3000, 4000],
    1.75: [50, 125, 500, 1000, 3000, 4000],
    3: [50, 125, 500, 1000, 3000, 4000],
    5: [50, 125, 3000, 4000],
    10: [50, 125, 500, 1000, 3000, 4000],
    25: [50, 500, 1000, 4000],
    50: [50, 100, 125, 250, 500, 1000, 2000, 3000, 3500, 4000],
}
CLEAN_VAL = {
    0.075: [125, 3000],
    0.1: [100, 250, 2000, 3500],
    0.25: [500, 1000],
    0.5: [100, 250, 2000, 3500],
    1.75: [100, 250, 2000, 3500],
    5: [500, 1000],
    10: [100, 250, 2000, 3500],
    25: [125, 3000],
}
CLEAN_TEST = {
    0.075: [100, 250, 2000, 3500],
    0.25: [100, 250, 2000, 3500],
    1: [100, 250, 2000, 3500],
    3: [100, 250, 2000, 3500],
    5: [100, 250, 2000, 3500],
    25: [100, 250, 2000, 3500],

}
PEAK_PARAMS = {
    0.01: 10,
    0.1: 7,
    1: 3,
}
PEAK_NREPEATS = 16
BLEACH_PARAMS: dict[Literal["mobile", "immobile", "both"], list[float]] = {
    "mobile": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
               0.12, 0.13, 0.14, 0.15, 0.16],
    "immobile": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
    "both": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
             0.12, 0.13, 0.14, 0.15, 0.16],
}
CLEAN_TRAIN_DETDROP = {
    0.05: [1000, 2000, 3000, 3500, 4000],
    0.075: [1000, 4000],
    0.1: [1000, 3000, 4000],
    0.25: [1000, 3000, 4000],
    0.5: [1000, 3000, 4000],
    1: [1000, 4000],
    1.75: [1000, 3000, 4000],
    3: [1000, 4000],
    5: [1000, 3000, 4000],
    10: [1000, 3000, 4000],
    25: [1000, 4000],
    50: [1000, 2000, 3000, 3500, 4000],
}
CLEAN_VAL_DETDROP = {
    0.075: [3000],
    0.1: [2000, 3500],
    0.5: [2000, 3500],
    1: [3000],
    1.75: [2000, 3500],
    3: [3000],
    10: [2000, 3500],
    25: [3000],
}
CLEAN_TEST_DETDROP = {
    0.075: [2000, 3500],
    0.25: [2000, 3500],
    1: [2000, 3500],
    3: [2000, 3500],
    5: [2000, 3500],
    25: [2000, 3500],
}
DETDROP_NREPEATS = 48

rng = np.random.default_rng(seed=42)
outdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
# -------------------------- PEAK ARTIFACTS --------------------------
sim_artifact = "peak_artifacts"
# training data
log.debug("Start simulating training traces with peak artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_TRAIN.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for peak_dmol, peak_nmol in PEAK_PARAMS.items():
            for _ in range(PEAK_NREPEATS):
                sim_params = fcsdc.FCSSimParams(
                    total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                    psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                    box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                    clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                    sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
                    peak_dmol=peak_dmol, peak_nmol=peak_nmol,
                   )
                dc = stwa.perform_simulation(sim_params, rng)
                df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-peak-artifacts-training.parquet")
df.write_parquet(out)
log.debug(f"Saved training traces with peak artifacts as {out}")
# validation data
log.debug("Start simulating validation traces with peak artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_VAL.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for peak_dmol, peak_nmol in PEAK_PARAMS.items():
            for _ in range(PEAK_NREPEATS):
                sim_params = fcsdc.FCSSimParams(
                    total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                    psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                    box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                    clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                    sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
                    peak_dmol=peak_dmol, peak_nmol=peak_nmol,
                   )
                dc = stwa.perform_simulation(sim_params, rng)
                df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-peak-artifacts-validation.parquet")
df.write_parquet(out)
log.debug(f"Saved validation traces with peak artifacts as {out}")
# testing data
log.debug("Start simulating testing traces with peak artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_TEST.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for peak_dmol, peak_nmol in PEAK_PARAMS.items():
            for _ in range(PEAK_NREPEATS):
                sim_params = fcsdc.FCSSimParams(
                    total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                    psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                    box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                    clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                    sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
                    peak_dmol=peak_dmol, peak_nmol=peak_nmol,
                   )
                dc = stwa.perform_simulation(sim_params, rng)
                df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-peak-artifacts-testing.parquet")
df.write_parquet(out)
log.debug(f"Saved testing traces with peak artifacts as {out}")
# -------------------------- DETECTOR DROPOUT --------------------------
sim_artifact = "detector_dropout"
# training data
log.debug("Start simulating training traces with detector dropout artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_TRAIN_DETDROP.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for _ in range(DETDROP_NREPEATS):
            sim_params = fcsdc.FCSSimParams(
                total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
               )
            dc = stwa.perform_simulation(sim_params, rng)
            df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-detector-dropout-training.parquet")
df.write_parquet(out)
log.debug(f"Saved training traces with detector dropout as {out}")
# validation data
log.debug("Start simulating validation traces with detector dropout artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_VAL_DETDROP.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for _ in range(DETDROP_NREPEATS):
            sim_params = fcsdc.FCSSimParams(
                total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
               )
            dc = stwa.perform_simulation(sim_params, rng)
            df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-detector-dropout-validation.parquet")
df.write_parquet(out)
log.debug(f"Saved validation traces with detector dropout as {out}")
# testing data
log.debug("Start simulating testing traces with detector dropout artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_TEST_DETDROP.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for _ in range(DETDROP_NREPEATS):
            sim_params = fcsdc.FCSSimParams(
                total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
               )
            dc = stwa.perform_simulation(sim_params, rng)
            df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-detector-dropout-testing.parquet")
df.write_parquet(out)
log.debug(f"Saved testing traces with detector dropout as {out}")
# -------------------------- PHOTOBLEACHING --------------------------
sim_artifact = "photobleaching"
# training data
log.debug("Start simulating training traces with photobleaching artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_TRAIN.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for bleach_type, bleach_exp_scale_list in BLEACH_PARAMS.items():
            for bleach_exp_scale in bleach_exp_scale_list:
                sim_params = fcsdc.FCSSimParams(
                    total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                    psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                    box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                    clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                    sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
                    bleach_type=bleach_type, bleach_exp_scale=bleach_exp_scale,
                   )
                dc = stwa.perform_simulation(sim_params, rng)
                df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-photobleaching-training.parquet")
df.write_parquet(out)
log.debug(f"Saved training traces with photobleaching as {out}")
# validation data
log.debug("Start simulating validation traces with photobleaching artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_VAL.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for bleach_type, bleach_exp_scale_list in BLEACH_PARAMS.items():
            for bleach_exp_scale in bleach_exp_scale_list:
                sim_params = fcsdc.FCSSimParams(
                    total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                    psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                    box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                    clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                    sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
                    bleach_type=bleach_type, bleach_exp_scale=bleach_exp_scale,
                   )
                dc = stwa.perform_simulation(sim_params, rng)
                df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-photobleaching-validation.parquet")
df.write_parquet(out)
log.debug(f"Saved validation traces with photobleaching as {out}")
# testing data
log.debug("Start simulating testing traces with photobleaching artifacts")
df = pl.DataFrame()
for clean_dmol, clean_nmol_list in CLEAN_TEST.items():
    log.debug(f"Simulating {clean_dmol=}")
    for clean_nmol in clean_nmol_list:
        for bleach_type, bleach_exp_scale_list in BLEACH_PARAMS.items():
            for bleach_exp_scale in bleach_exp_scale_list:
                sim_params = fcsdc.FCSSimParams(
                    total_sim_time=TOTAL_SIM_TIME, time_step=TIME_STEP,
                    psf_fwhm=PSF_FWHM, psf_distance=PSF_DISTANCE,
                    box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                    clean_dmol=clean_dmol, clean_nmol=clean_nmol,
                    sim_artifact=sim_artifact, sim_label_for=SIM_LABEL_FOR,
                    bleach_type=bleach_type, bleach_exp_scale=bleach_exp_scale,
                   )
                dc = stwa.perform_simulation(sim_params, rng)
                df = pl.concat([df, dc.to_polars()], how="vertical")
out = (f"{outdir}/{datetime.date.today()}-photobleaching-testing.parquet")
df.write_parquet(out)
log.debug(f"Saved testing traces with photobleaching as {out}")
