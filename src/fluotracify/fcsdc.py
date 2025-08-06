#!/usr/bin/env python3
import logging
import numpy as np
import polars as pl
import uuid

from dataclasses import dataclass, field, astuple, asdict
from typing import Literal, Any

logging.basicConfig(format="%(asctime)s - fcsdc - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def photon_counting_stats(
    time_series: np.ndarray, time_series_scale: np.ndarray
) -> tuple[float, float, float]:
    """returns counting statistics

    Notes
    -----
    - code is adopted from Dominic Waithe's Focuspoint package:
    https://github.com/dwaithe/FCS_point_correlator/blob/master/focuspoint/correlation_objects.py
    """
    unit = time_series_scale[-1] / len(time_series_scale)
    # Converts to counts per
    kcount_ch = np.average(time_series)
    # This is the unnormalised intensity count for int_time duration (the first
    # moment)
    raw_count = np.average(time_series)
    var_count = np.var(time_series)

    brightness_nandb_ch = ((var_count - raw_count) / (raw_count)) / (float(unit))
    if (var_count - raw_count) == 0:
        number_nandb_ch = 0
    else:
        number_nandb_ch = raw_count**2 / (var_count - raw_count)
    return float(kcount_ch), float(brightness_nandb_ch), float(number_nandb_ch)


def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays

    from https://stackoverflow.com/questions/51743827/how-to-compare-equality-of-dataclasses-holding-numpy-ndarray-boola-b-raises
    """
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    try:
        return a == b
    except TypeError:
        return NotImplemented


def dc_eq(dc1, dc2) -> bool:
    """checks if two dataclasses which hold numpy arrays are equal

    from https://stackoverflow.com/questions/51743827/how-to-compare-equality-of-dataclasses-holding-numpy-ndarray-boola-b-raises
    """
    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:
        return NotImplemented  # better than False
    t1 = astuple(dc1)
    t2 = astuple(dc2)
    return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))


@dataclass
class FCSPhotonDecay:
    name: str
    channel: int
    bin: float
    original: np.ndarray = field(default_factory=np.zeros(1), compare=False)
    scale: np.ndarray = field(default_factory=np.zeros(1), compare=False)
    no_offset: np.ndarray | None = field(default=None, compare=False)
    normalized: np.ndarray | None = field(default=None, compare=False)


@dataclass
class ProcessedFCSPhotonDecay(FCSPhotonDecay):
    processing_prediction: Literal["none", "threshold", "unet"] = "none"
    processing_scaling: Literal[
        "none", "standard", "robust", "maxabs", "quant_g", "minmax", "l1", "l2"
    ] = "none"
    processing_correction: Literal[
        "none",
        "set_to_zero",
        "cut_and_stitch",
        "averaging",
        "random_weights",
        "1-pred_weights",
        "constant_weight",
    ] = "none"


@dataclass
class FCSCorrelation:
    name: str
    method: Literal["tttr2xfcs", "multipletau"]
    channel1: int
    channel2: int
    count1: int = field(default=0, compare=False)
    count2: int = field(default=0, compare=False)
    kcount: int | float | None = None
    brightness_nandb: int | float | None = None
    number_nandb: int | float | None = None
    autotime: np.ndarray = field(default_factory=np.zeros(1), compare=False)
    autonorm: np.ndarray = field(default_factory=np.zeros(1), compare=False)


@dataclass
class ProcessedFCSCorrelation(FCSCorrelation):
    processing_prediction: Literal["none", "threshold", "unet"] = "none"
    processing_scaling: Literal[
        "none", "standard", "robust", "maxabs", "quant_g", "minmax", "l1", "l2"
    ] = "none"
    processing_correction: Literal[
        "none",
        "set_to_zero",
        "cut_and_stitch",
        "averaging",
        "random_weights",
        "1-pred_weights",
        "constant_weight",
    ] = "none"


@dataclass(eq=False, kw_only=True)
class FCSTimeSeries:
    """Holds exactly one FCS time-series including"""

    name: str
    channel: int
    bin: float
    size: int
    trace: np.ndarray
    scale: np.ndarray
    kcount: int | float | None = field(init=False)
    brightness_nandb: int | float | None = field(init=False)
    number_nandb: int | float | None = field(init=False)
    correlation: FCSCorrelation | ProcessedFCSCorrelation | None = None

    def __eq__(self, other):
        return dc_eq(self, other)

    def __post_init__(self):
        # counting statistics
        self.kcount, self.brightness_nandb, self.number_nandb = photon_counting_stats(
            self.trace, self.scale
        )

        log.debug(
            "FCSTimeSeries: kcount: %s, brightness: %s, number: %s",
            self.kcount,
            self.brightness_nandb,
            self.number_nandb,
        )

class FCSTimeSeriesLabel(FCSTimeSeries):
    kcount: int | float | None = None
    brightness_nandb: int | float | None = None
    number_nandb: int | float | None = None
    def __post_init__(self):
        pass

@dataclass
class ProcessedFCSTimeSeries(FCSTimeSeries):
    processing_prediction: Literal["none", "threshold", "unet"] = "none"
    processing_scaling: Literal[
        "none", "standard", "robust", "maxabs", "quant_g", "minmax", "l1", "l2"
    ] = "none"
    processing_correction: Literal[
        "none",
        "set_to_zero",
        "cut_and_stitch",
        "averaging",
        "random_weights",
        "1-pred_weights",
        "constant_weight",
    ] = "none"


@dataclass
class FCSSimParams:
    total_sim_time: int
    time_step: int | float
    psf_fwhm: int | float
    psf_distance: int | float
    box_width: int | float
    box_height: int | float
    clean_dmol: float
    clean_nmol: int
    sim_artifact: Literal[
        "none", "peak_artifacts", "detector_dropout", "photobleaching"
    ] = "none"
    sim_label_for: Literal["none", "restoration", "segmentation", "both"] = "none"
    pos_x: int = field(init=False)
    pos_y: int = field(init=False)
    bleach_type: Literal["immobile", "mobile", "both"] | None = None
    bleach_exp_scale: float | None = None
    dropout_n: int | None = None
    dropout_maxdrop: float | None = None
    peak_dmol: float | None = None
    peak_nmol: int | None = None
    peak_brightness: int | None = None

    def __post_init__(self):
        self.pos_x = int(self.box_width // 2)
        self.pos_y = int(self.box_height // 2)

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class SimulatedFCSTimeSeries():
    """Simulated FCS time-series based on brownian motion / random walk of a
    given number of molecules. Also supports artifacts
    """
    uuid: uuid.UUID
    sim_params: FCSSimParams
    record: dict[Literal["feature", "label_restoration", "label_segmentation"],
                 FCSTimeSeries | FCSTimeSeriesLabel] = (
        field(default_factory=dict, compare=False)
    )
    def to_polars(self):
        ts_schema = {}
        for key, rec in self.record.items():
            ts_schema = ts_schema | {
                key: pl.Array(pl.Float32, shape=(rec.size))
            }

        out = pl.DataFrame(
            {"uuid": str(self.uuid)} |
            {k: [v.trace] for k, v in self.record.items()} |
            {"sim_params": self.sim_params.to_dict()},
            schema={
                "uuid": pl.String
            } | ts_schema | {
                "sim_params": pl.Struct({
                    "total_sim_time": pl.Float32, "time_step": pl.Float32,
                    "psf_fwhm": pl.Float32, "box_width": pl.UInt32,
                    "box_height": pl.UInt32, "clean_dmol": pl.Float32,
                    "clean_nmol": pl.UInt32, "sim_artifact": pl.String,
                    "sim_label_for": pl.String, "pos_x": pl.UInt32,
                    "pos_y": pl.UInt32, "bleach_type": pl.String,
                    "bleach_exp_scale": pl.Float32, "dropout_n": pl.UInt32,
                    "dropout_maxdrop": pl.Float32, "peak_dmol": pl.Float32,
                    "peak_nmol": pl.UInt32, "peak_brightness": pl.UInt32,
                })
            }
        )
        return out




@dataclass
class FCSTimeSeriesCollection:
    name: str
    channel: int
    bin: float
    uuid: str
    pred_thresh: float | None = None
    record: dict[
        Literal[
            "original",
            "preprocessed",
            "predictions",
            "set_to_zero",
            "cut_and_stitch",
            "averaging",
            "random_weights",
            "1-pred_weights",
            "constant_weight",
        ],
        FCSTimeSeries | SimulatedFCSTimeSeries | ProcessedFCSTimeSeries,
    ] = field(default_factory=dict, compare=False)


@dataclass
class TCSPC:
    name: str
    resolution: float
    glob_res: float
    ptu_tags: list[tuple[str, Any]]
    ptu_num_records: int
    ch_present: np.ndarray = field(init=False, compare=False)
    num_ch: int = field(init=False)
    channels: np.ndarray = field(compare=False)
    macrotimes: np.ndarray = field(metadata={"unit": "ms"}, compare=False)
    microtimes: np.ndarray = field(metadata={"unit": "ns"}, compare=False)
    kcount: int | float | None = None
    brightness_nandb: int | float | None = None
    number_nandb: int | float | None = None
    correlations: list = field(default_factory=list, compare=False)
    photon_count_decays: list = field(default_factory=list, compare=False)

    def __post_init__(self):
        # How many channels there are in the files.
        ch_present = np.sort(np.unique(self.channels))
        num_ch = len(ch_present)
        for i in range(num_ch - 1, -1, -1):
            if ch_present[i] > 8:
                ch_present = np.delete(ch_present, i)

        log.debug("TCSPC: this file has %s channel(s): %s", num_ch, ch_present)
        self.ch_present = ch_present
        self.num_ch = num_ch


@dataclass
class ProcessedTCSPC(TCSPC):
    weights: np.ndarray | None = field(default=None, compare=False)
    channels_parts: list[np.ndarray] | None = field(default=None, compare=False)
    macrotimes_parts: list[np.ndarray] | None = field(default=None, compare=False)
    processing_prediction: Literal["none", "threshold", "unet"] = "none"
    processing_scaling: Literal[
        "none", "standard", "robust", "maxabs", "quant_g", "minmax", "l1", "l2"
    ] = "none"
    processing_correction: Literal[
        "none",
        "set_to_zero",
        "cut_and_stitch",
        "averaging",
        "random_weights",
        "1-pred_weights",
        "constant_weight",
    ] = "none"
    processing_bin: float | None = None
    processing_pred_thresh: float | None = None
