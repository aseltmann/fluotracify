#!/usr/bin/env python3
import lmfit
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
class FCSCorParams:
    method: Literal["multipletau", "tttr2xfcs"]
    multipletau_m: int | None = None
    multipletau_deltat: float | None = None
    multipletau_norm: bool | None = None
    multipletau_compress: Literal["average", "first", "second"] | None = None
    tttr2xfcs_ncascstart: int | None = None
    tttr2xfcs_ncascend: int | None = None
    tttr2xfcs_nsub: int | None = None

    def __post_init__(self):
        if (self.method == "multipletau") & (None in [
                self.multipletau_m, self.multipletau_deltat,
                self.multipletau_norm, self.multipletau_compress]):
            raise ValueError(
                f"if {self.method=}, the multipletau options can not be None."
            )
        elif (self.method == "tttr2xfcs") & (None in [
                self.tttr2xfcs_ncascstart, self.tttr2xfcs_ncascend,
                self.tttr2xfcs_nsub]):
            raise ValueError(
                f"if {self.method=}, the tttr2xfcs options can not be None."
            )

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class FCSFitParams:
    method: Literal["lmfit"]
    params: lmfit.Parameters
    equation_dim: Literal["2D", "3D"]
    equation_dspecies: Literal[1, 2, 3]
    offset: float
    # offset_init: float
    offset_min: float
    offset_max: float
    offset_vary: bool
    gn0: float
    # gn0_init: float
    gn0_min: float
    gn0_max: float
    gn0_vary: bool
    a1: float  # diffusion fraction of species 1
    # a1_init: float
    a1_min: float
    a1_max: float
    a1_vary: bool
    txy1: float  # diffusion time in lateral dimension
    # txy1_init: float
    txy1_min: float
    txy1_max: float
    txy1_vary: bool
    alpha1: float  # anomalous factor
    # alpha1_init: float
    alpha1_min: float
    alpha1_max: float
    alpha1_vary: bool
    equation_diff3d: Literal["none", "tau_z", "aspect_ratio"] = "none"
    equation_triplet: Literal[
        "none", "triplet_ratio", "triplet_fraction"
    ] = "none"
    equation_tspecies: Literal[1, 2, 3] | None = None
    a2: float | None = None
    a2_init: float | None = None
    a2_min: float | None = None
    a2_max: float | None = None
    a2_vary: bool | None = None
    a3: float | None = None
    a3_init: float | None = None
    a3_min: float | None = None
    a3_max: float | None = None
    a3_vary: bool | None = None
    txy2: float | None = None
    txy2_init: float | None = None
    txy2_min: float | None = None
    txy2_max: float | None = None
    txy2_vary: bool | None = None
    txy3: float | None = None
    txy3_init: float | None = None
    txy3_min: float | None = None
    txy3_max: float | None = None
    txy3_vary: bool | None = None
    alpha2: float | None = None
    alpha2_init: float | None = None
    alpha2_min: float | None = None
    alpha2_max: float | None = None
    alpha2_vary: bool | None = None
    alpha3: float | None = None
    alpha3_init: float | None = None
    alpha3_min: float | None = None
    alpha3_max: float | None = None
    alpha3_vary: bool | None = None
    tz1: float | None = None  # diffusion time in axial dimension
    tz1_init: float | None = None
    tz1_min: float | None = None
    tz1_max: float | None = None
    tz1_vary: bool | None = None
    tz2: float | None = None
    tz2_init: float | None = None
    tz2_min: float | None = None
    tz2_max: float | None = None
    tz2_vary: bool | None = None
    tz3: float | None = None
    tz3_init: float | None = None
    tz3_min: float | None = None
    tz3_max: float | None = None
    tz3_vary: bool | None = None
    ar1: float | None = None  # diffusion aspect ratio
    ar1_init: float | None = None
    ar1_min: float | None = None
    ar1_max: float | None = None
    ar1_vary: bool | None = None
    ar2: float | None = None
    ar2_init: float | None = None
    ar2_min: float | None = None
    ar2_max: float | None = None
    ar2_vary: bool | None = None
    ar3: float | None = None
    ar3_init: float | None = None
    ar3_min: float | None = None
    ar3_max: float | None = None
    ar3_vary: bool | None = None
    b1: float | None = None  # triplet ratio
    b1_init: float | None = None
    b1_min: float | None = None
    b1_max: float | None = None
    b1_vary: bool | None = None
    b2: float | None = None
    b2_init: float | None = None
    b2_min: float | None = None
    b2_max: float | None = None
    b2_vary: bool | None = None
    b3: float | None = None
    b3_init: float | None = None
    b3_min: float | None = None
    b3_max: float | None = None
    b3_vary: bool | None = None
    t1: float | None = None  # triplet fraction
    t1_init: float | None = None
    t1_min: float | None = None
    t1_max: float | None = None
    t1_vary: bool | None = None
    t2: float | None = None
    t2_init: float | None = None
    t2_min: float | None = None
    t2_max: float | None = None
    t2_vary: bool | None = None
    t3: float | None = None
    t3_init: float | None = None
    t3_min: float | None = None
    t3_max: float | None = None
    t3_vary: bool | None = None
    taut1: float | None = None  # triplet time
    taut1_init: float | None = None
    taut1_min: float | None = None
    taut1_max: float | None = None
    taut1_vary: bool | None = None
    taut2: float | None = None
    taut2_init: float | None = None
    taut2_min: float | None = None
    taut2_max: float | None = None
    taut2_vary: bool | None = None
    taut3: float | None = None
    taut3_init: float | None = None
    taut3_min: float | None = None
    taut3_max: float | None = None
    taut3_vary: bool | None = None


    def __post_init__(self):
        if not {"offset", "gn0", "a1", "txy1", "alpha1"}.issubset(
                self.params.keys()):
            raise ValueError(
                "for any fit, set 'offset', 'gn0', 'a1', 'txy1', and 'alpha1'"
            )
        if (self.equation_dspecies == 1) & (not set([
                self.a2, self.a3, self.txy2, self.txy3, self.alpha2, self.alpha3
        ]) == set([None])):
            raise ValueError(
                "for a 1 species fit, don't set a2, a3, txy2, txy3, alpha2, "
                "alpha3"
            )
        elif ((self.equation_dspecies == 2) &
              ((not set([self.a3, self.txy3, self.alpha3]) == set([None])) |
               (None in [self.a2, self.txy2, self.alpha2]))):
            raise ValueError(
                "for a 2 species fit, set a2, txy2 and alpha2, but don't set "
                "a3, txy3, alpha3"
            )
        elif (self.equation_dspecies == 3) & (
                None in [self.a2, self.a3, self.txy2, self.txy3, self.alpha2,
                         self.alpha3]):
            raise ValueError(
                "for a 3 species fit, set a1, a2, a3, txy1, txy2, txy3, alpha1",
                "alpha2 and alpha3"
            )

        if (self.equation_dim == "2D") & (self.equation_diff3d != "none"):
            raise ValueError(
                f"if {self.equation_dim=}, don't set equation_diff3d"
            )
        elif (self.equation_dim == "3D"):
            if self.equation_diff3d == "none":
                raise ValueError(
                    f"if {self.equation_dim=}, set equation_diff3d"
                )
            elif self.equation_diff3d == "tau_z":
                if not set([self.ar1, self.ar2, self.ar3]) == set([None]):
                    raise ValueError(
                        "for a 3D tau_z fit, don't set ar1, ar2 or ar3"
                    )
                if (self.equation_dspecies == 1) & (
                        (self.tz1 is None) | (self.tz2 is not None) |
                        (self.tz3 is not None)):
                    raise ValueError(
                        "for a 1 species 3D tau_z fit, set tz1, but don't set "
                        "tz2 and tz3"
                    )
                elif (self.equation_dspecies == 2) & (
                        (self.tz1 is None) | (self.tz2 is None) |
                        (self.tz3 is not None)):
                    raise ValueError(
                        "for a 2 species 3D tau_z fit, set tz1 and tz2, but "
                        "don't set tz3"
                    )
                elif (self.equation_dspecies == 3) & (
                        (None in [self.tz1, self.tz2, self.tz3])):
                    raise ValueError(
                        "for a 3 species 3D tau_z fit, set tz1, tz2 and tz3"
                    )
            elif self.equation_diff3d == "aspect_ratio":
                if not set([self.tz1, self.tz2, self.tz3]) == set([None]):
                    raise ValueError(
                        "for a 3D aspect ratio fit, don't set tz1, tz2 or tz3"
                    )
                if (self.equation_dspecies == 1) & (
                        (self.ar1 is None) | (self.ar2 is not None) |
                        (self.ar2 is not None)):
                    raise ValueError(
                        "for a 1 species 3D aspect ratio fit, set ar1, but "
                        "don't set ar2 and ar3"
                    )
                elif (self.equation_dspecies == 2) & (
                        (self.ar1 is None) | (self.ar2 is None) |
                        (self.ar3 is not None)):
                    raise ValueError(
                        "for a 2 species 3D aspect ratio fit, set ar1 and ar2, "
                        "but don't set ar3"
                    )
                elif ((self.equation_dspecies == 3) &
                      (None in [self.ar1, self.ar2, self.ar3])):
                    raise ValueError(
                        "for a 3 species 3D aspect ratio fit, set ar1, ar2 and "
                        "ar3 "
                    )
        if (self.equation_triplet == "none") & (not set([
                self.equation_tspecies, self.b1, self.b2, self.b3, self.t1,
                self.t2, self.t3, self.taut1, self.taut2, self.taut3
        ]) == set([None])):
            raise ValueError(
                f"If {self.equation_triplet=}, don't set equation_tspecies, b1,"
                "b2, b3, t1, t2, t3, taut1, taut2 or taut3"
            )
        else:
            if (self.equation_tspecies == 1) & (
                    (self.taut1 is None) | (self.taut2 is not None) |
                    (self.taut3 is not None)):
                raise ValueError(
                    "for a 1 species triplet fit, set taut1, but don't set "
                    "taut2 and taut3")

            elif (self.equation_tspecies == 2) & (
                    (self.taut1 is None) | (self.taut2 is None) |
                    (self.taut3 is not None)):
                raise ValueError(
                    "for a 2 species triplet fit, set taut1 and taut2, but "
                    "don't set taut3"
                )
            elif (self.equation_tspecies == 3) & (
                    None in [self.taut1, self.taut2, self.taut3]):
                raise ValueError(
                    "for a 3 species triplet fit, set taut1, taut2, and taut3"
                )
            if self.equation_triplet == "triplet_ratio":
                if ((self.equation_tspecies == 1) &
                    ((self.b1 is None) | (not set([
                        self.b2, self.b3, self.t1, self.t2, self.t3
                    ]) == set([None])))):
                    raise ValueError(
                        "for a 1 species triplet ratio equation fit, set b1, "
                        "but don't set b2, b3, t1, t2 and t3"
                    )
                elif ((self.equation_tspecies == 2) &
                      ((None in [self.b1, self.b2]) |
                       (not set([self.b3, self.t1, self.t2, self.t3]) ==
                        set([None])))):
                    raise ValueError(
                        "for a 2 species triplet ratio equation fit, set b1 "
                        "and b2, but don't set b3, t1, t2, and t3"
                    )
                elif ((self.equation_tspecies == 3) &
                      (None in [self.b1, self.b2, self.b3]) |
                      (not set([self.t1, self.t2, self.t3]) == set([None]))):
                    raise ValueError(
                        "for a 3 species triplet ratio equation fit, set b1, "
                        "b2, and b3, but don't set t1, t2 and t3"
                    )
            elif self.equation_triplet == "triplet_fraction":
                if ((self.equation_tspecies == 1) &
                    ((self.t1 is None) | (not set([
                        self.t2, self.t3, self.b1, self.b2, self.b3
                    ]) == set([None])))):
                    raise ValueError(
                        "for a 1 species triplet fraction equation fit, set "
                        "t1, but don't set t2, t3, b1, b2 and b3"
                    )
                elif ((self.equation_tspecies == 2) &
                      ((None in [self.t1, self.t2]) |
                       (not set([self.t3, self.b1, self.b2, self.b3]) ==
                        set([None])))):
                    raise ValueError(
                        "for a 2 species triplet fraction equation fit, set t1 "
                        "and t2, but don't set t3, b1, b2, and b3"
                    )
                elif ((self.equation_tspecies == 3) &
                      (None in [self.t1, self.t2, self.t3]) |
                      (not set([self.b1, self.b2, self.b3]) == set([None]))):
                    raise ValueError(
                        "for a 3 species triplet fraction equation fit, set "
                        "t1, t2, and t3, but don't set b1, b2 and b3"
                    )

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k not in "lmfit_params"}

    def get_equation(self, param, tc):
        """Returns output of theoretical FCS equations for fitting
        autocorrelation functions given the parameters of the dataclass

        Parameters
        ----------
        tc: lag time tau
        """
        if self.equation_dim == "2D":
            if self.equation_dspecies == 1:
                gdiff = param["A1"].value * ((1 + ((tc / param["txy1"].value)**param["alpha1"].value))**-1)
        if self.equation_triplet == "none":
            gt = 1
        return param["offset"].value + (param["GN0"].value * gdiff * gt)

    def get_residual(self, param, tc, cor):
        equ = self.get_equation(param, tc)
        return cor - equ




@dataclass
class SimulatedFCSCorrelationAndFit():
    """Multipletau correlation and lmfit fit of simulated FCS time-series based
    on brownian motion / random walk of a given number of molecules with a
    given diffusion coefficient.
    """
    uuid: uuid.UUID
    sim_params: FCSSimParams
    corr_params: FCSCorParams
    fit_params: FCSFitParams



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
