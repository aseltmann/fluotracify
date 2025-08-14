#!/usr/bin/env python3
import datetime
import lmfit
import logging
import multipletau
import numpy as np
import numpy.typing as npt
import polars as pl
import uuid as uuid_module

from dataclasses import dataclass, field, astuple, asdict
from pprint import pprint
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
    date: datetime.date | datetime.datetime
    channel: int
    bin: float
    size: int
    trace: np.ndarray
    scale: np.ndarray
    kcount: int | float | None = field(init=False)
    brightness_nandb: int | float | None = field(init=False)
    number_nandb: int | float | None = field(init=False)

    def __eq__(self, other):
        return dc_eq(self, other)

    def __post_init__(self):
        if isinstance(self.date, datetime.datetime):
            self.date = self.date.date()
        self.kcount, self.brightness_nandb, self.number_nandb = photon_counting_stats(
            self.trace, self.scale
        )

        # log.debug(
        #     "FCSTimeSeries: kcount: %s, brightness: %s, number: %s",
        #     self.kcount,
        #     self.brightness_nandb,
        #     self.number_nandb,
        # )
    @classmethod
    def from_polars(cls, df: pl.DataFrame):
        par = {k: v.to_numpy()[0]
               for k, v in df.to_dict().items() if k in ["scale", "trace"]}
        par = par | {k: v for k, v in df["ts_params"].item().items()
                     if not k in ["kcount", "brightness_nandb", "number_nandb"]}
        return cls(**par)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_polars(self) -> pl.DataFrame:
        ts_params = {k: v for k, v in self.to_dict().items()
                     if k not in ["scale", "trace"]}
        ts_params_schema = {
            "bin": pl.Float32, "channel": pl.UInt32, "date": pl.Date,
            "name": pl.String, "size": pl.UInt32, "kcount": pl.Float32,
            "brightness_nandb": pl.Float32, "number_nandb": pl.Float32,
        }
        out = pl.DataFrame(
            {"scale": [self.scale], "trace": [self.trace],
             "ts_params": ts_params},
            schema={
                "scale": pl.Array(pl.Float32, shape=(self.size)),
                "trace": pl.Array(pl.Float32, shape=(self.size)),
                "ts_params": pl.Struct(ts_params_schema)
            }
        )
        return out


class FCSTimeSeriesLabel(FCSTimeSeries):
    kcount: int | float | None = None
    brightness_nandb: int | float | None = None
    number_nandb: int | float | None = None
    def __post_init__(self):
        if isinstance(self.date, datetime.datetime):
            self.date = self.date.date()

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

    @classmethod
    def from_polars(cls, ser: pl.Series):
        # read in from polars Series which was exported with the to_polars()
        # method
        par = {k: v for k, v in ser.item().items()
               if k not in ["pos_x", "pos_y"]}
        # cleanup floating point errors
        par["clean_dmol"] = round(par["clean_dmol"], 3)
        par["peak_dmol"] = (None if par["peak_dmol"] is None
                            else round(par["peak_dmol"], 3))
        par["bleach_exp_scale"] = (None if par["bleach_exp_scale"] is None
                                   else round(par["bleach_exp_scale"], 2))
        return cls(**par)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

    def to_polars(self) -> pl.DataFrame:
        schema = {
             "total_sim_time": pl.Float32, "time_step": pl.Float32,
             "psf_fwhm": pl.Float32, "psf_distance": pl.Float32,
             "box_width": pl.UInt32, "box_height": pl.UInt32,
             "clean_dmol": pl.Float32, "clean_nmol": pl.UInt32,
             "sim_artifact": pl.String, "sim_label_for": pl.String,
             "pos_x": pl.UInt32, "pos_y": pl.UInt32, "bleach_type": pl.String,
             "bleach_exp_scale": pl.Float32, "dropout_n": pl.UInt32,
             "dropout_maxdrop": pl.Float32, "peak_dmol": pl.Float32,
             "peak_nmol": pl.UInt32, "peak_brightness": pl.UInt32,
        }
        out = pl.DataFrame(
            {"sim_params": self.to_dict()},
            schema={"sim_params": pl.Struct(schema)}
        )
        return out

@dataclass
class SimulatedFCSTimeSeries:
    """Simulated FCS time-series based on brownian motion / random walk of a
    given number of molecules. Also supports artifacts
    """
    uuid: uuid_module.UUID
    sim_params: FCSSimParams
    record: dict[Literal["feature", "label_restoration", "label_segmentation"],
                 FCSTimeSeries | FCSTimeSeriesLabel] = (
        field(default_factory=dict, compare=False)
    )
    def to_polars(self) -> pl.DataFrame:
        r = self.record.items()
        rec = [v.to_polars().select("trace").rename({"trace": k}) for k, v in r]
        if "feature" in self.record.keys():
            ts_params = self.record["feature"].to_polars().select("ts_params")
        else:
            ts_params = [v.to_polars().select("ts_params") for _, v in r][0]
        out = pl.DataFrame({"uuid": str(self.uuid)}, schema={"uuid": pl.String})
        out = pl.concat(
            [out, *rec, ts_params, self.sim_params.to_polars()],
            how="horizontal"
        )
        return out


@dataclass
class FCSCor:
    uuid: uuid_module.UUID
    method: Literal["multipletau", "tttr2xfcs"]
    multipletau_m: int | None = None
    multipletau_deltat: float | None = None
    multipletau_norm: bool | None = None
    multipletau_compress: Literal["average", "first", "second"] | None = None
    tttr2xfcs_ncascstart: int | None = None
    tttr2xfcs_ncascend: int | None = None
    tttr2xfcs_nsub: int | None = None
    tc: np.ndarray | None = None
    g: np.ndarray | None = None

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

    @classmethod
    def from_polars(cls, df: pl.DataFrame):
        par = {k: v.to_numpy()[0]
               for k, v in df.to_dict().items() if k in ["tc", "g"]}
        par = par | {k: v for k, v in df["cor_params"].item().items()}
        return cls(**par)

    def autocorrelate(
            self, input: np.ndarray
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if self.method == "multipletau":
            assert self.multipletau_m is not None
            assert self.multipletau_deltat is not None
            assert self.multipletau_norm is not None
            assert self.multipletau_compress is not None
            cor = multipletau.autocorrelate(
                input, m=self.multipletau_m,
                deltat=self.multipletau_deltat, normalize=self.multipletau_norm,
                compress=self.multipletau_compress
            )
            assert isinstance(cor, np.ndarray)
            self.tc = np.array(cor[1:, 0], dtype=np.float32)
            self.g = np.array(cor[1:, 1], dtype=np.float32)
        elif self.method == "tttr2xfcs":
            raise NotImplementedError("tttr2xfcs is not implemented yet")
        else:
            raise ValueError("method has to be 'multipletau' or 'tttr2xfcs'")
        return self.tc, self.g

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

    def to_polars(self) -> pl.DataFrame:
        cor_params = {k: v for k, v in self.to_dict().items()
                      if k not in ["tc", "g"]}
        cor_params_schema = {
            "uuid": pl.String, "method": pl.String, "multipletau_m": pl.UInt32,
            "multipletau_deltat": pl.Float32, "multipletau_norm": pl.Boolean,
            "multipletau_compress": pl.String, "tttr2xfcs_nsub": pl.UInt32,
            "tttr2xfcs_ncascstart": pl.UInt32, "tttr2xfcs_ncascend": pl.UInt32,
        }
        cor_schema = (pl.Null if self.g is None
                      else pl.Array(pl.Float32, shape=(self.g.size)))
        out = pl.DataFrame(
            {"tc": [self.tc], "g": [self.g], "cor_params": cor_params},
            schema={"tc": cor_schema, "g": cor_schema,
                    "cor_params": pl.Struct(cor_params_schema)}
        )
        return out

@dataclass
class SimulatedFCSTimeSeriesCor:
    uuid: uuid_module.UUID
    sim_params: FCSSimParams
    record: dict[Literal["feature", "label_restoration", "label_segmentation"],
                 FCSCor] = (
        field(default_factory=dict, compare=False)
    )
    def to_polars(self) -> pl.DataFrame:
        r = self.record.items()
        g = [v.to_polars().select("g").rename({"g": f"{k}_g"}) for k, v in r]
        tc = [v.to_polars().select("tc") for _, v in r][0]
        cor_params = [v.to_polars().select("cor_params") for _, v in r][0]
        out = pl.DataFrame({"uuid": str(self.uuid)}, schema={"uuid": pl.String})
        out = pl.concat(
            [out, tc, *g, cor_params, self.sim_params.to_polars()],
            how="horizontal"
        )
        return out



@dataclass
class FCSFit:
    uuid: uuid_module.UUID
    method: Literal["lmfit"]
    params: lmfit.Parameters
    equation_dim: Literal["2D", "3D"]
    equation_dspecies: Literal[1, 2, 3]
    equation_diff3d: Literal["none", "tau_z", "aspect_ratio"] = "none"
    equation_triplet: Literal[
        "none", "triplet_ratio", "triplet_fraction"
    ] = "none"
    equation_tspecies: Literal[1, 2, 3] | None = None
    result_minimizer: lmfit.minimizer.MinimizerResult | None = None
    tc: np.ndarray | None = None
    g: np.ndarray | None = None
    residual: np.ndarray | None = None

    def __post_init__(self):

        k = self.params.keys()
        if not set(k).issubset({
                "offset", "gn0", "a1", "a2", "a3", "txy1", "txy2", "txy3",
                "alpha1", "alpha2", "alpha3", "ar1", "ar2", "ar3", "tz1", "tz2",
                "tz3", "b1", "b2", "b3", "t1", "t2", "t3", "taut1", "taut2",
                "taut3"
        }):
            raise ValueError(
                f"At least one values of {self.params.keys()=} is unsupported. "
                "Only use: 'offset', 'gn0', 'a1', 'a2', 'a3', 'txy1', 'txy2', "
                "'txy3', 'alpha1', 'alpha2', 'alpha3', 'ar1', 'ar2', 'ar3', "
                "'tz1', 'tz2', 'tz3', 'b1', 'b2', 'b3', 't1', 't2', 't3', "
                "'taut1', 'taut2', 'taut3'"
            )
        if not {"offset", "gn0", "a1", "txy1", "alpha1"}.issubset(k):
            raise ValueError(
                "for any fit, set 'offset', 'gn0', 'a1', 'txy1', and 'alpha1'"
            )
        if self.equation_dspecies not in [1, 2, 3]:
            raise ValueError("set equation_dspecies to 1, 2 or 3")
        elif (self.equation_dspecies == 1) & (not {
                "a2", "a3", "txy2", "txy3", "alpha2", "alpha3"}.isdisjoint(k)):
            raise ValueError(
                "for a 1 species fit, don't set a2, a3, txy2, txy3, alpha2, "
                "alpha3"
            )
        elif (self.equation_dspecies == 2) & (
                (not {"a3", "txy3", "alpha3"}.isdisjoint(k)) |
                (not {"a2", "txy2", "alpha2"}.issubset(k))):
            raise ValueError(
                "for a 2 species fit, set a2, txy2 and alpha2, but don't set "
                "a3, txy3, alpha3"
            )
        elif (self.equation_dspecies == 3) & (not {
                "a2", "a3", "txy2", "txy3", "alpha2", "alpha3"}.issubset(k)):
            raise ValueError(
                "for a 3 species fit, set a1, a2, a3, txy1, txy2, txy3, "
                "alpha1, alpha2 and alpha3"
            )
        if self.equation_dim not in ["2D", "3D"]:
            raise ValueError("set equation_dim to '2D' or '3D'")
        elif (self.equation_dim == "2D") & (self.equation_diff3d != "none"):
            raise ValueError(
                f"if {self.equation_dim=}, don't set equation_diff3d"
            )
        elif (self.equation_dim == "3D"):
            if self.equation_diff3d not in ["none", "tau_z", "aspect_ratio"]:
                raise ValueError(
                    "set equation_diff3d to 'none', 'tau_z' or 'aspect_ratio'"
                )
            if self.equation_diff3d == "none":
                raise ValueError(
                    f"if {self.equation_dim=}, set equation_diff3d"
                )
            elif self.equation_diff3d == "tau_z":
                if not {"ar1", "ar2", "ar3"}.isdisjoint(k):
                    raise ValueError(
                        "for a 3D tau_z fit, don't set ar1, ar2 or ar3"
                    )
                if (self.equation_dspecies == 1) & (("tz1" not in k) | (
                        not {"tz2", "tz3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 1 species 3D tau_z fit, set tz1, but don't set "
                        "tz2 and tz3"
                    )
                elif (self.equation_dspecies == 2) & (("tz3" in k) | (
                        not {"tz1", "tz2"}.issubset(k))):
                    raise ValueError(
                        "for a 2 species 3D tau_z fit, set tz1 and tz2, but "
                        "don't set tz3"
                    )
                elif (self.equation_dspecies == 3) & (
                        not {"tz1", "tz2", "tz3"}.issubset(k)):
                    raise ValueError(
                        "for a 3 species 3D tau_z fit, set tz1, tz2 and tz3"
                    )
            elif self.equation_diff3d == "aspect_ratio":
                if not {"tz1", "tz2", "tz3"}.isdisjoint(k):
                    raise ValueError(
                        "for a 3D aspect ratio fit, don't set tz1, tz2 or tz3"
                    )
                if (self.equation_dspecies == 1) & (("ar1" not in k) | (
                        not {"ar2", "ar3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 1 species 3D aspect ratio fit, set ar1, but "
                        "don't set ar2 and ar3"
                    )
                elif (self.equation_dspecies == 2) & (("ar3" in k) | (
                        not {"ar1", "ar2"}.issubset(k))):
                    raise ValueError(
                        "for a 2 species 3D aspect ratio fit, set ar1 and ar2, "
                        "but don't set ar3"
                    )
                elif (self.equation_dspecies == 3) & (
                        not {"ar1", "ar2", "ar3"}.issubset(k)):
                    raise ValueError(
                        "for a 3 species 3D aspect ratio fit, set ar1, ar2 and "
                        "ar3 "
                    )
        if (self.equation_triplet not in
            ["none", "triplet_ratio", "triplet_fraction"]):
            raise ValueError(
                "set equation_triplet to 'none', 'triplet_ratio' or "
                "'triplet_fraction'"
            )
        elif (self.equation_triplet == "none") & (
                (not {"b1", "b2", "b3", "t1", "t2", "t3", "taut1", "taut2",
                      "taut3"}.isdisjoint(k)) |
                (self.equation_tspecies is not None)):
            raise ValueError(
                f"If {self.equation_triplet=}, don't set equation_tspecies, b1,"
                "b2, b3, t1, t2, t3, taut1, taut2 or taut3"
            )
        elif self.equation_triplet in ["triplet_ratio", "triplet_fraction"]:
            if self.equation_tspecies not in [1, 2, 3]:
                raise ValueError(
                    f"If {self.equation_triplet=}, set equation_tspecies to "
                    "1, 2 or 3"
                )
            if (self.equation_tspecies == 1) & (("taut1" not in k) | (
                    not {"taut2", "taut3"}.isdisjoint(k))):
                raise ValueError(
                    "for a 1 species triplet fit, set taut1, but don't set "
                    "taut2 and taut3"
                )
            elif (self.equation_tspecies == 2) & (("taut3" in k) | (
                    not {"taut1", "taut2"}.issubset(k))):
                raise ValueError(
                    "for a 2 species triplet fit, set taut1 and taut2, but "
                    "don't set taut3"
                )
            elif (self.equation_tspecies == 3) & (
                    not {"taut1", "taut2", "taut3"}.issubset(k)):
                raise ValueError(
                    "for a 3 species triplet fit, set taut1, taut2, and taut3"
                )
            if self.equation_triplet == "triplet_ratio":
                if (self.equation_tspecies == 1) & (("b1" not in k) | (
                        not {"b2", "b3", "t1", "t2", "t3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 1 species triplet ratio equation fit, set b1, "
                        "but don't set b2, b3, t1, t2 and t3"
                    )
                elif (self.equation_tspecies == 2) & (
                        (not {"b1", "b2"}.issubset(k)) |
                        (not {"b3", "t1", "t2", "t3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 2 species triplet ratio equation fit, set b1 "
                        "and b2, but don't set b3, t1, t2, and t3"
                    )
                elif (self.equation_tspecies == 3) & (
                      (not {"b1", "b2", "b3"}.issubset(k)) |
                      (not {"t1", "t2", "t3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 3 species triplet ratio equation fit, set b1, "
                        "b2, and b3, but don't set t1, t2 and t3"
                    )
            elif self.equation_triplet == "triplet_fraction":
                if (self.equation_tspecies == 1) & (("t1" not in k) | (
                        not {"t2", "t3", "b1", "b2", "b3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 1 species triplet fraction equation fit, set "
                        "t1, but don't set t2, t3, b1, b2 and b3"
                    )
                elif (self.equation_tspecies == 2) & (
                        (not {"t1", "t2"}.issubset(k)) |
                        (not {"t3", "b1", "b2", "b3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 2 species triplet fraction equation fit, set t1 "
                        "and t2, but don't set t3, b1, b2, and b3"
                    )
                elif (self.equation_tspecies == 3) & (
                        (not {"t1", "t2", "t3"}.issubset(k)) |
                        (not {"b1", "b2", "b3"}.isdisjoint(k))):
                    raise ValueError(
                        "for a 3 species triplet fraction equation fit, set "
                        "t1, t2, and t3, but don't set b1, b2 and b3"
                    )

    def get_equation(
            self, param: lmfit.Parameters, tc: np.ndarray
    ) -> npt.NDArray[np.float32]:
        """Returns output of theoretical FCS equations for fitting
        autocorrelation functions given the parameters of the dataclass

        Parameters
        ----------
        param: lmfit.Parameters() object
        tc: lag time tau
        """
        p = param
        if self.equation_dspecies == 2:
            p["a2"].value = 1.0 - p["a1"].value
        elif self.equation_dspecies == 3:
            p["a2"].value = 1.0 - p["a1"].value - p["a3"].value
            p["a3"].value = 1.0 - p["a2"].value - p["a1"].value

        if self.equation_dim == "2D":
            if self.equation_dspecies == 1:
                gdiff = p["a1"].value * (
                    (1 + ((tc / p["txy1"].value)**p["alpha1"].value))**-1)
            elif self.equation_dspecies == 2:
                gdiff = (
                    (p["a1"].value *
                     ((1 + (tc / p["txy1"].value)**p["alpha1"].value)**-1)) +
                    (p["a2"].value *
                     ((1 + (tc / p["txy2"].value)**p["alpha2"].value)**-1))
                )
            elif self.equation_dspecies == 3:
                gdiff = (
                    (p["a1"].value *
                     ((1 + (tc / p["txy1"].value)**p["alpha1"].value)**-1)) +
                    (p["a2"].value *
                     ((1 + (tc / p["txy2"].value)**p["alpha2"].value)**-1)) +
                    (p["a3"].value *
                     ((1 + (tc / p["txy3"].value)**p["alpha3"].value)**-1))
                )
            else:
                raise ValueError("equation_dspecies has to be 1, 2 or 3")
        elif self.equation_dim == "3D":
            if self.equation_diff3d == "tau_z":
                if self.equation_dspecies == 1:
                    gdiff = (
                        p["a1"].value * (
                            ((1 + ((tc / p["txy1"].value
                                    )**p["alpha1"].value))**-1) *
                            ((1 + (tc / p["tz1"].value))**-0.5)
                        )
                    )
                elif self.equation_dspecies == 2:
                    gdiff = (
                        (p["a1"].value * (
                            ((1 + ((tc / p["txy1"].value
                                    )**p["alpha1"].value))**-1) *
                            ((1 + (tc / p["tz1"].value))**-0.5)
                        )) +
                        (p["a2"].value * (
                            ((1 + ((tc / p["txy2"].value
                                    )**p["alpha2"].value))**-1) *
                            ((1 + (tc / p["tz2"].value))**-0.5)
                        ))
                    )
                elif self.equation_dspecies == 3:
                    gdiff = (
                        (p["a1"].value * (
                            ((1 + ((tc / p["txy1"].value
                                    )**p["alpha1"].value))**-1) *
                            ((1 + (tc / p["tz1"].value))**-0.5)
                        )) +
                        (p["a2"].value * (
                            ((1 + ((tc / p["txy2"].value
                                    )**p["alpha2"].value))**-1) *
                            ((1 + (tc / p["tz2"].value))**-0.5)
                        )) +
                        (p["a3"].value * (
                            ((1 + ((tc / p["txy3"].value
                                    )**p["alpha3"].value))**-1) *
                            ((1 + (tc / p["tz3"].value))**-0.5)
                        ))
                    )
                else:
                    raise ValueError("equation_dspecies has to be 1, 2 or 3")
            elif self.equation_diff3d == "aspect_ratio":
                if self.equation_dspecies == 1:
                    gdiff = (
                        p["a1"].value * (
                            ((1 + ((tc / p["txy1"].value
                                    )**p["alpha1"].value))**-1) *
                            ((1 + (tc / (p["txy1"].value *
                                         (p["ar1"].value**2))))**-0.5)
                        )
                    )
                elif self.equation_dspecies == 2:
                    gdiff = (
                        (p["a1"].value * (
                            ((1 + ((tc / p["txy1"].value
                                    )**p["alpha1"].value))**-1) *
                            ((1 + (tc / (p["txy1"].value *
                                         (p["ar1"].value**2))))**-0.5)
                        )) +
                        (p["a2"].value * (
                            ((1 + ((tc / p["txy2"].value
                                    )**p["alpha2"].value))**-1) *
                            ((1 + (tc / (p["txy2"].value *
                                         (p["ar2"].value**2))))**-0.5)
                        ))
                    )
                elif self.equation_dspecies == 3:
                    gdiff = (
                        (p["a1"].value * (
                            ((1 + ((tc / p["txy1"].value
                                    )**p["alpha1"].value))**-1) *
                            ((1 + (tc / (p["txy1"].value *
                                         (p["ar1"].value**2))))**-0.5)
                        )) +
                        (p["a2"].value * (
                            ((1 + ((tc / p["txy2"].value
                                    )**p["alpha2"].value))**-1) *
                            ((1 + (tc / (p["txy2"].value *
                                         (p["ar2"].value**2))))**-0.5)
                        )) +
                        (p["a3"].value * (
                            ((1 + ((tc / p["txy3"].value
                                    )**p["alpha3"].value))**-1) *
                            ((1 + (tc / (p["txy3"].value *
                                         (p["ar3"].value**2))))**-0.5)
                        ))
                    )
                else:
                    raise ValueError("equation_dspecies has to be 1, 2 or 3")
            else:
                raise ValueError(
                    "for a 3D fit, equation_diff3d has to be 'tau_z' or"
                    "'aspect_ratio'"
                )
        else:
            raise ValueError("equation_dim has to be '2D' or '3D'")
        if self.equation_triplet == "none":
            gt = 1
        elif self.equation_triplet == "triplet_ratio":
            if self.equation_tspecies == 1:
                gt = 1 + (p["b1"].value * np.exp(-tc / p["taut1"].value))
            elif self.equation_tspecies == 2:
                gt = (1 +
                      (p["b1"].value * np.exp(-tc / p["taut1"].value)) +
                      (p["b2"].value * np.exp(-tc / p["taut2"].value))
                      )
            elif self.equation_tspecies == 3:
                gt = (1 +
                      (p["b1"].value * np.exp(-tc / p["taut1"].value)) +
                      (p["b2"].value * np.exp(-tc / p["taut2"].value)) +
                      (p["b3"].value * np.exp(-tc / p["taut3"].value))
                      )
            else:
                raise ValueError("equation_tspecies has to be 1, 2 or 3")
        elif self.equation_triplet == "triplet_fraction":
            if self.equation_tspecies == 1:
                gt = (1 - p["t1"].value +
                      (p["t1"].value * np.exp(-tc / p["taut1"].value))
                      )
            elif self.equation_tspecies == 2:
                gt = (1 - (p["t1"].value + p["t2"].value) +
                      ((p["t1"].value * np.exp(-tc / p["taut1"].value)) +
                       (p["t2"].value * np.exp(-tc / p["taut2"].value)))
                      )
            elif self.equation_tspecies == 3:
                gt = (1 - (p["t1"].value + p["t2"].value + p["t3"].value) +
                      ((p["t1"].value * np.exp(-tc / p["taut1"].value)) +
                       (p["t2"].value * np.exp(-tc / p["taut2"].value)) +
                       (p["t3"].value * np.exp(-tc / p["taut3"].value)))
                      )
            else:
                raise ValueError("equation_tspecies has to be 1, 2 or 3")
        else:
            raise ValueError(
                "equation_triplet has to be 'none', 'triplet_ratio' or "
                "'triplet_equation'"
            )
        return np.array(p["offset"].value + (p["gn0"].value * gdiff * gt),
                        dtype=np.float32)

    def get_residual(
            self, param: lmfit.Parameters, tc: np.ndarray, cor_g: np.ndarray
    ) -> npt.NDArray[np.float32]:
        equ = self.get_equation(param, tc)
        return np.array(cor_g - equ, dtype=np.float32)

    def minimize(
            self, tc: np.ndarray, cor_g: np.ndarray
    ) -> lmfit.minimizer.MinimizerResult:
        if self.method != "lmfit":
            raise ValueError("Currently only fitting via lmfit is supported.")
        self.result_minimizer = lmfit.minimize(
            self.get_residual, self.params, args=(tc, cor_g)
        )
        if self.result_minimizer is not None:
            self.tc = np.array(tc, dtype=np.float32)
            self.g = np.array(self.get_equation(
                getattr(self.result_minimizer, "params"), self.tc
            ), dtype=np.float32)
            self.residual = self.get_residual(
                getattr(self.result_minimizer, "params"), self.tc, cor_g
            )
        return self.result_minimizer

    def get_minimizer_params(self) -> dict | None:
        if (rm_params_dict := getattr(self.result_minimizer, "params", None)
            ) is not None:
            rm_params_dict = {k: v for k, v in rm_params_dict.items()}
        rm_fitstats = {k: getattr(self.result_minimizer, k, None) for k in [
            "nfev", "nvarys", "ndata", "nfree", "chisqr", "redchi", "aic", "bic"
        ]}
        rm_callkws = getattr(self.result_minimizer, "call_kws", None)
        rm_dict = {"aborted": getattr(self.result_minimizer, "aborted", None),
                   "success": getattr(self.result_minimizer, "success", None),
                   "message": getattr(self.result_minimizer, "message", None),
                   "params": rm_params_dict,
                   "fit_report": lmfit.fit_report(self.result_minimizer),
                   "fit_stats": rm_fitstats,
                   "call_kws": rm_callkws}
        rm_dict = None if self.result_minimizer is None else rm_dict
        return rm_dict

    def to_dict(self) -> dict:
        rm_dict = self.get_minimizer_params()
        out = {
            k: (v if k not in ["params", "result_minimizer"]
                else (dict(self.params) if k == "params" else rm_dict))
            for k, v in asdict(self).items()
        }
        return out

    def lmfit_parameters_to_dict(self, params: lmfit.Parameters) -> dict:
        params_list = [
            "offset", "gn0", "a1", "a2", "a3", "txy1", "txy2", "txy3",
            "alpha1", "alpha2", "alpha3", "ar1", "ar2", "ar3", "tz1", "tz2",
            "tz3", "b1", "b2", "b3", "t1", "t2", "t3", "taut1", "taut2", "taut3"
        ]
        param_empty = {
            "value": None, "min": None, "max": None, "vary": None,
            "stderr": None
        }
        params_dict = {}
        for p in params_list:
            v = params[p] if p in params.keys() else None
            if v is None:
                param_dict = param_empty
            else:
                param_dict = {
                    "value": v.value, "min": v.min, "max": v.max,
                    "vary": v.vary, "stderr": v.stderr
                }
            params_dict[p] = param_dict
        return params_dict

    def to_polars(self):
        params_list = [
            "offset", "gn0", "a1", "a2", "a3", "txy1", "txy2", "txy3",
            "alpha1", "alpha2", "alpha3", "ar1", "ar2", "ar3", "tz1", "tz2",
            "tz3", "b1", "b2", "b3", "t1", "t2", "t3", "taut1", "taut2", "taut3"
        ]

        param_schema = {"value": pl.Float32, "min": pl.Float32, "max": pl.Float32,
                        "vary": pl.Boolean, "stderr": pl.Float32}
        params_schema = {p: pl.Struct(param_schema) for p in params_list}

        params = self.lmfit_parameters_to_dict(self.params)
        initial_params = {
            "method": self.method, "equation_dim": self.equation_dim,
            "equation_dspecies": self.equation_dspecies,
            "equation_diff3d": self.equation_diff3d,
            "equation_triplet": self.equation_triplet,
            "equation_tspecies": self.equation_tspecies, "params": params
        }
        initial_schema = {
            "method": pl.String, "equation_diff3d": pl.String,
            "equation_dim": pl.String, "equation_dspecies": pl.UInt32,
            "equation_triplet": pl.String, "equation_tspecies": pl.UInt32,
            "params": pl.Struct(params_schema)
        }
        minimizer_params = self.get_minimizer_params()
        if minimizer_params is not None:
            minimizer_params["params"] = self.lmfit_parameters_to_dict(
                minimizer_params["params"]
            )
            del minimizer_params["call_kws"]
        minimizer_schema = {
            "aborted": pl.Boolean, "success": pl.Boolean, "message": pl.String,
            "fit_stats": pl.Struct({
                "aic": pl.Float32, "bic": pl.Float32, "chisqr": pl.Float32,
                "ndata": pl.UInt32, "nfev": pl.UInt32, "nfree": pl.UInt32,
                "nvarys": pl.UInt32, "redchi": pl.Float32,
            }),
            "fit_report": pl.String, "params": pl.Struct(params_schema),
        }
        out = pl.DataFrame(
            {"uuid": str(self.uuid)} |
            {"tc": [self.tc]} |
            {"g": [self.g]} |
            {"residual": [self.residual]} |
            {"initial_params": initial_params} |
            {"minimizer_params": minimizer_params},
            schema={
                "uuid": pl.String,
                "tc": (
                    pl.Array(pl.Float32, self.tc.size)
                    if self.tc is not None else pl.Null
                ),
                "g": (
                    pl.Array(pl.Float32, self.g.size)
                    if self.g is not None else pl.Null
                ),
                "residual": (
                    pl.Array(pl.Float32, self.residual.size)
                    if self.residual is not None else pl.Null
                ),
                "initial_params": pl.Struct(initial_schema),
                "minimizer_params": (
                    pl.Struct(minimizer_schema)
                    if minimizer_params is not None else pl.Null
                ),
            }
        )
        return out



@dataclass
class SimulatedFCSTimeSeriesFit:
    uuid: uuid_module.UUID
    sim_params: FCSSimParams
    record: dict[Literal["feature", "label_restoration", "label_segmentation"],
                 FCSFit] = (
        field(default_factory=dict, compare=False)
    )

    def to_polars(self) -> pl.DataFrame:
        r = self.record.items()
        tc = [v.to_polars().select("tc") for _, v in r][0]
        g = [v.to_polars().select("g").rename({"g": f"{k}_g"}) for k, v in r]
        res = [v.to_polars().select("residual")
               .rename({"residual": f"{k}_residual"}) for k, v in r]
        ini_p = [v.to_polars().select("initial_params")
                 .rename({"initial_params": f"{k}_initial_params"})
                 for k, v in r]
        min_p = [v.to_polars().select("minimizer_params")
                 .rename({"minimizer_params": f"{k}_minimizer_params"})
                 for k, v in r]
        out = pl.DataFrame({"uuid": str(self.uuid)}, schema={"uuid": pl.String})
        out = pl.concat(
            [out, tc, *g, *res, *ini_p, *min_p, self.sim_params.to_polars()],
            how="horizontal"
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
