"""script written for tensorflow 2.19 / keras 2.19 based environments
this is a backport of the script for tensorflow 1.19.1 / keras 3. Notably,
it is harder to implement custom metrics, which were left out, and the
convenient keras.ops module is missing - so tf alternatives had to be used.
"""
#/usr/bin/env python3

import logging
import os
import sys

# needed for tf to substitute tf.keras calls which by default use keras 3 with
# the tf_keras module which is keras 2, see https://keras.io/getting_started/
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import mlflow
import mlflow.client
import mlflow.entities
import numpy as np
import polars as pl
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import tensorflow as tf
import tf_keras as keras
import tensorflow.python.platform.build_info as build

from datetime import datetime

tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=False)
keras.saving.get_custom_objects().clear()

logging.basicConfig(format="%(asctime)s - hparams - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

log.debug("Python version: %s", sys.version)
log.debug("Tensorflow version: %s", tf.__version__)
log.debug("Keras version: %s", keras.__version__)
log.debug("Cudnn version: %s", build.build_info.get("cudnn_version"))
log.debug("Cuda version: %s", build.build_info.get("cuda_version"))
# Workaround for a "No algorithm worked" bug on GPUs
# see https://github.com/tensorflow/tensorflow/issues/45044
physical_devices = tf.config.list_physical_devices("GPU")
log.debug("GPUs: %s. Trying to set memory growth to 'True'...",
          physical_devices)
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    log.debug("Setting memory growth successful.")
except IndexError:
    log.debug("No GPU was found on this machine. ")

os.chdir("/home/alva/Programs/drmed-git")

inputdir = "data/exp-250327-masters/2025-05-28-simulations/parquet"
workdir = "data/exp-250327-masters/2025-12-19-simulations-segmentation"

exp_dict = {
    "peak-artifacts-testing": "peak_2",
    "photobleaching-testing": "bleach_2",
    "detector-dropout-testing": "dropout_2",
}


def get_runs(exp_name: str):
    client = mlflow.client.MlflowClient("file:data/mlruns")
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is not None:
        log.debug(f"found experiment {exp_name}: {experiment.experiment_id}")
        runs = client.search_runs(experiment.experiment_id)
    else:
        raise ValueError(f"experiment {exp_name} is None.")
    return client, runs


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
    return df, out_first


def tfds_from_pldf(feature: pl.Series) -> tf.data.Dataset:
    """TensorFlow Dataset from polars Series

    Parameters
    ----------
    feature, label: polars DataFrames
        Contain row-wise features

    Returns
    -------
    dataset : TensorFlow Dataset
        Contains features
    num_examples : int
        Number of examples
    """

    X_tensor = tf.convert_to_tensor(value=feature)
    X_tensor = tf.where(
        tf.math.is_nan(X_tensor), tf.zeros_like(X_tensor), X_tensor
    )

    num_total_examples = X_tensor.shape[0]
    X_tensor = tf.reshape(tensor=X_tensor, shape=(num_total_examples, -1, 1))

    dataset = tf.data.Dataset.from_tensor_slices((X_tensor))
    return dataset


def tfds_scale(feature, scaler):
    """Part of tf.data pipeline. Wrapper function to be able to .map()
    scale_feature()
    """
    feature_shape = feature.shape
    [feature, ] = tf.py_function(  # type: ignore
        func=scale_feature, inp=[feature, scaler], Tout=[tf.float32]
    )
    feature.set_shape(feature_shape)
    return feature


def scale_feature(feature, scaler):
    """Part of tf.data pipeline. Scale / normalize the input feature.

    Parameters:
    -----------
    feature : np.array, pl.DataFrame, pd.DataFrame or tf.Tensor
        1D-Feature
    scaler : ('standard', 'robust', 'maxabs', 'quant_g', 'minmax', l1', 'l2')
        Selected scalers from sklearn.preprocessing

    Returns:
    --------
    feature : np.array
        Scaled / normalized feature.

    Raises:
    -------
    ValueError
        If the value for scaler is not in ('standard', 'robust', 'maxabs',
        'quant_g', 'minmax', 'l1', 'l2')
    """
    scaler = tf.convert_to_tensor(scaler)
    if scaler == tf.convert_to_tensor("standard"):
        feature = skp.StandardScaler().fit_transform(feature)
    elif scaler == tf.convert_to_tensor("robust"):
        feature = skp.RobustScaler(
            quantile_range=(25, 75)
        ).fit_transform(feature)
    elif scaler == tf.convert_to_tensor("maxabs"):
        feature = skp.MaxAbsScaler().fit_transform(feature)
    elif scaler == tf.convert_to_tensor("quant_g"):
        feature = skp.QuantileTransformer(
            output_distribution="normal").fit_transform(feature)
    elif scaler == tf.convert_to_tensor("minmax"):
        feature = skp.MinMaxScaler().fit_transform(feature)
    elif scaler == tf.convert_to_tensor("l1"):
        feature = skp.normalize(X=feature, norm="l1", axis=0)
    elif scaler == tf.convert_to_tensor("l2"):
        feature = skp.normalize(X=feature, norm="l2", axis=0)
    else:
        raise ValueError(
            "scaler has to be a string. currently supported are:"
            "'standard', 'robust', 'maxabs', 'quant_g', 'minmax', 'l1', 'l2'"
        )
    return feature


def tfds_pad(feature: tf.Tensor) -> tf.Tensor:
    """Part of tf.data pipeline. Pad the end of the feature with the
    median of the feature. Set the label for this pad to 0 (no artifact)

    Notes
    -----
    - at the moment, the implementation of `tf.experimental.numpy.pad` does not
    support the mode `median` (see
    https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/pad)
    - that's why an own pure tf implementation of a median is used (see
    https://stackoverflow.com/questions/43824665/tensorflow-median-value)
    """
    pad_size, pad_median = _get_pad_size_and_value(feature)
    feature = tf.experimental.numpy.pad(
        feature,
        pad_width=[[0, pad_size], [0, 0]],
        mode="constant",
        constant_values=pad_median
    )
    feature_shape = feature.shape
    feature.set_shape(feature_shape)
    return feature


def _get_pad_size_and_value(trace):
    """Get pad size and pad value. """
    def get_median(v):
        v = tf.reshape(v, [-1])
        mid = v.get_shape()[0] // 2 + 1
        return tf.nn.top_k(v, mid).values[-1]

    trace_size = trace.size
    if trace_size < 1024:
        input_size = 1024
    else:
        # new size is the next biggest power of 2 → this is important for the
        # skip connections of the UNET
        input_size = 2**tf.experimental.numpy.ceil(tf.experimental.numpy.log2(trace_size))
    input_size = tf.cast(input_size, tf.int32)
    pad_size = input_size - trace_size

    # pad trace
    pad_value = get_median(trace)
    return pad_size, pad_value


def tfds_prepare(ds: tf.data.Dataset, params: dict) -> tf.data.Dataset:
    return (
        ds
        .map(lambda feature: tfds_scale(feature, params["scaler"]),
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda feature: tfds_pad(feature),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(1)
        .prefetch(tf.data.AUTOTUNE)
    )

# loading the trained model needs registering some custom functions

@keras.saving.register_keras_serializable()
def Adam(*args, **kwargs):
    return keras.optimizers.Adam(*args, **kwargs)


# define custom loss functions
def binary_ce_dice_loss_coef(y_true, y_pred, axis, smooth):
    def dice_loss(y_true, y_pred, axis, smooth):
        """Soft dice coefficient for comparing the similarity of two batches
        of data, usually used for binary image segmentation

        For binary labels, the dice loss will be between 0 and 1 where 1 is a
        total match. Reshaping is needed to combine the global dice loss with
        the local binary_crossentropy
        """
        numerator = 2 * tf.math.reduce_sum(
            input_tensor=y_true * y_pred, axis=axis, keepdims=True)
        denominator = tf.math.reduce_sum(input_tensor=y_true + y_pred,
                                         axis=axis,
                                         keepdims=True)

        return 1 - (numerator + smooth) / (denominator + smooth)

    return keras.backend.binary_crossentropy(y_true, y_pred) + dice_loss(
        y_true, y_pred, axis, smooth)

@keras.saving.register_keras_serializable(name="binary_ce_dice")
def binary_ce_dice_loss(axis=-1, smooth=1e-5):
    """Combination of binary crossentropy and dice loss

    Parameters
    -----------
    y_true : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_pred : Tensor
        The y_pred distribution, format the same with `y_true`.
    axis : int or tuple of int
        All dimensions are reduced, default ``-1``
    smooth : float, optional
        Will be added to the numerator and denominator of the dice loss.
        - If both y_true and y_pred are empty, it makes sure dice is 1.
        - If either y_true or y_pred are empty (all pixels are background),
        dice = ```smooth/(small_value + smooth)``
        - Smoothing is not really necessary for combined losses (so standard
        value is 0)

    Notes
    -----
    - this function was influenced by code from
        - TensorLayer project
        https://tensorlayer.readthedocs.io/en/latest/modules/cost.html#tensorlayer.cost.dice_coe,
        - Lars Nieradzik
        https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
        - Code by Stefan Hoffmann, Applied Systems Biology group,
        Hans-Knöll-Institute Jena
    - To be able to load the custom loss function in Keras, it must only take
      (y_true, y_pred) as parameters - that is why this setup seems so
      complicated.
    - binary crossentropy returns a tensor with loss for each 1D step of a
    trace, bringing local info
    - dice loss returns a scalar for each 1D trace, bringing global info
    """
    def binary_ce_dice(y_true, y_pred):
        return binary_ce_dice_loss_coef(y_true, y_pred, axis, smooth)

    return binary_ce_dice


def predict_unet(trace: pl.Series, run: mlflow.entities.Run) -> np.ndarray:
    params = {
        "input_size": None,  # don't restrict input size for test data
        "scaler": run.data.params["hp_scaler"],
        "batch_size": int(run.data.params["hp_batch_size"]),
    }
    ds = tfds_from_pldf(trace)
    ds = tfds_prepare(ds, params)
    model_path = (
        f"./data/mlruns/{run.info.experiment_id}/{run.info.run_id}/"
        "artifacts/model/data/model"
    )
    with keras.saving.custom_object_scope(
            {"binary_ce_dice": binary_ce_dice_loss()}
    ):
        model = keras.models.load_model(model_path, compile=False)
    if model is not None:
        out = model.predict(x=ds, verbose=0)
        out = out.squeeze(-1)
    else:
        raise ValueError(f"problem in loading model of run {run.info.run_id} ")
    return out


def jaccard(
        true: list, pred: list, precision: float, recall: float, average: str
) -> float:
    if np.isnan(precision) | np.isnan(recall):
        out = np.nan
    else:
        out = float(skm.jaccard_score(true, pred, average=average))
    return out


for myfile in [
        "2025-05-28-detector-dropout-testing.parquet",
        "2025-05-28-peak-artifacts-testing.parquet",
        "2025-05-28-photobleaching-testing.parquet",
]:
    log.debug(f"Perform and evaluate unet segmentation for {myfile} ...")
    df, out_file = get_data(myfile)
    exp = exp_dict[out_file]
    out_date = datetime.today().date()
    out_file = f"{out_date}-{out_file}-unet.parquet"
    client, runs = get_runs(exp)
    experiment = client.get_experiment_by_name(exp)
    if experiment is not None:
        exp_id = experiment.experiment_id
    else:
        raise ValueError(f"experiment {exp} is None.")

    for r in runs:
        run_id = f"{r.info.run_id:.5}"
        full_id = f"{run_id}_full-id"
        pred = f"{run_id}_pred"
        seg = f"{run_id}_seg"
        cm = f"{run_id}_cm"
        precision = f"{run_id}_precision"
        recall = f"{run_id}_recall"
        fbeta2 = f"{run_id}_fbeta2"
        biniou = f"{run_id}_biniou"
        meaniou = f"{run_id}_meaniou"
        overlap = f"{run_id}_overlap"
        if r.data.metrics.get("loss") is None:
            log.debug(f"run {run_id}: skipped")
            continue
        log.debug(f"run {run_id}: auc {str(r.data.metrics.get('auc')):.5}")
        df = df.with_columns(
            (pl.lit(r.info.run_id)).alias(full_id),
            (pl.struct("feature").map_batches(
                lambda x: predict_unet(x.struct.field("feature"), r)
            )).alias(pred),
        )
        df = df.with_columns(
            pl.col(pred).arr.to_list().list.eval(pl.element() > 0.5)
            .cast(pl.Array(pl.Boolean, shape=(16384))).alias(seg)
        )
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
             pl.min_horizontal(pl.col.label_ground_truth.arr.sum(),
                               pl.col(seg).arr.sum())
             ).alias(overlap),
        )

    df = df.drop("feature")
    df.write_parquet(f"{workdir}/parquet/{out_file}")
