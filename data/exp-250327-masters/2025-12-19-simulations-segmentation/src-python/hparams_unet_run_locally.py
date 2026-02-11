"""script written for tensorflow 2.19 / keras 2.19 based environments
this is a backport of the script for tensorflow 1.19.1 / keras 3. Notably,
it is harder to implement custom metrics, which were left out, and the
convenient keras.ops module is missing - so tf alternatives had to be used.
"""
#/usr/bin/env python3

import logging
import os
import random
import sys

# needed for tf to substitute tf.keras calls which by default use keras 3 with
# the tf_keras module which is keras 2, see https://keras.io/getting_started/
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import click
import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import polars as pl
import sklearn.preprocessing as skp
import tensorflow as tf
import tf_keras as keras
import tensorflow.python.platform.build_info as build

from datetime import datetime
# from tf_keras.src.metrics import metrics_utils
# from mlflow.keras.callback import MlflowCallback
from tensorboard.plugins.hparams import api as hp

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

HP_EPOCHS = hp.HParam("hp_epochs", hp.Discrete([20], dtype=int))
HP_BATCH_SIZE = hp.HParam("hp_batch_size", hp.IntInterval(4, 30))
HP_SCALER = hp.HParam(
    "hp_scaler",
    hp.Discrete(
        ["robust", "minmax", "maxabs", "quant_g", "standard", "l1", "l2"],
        dtype=str))
HP_N_LEVELS = hp.HParam("hp_n_levels", hp.IntInterval(1, 9))
HP_FIRST_FILTERS = hp.HParam("hp_first_filters", hp.IntInterval(1, 128))
HP_POOL_SIZE = hp.HParam("hp_pool_size", hp.Discrete([2, 4, 8], dtype=int))
HP_INPUT_SIZE = hp.HParam("hp_input_size", hp.Discrete([14000], dtype=int))
HP_LR_START = hp.HParam("hp_lr_start", hp.RealInterval(1e-6, 0.06))
HP_LR_POWER = hp.HParam("hp_lr_power", hp.IntInterval(1, 7))

HPARAMS = [
    HP_EPOCHS, HP_BATCH_SIZE, HP_SCALER, HP_N_LEVELS, HP_FIRST_FILTERS,
    HP_POOL_SIZE, HP_INPUT_SIZE, HP_LR_START, HP_LR_POWER
]

LOG_DIR = "../tmp/tb-" + datetime.now().strftime("%Y%m%d-%H%M%S")

SESSIONS_PER_GROUP = 2


def tfds_from_pldf(
        feature: pl.Series, label: pl.Series
) -> tuple[tf.data.Dataset, int]:
    """TensorFlow Dataset from polars Series

    Parameters
    ----------
    feature, label: polars DataFrames
        Contain row-wise features / labels

    Returns
    -------
    dataset : TensorFlow Dataset
        Contains features and labels
    num_total_examples : int
        Number of test examples
    """

    X_tensor = tf.convert_to_tensor(value=feature)
    X_tensor = tf.where(
        tf.math.is_nan(X_tensor), tf.zeros_like(X_tensor), X_tensor
    )

    y_tensor = tf.convert_to_tensor(value=label)
    y_tensor = tf.cast(y_tensor, tf.float32)
    y_tensor = tf.where(
        tf.math.is_nan(y_tensor), tf.zeros_like(y_tensor), y_tensor
    )

    num_total_examples = X_tensor.shape[0]
    X_tensor = tf.reshape(tensor=X_tensor, shape=(num_total_examples, -1, 1))
    y_tensor = tf.reshape(tensor=y_tensor, shape=(num_total_examples, -1, 1))

    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

    log.debug("number of examples: %s", num_total_examples)
    return (dataset, num_total_examples)


def get_data(file_features: str, file_labels: str) -> pl.DataFrame:
    df = pl.concat(
        [pl.read_parquet(file_features),
         (pl.read_parquet(file_labels)
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
    return df


def tfds_crop(feature, label, length_delimiter):
    """Part of tf.data pipeline. Crop feature and label to a maximum length of
    length_delimiter
    """
    feature = feature[:length_delimiter]
    label = label[:length_delimiter]
    trace_shape = feature.shape
    label_shape = label.shape
    feature.set_shape(trace_shape)
    label.set_shape(label_shape)
    return feature, label

def tfds_scale(feature, label, scaler):
    """Part of tf.data pipeline. Wrapper function to be able to .map()
    scale_feature()
    """
    feature_shape = feature.shape
    [feature, ] = tf.py_function(  # type: ignore
        func=scale_feature, inp=[feature, scaler], Tout=[tf.float32]
    )
    feature.set_shape(feature_shape)
    return feature, label


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


def tfds_pad(feature: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
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
    feature = tf.experimental.numpy.pad(feature, pad_width=[[0, pad_size], [0, 0]],
                    mode="constant",
                    constant_values=pad_median)
    label = tf.experimental.numpy.pad(label, pad_width=[[0, pad_size], [0, 0]],
                    mode="constant",
                    constant_values=0)

    feature_shape = feature.shape
    label_shape = label.shape
    feature.set_shape(feature_shape)
    label.set_shape(label_shape)
    return feature, label


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


def tfds_prepare(
        ds: tf.data.Dataset, params: dict, num_examples: int
) -> tf.data.Dataset:
    return (
        ds
        .map(lambda feature, label: tfds_crop(
            feature, label, params["input_size"]
        ), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda feature, label: tfds_scale(
            feature, label, params["scaler"]
        ), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda feature, label: tfds_pad(feature, label),
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=num_examples)
        .repeat()
        .batch(params["batch_size"], drop_remainder=True)
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
