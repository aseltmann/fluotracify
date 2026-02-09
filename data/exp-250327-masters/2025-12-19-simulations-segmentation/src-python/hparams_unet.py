#!/usr/bin/env python3

import logging
import os
import random
import sys

import click
import keras
import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import polars as pl
import sklearn.preprocessing as skp
import tensorflow as tf
import tensorflow.python.platform.build_info as build

from collections.abc import Iterable
from datetime import datetime
from keras.src.metrics import metrics_utils
from tensorboard.plugins.hparams import api as hp

tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=False)

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

SEED = 42

rng = random.Random(SEED)


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


# define model layers
def convtrans(filters, name, kernel_size, strides):
    """Sequential API: Conv1DTranspose, BatchNorm"""
    upsamp = keras.Sequential(name=name)
    upsamp.add(
        keras.layers.Conv1DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides))
    upsamp.add(keras.layers.BatchNormalization())
    upsamp.add(keras.layers.Activation("relu"))
    return upsamp


def twoconv(filters, name):
    """Sequential API: Conv1D, BatchNorm, Conv1D, BatchNorm"""
    conv = keras.Sequential(name=name)
    conv.add(
        keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same"))
    conv.add(keras.layers.BatchNormalization())
    conv.add(keras.layers.Activation("relu"))

    conv.add(
        keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same"))
    conv.add(keras.layers.BatchNormalization())
    conv.add(keras.layers.Activation("relu"))
    return conv


def encoder(input_tensor, filters, name, pool_size=2):
    """Functional API: Two Conv1D incl BatchNorm, MaxPool1D"""
    encode = twoconv(filters=filters, name=name)(input_tensor)
    encode_pool = keras.layers.MaxPool1D(pool_size=pool_size,
                                            name="mp_{}".format(name))(encode)
    return encode_pool, encode


def decoder(input_tensor,
            concat_tensor,
            filters,
            name,
            kernel_size=2,
            strides=2):
    """Functional API: Conv1DTrans, BatchNorm, Concat, Two Conv incl BatchNorm
    """
    decode = convtrans(filters=filters,
                       name="conv_transpose_{}".format(name),
                       kernel_size=kernel_size,
                       strides=strides)(input_tensor)
    decode = keras.layers.concatenate([concat_tensor, decode],
                                         axis=-1,
                                         name=name)
    decode = twoconv(filters=filters, name="two_conv_{}".format(name))(decode)
    return decode


# define custom loss functions
class BinaryCrossentropyDice(keras.Loss):
    def call(self, y_true, y_pred):
        loss = (keras.losses.Dice(axis=-1).call(y_true, y_pred) +
                keras.losses.BinaryCrossentropy(axis=-1).call(y_true, y_pred))
        return loss

# define custom metric
class MyFBetaScore(keras.Metric):
    """Had to re-implement FBeta due to https://github.com/keras-team/tf-keras/issues/771

    Note: reimplementation based on tf.keras.metrics.Recall and tf.keras.metrics.Precision
    """
    def __init__(
        self, thresholds=None, top_k=None, class_id=None, beta=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.beta = beta

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=keras.initializers.Zeros(),
            name="true_positives",
        )
        self.false_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=keras.initializers.Zeros(),
            name="false_positives",
        )
        self.true_negatives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=keras.initializers.Zeros(),
            name="true_negatives",
        )
        self.false_negatives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=keras.initializers.Zeros(),
            name="false_negatives",
        )
    def reset_state(self):
        num_thresholds = len(self.thresholds)
        self.true_positives.assign(keras.ops.zeros((num_thresholds,)))
        self.false_positives.assign(keras.ops.zeros((num_thresholds,)))
        self.true_negatives.assign(keras.ops.zeros((num_thresholds,)))
        self.false_negatives.assign(keras.ops.zeros((num_thresholds,)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to `1`.
                Can be a tensor whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
        """
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        betasq = keras.ops.power(self.beta, 2)
        numerator = keras.ops.multiply(
            keras.ops.add(1, betasq), self.true_positives
        )
        denominator = keras.ops.add(
            keras.ops.multiply(
                betasq,
                keras.ops.add(self.true_positives, self.false_negatives)
            ),
            keras.ops.add(self.true_positives, self.false_positives)
        )
        result = keras.ops.divide_no_nan(numerator, denominator)
        return result[0] if len(self.thresholds) == 1 else result

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
            "beta": self.beta,
        }
        base_config = keras.Metric().get_config()
        return {**base_config, **config}

class MyOverlap(keras.Metric):
    """Overlap coefficient. Currently only for target cla

    see https://en.wikipedia.org/wiki/Overlap_coefficient
    """
    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=keras.initializers.Zeros(),
            name="true_positives",
        )
        self.sum_true = self.add_variable(
            shape=(1,),
            initializer=keras.initializers.Zeros(),
            name="sum_true",
        )
        self.sum_pred = self.add_variable(
            shape=(1,),
            initializer=keras.initializers.Zeros(),
            name="sum_pred",
        )

    def reset_state(self):
        num_thresholds = len(self.thresholds)
        self.true_positives.assign(keras.ops.zeros((num_thresholds,)))
        self.sum_true.assign(keras.ops.zeros((1,)))
        self.sum_pred.assign(keras.ops.zeros((1,)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to `1`.
                Can be a tensor whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
        """
        y_true = keras.ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = keras.ops.convert_to_tensor(y_pred, dtype=self.dtype)
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )
        sum_true = keras.ops.add(self.sum_true, keras.ops.sum(y_true))
        sum_pred = keras.ops.add(self.sum_pred, keras.ops.sum(y_pred))
        self.sum_true.assign(sum_true)
        self.sum_pred.assign(sum_pred)

    def result(self):
        denominator = keras.ops.minimum(self.sum_true, self.sum_pred)
        result = keras.ops.divide_no_nan(self.true_positives, denominator)
        return result[0] if len(self.thresholds) == 1 else result

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = keras.Metric().get_config()
        return {**base_config, **config}

def unet_metrics(metrics_thresholds):
    """Returns a selection of metrics for model training

    Parameters
    ----------
    metrics_thresholds: list of float between 0 and 1

    Returns
    -------
    list of metrics
    """
    metrics = []
    for thresh in metrics_thresholds:
        metrics.append(keras.metrics.TruePositives(
            name=f"tp_{thresh}", thresholds=thresh
        ))
        metrics.append(keras.metrics.FalsePositives(
            name=f"fp_{thresh}", thresholds=thresh
        ))
        metrics.append(keras.metrics.TrueNegatives(
            name=f"tn_{thresh}", thresholds=thresh
        ))
        metrics.append(keras.metrics.FalseNegatives(
            name=f"fn_{thresh}", thresholds=thresh
        ))
        metrics.append(keras.metrics.Precision(
            name=f"precision_{thresh}", thresholds=thresh
        ))
        metrics.append(keras.metrics.Recall(
            name=f"recall_{thresh}", thresholds=thresh
        ))
        metrics.append(MyFBetaScore(
            name=f"fbeta2_{thresh}", beta=2., thresholds=thresh
        ))
        metrics.append(keras.metrics.BinaryIoU(
            name=f"biniou1_{thresh}", target_class_ids=[1], threshold=thresh
        ))
        metrics.append(keras.metrics.BinaryIoU(
            name=f"biniou0_{thresh}", target_class_ids=[0], threshold=thresh
        ))
        metrics.append(MyOverlap(name=f"ovl_{thresh}", thresholds=thresh))
    metrics.append(keras.metrics.MeanIoU(name=f"meaniou", num_classes=2))
    metrics.append(keras.metrics.AUC(name=f"auc", curve="PR"))
    return metrics


def unet_1d_hparams(hparams):
    """Defines compiled U-Net. Includes option to define various hyperparameters
    and the more abstract parameter of unet levels.

    Parameters
    ----------
    input_size : int
        Input vector size
    n_levels : int
        Number of levels or steps in the Unet
    first_filters : int
        The number of filters in the first level. Every deeper level
        will be twice as many filters till a maximum of 512 is reached.
        Filters will be clipped if smaller than 1 or bigger than 512
    pool_size : int, Optional. Default: 2
        Pool size of the MaxPool1D layer, as well as kernel size and
        strides of the Conv1DTranspose layer
    metrics_thresholds : list of float between 0 and 1
        compute metrics with these prediction thresholds

    Returns
    -------
    Compiled Model as described by the tensorflow.keras Functional API

    Notes
    -----
    - Paper: https://arxiv.org/pdf/1505.04597.pdf
    - conceptually different approach than in the paper is the use of
    transposed convolution opposed to a up"-convolution" consisting of
    bed-of-nails upsampling and a 2x2 convolution
    - this implementation was influenced by:
    https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    filters = [hparams[HP_FIRST_FILTERS]]
    nextfilters = hparams[HP_FIRST_FILTERS]
    for _ in range(1, hparams[HP_N_LEVELS] + 1):
        nextfilters *= 2
        filters.append(nextfilters)
    filters = tf.experimental.numpy.clip(filters, a_min=1, a_max=512)
    filters = tf.cast(filters, tf.int32).numpy()  # type: ignore

    ldict = {}

    inputs = keras.layers.Input(shape=(None, 1))

    # Downsampling through model
    ldict["x0_pool"], ldict["x0"] = encoder(
        input_tensor=inputs,
        filters=filters[0],
        name="encode0",
        pool_size=hparams[HP_POOL_SIZE]
    )
    for i in range(1, hparams[HP_N_LEVELS]):
        ldict["x{}_pool".format(i)], ldict["x{}".format(i)] = encoder(
            input_tensor=ldict["x{}_pool".format(i - 1)],
            filters=filters[i],
            name="encode{}".format(i),
            pool_size=hparams[HP_POOL_SIZE]
        )

    # Center
    center = twoconv(
        2 * filters[hparams[HP_N_LEVELS] - 1], name="two_conv_center"
    )(
        ldict["x{}_pool".format(hparams[HP_N_LEVELS] - 1)]
    )

    # Upsampling through model
    ldict["y{}".format(hparams[HP_N_LEVELS] - 1)] = decoder(
        input_tensor=center,
        concat_tensor=ldict["x{}".format(hparams[HP_N_LEVELS] - 1)],
        filters=filters[-1],
        name="decoder{}".format(hparams[HP_N_LEVELS] - 1),
        kernel_size=hparams[HP_POOL_SIZE],
        strides=hparams[HP_POOL_SIZE]
    )

    for j in range(1, hparams[HP_N_LEVELS]):
        ldict["y{}".format(hparams[HP_N_LEVELS] - 1 - j)] = decoder(
            input_tensor=ldict["y{}".format(hparams[HP_N_LEVELS] - j)],
            concat_tensor=ldict["x{}".format(hparams[HP_N_LEVELS] - 1 - j)],
            filters=filters[-1 - j],
            name="decoder{}".format(hparams[HP_N_LEVELS] - 1 - j),
            kernel_size=hparams[HP_POOL_SIZE],
            strides=hparams[HP_POOL_SIZE]
        )

    # create "binary" output vector
    outputs = keras.layers.Conv1D(
        filters=1, kernel_size=1, activation="sigmoid"
    )(
        ldict["y0"]
    )

    log.debug("unet: input shape: %s, output shape: %s", inputs.shape,  # type: ignore
              outputs.shape)

    unet = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="unet_depth{}".format(hparams[HP_N_LEVELS])
    )

    optimizer = keras.optimizers.Adam()
    loss = BinaryCrossentropyDice()
    metrics_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    metrics = unet_metrics(metrics_thresholds)
    unet.compile(loss=loss, optimizer=optimizer, metrics=metrics)  # type: ignore
    return unet


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


def _get_pad_size_and_value(feature: tf.Tensor) -> tuple[int, float]:
    """Get pad size and pad value. """
    def get_median(v: tf.Tensor) -> float:
        v = tf.reshape(v, [-1])
        mid = v.get_shape()[0] // 2 + 1  # type: ignore
        return tf.nn.top_k(v, mid).values[-1]

    feature_size = feature.size
    if feature_size < 1024:  # type: ignore
        input_size = 1024
    else:
        # new size is the next biggest power of 2 → this is important for the
        # skip connections of the UNET
        input_size = 2**keras.ops.ceil(keras.ops.log2(feature_size))  # type: ignore
    pad_size = input_size - feature_size   # type: ignore

    # pad trace
    pad_value = get_median(feature)
    return pad_size, pad_value   # type: ignore


def tfds_prepare(
        ds: tf.data.Dataset, hparams: dict, num_examples: int
) -> tf.data.Dataset:
    return (
        ds
        .map(lambda feature, label: tfds_crop(
            feature, label, hparams[HP_INPUT_SIZE]
        ), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda feature, label: tfds_scale(
            feature, label, hparams[HP_SCALER]
        ), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda feature, label: tfds_pad(feature, label),
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=num_examples)
        .repeat()
        .batch(hparams[HP_BATCH_SIZE], drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def tfds_plot(
        dataset: tf.data.Dataset, ntraces: int, model: keras.Model
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(
        ntraces, figsize=(16, ntraces * 2), facecolor="white"
    )
    pred_iterator = dataset.unbatch().take(ntraces).as_numpy_iterator()

    for i in range(ntraces):
        pred_data = pred_iterator.next()
        pred_trace = pred_data[0].reshape(1, -1, 1)
        prediction = model.predict(pred_trace)
        prediction = prediction.flatten()
        pred_trace = pred_trace.flatten()
        pred_label = pred_data[1].flatten()
        ax[i].plot(pred_trace / tf.experimental.numpy.max(pred_trace))
        ax[i].plot(prediction)
        ax[i].plot(pred_label)
    plt.tight_layout()
    return fig


def log_plots(
    epoch: int, logs, dataset: tf.data.Dataset, model: keras.Model
) -> None:
    """Image logging function for tf.keras.callbacks.LambdaCallback

    Notes
    -----
    - `tf.keras.callbacks.LambdaCallback` expects two positional
      arguments `epoch` and `logs`, if `on_epoch_end` is being used
    - see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback
    """
    figure = tfds_plot(dataset=dataset, ntraces=5, model=model)
    mlflow.log_figure(
        figure=figure, artifact_file=f"predplots/plot_{epoch}.png")


def lr_schedule(epoch: int, hparams: dict):
    """
    Returns a custom learning rate that decreases as epochs progress.

    Notes
    -----
    - function is supposed to be used with
      `tf.keras.callbacks.LearningRateScheduler`. It takes an epoch
      index as input (integer, indexed from 0) and returns a new
      learning rate as output (float)
    """
    # power: 1 == linear decay, higher, e.g. 5 == polynomial decay
    lr_list = [
        hparams[HP_LR_START] *
        (1 - i / hparams[HP_EPOCHS])**hparams[HP_LR_POWER]
        for i in range(hparams[HP_EPOCHS])
    ]

    if epoch == 0:
        mlflow.log_param("lr schedule", value=str(lr_list))
    return lr_list[epoch]


def get_callbacks(
        ds_val: tf.data.Dataset, model: keras.Model, hparams: dict,
        hp_logdir: str, session_id: int
) -> list:
    def log_plots_wrapper(epoch, logs):
        return log_plots(epoch, logs, ds_val, model)

    def lr_schedule_wrapper(epoch):
        return lr_schedule(epoch, hparams)

    logdir = os.path.join(hp_logdir, str(session_id))

    image_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=log_plots_wrapper
    )
    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule_wrapper)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=5,
        write_graph=False,
        write_images=False,
        update_freq="epoch",
        profile_batch=0,  # workaround for issue #2084
    )
    hparams_callback = hp.KerasCallback(logdir, hparams)  # logs hparams
    mlflow_callback = mlflow.keras.callback.MlflowCallback()
    return [tensorboard_callback, lr_callback, image_callback, hparams_callback,
            mlflow_callback]


def run_one(
        ds_train: tf.data.Dataset, ds_val: tf.data.Dataset, hp_logdir: str,
        session_id: int, hparams: dict, num_train_examples: int,
        num_val_examples: int, best_auc_val: float
):
    """Run a training/validation session.

    Parameters:
    -----------
    train_ds, val_ds : tf.Dataset
        Train and validation data as tf.Datasets.
    hp_logdir : str
        The top-level logdir to which to write summary data.
    session_id : float, int, str
        A unique ID for this session.
    hparams : dict
        A dict mapping hyperparameters in `HPARAMS` to values.
    num_train_examples, num_val_examples : int
        number of train and validation examples
    best_auc_val : float
        Best validation AUC. If the trained model is currently the best,
        it is saved.

    Returns:
    --------
    best_auc_val : float
        Best validation AUC (currently)
    """

    ds_train = tfds_prepare(ds_train, hparams, num_train_examples)
    ds_val = tfds_prepare(ds_val, hparams, num_val_examples)

    model = unet_1d_hparams(hparams=hparams)

    steps_train = num_train_examples // hparams[HP_BATCH_SIZE]
    steps_val = num_val_examples // hparams[HP_BATCH_SIZE]

    callbacks = get_callbacks(ds_val, model, hparams, hp_logdir, session_id)

    result = model.fit(
        x=ds_train,
        epochs=2,  # hparams[HP_EPOCHS],
        steps_per_epoch=3,  # steps_trani
        validation_data=ds_val,
        validation_steps=2,  # steps_val
        callbacks=callbacks,
    )

    if result.history["auc"][-1] > best_auc_val:
        mlflow.keras.save.log_model(
            model=model,
            artifact_path="model",
            # conda_env=mlflow.keras.get_default_conda_env(
            #     keras_module=keras),
            custom_objects={"BinaryCrossentropyDice": BinaryCrossentropyDice()},
            # keras_module=keras,
            )
        best_auc_val = result.history["auc"][-1]

    return best_auc_val


@click.command()
@click.option("--num_session_groups",
              type=int,
              default=2,
              help="number of sessions for random search")
@click.option(
    "--file_train_feature",
    type=str,
    default="2025-05-28-peak-artifacts-training.parquet"
)
@click.option(
    "--file_train_label",
    type=str,
    default="2025-05-28-peak-artifacts-training.parquet"
)
@click.option(
    "--file_val_feature",
    type=str,
    default="2025-05-28-peak-artifacts-validation.parquet"
)
@click.option(
    "--file_val_label",
    type=str,
    default="2025-05-28-peak-artifacts-validation.parquet"
)
@click.option("--mlflow_tracking_uri", type=str, default="file:./data/mlruns")
@click.option("--experiment_name", type=str, default="hparams_unet")
@click.option("--is_remote", type=bool, default=False)
def hparams_run(
        num_session_groups, file_train_feature, file_train_label,
        file_val_feature, file_val_label, mlflow_tracking_uri, experiment_name,
        is_remote, rng=rng
):
    if not is_remote:
        os.chdir("/home/alva/Programs/drmed-git")

    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    df_train = get_data(file_train_feature, file_train_label)
    df_train = df_train.head()
    ds_train, num_train_ex = tfds_from_pldf(
        df_train["feature"], df_train["label_ground_truth"]
       )
    file_val = "2025-05-28-peak-artifacts-validation.parquet"
    df_val = get_data(file_val_feature, file_val_label)
    df_val = df_val.head()
    ds_val, num_val_ex = tfds_from_pldf(
        df_val["feature"], df_val["label_ground_truth"]
       )

    with mlflow.start_run(experiment_id=experiment.experiment_id) as parent_run:
        # mlflow.tensorflow.autolog(every_n_iter=1, log_models=False)
        best_auc_val = float(tf.experimental.numpy.finfo(
            tf.experimental.numpy.float64
           ).min)

        num_sessions = num_session_groups * SESSIONS_PER_GROUP

        # experiment_params_train.to_csv(EXP_PARAM_PATH_TRAIN)
        # experiment_params_val.to_csv(EXP_PARAM_PATH_VAL)

        session_index = 0  # across all session groups
        for _ in range(num_session_groups):
            hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
            hparams_mlflow = {h.name: hparams[h] for h in hparams.keys()}
            for repeat_index in range(SESSIONS_PER_GROUP):
                log.debug(f"--- Running training session {session_index + 1}/"
                          f"{num_sessions}")
                log.debug(hparams_mlflow)
                log.debug(f"--- repeat #: {repeat_index + 1}")
                with mlflow.start_run(nested=True) as _:
                    mlflow.log_params(hparams_mlflow)
                    mlflow.log_params({
                        "num_train_examples": num_train_ex,
                        "num_val_examples": num_val_ex
                       })
                    critical_hparam_combi = 2 * hparams_mlflow[
                        "hp_pool_size"]**hparams_mlflow["hp_n_levels"]
                    if critical_hparam_combi <= hparams_mlflow["hp_input_size"]:
                        best_auc_val = run_one(
                            ds_train=ds_train,
                            ds_val=ds_val,
                            hp_logdir=LOG_DIR,
                            session_id=session_index,
                            hparams=hparams,
                            num_train_examples=num_train_ex,
                            num_val_examples=num_val_ex,
                            best_auc_val=best_auc_val)
                    else:
                        log.debug(
                            "This run is skipped, because the following "
                            "condition (needed to build the model) was not "
                            "given: 2 * pool_size**n_levels <= input_size"
                        )
                session_index += 1

        # Now log best values in parent run
        client = mlflow.tracking.client.MlflowClient()
        runs = client.search_runs(
            [parent_run.info.experiment_id],
            f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'")
        best_auc_val = float(tf.experimental.numpy.finfo(
            tf.experimental.numpy.float64).min)
        best_run = None
        best_auc_train = best_auc_val
        best_auc_val = best_auc_val
        for r in runs:
            if r.data.metrics["val_auc"] > best_auc_val:
                best_run = r
                best_auc_train = r.data.metrics["auc"]
                best_auc_val = r.data.metrics["val_auc"]
        try:
            mlflow.set_tag("best_run", best_run.info.run_id)
        except AttributeError:
            log.debug("Logging the best run failed. Maybe check if MlflowClient"
                      " is set up correctly")
        mlflow.log_metrics({
            "best_auc": best_auc_train,
            "best_auc_val": best_auc_val
        })

# small workaround to check for 'get_ipython' to not cause error when transcluding
# the file in emacs
if __name__ == "__main__" and "get_ipython" not in dir():
    hparams_run()
