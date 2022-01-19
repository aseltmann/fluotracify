"""This module contains functions to build neural networks."""

import logging
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

logging.basicConfig(format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def vgg10_1d(win_len, col_no):
    """Defines a VGG16-inspired model architecture with 10 weight layers"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=3,
                               padding='same',
                               activation='relu',
                               input_shape=(win_len, col_no)),
        tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def vgg16_1d(win_len, col_no):
    """Defines a VGG16-inspired model architecture with 16 weight layers"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=3,
                               padding='same',
                               activation='relu',
                               input_shape=(win_len, col_no)),
        tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# Block definitions for U-Net
def conv_block(input_tensor, num_filters):
    """Convolutional layers incl BatchNorm"""
    conv = tf.keras.layers.Conv1D(filters=num_filters,
                                  kernel_size=3,
                                  padding='same')(input_tensor)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    conv = tf.keras.layers.Conv1D(num_filters, 3, padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    return conv


def encoder_block(input_tensor, num_filters):
    """Convolutional layers plus max pool layers"""
    encode = conv_block(input_tensor, num_filters)
    encode_pool = tf.keras.layers.MaxPool1D(pool_size=2)(encode)
    return encode_pool, encode


class Conv1DTranspose(tf.keras.layers.Layer):
    """The 1D implementation of Conv2DTranspose

    Notes
    -----
    Implementation taken from this thread:
    https://github.com/tensorflow/tensorflow/issues/30309

    There is already a implementation of this function in
    tf.nn.conv1d_transpose`, but still no tf.keras.Conv1DTranspose.

    The arithmetics of a transposed convolution are nicely described here:
    https://arxiv.org/pdf/1603.07285.pdf. The original U-Net publication by
    Ronneberger et. al. uses bed-of-nails upsampling combined with a regular
    convolution. In a transposed convolution, these steps are combined (not
    literally, but conceptually)
    """
    def __init__(self, filters, kernel_size, strides=2, padding='same'):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding=padding)

    def call(self, x):
        x = tf.expand_dims(input=x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(input=x, axis=2)
        return x


def decoder_block(input_tensor, concat_tensor, num_filters):
    """Transposed convolution and concatenation with output of corresponding
    encoder level, then convolution of both reducing the depth
    """
    decode = Conv1DTranspose(filters=num_filters, kernel_size=2)(input_tensor)
    decode = tf.keras.layers.concatenate([concat_tensor, decode], axis=-1)
    decode = tf.keras.layers.BatchNormalization()(decode)
    decode = tf.keras.layers.Activation('relu')(decode)
    decode = conv_block(input_tensor=decode, num_filters=num_filters)
    return decode


def unet_1d(input_size):
    """U-Net as described by Ronneberger et al.

    Parameters
    ----------
    input_size : int
        Input vector size

    Returns
    -------
    Model as described by the tensorflow.keras Functional API

    Notes
    -----
    - Paper: https://arxiv.org/pdf/1505.04597.pdf
    - conceptually different approach than in the paper is the use of
    transposed convolution opposed to a "up-convolution" consisting of
    bed-of-nails upsampling and a 2x2 convolution
    """
    inputs = tf.keras.layers.Input(shape=(input_size, 1))

    # down_sampling
    encoder0_pool, encoder0 = encoder_block(input_tensor=inputs,
                                            num_filters=64)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 128)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 256)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 512)

    center = conv_block(input_tensor=encoder3_pool, num_filters=1024)

    # up_sampling
    decoder3 = decoder_block(input_tensor=center,
                             concat_tensor=encoder3,
                             num_filters=512)
    decoder2 = decoder_block(decoder3, encoder2, 256)
    decoder1 = decoder_block(decoder2, encoder1, 128)
    decoder0 = decoder_block(decoder1, encoder0, 64)

    # create 'binary' output vector
    outputs = tf.keras.layers.Conv1D(filters=1,
                                     kernel_size=1,
                                     activation='sigmoid')(decoder0)

    print('input - shape:\t', inputs.shape)
    print('output - shape:\t', outputs.shape)

    unet = tf.keras.Model(inputs=inputs, outputs=outputs)
    return unet


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

    return tf.keras.backend.binary_crossentropy(y_true, y_pred) + dice_loss(
        y_true, y_pred, axis, smooth)


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


# notes
# - Dropout could be added


# Alternative U-Net definition
def convtrans_old(filters, name):
    """Sequential API: Conv1DTranspose, BatchNorm"""
    upsamp = tf.keras.Sequential(name=name)
    upsamp.add(Conv1DTranspose(filters=filters, kernel_size=2, strides=2))
    upsamp.add(tf.keras.layers.BatchNormalization())
    upsamp.add(tf.keras.layers.Activation('relu'))
    return upsamp


def convtrans(filters, name, kernel_size, strides):
    """Sequential API: Conv1DTranspose, BatchNorm"""
    upsamp = tf.keras.Sequential(name=name)
    upsamp.add(
        tf.keras.layers.Conv1DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides))
    upsamp.add(tf.keras.layers.BatchNormalization())
    upsamp.add(tf.keras.layers.Activation('relu'))
    return upsamp


def twoconv(filters, name):
    """Sequential API: Conv1D, BatchNorm, Conv1D, BatchNorm"""
    conv = tf.keras.Sequential(name=name)
    conv.add(
        tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding='same'))
    conv.add(tf.keras.layers.BatchNormalization())
    conv.add(tf.keras.layers.Activation('relu'))

    conv.add(
        tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding='same'))
    conv.add(tf.keras.layers.BatchNormalization())
    conv.add(tf.keras.layers.Activation('relu'))
    return conv


def encoder(input_tensor, filters, name, pool_size=2):
    """Functional API: Two Conv1D incl BatchNorm, MaxPool1D"""
    encode = twoconv(filters=filters, name=name)(input_tensor)
    encode_pool = tf.keras.layers.MaxPool1D(pool_size=pool_size,
                                            name='mp_{}'.format(name))(encode)
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
                       name='conv_transpose_{}'.format(name),
                       kernel_size=kernel_size,
                       strides=strides)(input_tensor)
    decode = tf.keras.layers.concatenate([concat_tensor, decode],
                                         axis=-1,
                                         name=name)
    decode = twoconv(filters=filters, name='two_conv_{}'.format(name))(decode)
    return decode


# length of your training timeline (needs to be constant during training, can
# be anything when predicting) corresponding to the depth of your U-net
# (number of down- and upsamplings) the minimum lenght should be about 30 time
# steps or less


def unet_1d_alt(input_size):
    """U-Net as described by Ronneberger et al.

    Parameters
    ----------
    input_size : int
        Input vector size

    Returns
    -------
    Model as described by the tensorflow.keras Functional API

    Notes
    -----
    - Paper: https://arxiv.org/pdf/1505.04597.pdf
    - conceptually different approach than in the paper is the use of
    transposed convolution opposed to a "up-convolution" consisting of
    bed-of-nails upsampling and a 2x2 convolution
    - this implementation was influenced by:
    https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    inputs = tf.keras.layers.Input(shape=(input_size, 1))  # (bs, 16384, 1)

    # Downsampling through model
    x0_pool, x0 = encoder(inputs, 64, name='encode0')  # (bs, 8192, 64)
    x1_pool, x1 = encoder(x0_pool, 128, name='encode1')  # (bs, 4096, 128)
    x2_pool, x2 = encoder(x1_pool, 256, name='encode2')  # (bs, 2048, 256)
    x3_pool, x3 = encoder(x2_pool, 512, name='encode3')  # (bs, 1024, 512)
    x4_pool, x4 = encoder(x3_pool, 512, name='encode4')  # (bs, 512, 512)
    x5_pool, x5 = encoder(x4_pool, 512, name='encode5')  # (bs, 256, 512)
    x6_pool, x6 = encoder(x5_pool, 512, name='encode6')  # (bs, 128, 512)
    x7_pool, x7 = encoder(x6_pool, 512, name='encode7')  # (bs, 64, 512)
    x8_pool, x8 = encoder(x7_pool, 512, name='encode8')  # (bs, 32, 512)

    # Center
    center = twoconv(1024, name='two_conv_center')(x8_pool)  # (bs, 32, 1024)

    # Upsampling through model
    y8 = decoder(center, x8, 512, name='decoder8')  # (bs, 64, 1024)
    y7 = decoder(y8, x7, 512, name='decoder7')  # (bs, 128, 1024)
    y6 = decoder(y7, x6, 512, name='decoder6')  # (bs, 256, 1024)
    y5 = decoder(y6, x5, 512, name='decoder5')  # (bs, 512, 1024)
    y4 = decoder(y5, x4, 512, name='decoder4')  # (bs, 1024, 1024)
    y3 = decoder(y4, x3, 512, name='decoder3')  # (bs, 2048, 1024)
    y2 = decoder(y3, x2, 256, name='decoder2')  # (bs, 4096, 512)
    y1 = decoder(y2, x1, 128, name='decoder1')  # (bs, 8192, 128)
    y0 = decoder(y1, x0, 64, name='decoder0')  # (bs, 16384, 64)

    # create 'binary' output vector
    outputs = tf.keras.layers.Conv1D(filters=1,
                                     kernel_size=1,
                                     activation='sigmoid')(y0)

    print('input - shape:\t', inputs.shape)
    print('output - shape:\t', outputs.shape)

    unet = tf.keras.Model(inputs=inputs, outputs=outputs)
    return unet


def unet_1d_alt2(input_size, n_levels, first_filters, pool_size):
    """U-Net as described by Ronneberger et al.

    Parameters
    ----------
    input_size : int
        Input vector size
    n_levelsb : int
        Number of levels or steps in the Unet
    first_filters : int
        The number of filters in the first level. Every deeper level
        will be twice as many filters till a maximum of 512 is reached.
        Filters will be clipped if smaller than 1 or bigger than 512
    pool_size : int, Optional. Default: 2
        Pool size of the MaxPool1D layer, as well as kernel size and
        strides of the Conv1DTranspose layer

    Returns
    -------
    Model as described by the tensorflow.keras Functional API

    Raises
    ------
    - ValueError: the number of filters in filters_ls has to be equal
    to n_levels

    Notes
    -----
    - Paper: https://arxiv.org/pdf/1505.04597.pdf
    - conceptually different approach than in the paper is the use of
    transposed convolution opposed to a up"-convolution" consisting of
    bed-of-nails upsampling and a 2x2 convolution
    - this implementation was influenced by:
    https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    filters = [first_filters]
    nextfilters = first_filters
    for _ in range(1, n_levels + 1):
        nextfilters *= 2
        filters.append(nextfilters)
    filters = tf.experimental.numpy.clip(filters, a_min=1, a_max=512).numpy()
    filters = tf.cast(filters, tf.int32).numpy()

    ldict = {}

    inputs = tf.keras.layers.Input(shape=(input_size, 1))

    # Downsampling through model
    ldict['x0_pool'], ldict['x0'] = encoder(inputs,
                                            filters[0],
                                            name='encode0',
                                            pool_size=pool_size)
    for i in range(1, n_levels):
        ldict['x{}_pool'.format(i)], ldict['x{}'.format(i)] = encoder(
            input_tensor=ldict['x{}_pool'.format(i - 1)],
            filters=filters[i],
            name='encode{}'.format(i),
            pool_size=pool_size)

    # Center
    center = twoconv(2 * filters[n_levels - 1], name='two_conv_center')(
        ldict['x{}_pool'.format(n_levels - 1)])

    # Upsampling through model
    ldict['y{}'.format(n_levels - 1)] = decoder(
        input_tensor=center,
        concat_tensor=ldict['x{}'.format(n_levels - 1)],
        filters=filters[-1],
        name='decoder{}'.format(n_levels - 1),
        kernel_size=pool_size,
        strides=pool_size)

    for j in range(1, n_levels):
        ldict['y{}'.format(n_levels - 1 - j)] = decoder(
            input_tensor=ldict['y{}'.format(n_levels - j)],
            concat_tensor=ldict['x{}'.format(n_levels - 1 - j)],
            filters=filters[-1 - j],
            name='decoder{}'.format(n_levels - 1 - j),
            kernel_size=pool_size,
            strides=pool_size)

    # create 'binary' output vector
    outputs = tf.keras.layers.Conv1D(filters=1,
                                     kernel_size=1,
                                     activation='sigmoid')(ldict['y0'])

    print('input - shape:\t', inputs.shape)
    print('output - shape:\t', outputs.shape)

    unet = tf.keras.Model(inputs=inputs,
                          outputs=outputs,
                          name='unet_depth{}'.format(n_levels))
    return unet


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", from_logits=False, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(from_logits)
        self.recall = tf.keras.metrics.Recall(from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return (2 * p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def unet_metrics(metrics_thresholds):
    """Returns a selection of metrics for model training

    Currently these metrics are True Positives, False Positives, True
    Negatives, False Negatives, Preciesion, Recall, Accuracy, AUC

    Parameters
    ----------
    metrics_thresholds: list of float between 0 and 1

    Returns
    -------
    list of metrics
    """
    metrics = []
    for thresh in metrics_thresholds:
        metrics.append(
            tf.keras.metrics.TruePositives(name='tp{}'.format(thresh),
                                           thresholds=thresh))
        metrics.append(
            tf.keras.metrics.FalsePositives(name='fp{}'.format(thresh),
                                            thresholds=thresh))
        metrics.append(
            tf.keras.metrics.TrueNegatives(name='tn{}'.format(thresh),
                                           thresholds=thresh))
        metrics.append(
            tf.keras.metrics.FalseNegatives(name='fn{}'.format(thresh),
                                            thresholds=thresh))
        metrics.append(
            tf.keras.metrics.Precision(name='precision{}'.format(thresh),
                                       thresholds=thresh))
        metrics.append(
            tf.keras.metrics.Recall(name='recall{}'.format(thresh),
                                    thresholds=thresh))
    metrics.append(
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5))
    metrics.append(tf.keras.metrics.AUC(name='auc', num_thresholds=100))
    metrics.append(F1Score(name='f1'))
    return metrics


def unet_hp_metrics(metrics_thresholds):
    """Returns unet metrics Tensorboard HParams objects for logging"""
    metrics = []
    for thresh in metrics_thresholds:
        metrics.append(hp.Metric("tp{}".format(thresh)))
        metrics.append(hp.Metric("fp{}".format(thresh)))
        metrics.append(hp.Metric("tn{}".format(thresh)))
        metrics.append(hp.Metric("fn{}".format(thresh)))
        metrics.append(hp.Metric("precision{}".format(thresh)))
        metrics.append(hp.Metric("recall{}".format(thresh)))
    metrics.append(hp.Metric("accuracy{}".format(0.5)))
    metrics.append(hp.Metric("auc"))
    metrics.append(hp.Metric("f1"))
    return metrics


def prepare_model(model):
    """Iterate through multiple input sizes to make tensorflow accept them

    Notes
    -----
    Conceptually, the UNET is a fully convolutional model and should accept
    different input sizes. But after calling the model on an input size it was
    not trained on the first time, it will throw an error. After iterating
    through more different input sizes, the error will be gone.
    """
    for ldelim in [2**14, 2**13, 2**12, 2**13]:
        test_features = tf.experimental.numpy.zeros((ldelim))
        test_features = tf.experimental.numpy.reshape(a=test_features,
                                                      newshape=(1, -1, 1))
        try:
            predictions = model.predict(test_features, verbose=0).flatten()
            log.debug('prepare_model: test shape %s, e.g. %s',
                      test_features.shape, predictions[:5])
        except ValueError:
            log.debug('prepare_model: test shape %s. prediction failed '
                      'as expected. Retry...', test_features.shape)
    log.debug('prepare_model: UNET ready for different trace lengths')
