import tensorflow as tf


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
    """Convolutional layers"""
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
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = tf.keras.layers.MaxPool1D(pool_size=2)(encoder)
    return encoder_pool, encoder


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
    decoder = Conv1DTranspose(filters=num_filters, kernel_size=2)(input_tensor)
    decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    decoder = conv_block(input_tensor=decoder, num_filters=num_filters)
    return decoder


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
def binary_ce_dice_loss(y_true, y_pred, axis=-1, smooth=1e-5):
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
    this function was influenced by code from
        - TensorLayer project
        https://tensorlayer.readthedocs.io/en/latest/modules/cost.html#tensorlayer.cost.dice_coe,
        - Lars Nieradzik
        https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
        - Code by Stefan Hoffmann, Applied Systems Biology group,
        Hans-Knöll-Institute Jena

    Note
    ----
    - binary crossentropy returns a tensor with loss for each 1D step of a
    trace, bringing local info
    - dice loss returns a scalar for each 1D trace, bringing global info
    """
    def dice_loss(y_true, y_pred, axis=axis, smooth=smooth):
        """Soft dice coefficient for comparing the similarity of two batches
        of data, usually used for binary image segmentation

        For binary labels, the dice loss will be between 0 and 1 where 1 is a
        total match. Reshaping is needed to combine the global dice loss with
        the local binary_crossentropy
        """
        numerator = 2 * tf.math.reduce_sum(input_tensor=y_true * y_pred,
                                           axis=axis, keepdims=True)
        denominator = tf.math.reduce_sum(input_tensor=y_true + y_pred,
                                         axis=axis, keepdims=True)

        return 1 - (numerator + smooth) / (denominator + smooth)

    return tf.keras.backend.binary_crossentropy(y_true, y_pred) + dice_loss(
        y_true, y_pred)
