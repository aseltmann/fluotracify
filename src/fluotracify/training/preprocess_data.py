import numpy as np
import pandas as pd
import tensorflow as tf


def tfds_from_pddf_for_unet(features_df,
                            labels_df,
                            is_training,
                            batch_size,
                            length_delimiter=None,
                            frac_val=0.2):
    """TensorFlow Dataset from pandas DataFrame for UNET

    This function was created to take pandas DataFrames containing simulated
    fluorescence traces with artifacts (features) and the ground truth about
    the artifacts (labels) as an input and to prepare the data for the training
    pipeline (batch, shuffle, repeat, split in train + validation dataset).

    Parameters
    ----------
    features_df, labels_df : pandas DataFrames
        Contain features / labels ordered columnwise in the manner: feature_1,
        feature_2, ... / label_1, label_2, ...
    batch_size : int
        Batch size for dataset creation (machine learning terminology)
    is_training : bool
        if True: Dataset will be repeated, shuffled and batched + a validation
        dataset will be created (see frac_val).
        If False: Dataset will only be batched, since it is for testing.
    length_delimiter : int, optional
        Length of the output traces in the returned dataset. If None, then the
        whole length of the DataFrame is used
    frac_val : float, optional
        Fraction of training data used for validation (default 0.2). Only
        relevant if is_training = True.

    Returns
    -------
    dataset : TensorFlow Dataset
        Contains features and labels, already batched according to BATCH_SIZE
        (2 datasets (training, validation) if is_training = True)
    num_train_examples / num_val_examples / num_total_examples : int
        Number of examples in the dataset (2 numbers (training, validiation)
        if is_training = True, 1 number (for test) if is_training = False)
    """
    features_cropped = features_df.iloc[:length_delimiter, :]
    labels_cropped = labels_df.iloc[:length_delimiter, :]

    X_tensor = tf.convert_to_tensor(value=features_cropped.values)
    X_tensor = tf.transpose(a=X_tensor, perm=[1, 0])
    X_tensor_norm = min_max_normalize_tensor(X_tensor, axis=1)

    y_tensor = tf.convert_to_tensor(value=labels_cropped.values)
    y_tensor = tf.transpose(a=y_tensor, perm=[1, 0])
    y_tensor = tf.cast(y_tensor, tf.float32)

    num_total_examples = X_tensor_norm.shape[0]
    X_tensor_norm = tf.reshape(tensor=X_tensor_norm,
                               shape=(num_total_examples, -1, 1))
    y_tensor = tf.reshape(tensor=y_tensor, shape=(num_total_examples, -1, 1))

    if is_training:
        # for training: split dataset in training and validation set
        num_val_examples = round(frac_val * num_total_examples)
        num_train_examples = num_total_examples - num_val_examples

        dataset = tf.data.Dataset.from_tensor_slices((X_tensor_norm, y_tensor))
        dataset = dataset.shuffle(buffer_size=num_total_examples)
        dataset_train = dataset.take(num_train_examples).repeat().batch(
            batch_size)
        dataset_val = dataset.skip(num_train_examples).repeat().batch(
            batch_size)

        print('number of training examples: {}, number of validation '
              'examples: {}\n\n------------------------'.format(
                  num_train_examples, num_val_examples))
        return (dataset_train, dataset_val, num_train_examples,
                num_val_examples)
    # if we create a test dataset: no validation set, no shuffling
    dataset_test = tf.data.Dataset.from_tensor_slices(
        (X_tensor_norm, y_tensor)).batch(batch_size)

    print('number of test examples: {}\n'.format(num_total_examples))
    return dataset_test, num_total_examples


def tfds_from_pddf_for_vgg(features_df,
                           labels_df,
                           win_len,
                           ntraces_index,
                           ntraces_delimiter,
                           zoomvector,
                           label_threshold,
                           BATCH_SIZE,
                           is_training,
                           frac_val=0.2,
                           verbose=True,
                           _NUM_CLASSES=None):
    """TensorFlow Dataset from pandas DataFrame for VGG-style neural net

    This function was created to handle pandas DataFrames containing simulated
    fluorescence traces with artifacts (features) and the ground truth about
    the artifacts (labels) while providing options to change the input data
    (zoom levels, window length) and to prepare the data for the training
    pipeline (batch, shuffle, repeat, split in train + validation dataset).

    Parameters
    ----------
    features_df, labels_df : pandas DataFrames
        Contain features / labels ordered columnwise in the manner: feature_1,
        feature_2, ... / label_1, label_2, ...
    win_len : int
        Length of feature vector or length of trace the network should inspect.
    ntraces_index : int
        Index of trace used (which column to pick out of features_df)
    ntraces_delimiter : int or None
        Number of traces used / end value in slicing / None: no delimiter =
        use all traces
    zoomvector : tuple or list of uneven, positive integers
        Multiplier for each zoom level (zoom window = zoomvector * win_len)

    label_threshold : float
        Threshold defining when a trace is to be labelled as corrupted
    BATCH_SIZE : int
        Batch size for dataset creation (machine learning terminology)
    is_training : bool
        if True: Dataset will be repeated, shuffled and batched + a validation
        dataset will be created (see frac_val).
        If False: Dataset will only be batched, since it is for testing.
    frac_val : float, optional
        Fraction of training data used for validation (default 0.2)

    Returns
    -------
    dataset : TensorFlow Dataset
        Contains features and labels, already batched according to BATCH_SIZE
        (2 datasets (training, validation) in case of is_training)
    _NUM_EXAMPLES : int
        Number of examples in the dataset (2 numbers (training, validiation)
        in case of is_training)

    Notes
    -----
    A parameter which could be implemented to deal with simulations of
    multiple artifacts in one fluorescence trace.
    _NUM_CLASSES : int
        Number of classes / labels for one-hot operation on y_tensor (not yet
        implemented)

    """

    features_df = features_df.iloc[:, ntraces_index:(ntraces_index +
                                                     ntraces_delimiter)]

    if zoomvector is None:
        features_df_dict = []
    else:
        features_df_dict = _get_smoothed_traces_from_pandasdf(
            zoomvector=zoomvector,
            features_df=features_df,
            win_len=win_len,
            verbose=verbose)

    col_no = len(features_df_dict) + 1

    X = _get_windowed_X_features_from_pandasdf(
        index=0,
        features_df_dict=features_df_dict,
        col_no=col_no,
        win_len=win_len,
        features_df=features_df,
        ntraces_delimiter=ntraces_delimiter,
        zoomvector=zoomvector)
    y = _get_windowed_y_labels_from_pandasdf(
        index=0,
        labels_df=labels_df,
        win_len=win_len,
        ntraces_delimiter=ntraces_delimiter,
        label_threshold=label_threshold)

    X_tensor = tf.convert_to_tensor(value=X.values)
    y_tensor = tf.convert_to_tensor(value=y.values)

    for idx in np.arange(1, len(features_df) // win_len):
        X_window = _get_windowed_X_features_from_pandasdf(
            index=idx,
            features_df_dict=features_df_dict,
            col_no=col_no,
            win_len=win_len,
            features_df=features_df,
            ntraces_delimiter=ntraces_delimiter,
            zoomvector=zoomvector)
        X_temp = tf.convert_to_tensor(value=X_window.values)
        X_tensor = tf.concat(values=[X_tensor, X_temp], axis=0)

        y_window = _get_windowed_y_labels_from_pandasdf(
            index=idx,
            labels_df=labels_df,
            win_len=win_len,
            ntraces_delimiter=ntraces_delimiter,
            label_threshold=label_threshold)
        y_temp = tf.convert_to_tensor(value=y_window.values)
        y_tensor = tf.concat(values=[y_tensor, y_temp], axis=0)

    if verbose:
        print('this is the shape, dtype and an example of the features after '
              'concatenation\n{} {}\n{}\n'.format(X_tensor.shape,
                                                  X_tensor.dtype,
                                                  X_tensor.numpy()[:5, :5]))

    # Do preprocessing: min-max feature scaling (normalization), reshaping for
    # model compatibility, downcasting because of memory
    _NUM_EXAMPLES_total = len(y_tensor.numpy())
    X_tensor = min_max_normalize_tensor(X_tensor, axis=1)
    X_tensor = tf.reshape(X_tensor,
                          shape=(_NUM_EXAMPLES_total, col_no, win_len))
    X_tensor = tf.transpose(a=X_tensor, perm=[0, 2, 1])
    X_tensor = tf.cast(X_tensor, tf.float32)
    y_tensor = tf.cast(y_tensor, tf.float32)

    if verbose:
        print('this is the shape, dtype and an example of the features ready '
              'to feed into the TensorFlow pipeline\n{} {}\n{}\n'.format(
                  X_tensor.shape, X_tensor.dtype,
                  X_tensor.numpy()[0, :5, :]))
        print('shape {}, dtype {} of the labels, an example:\n{}\n all in all '
              'there are {:03.2f}% of the traces corrupt.\n'.format(
                  y_tensor.shape, y_tensor.dtype, y_tensor.numpy(),
                  sum(y_tensor.numpy()) * 100 / len(y_tensor.numpy())))

    if is_training:
        # for training: split dataset in training and validation set
        _NUM_EXAMPLES_val = int(round(frac_val * _NUM_EXAMPLES_total, 0))
        _NUM_EXAMPLES_train = int(
            round((1 - frac_val) * _NUM_EXAMPLES_total, 0))
        # shuffle first, otherwise structural error in data (no data with
        # padding at the end)
        dataset = tf.data.Dataset.from_tensor_slices(
            (X_tensor, y_tensor)).shuffle(buffer_size=_NUM_EXAMPLES_total)
        dataset_train = dataset.take(_NUM_EXAMPLES_train).repeat().batch(
            BATCH_SIZE)
        dataset_val = dataset.skip(_NUM_EXAMPLES_train).repeat().batch(
            BATCH_SIZE)

        print('number of training examples: {}, number of validation '
              'examples: {}\n\n------------------------'.format(
                  _NUM_EXAMPLES_train, _NUM_EXAMPLES_val))
        return (dataset_train, dataset_val, _NUM_EXAMPLES_train,
                _NUM_EXAMPLES_val, col_no)
    # if we create a test dataset: no validation set, no shuffling
    dataset_test = tf.data.Dataset.from_tensor_slices(
        (X_tensor, y_tensor)).batch(BATCH_SIZE)

    print('number of test examples: {}\n'.format(_NUM_EXAMPLES_total))
    return dataset_test, _NUM_EXAMPLES_total


def unet_preprocessing(features_df, length_delimiter, ntraces_index):
    """Preprocessing for UNET for application / deployment

    This function preprocesses one trace from a pandas DataFrames containing
    1D fluorescence traces and preprocesses it so that it can be fed for
    prediction to the UNET trained in the fluotracfiy project

    Parameters
    ----------
    features_df : pandas DataFrames
        Contain features ordered columnwise in the manner: feature_1,
        feature_2, ...
    length_delimiter : int, optional
        Length of the output traces in the returned dataset. If None, then the
        whole length of the DataFrame is used
    ntraces_index : int
        Index of trace used (which column to pick out of features_df)

    Returns
    -------
    X : numpy array
        Contains input features with original input values
    X_norm : numpy array
        Contains features after preprocessing for model application
    """
    # handle different lengths of traces of experimental data
    features_df = features_df.iloc[:length_delimiter,
                                   ntraces_index:(ntraces_index + 1)]
    features_df = features_df.dropna()

    X = np.array(features_df).flatten()
    X_norm = min_max_normalize_tensor(X, axis=0)
    X_norm = np.reshape(X_norm, newshape=(1, -1, 1))

    return X, X_norm

# FIXME: Rename to vgg_preprocessing()
def pandasdf_preprocessing(features_df,
                           win_len,
                           ntraces_index,
                           ntraces_delimiter,
                           zoomvector,
                           verbose=True):
    """Does the preprocessing like in the training pipeline (zoom levels,
    window length) for arbitrary fluorescence traces

    Parameters
    ----------
    features_df : pandas DataFrame
        Contains features ordered columnwise in the manner: features_1,
        features_2, ...
    win_len : int
        Length of feature vector or length of trace the network should inspect
    ntraces_index : int
        Index of trace used (which column to pick out of features_df)
    ntraces_delimiter : int
        Number of traces used / end value in slicing /
        None: no delimiter = use all
    zoomvector : tuple or list of positive, uneven int
        Multiplier for each zoom level (zoom window = zoomvector x win_len)

    Returns
    -------
    X : numpy array
        Contains input features with original input values
    X_norm : numpy array
        Contains features after preprocessing for model application
    _NUM_EXAMPLES : int
        Number of examples in the dataset
    col_no : int
        Number of columns per feature
    """
    assert ntraces_delimiter == 1

    # handle different lengths of traces of experimental data
    features_df = features_df.iloc[:, ntraces_index:(ntraces_index +
                                                     ntraces_delimiter)]
    features_df = features_df.dropna()

    if zoomvector is None:
        features_df_dict = []
    else:
        features_df_dict = _get_smoothed_traces_from_pandasdf(
            zoomvector=zoomvector,
            features_df=features_df,
            win_len=win_len,
            verbose=verbose)

    col_no = len(features_df_dict) + 1

    X = _get_windowed_X_features_from_pandasdf(
        index=0,
        features_df_dict=features_df_dict,
        col_no=col_no,
        win_len=win_len,
        features_df=features_df,
        ntraces_delimiter=ntraces_delimiter,
        zoomvector=zoomvector)

    for idx in np.arange(1, len(features_df) // win_len):
        X_temp = _get_windowed_X_features_from_pandasdf(
            index=idx,
            features_df_dict=features_df_dict,
            col_no=col_no,
            win_len=win_len,
            features_df=features_df,
            ntraces_delimiter=ntraces_delimiter,
            zoomvector=zoomvector)
        X = pd.concat([X, X_temp], axis=0)

    if verbose:
        print('this is the shape and an example of the features after '
              'concatenation\n{}\n{}\n'.format(X.shape, X.iloc[:5, :5].values))

    _NUM_EXAMPLES = int(len(X) / col_no)
    X = np.array(X)
    X_norm = min_max_normalize_tensor(X, axis=1)
    X_norm = np.reshape(X_norm, newshape=(_NUM_EXAMPLES, col_no, win_len))
    X_norm = np.transpose(X_norm, axes=[0, 2, 1])
    X = np.reshape(X, newshape=(_NUM_EXAMPLES, col_no, win_len))
    X = np.transpose(X, axes=[0, 2, 1])

    if verbose:
        print('this is the shape and an example of the features after min-max '
              ' normalization\n{}\n{}\n'.format(X_norm.shape,
                                                X_norm[0, :5, :]))
        print('this is the shape and an example of the features without min-'
              'max normalization\n{}\n{}\n'.format(X.shape, X[0, :5, :]))
        print('number of examples: {}, formed out of {} trace(s)'.format(
            _NUM_EXAMPLES, features_df.shape[1]))

    return X, X_norm, _NUM_EXAMPLES, col_no


# Helper functions
def _get_smoothed_traces_from_pandasdf(zoomvector, features_df, win_len,
                                       verbose):
    """creates zoom levels defined by zoomvector by applying a rolling
    mean. Note: this is the most expensive computational step!
    """
    features_df_dict = {}
    padding_zoom_dict = {}

    for idx, zoomvec in enumerate(zoomvector):
        # padding is for easier downsampling later
        padding_zoom_dict['padding_zoom{}'.format(zoomvec)] = pd.DataFrame(
            np.zeros(shape=(win_len * (zoomvec // 2),
                            len(features_df.columns))))
        pad = padding_zoom_dict['padding_zoom{}'.format(zoomvec)]
        features_df_dict['features_zoom{}'.format(zoomvec)] = pd.DataFrame(
            np.concatenate((pad.values, features_df.values, pad.values),
                           axis=0))
        feat = features_df_dict['features_zoom{}'.format(zoomvec)]

        feat.index = pd.to_datetime(feat.index, unit='ms')
        feat = feat.rolling(window=zoomvec,
                            win_type='gaussian').mean(std=1).fillna(0)

        if verbose:
            print('Created {}. zoom level of shape {}, which should create '
                  'windows of {}ms\n'.format(idx + 1, feat.shape,
                                             2 * len(pad) + win_len))

    return features_df_dict


def _get_windowed_X_features_from_pandasdf(index, features_df_dict, col_no,
                                           win_len, features_df,
                                           ntraces_delimiter, zoomvector):
    """Cuts the indexed part of the 20,000 rows long traces into chunks of
    length win_len, also downsamples the zoom levels using a median to
    chunks of length win_len, then outputs one big array for a TensorFlow
    pipeline

    Returns
    -------
    X : pandas DataFrame
        Features ordered row-wise. So if 1 is the first window with three
        zoom levels A, B, C, then output is in form: row1 = 1A, row2 = 1B,
        row3 = 1C, row4 = 2A, ...
    """
    if ntraces_delimiter is None:
        # create zero-array height x width where height = win_len and
        # width = original trace + all zoom levels
        X = pd.DataFrame(
            np.zeros(shape=(win_len, len(features_df.columns) * col_no)))
    elif ntraces_delimiter is not None:
        X = pd.DataFrame(np.zeros(shape=(win_len, ntraces_delimiter * col_no)))

    # first trace: the original trace with length win_len
    X.iloc[:, ::col_no] += features_df.iloc[win_len * index:win_len *
                                            (index +
                                             1), :ntraces_delimiter].values

    if features_df_dict == []:
        # ends the function, if we don't want any zoom levels
        return X.T

    # second to ... trace: the zoomed traces with length win_len
    for jdx, (zoomvec, feat) in enumerate(zip(zoomvector, features_df_dict),
                                          start=1):
        # start and end should be integers (for slicing) and multiples of
        # zoomvec (for pd.DataFrame.resample to work as expected)
        start = int(zoomvec * round(win_len * index / zoomvec))
        end = int(zoomvec * round(win_len * (zoomvec + index) / zoomvec))
        feat_window = features_df_dict[feat].iloc[
            start:end, :ntraces_delimiter]
        X.iloc[:, jdx::col_no] += feat_window.resample(rule=str(zoomvec) +
                                                       'ms').median().values

    return X.T


def _get_windowed_y_labels_from_pandasdf(index, labels_df, win_len,
                                         ntraces_delimiter, label_threshold):
    """Create one label for every X chunk including its zoom levels"""
    # if more than label_threshold time steps are corrupted, label the
    # trace as corrupted (remember: 1 timestep is corrupted, if the
    # intensity in the simulated corruption trace is above a certain
    # threshold)
    y = labels_df.iloc[win_len * index:win_len *
                       (index + 1), :ntraces_delimiter]
    y.iloc[0, :] = y.sum(axis=0) > label_threshold
    return y.iloc[0, :]


def min_max_normalize_tensor(tensor, axis):
    """Rescale the range in [0, 1]

    Parameters
    ----------
    tensor : tensorflow tensor
        input feature to be normalized
    axis : int
        axis along which `tf.math.reduce_min` and `tf.math.reduce.max` are
        performing the respective operation

    Returns
    -------
    normalized input tensor
    """
    tensor_min = tf.math.reduce_min(input_tensor=tensor,
                                    axis=axis,
                                    keepdims=True)
    tensor_max = tf.math.reduce_max(input_tensor=tensor,
                                    axis=axis,
                                    keepdims=True)
    return (tensor - tensor_min) / (tensor_max - tensor_min)
