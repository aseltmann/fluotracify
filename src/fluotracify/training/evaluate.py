import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def plot_history(history):
    """Plot training history"""
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Accuracy')
    plt.ylim([0.8, 1])
    plt.legend()

    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Loss')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.show()

    return hist


def predict_traces(dataset, model, num_rows, num_cols, batch_to_sample,
                   batch_size, width_ratio, figlabels, figlegloc):
    """Plot a sample of traces with its predictions

    Parameters
    ----------
    dataset : tf.Dataset
        Contains features and labels
    model
        The model trained on the dataset to predict the traces
    num_rows, num_cols : int
        Number of rows / columns with trace-bar-pair-plots
    batch_to_sample : int
        Batch to take from Test Dataset for prediction
    batch_size : int
        Maximum number of predictions to create with this function
    width_ratio : tuple or list of int
        Ratio of trace and bar plots
    figlabels : list of str
        Contains description of handles (e.g. 1 original trace + 2 zoom levels)

    Outputs:
    A figure with num_rows * num_cols subplots sampled from a tf.Dataset
    """
    def _plot_trace(i, predictions_array, true_labels, traces):
        predicted_label = predictions_array[i]
        true_label = true_labels[i]
        trace = traces[i]
        predicted_label = predicted_label > 0.95
        predicted_percent = float(100 * predictions_array[i])

        lines = plt.plot(trace)

        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.ylabel('norm. Int. in AU')
        plt.xlabel('time steps in ms')
        plt.title(
            'Trace {}. True Corruption: {}. Predicted Corruption: {:2.0f}%'.
            format(i + 1, true_label, predicted_percent),
            color=color)
        plt.figlegend(handles=lines[:], labels=figlabels, loc=figlegloc)

    def _plot_probability_bar(i, predictions_array, true_labels):
        predicted_label, true_label = predictions_array[i], true_labels[i]
        plt.xticks([])
        thisplot = plt.bar(range(len(predictions_array[0]) + 1),
                           (predicted_label, true_label),
                           color='#777777')
        plt.ylim([0, 1])
        predicted_label = predicted_label > 0.95
        thisplot[int(predicted_label)].set_color('red')
        thisplot[int(true_label)].set_color('blue')

    num_traces = num_rows * num_cols

    if num_traces > batch_size:
        raise ValueError('num_traces (= num_rows * num_cols) must not be '
                         ' bigger than batch_size')
    if batch_to_sample == 0:
        raise ValueError('0 is not a valid number. you need at least 1 '
                         'element from the dataset (-1 takes all elements)')

    traces = None
    names = None
    for traces, names in dataset.take(batch_to_sample):
        # dataset.take(1) takes first batch (e.g. = 32 Examples)
        traces = traces.numpy()
        names = names.numpy()
        predictions = model.predict(traces)

    fig = plt.figure(figsize=(4 * 2 * num_cols, 2 * num_rows),
                     constrained_layout=True)
    spec = fig.add_gridspec(nrows=num_rows,
                            ncols=2 * num_cols,
                            width_ratios=width_ratio * num_cols)
    fig.suptitle(
        'Blue: Prediction = Ground Truth. Red: Corruption True, but '
        'Prediction < 95% (assumed as no corruption) \nBarplot shows '
        'prediction on the left and ground truth on the right',
        y=1.1,
        fontsize=15)

    for i in range(num_traces):
        plt.subplot(spec[2 * i])
        _plot_trace(i, predictions, names, traces)
        plt.subplot(spec[2 * i + 1])
        _plot_probability_bar(i, predictions, names)

    plt.show()

def plot_trace_and_pred_from_df(df, ntraces, model):
    fig, ax = plt.subplots(ntraces, figsize=(16, ntraces*2))

    for i in range(ntraces):
        pred_trace = df.iloc[:16384, i].to_numpy().reshape(1, -1, 1)
        prediction = model.predict(pred_trace)
        prediction = prediction.flatten()
        pred_trace = pred_trace.flatten()
        ax[i].plot(pred_trace / np.max(pred_trace))
        ax[i].plot(prediction)
    return fig

def plot_trace_and_pred_from_tfds(dataset, ntraces, model):
    fig, ax = plt.subplots(ntraces, figsize=(16, ntraces*2))
    pred_iterator = dataset.unbatch().take(ntraces).as_numpy_iterator()

    for i in range(ntraces):
        pred_data = pred_iterator.next()
        pred_trace = pred_data[0].reshape(1, -1, 1)
        prediction = model.predict(pred_trace)
        prediction = prediction.flatten()
        pred_trace = pred_trace.flatten()
        pred_label = pred_data[1].flatten()
        ax[i].plot(pred_trace / np.max(pred_trace))
        ax[i].plot(prediction)
        ax[i].plot(pred_label)
    plt.tight_layout()
    return fig

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.

    Notes
    -----
    - Stolen from https://www.tensorflow.org/tensorboard/image_summaries
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
