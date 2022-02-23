"""This module contains a script to streamline the training of a neural network
on simulated fluroescence traces using mlflow. It is meant to be put into the
`MLproject` file of a mlflow setup or executed via `mlflow run`."""

import datetime
import logging
import sys

import click
import matplotlib
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow.python.platform.build_info as build

# fixes a problem when calling plotting functions on the server
matplotlib.use('agg')
# use logging
logging.basicConfig(format='%(asctime)s - train -  %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

log.debug("Python version: %s", sys.version)
log.debug("Tensorflow version: %s", tf.__version__)
log.debug("tf.keras version: %s", tf.keras.__version__)
log.debug('Cudnn version: %s', build.build_info['cudnn_version'])
log.debug('Cuda version: %s', build.build_info['cuda_version'])
# Workaround for a "No algorithm worked" bug on GPUs
# see https://github.com/tensorflow/tensorflow/issues/45044
physical_devices = tf.config.list_physical_devices('GPU')
log.debug('GPUs: %s. Trying to set memory growth to "True"...',
          physical_devices)
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    log.debug('Setting memory growth successful.')
except IndexError:
    log.debug('No GPU was found on this machine. ')

# logging with mlflow: most of the logging happens with the autolog feature,
# which is already quite powerful. Since I want to add custom logs, I have to
# start and end the log manually with mlflow.start_run() and mlflow.end_run()


@click.command()
@click.option('--batch_size', type=int, default=5)
@click.option('--input_size', default=16384)
@click.option('--lr_start', type=float, default=1e-5)
@click.option('--lr_power', type=float, default=1)
@click.option('--epochs', type=int, default=10)
@click.option(
    '--csv_path_train',
    type=click.Path(exists=True),
    default= '~/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/')
@click.option(
    '--csv_path_val',
    type=click.Path(exists=True),
    default= '~/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/')
@click.option('--col_per_example', type=int, default=3)
@click.option('--scaler', type=click.Choice(['standard', 'robust', 'maxabs',
                                             'quant_g', 'minmax', 'l1', 'l2']),
              default='robust')
@click.option('--n_levels', type=int, default=9)
@click.option('--first_filters', type=int, default=64)
@click.option('--pool_size', type=int, default=2)
@click.option('--fluotracify_path',
              type=click.Path(exists=True),
              default='~/Programme/drmed-git/src/')
def mlflow_run(batch_size, input_size, lr_start, lr_power, epochs,
               csv_path_train, csv_path_val, col_per_example, scaler, n_levels,
               first_filters, pool_size, fluotracify_path):
    sys.path.append(fluotracify_path)
    if True:  # isort workaround
        from fluotracify.simulations import import_simulation_from_csv as isfc
        from fluotracify.training import (build_model as bm,
                                          preprocess_data as ppd)
        from fluotracify.training import evaluate

    LOG_DIR = "../tmp/tb-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LABEL_THRESH = 0.04
    # FIXME (PENDING): at some point, I want to plot metrics vs thresholds
    # from TF side, this is possible by providing the `thresholds`
    # argument as a list of thresholds
    # but currently, mlflow does not support logging lists, so I log the
    # elements of the list one by one
    METRICS_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
    EXP_PARAM_PATH_TRAIN = '../tmp/experiment_params_train.csv'
    EXP_PARAM_PATH_VAL = '../tmp/experiment_params_val.csv'

    def run_one(dataset_train, dataset_val, logdir, num_train_examples,
                num_val_examples):
        """Run a training/validation session.

        Parameters:
        -----------
        dataset_train, dataset_val : tf.Dataset
            Train and validation data as tf.Datasets.
        logdir : str
            The top-level logdir to which to write summary data.
        num_train_examples, num_val_examples : int
            number of train and validation examples

        Returns:
        --------
        """
        ds_train_prep = dataset_train.map(
            lambda trace, label: ppd.tf_crop_trace(trace, label, input_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.map(
            lambda trace, label: ppd.tf_scale_trace(trace, label, scaler),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.shuffle(
            buffer_size=num_train_examples).repeat().batch(
                batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        ds_val_prep = dataset_val.map(
            lambda trace, label: ppd.tf_crop_trace(trace, label, input_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.map(
            lambda trace, label: ppd.tf_scale_trace(trace, label, scaler),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.shuffle(
            buffer_size=num_val_examples).repeat().batch(
                batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        model = bm.unet_1d_alt2(input_size=input_size,
                                n_levels=n_levels,
                                first_filters=first_filters,
                                pool_size=pool_size,
                                metrics_thresholds=METRICS_THRESHOLDS)

        def log_plots(epoch, logs):
            """Image logging function for tf.keras.callbacks.LambdaCallback

            Notes
            -----
            - `tf.keras.callbacks.LambdaCallback` expects two positional
              arguments `epoch` and `logs`, if `on_epoch_end` is being used
            - see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback
            """
            figure = evaluate.plot_trace_and_pred_from_tfds(
                dataset=ds_val_prep, ntraces=5, model=model)
            # Convert matplotlib figure to image
            mlflow.log_figure(
                figure=figure,
                artifact_file='predplots/plot{}.png'.format(epoch))

        def lr_schedule(epoch):
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
            lr_list = [lr_start * (1 - i / epochs)**lr_power
                       for i in range(epochs)]

            # log in mlflow
            if epoch == 0:
                mlflow.log_param('lr schedule', value=str(lr_list))
            return lr_list[epoch]

        tensorboard_callback = tf.keras.callbacks.TensorBoard(  # logs metrics
            log_dir=logdir,
            histogram_freq=5,
            write_graph=False,
            write_images=False,
            update_freq='epoch',
            profile_batch=0,  # workaround for issue #2084
        )
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        image_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_plots)

        steps_train = num_train_examples // batch_size
        steps_val = num_val_examples // batch_size

        model.fit(
            x=ds_train_prep,
            epochs=epochs,
            steps_per_epoch=steps_train,
            validation_data=ds_val_prep,
            validation_steps=steps_val,
            callbacks=[tensorboard_callback, lr_callback, image_callback],
        )

        mlflow.keras.log_model(
            keras_model=model,
            artifact_path='model',
            conda_env=mlflow.keras.get_default_conda_env(
                keras_module=tf.keras),
            custom_objects={'binary_ce_dice': bm.binary_ce_dice_loss()},
            keras_module=tf.keras)

    train, _, nsamples_train, experiment_params_train = isfc.import_from_csv(
        folder=csv_path_train,
        header=12,
        frac_train=1,
        col_per_example=col_per_example,
        dropindex=None,
        dropcolumns=None)

    val, _, nsamples_val, experiment_params_val = isfc.import_from_csv(
        folder=csv_path_val,
        header=12,
        frac_train=1,
        col_per_example=col_per_example,
        dropindex=None,
        dropcolumns=None)

    train_sep = isfc.separate_data_and_labels(array=train,
                                              nsamples=nsamples_train,
                                              col_per_example=col_per_example)

    val_sep = isfc.separate_data_and_labels(array=val,
                                            nsamples=nsamples_val,
                                            col_per_example=col_per_example)

    # '0': trace with artifact
    # '1': just the simulated artifact (label for unet)
    # '2': whole trace without artifact (label for vae)

    train_data = train_sep['0']
    train_labels = train_sep['1']
    train_labels_bool = train_labels > LABEL_THRESH

    val_data = val_sep['0']
    val_labels = val_sep['1']
    val_labels_bool = val_labels > LABEL_THRESH

    # Cleanup
    del train, val, train_sep, val_sep

    dataset_train, num_train_examples = ppd.tfds_from_pddf(
        features_df=train_data, labels_df=train_labels_bool, frac_val=False)

    dataset_val, num_val_examples = ppd.tfds_from_pddf(
        features_df=val_data, labels_df=val_labels_bool, frac_val=False)

    with mlflow.start_run() as _:
        mlflow.tensorflow.autolog(every_n_iter=1, log_models=False)

        experiment_params_train.to_csv(EXP_PARAM_PATH_TRAIN)
        experiment_params_val.to_csv(EXP_PARAM_PATH_VAL)
        mlflow.log_artifact(EXP_PARAM_PATH_TRAIN)
        mlflow.log_artifact(EXP_PARAM_PATH_VAL)
        mlflow.log_params({
            'num_train_examples': num_train_examples,
            'num_val_examples': num_val_examples
        })

        run_one(dataset_train=dataset_train,
                dataset_val=dataset_val,
                logdir=LOG_DIR,
                num_train_examples=num_train_examples,
                num_val_examples=num_val_examples)


if __name__ == "__main__":
    mlflow_run()
