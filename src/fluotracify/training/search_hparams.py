"""This module contains a script to streamline the training of a neural network
on simulated fluroescence traces using mlflow. It is meant to be put into the
`MLproject` file of a mlflow setup or executed via `mlflow run`."""

import datetime
import os
import random
import sys

import click
import matplotlib
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow.python.platform.build_info as build
from tensorboard.plugins.hparams import api as hp

# fixes a problem when calling plotting functions on the server
matplotlib.use('agg')

print("Python version: ", sys.version)
print("Tensorflow version: ", tf.__version__)
print("tf.keras version:", tf.keras.__version__)
print('Cudnn version: ', build.build_info['cudnn_version'])
print('Cuda version: ', build.build_info['cuda_version'])
# Workaround for a "No algorithm worked" bug on GPUs
# see https://github.com/tensorflow/tensorflow/issues/45044
physical_devices = tf.config.list_physical_devices('GPU')
print('GPUs: {}. Trying to set memory growth to "True"...'.format(
    physical_devices))
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except IndexError:
    print('No GPU was found on this machine.')

# logging with mlflow: most of the logging happens with the autolog feature,
# which is already quite powerful. Since I want to add custom logs, I have to
# start and end the log manually with mlflow.start_run() and mlflow.end_run()


@click.command()
@click.option('--num_session_groups',
              type=int,
              default=2,
              help='number of sessions for random search')
@click.option(
    '--csv_path_train',
    type=str,
    default=
    '/home/lex/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/')
@click.option(
    '--csv_path_val',
    type=str,
    default=
    '/home/lex/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/')
@click.option('--fluotracify_path',
              type=str,
              default='~/Programme/drmed-git/src/')
@click.option('--col_per_example', type=int, default=3)
def hparams_run(num_session_groups, csv_path_train, csv_path_val,
                col_per_example, fluotracify_path):
    sys.path.append(fluotracify_path)

    if True:  # FIXME (PENDING): isort workaround
        from fluotracify.simulations import import_simulation_from_csv as isfc
        from fluotracify.training import build_model as bm, preprocess_data as ppd
        from fluotracify.training import evaluate

    # FIXME (PENDING): at some point, I want to plot metrics vs thresholds
    # from TF side, this is possible by providing the `thresholds`
    # argument as a list of thresholds
    # but currently, mlflow does not support logging lists, so I log the
    # elements of the list one by one
    HP_EPOCHS = hp.HParam('hp_epochs', hp.Discrete([20], dtype=int))
    HP_BATCH_SIZE = hp.HParam('hp_batch_size', hp.IntInterval(2, 20))
    HP_SCALER = hp.HParam(
        'hp_scaler',
        hp.Discrete(
            ['robust', 'minmax', 'maxabs', 'quant_g', 'standard', 'l1', 'l2'],
            dtype=str))
    HP_N_LEVELS = hp.HParam('hp_n_levels', hp.Discrete([3, 5, 7, 9],
                                                       dtype=int))
    HP_FIRST_FILTERS = hp.HParam('hp_first_filters',
                                 hp.Discrete([16, 32, 64, 128], dtype=int))
    HP_POOL_SIZE = hp.HParam('hp_pool_size', hp.Discrete([2, 4], dtype=int))
    HP_INPUT_SIZE = hp.HParam('hp_input_size',
                              hp.Discrete([2**12, 2**13, 2**14], dtype=int))
    HP_LR_START = hp.HParam('hp_lr_start', hp.RealInterval(1e-5, 1e-1))
    HP_LR_POWER = hp.HParam('hp_lr_power', hp.Discrete([1.0, 5.0],
                                                       dtype=float))

    HPARAMS = [
        HP_EPOCHS, HP_BATCH_SIZE, HP_SCALER, HP_N_LEVELS, HP_FIRST_FILTERS,
        HP_POOL_SIZE, HP_INPUT_SIZE, HP_LR_START, HP_LR_POWER
    ]

    LOG_DIR = "../tmp/tb-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    EXP_PARAM_PATH_TRAIN = '../tmp/experiment_params_train.csv'
    EXP_PARAM_PATH_VAL = '../tmp/experiment_params_val.csv'

    SESSIONS_PER_GROUP = 2
    LABEL_THRESH = 0.04
    METRICS_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

    def unet_1d_hparams(hparams):
        """U-Net as described by Ronneberger et al.

        Parameters
        ----------
        input_size : int
            Input vector size
        hparams: dict
            A dict mapping hyperparameters in `HPARAMS` to values.

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
        filters = [hparams[HP_FIRST_FILTERS]]
        nextfilters = hparams[HP_FIRST_FILTERS]
        for _ in range(1, hparams[HP_N_LEVELS] + 1):
            nextfilters *= 2
            filters.append(nextfilters)
        filters = tf.experimental.numpy.clip(filters, a_min=1,
                                             a_max=512).numpy()
        filters = tf.cast(filters, tf.int32).numpy()

        ldict = {}

        inputs = tf.keras.layers.Input(shape=(hparams[HP_INPUT_SIZE], 1))

        # Downsampling through model
        ldict['x0_pool'], ldict['x0'] = bm.encoder(
            input_tensor=inputs,
            filters=filters[0],
            name='encode0',
            pool_size=hparams[HP_POOL_SIZE])
        for i in range(1, hparams[HP_N_LEVELS]):
            ldict['x{}_pool'.format(i)], ldict['x{}'.format(i)] = bm.encoder(
                input_tensor=ldict['x{}_pool'.format(i - 1)],
                filters=filters[i],
                name='encode{}'.format(i),
                pool_size=hparams[HP_POOL_SIZE])

        # Center
        center = bm.twoconv(2 * filters[hparams[HP_N_LEVELS] - 1],
                            name='two_conv_center')(
                                ldict['x{}_pool'.format(hparams[HP_N_LEVELS] -
                                                        1)])

        # Upsampling through model
        ldict['y{}'.format(hparams[HP_N_LEVELS] - 1)] = bm.decoder(
            input_tensor=center,
            concat_tensor=ldict['x{}'.format(hparams[HP_N_LEVELS] - 1)],
            filters=filters[-1],
            name='decoder{}'.format(hparams[HP_N_LEVELS] - 1),
            kernel_size=hparams[HP_POOL_SIZE],
            strides=hparams[HP_POOL_SIZE])

        for j in range(1, hparams[HP_N_LEVELS]):
            ldict['y{}'.format(hparams[HP_N_LEVELS] - 1 - j)] = bm.decoder(
                input_tensor=ldict['y{}'.format(hparams[HP_N_LEVELS] - j)],
                concat_tensor=ldict['x{}'.format(hparams[HP_N_LEVELS] - 1 -
                                                 j)],
                filters=filters[-1 - j],
                name='decoder{}'.format(hparams[HP_N_LEVELS] - 1 - j),
                kernel_size=hparams[HP_POOL_SIZE],
                strides=hparams[HP_POOL_SIZE])

        # create 'binary' output vector
        outputs = tf.keras.layers.Conv1D(filters=1,
                                         kernel_size=1,
                                         activation='sigmoid')(ldict['y0'])

        print('input - shape:\t', inputs.shape)
        print('output - shape:\t', outputs.shape)

        unet = tf.keras.Model(inputs=inputs,
                              outputs=outputs,
                              name='unet_depth{}'.format(hparams[HP_N_LEVELS]))

        optimizer = tf.keras.optimizers.Adam()
        loss = bm.binary_ce_dice_loss()
        metrics = bm.unet_metrics(METRICS_THRESHOLDS)
        unet.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return unet

    def run_one(dataset_train, dataset_val, hp_logdir, session_id, hparams,
                num_train_examples, num_val_examples, best_auc_val):
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
        logdir = os.path.join(hp_logdir, str(session_id))

        ds_train_prep = dataset_train.map(
            lambda trace, label: ppd.tf_crop_trace(trace, label, hparams[
                HP_INPUT_SIZE]),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.map(
            lambda trace, label: ppd.tf_scale_trace(trace, label, hparams[
                HP_SCALER]),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.shuffle(
            buffer_size=num_train_examples).repeat().batch(
                hparams[HP_BATCH_SIZE],
                drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        ds_val_prep = dataset_val.map(lambda trace, label: ppd.tf_crop_trace(
            trace, label, hparams[HP_INPUT_SIZE]),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.map(lambda trace, label: ppd.tf_scale_trace(
            trace, label, hparams[HP_SCALER]),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.shuffle(
            buffer_size=num_val_examples).repeat().batch(
                hparams[HP_BATCH_SIZE],
                drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        model = unet_1d_hparams(hparams=hparams)

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
            lr_list = [
                hparams[HP_LR_START] *
                (1 - i / hparams[HP_EPOCHS])**hparams[HP_LR_POWER]
                for i in range(hparams[HP_EPOCHS])
            ]

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
        hparams_callback = hp.KerasCallback(logdir, hparams)  # logs hparams
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        image_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_plots)
        steps_train = num_train_examples // hparams[HP_BATCH_SIZE]
        steps_val = num_val_examples // hparams[HP_BATCH_SIZE]

        result = model.fit(
            x=ds_train_prep,
            epochs=hparams[HP_EPOCHS],
            steps_per_epoch=steps_train,
            validation_data=ds_val_prep,
            validation_steps=steps_val,
            callbacks=[
                tensorboard_callback, hparams_callback, lr_callback,
                image_callback
            ],
        )

        if result.history['val_auc'][-1] > best_auc_val:
            mlflow.keras.log_model(
                keras_model=model,
                artifact_path='model',
                conda_env=mlflow.keras.get_default_conda_env(
                    keras_module=tf.keras),
                custom_objects={'binary_ce_dice': bm.binary_ce_dice_loss()},
                keras_module=tf.keras)
            best_auc_val = result.history['val_auc'][-1]

        return best_auc_val

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
    del train, train_sep

    dataset_train, num_train_examples = ppd.tfds_from_pddf(
        features_df=train_data, labels_df=train_labels_bool, frac_val=False)

    dataset_val, num_val_examples = ppd.tfds_from_pddf(
        features_df=val_data, labels_df=val_labels_bool, frac_val=False)

    with mlflow.start_run() as parent_run:
        mlflow.tensorflow.autolog(every_n_iter=1, log_models=False)
        rng = random.Random(1)
        best_auc_val = tf.experimental.numpy.finfo(
            tf.experimental.numpy.float64).min

        num_sessions = num_session_groups * SESSIONS_PER_GROUP

        experiment_params_train.to_csv(EXP_PARAM_PATH_TRAIN)
        experiment_params_val.to_csv(EXP_PARAM_PATH_VAL)

        session_index = 0  # across all session groups
        for _ in range(num_session_groups):
            hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
            hparams_mlflow = {h.name: hparams[h] for h in hparams.keys()}
            for repeat_index in range(SESSIONS_PER_GROUP):
                print("--- Running training session {}/{}".format(
                    session_index + 1, num_sessions))
                print(hparams_mlflow)
                print("--- repeat #: {}".format(repeat_index + 1))
                with mlflow.start_run(nested=True) as _:
                    mlflow.log_artifact(EXP_PARAM_PATH_TRAIN)
                    mlflow.log_artifact(EXP_PARAM_PATH_VAL)
                    mlflow.log_params(hparams_mlflow)
                    mlflow.log_params({
                        'num_train_examples': num_train_examples,
                        'num_val_examples': num_val_examples
                    })
                    critical_hparam_combi = 2 * hparams_mlflow[
                        'hp_pool_size']**hparams_mlflow['hp_n_levels']
                    if critical_hparam_combi <= hparams_mlflow['hp_input_size']:
                        best_auc_val = run_one(
                            dataset_train=dataset_train,
                            dataset_val=dataset_val,
                            hp_logdir=LOG_DIR,
                            session_id=session_index,
                            hparams=hparams,
                            num_train_examples=num_train_examples,
                            num_val_examples=num_val_examples,
                            best_auc_val=best_auc_val)
                    else:
                        print('This run is skipped, because the following '
                              'condition (needed to build the model) was not '
                              'given: 2 * pool_size**n_levels <= input_size')
                session_index += 1

        # Now log best values in parent run
        client = mlflow.tracking.client.MlflowClient()
        runs = client.search_runs(
            [parent_run.info.experiment_id],
            "tags.mlflow.parentRunId = '{run_id}' ".format(
                run_id=parent_run.info.run_id))
        best_auc_val = tf.experimental.numpy.finfo(
            tf.experimental.numpy.float64).min
        best_run = None
        for r in runs:
            if r.data.metrics["val_auc"] > best_auc_val:
                best_run = r
                best_auc_train = r.data.metrics["auc"]
                best_auc_val = r.data.metrics["val_auc"]
        try:
            mlflow.set_tag("best_run", best_run.info.run_id)
        except AttributeError:
            print('Logging the best run failed. Maybe check if MlflowClient'
                  ' is set up correctly')
        mlflow.log_metrics({
            "best_auc": best_auc_train,
            "best_auc_val": best_auc_val
        })


if __name__ == "__main__":
    hparams_run()
