"""This module contains a script to streamline the training of a neural network
on simulated fluroescence traces using mlflow. It is meant to be put into the
`MLproject` file of a mlflow setup or executed via `mlflow run`."""

import os
import random
import sys

import click
import matplotlib
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# fixes a problem when calling plotting functions on the server
matplotlib.use('agg')

print(tf.version.VERSION)
print('GPUs: ', tf.config.list_physical_devices('GPU'))

# logging with mlflow: most of the logging happens with the autolog feature,
# which is already quite powerful. Since I want to add custom logs, I have to
# start and end the log manually with mlflow.start_run() and mlflow.end_run()


@click.command()
@click.option('--frac_val', type=click.FloatRange(0, 1), default=0.2)
@click.option('--length_delimiter', type=int, default=16384)
@click.option('--learning_rate', type=float, default=1e-5)
@click.option('--num_session_groups',
              type=int,
              default=2,
              help='number of sessions for grid search')
@click.option(
    '--csv_path_train',
    type=str,
    default=
    '/home/lex/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/')
@click.option(
    '--csv_path_test',
    type=str,
    default=
    '/home/lex/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/')
@click.option('--fluotracify_path',
              type=str,
              default='~/Programme/drmed-git/src/')
@click.option('--col_per_example', type=int, default=3)
@click.option('--epochs', type=int, default=10)
@click.option('--batch_size', type=int, default=5)
@click.option('--steps_per_epoch', type=int, default=10)
@click.option('--validation_steps', type=int, default=10)
@click.option('--scaler', type=str, default='robust')
@click.option('--n_levels', type=int, default=9)
@click.option('--first_filters', type=int, default=64)
@click.option('--pool_size', type=int, default=2)
def hparams_run(batch_size, frac_val, length_delimiter, learning_rate,
                num_session_groups, epochs, csv_path_train, csv_path_test,
                col_per_example, steps_per_epoch, validation_steps, scaler,
                n_levels, first_filters, pool_size, fluotracify_path):
    sys.path.append(fluotracify_path)

    if True:  # isort workaround
        from fluotracify.simulations import import_simulation_from_csv as isfc
        from fluotracify.training import build_model as bm, preprocess_data as ppd
        from fluotracify.training import evaluate

    LOG_DIR_HP = "/tmp/tb/hparams"
    EXP_PARAM_PATH_TRAIN = '/tmp/experiment_params_train.csv'
    LABEL_THRESH = 0.04
    # FIXME (PENDING): at some point, I want to plot metrics vs thresholds
    # from TF side, this is possible by providing the `thresholds`
    # argument as a list of thresholds
    # but currently, mlflow does not support logging lists, so I log the
    # elements of the list one by one
    METRICS_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete([2], dtype=int))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([5], dtype=int))
    HP_STEPS_PER_EPOCH = hp.HParam('steps_per_epoch',
                                   hp.Discrete([650], dtype=int))
    HP_VALIDATION_STEPS = hp.HParam('validation_steps',
                                    hp.Discrete([100], dtype=int))
    HP_SCALER = hp.HParam('scaler', hp.Discrete(['robust', 'minmax'], dtype=str))
    HP_N_LEVELS = hp.HParam('n_levels', hp.Discrete([3, 9], dtype=int))
    HP_FIRST_FILTERS = hp.HParam('first_filteres',
                                 hp.Discrete([64], dtype=int))
    HP_POOL_SIZE = hp.HParam('pool_size', hp.Discrete([2], dtype=int))

    HPARAMS = [
        HP_EPOCHS, HP_BATCH_SIZE, HP_STEPS_PER_EPOCH, HP_VALIDATION_STEPS,
        HP_SCALER, HP_N_LEVELS, HP_FIRST_FILTERS, HP_POOL_SIZE
    ]

    def unet_1d_hparams(hparams, input_size):
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

        inputs = tf.keras.layers.Input(shape=(input_size, 1))

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
                input_size, num_train_examples, num_val_examples):
        """Run a training/validation session.

        Returns:
        --------
        train_ds, val_ds : tf.Dataset
            Train and validation data as tf.Datasets.
        hp_logdir : str
            The top-level logdir to which to write summary data.
        session_id : str
            A unique string ID for this session.
        hparams : dict
            A dict mapping hyperparameters in `HPARAMS` to values.
        """
        logdir = os.path.join(hp_logdir, session_id)

        ds_train_prep = dataset_train.map(
            lambda trace, label: ppd.tf_crop_trace(trace, label, input_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.map(
            lambda trace, label: ppd.tf_scale_trace(trace, label, hparams[
                HP_SCALER]),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.shuffle(
            buffer_size=num_train_examples).repeat().batch(
                hparams[HP_BATCH_SIZE]).prefetch(tf.data.AUTOTUNE)

        ds_val_prep = dataset_val.map(
            lambda trace, label: ppd.tf_crop_trace(trace, label, input_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.map(lambda trace, label: ppd.tf_scale_trace(
            trace, label, hparams[HP_SCALER]),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.shuffle(
            buffer_size=num_val_examples).repeat().batch(
                hparams[HP_BATCH_SIZE]).prefetch(tf.data.AUTOTUNE)

        file_writer_image = tf.summary.create_file_writer(logdir + '/image')
        file_writer_lr = tf.summary.create_file_writer(logdir + "/lr")

        model = unet_1d_hparams(hparams=hparams, input_size=input_size)

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
            image = evaluate.plot_to_image(figure)
            # Log the image as an image summary
            with file_writer_image.as_default():
                tf.summary.image('Prediction plots', image, step=epoch)
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
            learning_rate = 0.2
            if epoch > 1:
                learning_rate = 0.02
            if epoch > 3:
                learning_rate = 0.01
            if epoch > 5:
                learning_rate = 0.001
            if epoch > 10:
                learning_rate = 1e-5
            with file_writer_lr.as_default():
                # log in tensorflow
                tf.summary.scalar('learning rate',
                                  data=learning_rate,
                                  step=epoch)
            # log in mlflow
            mlflow.log_metric('learning rate', value=learning_rate, step=epoch)
            return learning_rate

        tensorboard_callback = tf.keras.callbacks.TensorBoard(  # logs metrics
            log_dir=logdir,
            histogram_freq=5,
            write_graph=False,
            write_images=True,
            update_freq=600,
            profile_batch=0,  # workaround for issue #2084
        )
        hparams_callback = hp.KerasCallback(logdir, hparams)  # logs hparams
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        image_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_plots)

        result = model.fit(
            x=ds_train_prep,
            epochs=hparams[HP_EPOCHS],
            steps_per_epoch=hparams[HP_STEPS_PER_EPOCH],
            validation_data=ds_val_prep,
            validation_steps=hparams[HP_VALIDATION_STEPS],
            callbacks=[
                tensorboard_callback, hparams_callback, lr_callback,
                image_callback
            ],
        )

    train, _, nsamples_train, experiment_params_train = isfc.import_from_csv(
        folder=csv_path_train,
        header=12,
        frac_train=1,
        col_per_example=col_per_example,
        dropindex=None,
        dropcolumns=None)

    train_sep = isfc.separate_data_and_labels(array=train,
                                              nsamples=nsamples_train,
                                              col_per_example=col_per_example)

    # '0': trace with artifact
    # '1': just the simulated artifact (label for unet)
    # '2': whole trace without artifact (label for vae)

    train_data = train_sep['0']
    train_labels = train_sep['1']
    train_labels_bool = train_labels > LABEL_THRESH

    # Cleanup
    del train, train_sep

    dataset_train, dataset_val, num_train_examples, num_val_examples = ppd.tfds_from_pddf(
        features_df=train_data, labels_df=train_labels_bool, frac_val=frac_val)

    with mlflow.start_run() as parent_run:
        mlflow.tensorflow.autolog(every_n_iter=1)
        rng = random.Random(0)
        experiment_id = parent_run.info.experiment_id

        experiment_params_train.to_csv(EXP_PARAM_PATH_TRAIN)
        mlflow.log_artifact(EXP_PARAM_PATH_TRAIN)

        mlflow.log_params({
            'num_train_examples': num_train_examples,
            'num_val_examples': num_val_examples
        })

        file_writer_hparams = tf.summary.create_file_writer(LOG_DIR_HP)

        with file_writer_hparams.as_default():
            hp.hparams_config(hparams=HPARAMS,
                              metrics=bm.unet_hp_metrics(METRICS_THRESHOLDS))

        sessions_per_group = 2
        num_sessions = num_session_groups * sessions_per_group
        session_index = 0  # across all session groups
        for group_index in range(num_session_groups):
            hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
            hparams_string = str(hparams)
            for repeat_index in range(sessions_per_group):
                session_id = str(session_index)
                session_index += 1
                print("--- Running training session {}/{}".format(
                    session_index, num_sessions))
                print(hparams_string)
                print("--- repeat #: {}".format(repeat_index + 1))
                with mlflow.start_run(nested=True) as child_run:
                    run_one(dataset_train=dataset_train,
                            dataset_val=dataset_val,
                            hp_logdir=LOG_DIR_HP,
                            session_id=session_id,
                            hparams=hparams,
                            input_size=length_delimiter,
                            num_train_examples=num_train_examples,
                            num_val_examples=num_val_examples)

        # Search all child runs with a parent id
        query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
        results = mlflow.search_runs(filter_string=query)
        print(results[["run_id", "params.child", "tags.mlflow.runName"]])


#        # find the best run, log its metrics as the final metrics of this run.
#        client = mlflow.tracking.client.MlflowClient()
#        runs = client.search_runs(
#            [experiment_id], "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=parent_run.info.run_id)
#        )
#        best_val_train = _inf
#        best_val_valid = _inf
#        best_val_test = _inf
#        best_run = None
#        for r in runs:
#            if r.data.metrics["val_rmse"] < best_val_valid:
#                best_run = r
#                best_val_train = r.data.metrics["train_rmse"]
#                best_val_valid = r.data.metrics["val_rmse"]
#                best_val_test = r.data.metrics["test_rmse"]
#        mlflow.set_tag("best_run", best_run.info.run_id)
#        mlflow.log_metrics(
#            {
#                "train_{}".format(metric): best_val_train,
#                "val_{}".format(metric): best_val_valid,
#                "test_{}".format(metric): best_val_test,
#            }
#        )

if __name__ == "__main__":
    hparams_run()
