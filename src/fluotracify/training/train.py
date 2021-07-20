"""This module contains a script to streamline the training of a neural network
on simulated fluroescence traces using mlflow. It is meant to be put into the
`MLproject` file of a mlflow setup or executed via `mlflow run`."""

import sys

import click
import matplotlib
import mlflow
import mlflow.tensorflow
import tensorflow as tf

FLUOTRACIFY_PATH = sys.argv[1] if len(
    sys.argv) > 1 else '~/Programme/drmed-git/src/'
sys.path.append(FLUOTRACIFY_PATH)

if True:  # isort workaround
    from fluotracify.simulations import import_simulation_from_csv as isfc
    from fluotracify.training import build_model as bm, preprocess_data as ppd
    from fluotracify.training import evaluate

# fixes a problem when calling plotting functions on the server
matplotlib.use('agg')

print(tf.version.VERSION)
print('GPUs: ', tf.config.list_physical_devices('GPU'))

# logging with mlflow: most of the logging happens with the autolog feature,
# which is already quite powerful. Since I want to add custom logs, I have to
# start and end the log manually with mlflow.start_run() and mlflow.end_run()


@click.command()
@click.option('--batch_size', type=int, default=5)
@click.option('--frac_val', type=click.FloatRange(0, 1), default=0.2)
@click.option('--length_delimiter',
              type=int,
              default=16384,
              help='number of time steps after which to crop your trace')
@click.option('--learning_rate', type=float, default=1e-5)
@click.option('--epochs', type=int, default=10)
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
@click.option('--col_per_example', type=int, default=3)
@click.option('--steps_per_epoch', type=int, default=10)
@click.option('--validation_steps', type=int, default=10)
@click.option('--scaler', type=str, default='robust')
@click.option('--n_levels', type=int, default=9)
@click.option('--first_filters', type=int, default=64)
@click.option('--pool_size', type=int, default=2)
def mlflow_run(batch_size, frac_val, length_delimiter, learning_rate, epochs,
               csv_path_train, csv_path_test, col_per_example, steps_per_epoch,
               validation_steps, scaler, n_levels, first_filters, pool_size):
    with mlflow.start_run() as _:
        mlflow.tensorflow.autolog(every_n_iter=1)

        LOG_DIR_TB = "/tmp/tb"
        LABEL_THRESH = 0.04
        # FIXME (PENDING): at some point, I want to plot metrics vs thresholds
        # from TF side, this is possible by providing the `thresholds`
        # argument as a list of thresholds
        # but currently, mlflow does not support logging lists, so I log the
        # elements of the list one by one
        METRICS_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
        EXP_PARAM_PATH_TRAIN = '/tmp/experiment_params_train.csv'
        EXP_PARAM_PATH_TEST = '/tmp/experiment_params_test.csv'


        train, _, nsamples_train, experiment_params_train = isfc.import_from_csv(
            folder=csv_path_train,
            header=12,
            frac_train=1,
            col_per_example=col_per_example,
            dropindex=None,
            dropcolumns=None)

        test, _, nsamples_test, experiment_params_test = isfc.import_from_csv(
            folder=csv_path_test,
            header=12,
            frac_train=1,
            col_per_example=col_per_example,
            dropindex=None,
            dropcolumns=None)

        experiment_params_train.to_csv(EXP_PARAM_PATH_TRAIN)
        experiment_params_test.to_csv(EXP_PARAM_PATH_TEST)
        mlflow.log_artifact(EXP_PARAM_PATH_TRAIN)
        mlflow.log_artifact(EXP_PARAM_PATH_TEST)

        train_sep = isfc.separate_data_and_labels(
            array=train,
            nsamples=nsamples_train,
            col_per_example=col_per_example)

        test_sep = isfc.separate_data_and_labels(
            array=test,
            nsamples=nsamples_test,
            col_per_example=col_per_example)

        # '0': trace with artifact
        # '1': just the simulated artifact (label for unet)
        # '2': whole trace without artifact (label for vae)

        train_data = train_sep['0']
        train_labels = train_sep['1']
        train_labels_bool = train_labels > LABEL_THRESH

        test_data = test_sep['0']
        test_labels = test_sep['1']
        test_labels_bool = test_labels > LABEL_THRESH

        print('\nfor each {} timestap trace there are the following numbers '
              'of corrupted timesteps:\n{}'.format(
                  length_delimiter,
                  test_labels_bool.sum(axis=0).head()))

        # Cleanup
        del train, test, train_sep, test_sep

        dataset_train, dataset_val, num_train_examples, num_val_examples = ppd.tfds_from_pddf(
            features_df=train_data,
            labels_df=train_labels_bool,
            frac_val=frac_val)

        dataset_test, num_test_examples = ppd.tfds_from_pddf(
            features_df=test_data, labels_df=test_labels_bool)

        ds_train_prep = dataset_train.map(
            lambda trace, label: ppd.tf_crop_trace(trace, label,
                                                   length_delimiter),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.map(
            lambda trace, label: ppd.tf_scale_trace(trace, label, scaler),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_prep = ds_train_prep.shuffle(
            buffer_size=num_train_examples).repeat().batch(
                batch_size).prefetch(tf.data.AUTOTUNE)

        ds_val_prep = dataset_val.map(lambda trace, label: ppd.tf_crop_trace(
            trace, label, length_delimiter),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.map(
            lambda trace, label: ppd.tf_scale_trace(trace, label, scaler),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_prep = ds_val_prep.shuffle(
            buffer_size=num_val_examples).repeat().batch(batch_size).prefetch(
                tf.data.AUTOTUNE)

        ds_test_prep = dataset_test.map(lambda trace, label: ppd.tf_crop_trace(
            trace, label, length_delimiter),
                                        num_parallel_calls=tf.data.AUTOTUNE)
        ds_test_prep = ds_test_prep.map(
            lambda trace, label: ppd.tf_scale_trace(trace, label, scaler),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds_test_prep = ds_test_prep.batch(batch_size)

        mlflow.log_params({
            'num_train_examples': num_train_examples,
            'num_val_examples': num_val_examples,
            'num_test_examples': num_test_examples
        })

        model = bm.unet_1d_alt2(input_size=length_delimiter,
                                n_levels=n_levels,
                                first_filters=first_filters,
                                pool_size=pool_size)
        optimizer = tf.keras.optimizers.Adam()
        loss = bm.binary_ce_dice_loss()

        file_writer_image = tf.summary.create_file_writer(LOG_DIR_TB +
                                                          '/image')
        file_writer_lr = tf.summary.create_file_writer(LOG_DIR_TB + "/metrics")
        file_writer_lr.set_as_default()

        def log_plots(epoch, logs):
            """Image logging function for tf.keras.callbacks.LambdaCallback

            Notes
            -----
            - `tf.keras.callbacks.LambdaCallback` expects two positional
              arguments `epoch` and `logs`, if `on_epoch_end` is being used
            - see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback
            """
            figure = evaluate.plot_trace_and_pred_from_tfds(
                dataset=ds_test_prep, ntraces=5, model=model)
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

            # log in tensorflow
            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            # log in mlflow
            mlflow.log_metric('learning rate', value=learning_rate, step=epoch)
            return learning_rate

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR_TB,
            histogram_freq=5,
            write_graph=False,
            write_images=True,
            update_freq='epoch')
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        image_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_plots)

        metrics = bm.unet_metrics(METRICS_THRESHOLDS)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model.fit(
            x=ds_train_prep,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=ds_val_prep,
            validation_steps=validation_steps,
            callbacks=[tensorboard_callback, image_callback, lr_callback])

        model.evaluate(ds_test_prep,
                       steps=tf.math.ceil(num_test_examples / batch_size))

        mlflow.keras.log_model(
            keras_model=model,
            artifact_path='model',
            conda_env=mlflow.keras.get_default_conda_env(
                keras_module=tf.keras),
            custom_objects={'binary_ce_dice': bm.binary_ce_dice_loss()},
            keras_module=tf.keras)


if __name__ == "__main__":
    mlflow_run()
