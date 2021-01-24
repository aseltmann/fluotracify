"""This module contains a script to streamline the training of a neural network
on simulated fluroescence traces using mlflow. It is meant to be put into the
`MLproject` file of a mlflow setup or executed via `mlflow run`."""

import sys

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

mlflow.tensorflow.autolog(every_n_iter=1)

if __name__ == "__main__":
    BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    FRAC_VAL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    LENGTH_DELIMITER = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
    LEARNING_RATE = sys.argv[5] if len(sys.argv) > 5 else 1e-5
    EPOCHS = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    CSV_PATH = sys.argv[7] if len(
        sys.argv) > 7 else '/home/lex/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/'
    COL_PER_EXAMPLE = int(sys.argv[8]) if len(sys.argv) > 8 else 3
    STEPS_PER_EPOCH = int(sys.argv[9]) if len(sys.argv) > 9 else 10
    VALIDATION_STEPS = int(sys.argv[10]) if len(sys.argv) > 10 else 10
    LOG_DIR_TB = "/tmp/tb"
    LABEL_THRESH = 0.04
    # FIXME (PENDING): at some point, I want to plot metrics vs thresholds
    # from TF side, this is possible by providing the `thresholds`
    # argument as a list of thresholds
    # but currently, mlflow does not support logging lists
    METRICS_THRESHOLDS = 0.5
    EXP_PARAM_PATH = '/tmp/experiment_params.csv'

    train, test, nsamples, experiment_params = isfc.import_from_csv(
        folder=CSV_PATH,
        header=12,
        frac_train=0.8,
        col_per_example=COL_PER_EXAMPLE,
        dropindex=None,
        dropcolumns=None)

    experiment_params.to_csv(EXP_PARAM_PATH)

    train_sep = isfc.separate_data_and_labels(array=train,
                                              nsamples=nsamples,
                                              col_per_example=COL_PER_EXAMPLE)

    test_sep = isfc.separate_data_and_labels(array=test,
                                             nsamples=nsamples,
                                             col_per_example=COL_PER_EXAMPLE)

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
              LENGTH_DELIMITER,
              test_labels_bool.sum(axis=0).head()))

    # Cleanup
    del train, test, train_sep, test_sep

    dataset_train, dataset_val, num_train_examples, num_val_examples = ppd.tfds_from_pddf_for_unet(
        features_df=train_data,
        labels_df=train_labels_bool,
        is_training=True,
        batch_size=BATCH_SIZE,
        length_delimiter=LENGTH_DELIMITER,
        frac_val=FRAC_VAL)

    dataset_test, num_test_examples = ppd.tfds_from_pddf_for_unet(
        features_df=test_data,
        labels_df=test_labels_bool,
        is_training=False,
        batch_size=BATCH_SIZE,
        length_delimiter=LENGTH_DELIMITER)

    model = bm.unet_1d_alt(input_size=LENGTH_DELIMITER)
    optimizer = tf.keras.optimizers.Adam()
    loss = bm.binary_ce_dice_loss()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR_TB,
                                                          histogram_freq=5,
                                                          write_graph=False,
                                                          write_images=True,
                                                          update_freq='epoch')

    file_writer_image = tf.summary.create_file_writer(LOG_DIR_TB + '/image')
    file_writer_lr = tf.summary.create_file_writer(LOG_DIR_TB + "/metrics")
    file_writer_lr.set_as_default()

    def log_plots(epoch, logs):
        """Image logging function for tf.keras.callbacks.LambdaCallback

        Notes
        -----
        - `tf.keras.callbacks.LambdaCallback` expects two positional arguments
          `epoch` and `logs`, if `on_epoch_end` is being used
        - see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback
        """
        figure = evaluate.plot_trace_and_pred_from_tfds(dataset=dataset_test,
                                                        ntraces=5,
                                                        model=model)
        # Convert matplotlib figure to image
        image = evaluate.plot_to_image(figure)
        # Log the image as an image summary
        with file_writer_image.as_default():
            tf.summary.image('Prediction plots', image, step=epoch)

    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.

        Notes
        -----
        - function is supposed to be used with
          `tf.keras.callbacks.LearningRateScheduler`. It takes an epoch index
          as input (integer, indexed from 0) and returns a new learning rate
          as output (float)
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

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    image_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_plots)

    metrics = [
        tf.keras.metrics.TruePositives(name='tp',
                                       thresholds=METRICS_THRESHOLDS),
        tf.keras.metrics.FalsePositives(name='fp',
                                        thresholds=METRICS_THRESHOLDS),
        tf.keras.metrics.TrueNegatives(name='tn',
                                       thresholds=METRICS_THRESHOLDS),
        tf.keras.metrics.FalseNegatives(name='fn',
                                        thresholds=METRICS_THRESHOLDS),
        tf.keras.metrics.Precision(name='precision',
                                   thresholds=METRICS_THRESHOLDS),
        tf.keras.metrics.Recall(name='recall', thresholds=METRICS_THRESHOLDS),
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
        tf.keras.metrics.AUC(num_thresholds=100, name='auc')
    ]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit(x=dataset_train,
              epochs=EPOCHS,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=dataset_val,
              validation_steps=VALIDATION_STEPS,
              callbacks=[tensorboard_callback, image_callback, lr_callback])

    model.evaluate(dataset_test,
                   steps=tf.math.ceil(num_test_examples / BATCH_SIZE))

    mlflow.keras.log_model(
        keras_model=model,
        artifact_path='model',
        conda_env=mlflow.keras.get_default_conda_env(keras_module=tf.keras),
        custom_objects={'binary_ce_dice': bm.binary_ce_dice_loss()},
        keras_module=tf.keras)

    mlflow.log_artifact(EXP_PARAM_PATH)
    mlflow.log_params({'num_train_examples': num_train_examples,
                       'num_val_examples': num_val_examples,
                       'num_test_examples': num_test_examples})
