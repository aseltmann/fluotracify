import sys

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

print(tf.version.VERSION)
print('GPUs: ', tf.config.list_physical_devices('GPU'))

mlflow.tensorflow.autolog(every_n_iter=1)

if __name__ == "__main__":
    BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    FRAC_VAL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    LENGTH_DELIMITER = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
    LEARNING_RATE = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
    EPOCHS = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    CSV_PATH = sys.argv[7] if len(
        sys.argv
    ) > 7 else '/home/lex/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/'

    train, test, nsamples, experiment_params = isfc.import_from_csv(
        path=CSV_PATH,
        header=12,
        frac_train=0.8,
        col_per_example=2,
        dropindex=None,
        dropcolumns='Unnamed: 200')

    train_data, train_labels = isfc.separate_data_and_labels(array=train,
                                                             nsamples=nsamples)
    test_data, test_labels = isfc.separate_data_and_labels(array=test,
                                                           nsamples=nsamples)

    # get bool as ground truth
    train_labels_bool = train_labels > 0.04

    test_labels_bool = test_labels > 0.04
    print('\nfor each 20,000 timestap trace there are the following numbers '
          'of corrupted timesteps:\n', test_labels_bool.sum(axis=0).head())

    # Cleanup
    del train, test

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

    log_dir = "/tmp/tb"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=5,
                                                          write_graph=False,
                                                          write_images=True,
                                                          update_freq='epoch')

    file_writer_image = tf.summary.create_file_writer(log_dir + '/image')
    file_writer_lr = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer_lr.set_as_default()

    def log_plots(epoch, logs):
        figure = evaluate.plot_trace_and_pred_from_tfds(dataset=dataset_test,
                                                        ntraces=5)
        # Convert matplotlib figure to image
        image = evaluate.plot_to_image(figure)
        # Log the image as an image summary
        with file_writer_image.as_default():
            tf.summary.image('Prediction plots', image, step=epoch)

    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
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

    model = bm.unet_1d_alt(input_size=LENGTH_DELIMITER)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = bm.binary_ce_dice_loss()
    thresholds = 0.5  # currently, mlflow does not support logging lists
    metrics = [
        tf.keras.metrics.TruePositives(name='tp', thresholds=thresholds),
        tf.keras.metrics.FalsePositives(name='fp', thresholds=thresholds),
        tf.keras.metrics.TrueNegatives(name='tn', thresholds=thresholds),
        tf.keras.metrics.FalseNegatives(name='fn', thresholds=thresholds),
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
        tf.keras.metrics.Precision(name='precision', thresholds=thresholds),
        tf.keras.metrics.Recall(name='recall', thresholds=thresholds),
        tf.keras.metrics.AUC(num_thresholds=100, name='auc')
    ]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit(x=dataset_train,
              epochs=EPOCHS,
              steps_per_epoch=int(
                  tf.math.ceil(num_train_examples / BATCH_SIZE).numpy()),
              validation_data=dataset_val,
              validation_steps=int(
                  tf.math.ceil(num_val_examples / BATCH_SIZE).numpy()),
              callbacks=[tensorboard_callback, image_callback, lr_callback])

    model.evaluate(dataset_test,
                   steps=tf.math.ceil(num_test_examples / BATCH_SIZE))

    mlflow.keras.log_model(
        keras_model=model,
        artifact_path='model',
        conda_env=mlflow.keras.get_default_conda_env(keras_module=tf.keras),
        custom_objects={'binary_ce_dice': bm.binary_ce_dice_loss()},
        keras_module=tf.keras)
