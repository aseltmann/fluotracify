import sys

import mlflow
import mlflow.tensorflow
import tensorflow as tf

sys.path.append('/home/lex/Programme/drmed-git/src/')
from fluotracify.simulations import import_simulation_from_csv as isfc
from fluotracify.training import build_model as bm, preprocess_data as ppd

print(tf.__version__)

mlflow.tensorflow.autolog()

if __name__ == "__main__":
    train, test, nsamples, experiment_params = isfc.import_from_csv(
        path='/home/lex/Programme/Jupyter/DOKTOR/saves/firstartefact/subsample_rand/',
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
    print(
        '\nfor each 20,000 timestap trace there are the following numbers '
        'of corrupted timesteps:\n',
        test_labels_bool.sum(axis=0).head())

    # Cleanup
    del train, test

    batch_size = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    batch_size = int(batch_size)
    frac_val = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    length_delimiter = float(sys.argv[3]) if len(sys.argv) > 3 else 16384
    length_delimiter = int(length_delimiter)
    learning_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    epochs = float(sys.argv[5]) if len(sys.argv) > 5 else 10
    epochs = int(epochs)

    dataset_train, dataset_val, num_train_examples, num_val_examples = ppd.tfds_from_pddf_for_unet(
        features_df=train_data,
        labels_df=train_labels_bool,
        is_training=True,
        batch_size=batch_size,
        length_delimiter=length_delimiter,
        frac_val=frac_val)

    dataset_test, num_test_examples = ppd.tfds_from_pddf_for_unet(
        features_df=test_data,
        labels_df=test_labels_bool,
        is_training=False,
        batch_size=batch_size,
        length_delimiter=length_delimiter)

    with mlflow.start_run():
        model = bm.unet_1d_alt(input_size=length_delimiter)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = bm.binary_ce_dice_loss

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[
                          tf.keras.metrics.MeanIoU(num_classes=2),
                          tf.keras.metrics.Precision(),
                          tf.keras.metrics.Recall()
                      ])

        model.fit(x=dataset_train,
                  epochs=epochs,
                  steps_per_epoch=tf.math.ceil(num_train_examples /
                                               batch_size),
                  validation_data=dataset_val,
                  validation_steps=tf.math.ceil(num_val_examples / batch_size))
        model.evaluate(dataset_test,
                       steps=tf.math.ceil(num_test_examples / batch_size))
        model.predict(dataset_test)
        model.reset_metrics()
        model.save('data/exp-devtest/unet.tf', save_format='tf')
