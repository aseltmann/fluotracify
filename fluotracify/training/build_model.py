import tensorflow as tf


def get_vgg10_1d_model(win_len, col_no):
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

def get_vgg16_1d_model(win_len, col_no):
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
