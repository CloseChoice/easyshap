"""
This is a working lstm shap example with random input using the following versions/settings:
tf 2.3.0
shap 0.41.0
np 1.19.5
python 3.7.16 (default, Jan 17 2023, 16:06:28) [MSC v.1916 64 bit (AMD64)]
### disable tensorflow 2 behaviour tf.compat.v1.disable_v2_behavior()

with tf > 2.4.1 it will not work
"""

import datetime
import os
import sys

import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, Activation, Concatenate, BatchNormalization, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

project_name = 'lstm_test'

embedding_layer = True
use_event_attributes = True
dense_layer = False
additional_dense_layer = False

n = 1000
n_num_event_attributes = 5
voc_size = 100
sequence_length = 80
n_features = 10

X_events = np.random.randint(voc_size, size=(n, sequence_length))
X_numerical_attributes = np.random.randint(100, size=(n, sequence_length, n_num_event_attributes))
X_features = np.random.randint(100, size=(n, n_features))
y = np.random.randint(2, size=n)

# pip install numpy==1.19.5
# pip install tensorflow==2.3.0

def define_callbacks():
    dirpath = os.path.dirname(__file__)
    model_file_name = os.path.join(dirpath, project_name + '.h5')
    print("model file name: ", model_file_name)
    tensorboard_dict = os.path.expanduser('~') + '/' + datetime.date.today().strftime('%Y-%m-%d')
    experiment_log_dir = dirpath + 'tensorboard_logs/'
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=True, mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
    callbacks = [early_stopping, model_checkpoint]
    return callbacks, model_file_name


def main():
    print("tf", tf.__version__)
    print("shap", shap.__version__)
    print("np", np.__version__)
    print("python", sys.version)
    print("### disable tensorflow 2 behaviour", "tf.compat.v1.disable_v2_behavior()")
    tf.compat.v1.disable_v2_behavior()
    X = create_input()
    model = create_model(X)
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()

    callbacks, model_file_name = define_callbacks()

    model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2, verbose=2, shuffle=True, callbacks=callbacks)
    model = tf.keras.models.load_model(model_file_name)  # load best model

    n_example = 100
    n_explainer = 100

    arr_rand_choice_explainer = np.random.choice(np.arange(0, n), size=n_explainer, replace=False)
    arr_rand_choice = np.random.choice(np.arange(0, n), size=n_example, replace=False)

    X_shap_explainer = [X[0][arr_rand_choice], X[1][arr_rand_choice]]
    X_shap = [X[0][arr_rand_choice_explainer], X[1][arr_rand_choice_explainer]]

    explainer = shap.DeepExplainer(model, X_shap_explainer[:n_explainer])
    print(f'explainer - expected value: {explainer.expected_value[0]}')

    print('calculate shap values')
    shap_values = explainer.shap_values(X_shap, check_additivity=True)[0]

    print(shap_values)


def create_input():
    if embedding_layer:
        if use_event_attributes:
            if dense_layer:
                X = [X_events, X_numerical_attributes, X_features]
            else:
                X = [X_events, X_numerical_attributes]
        else:
            if dense_layer:
                X = [X_events, X_features]
            else:
                X = [X_events]
    else:
        event_input = to_categorical(X_events, num_classes=voc_size, dtype='int32')
        if use_event_attributes:
            event_input = np.concatenate((event_input, X_numerical_attributes), 2)
        if dense_layer:
            X = [event_input, X_features]
        else:
            X = [event_input]
    return X


def create_model(X):
    tf.keras.backend.clear_session()
    neurons_dense = 27 * 3
    metric: str = "accuracy"
    emb_rate_of_dim_reductions = 1
    dropout = 0.4
    activation_setting = "tanh"  # 'relu'
    neurons = 64

    if embedding_layer:
        input_shape = [sequence_length, ]
        input = Input(input_shape)
        hidden = Embedding(voc_size, int(voc_size * emb_rate_of_dim_reductions))(input)
        if use_event_attributes:
            input_shape_2 = [sequence_length, n_num_event_attributes]
            input_2 = Input(input_shape_2)
            hidden = Concatenate(axis=2)([hidden, input_2])
        hidden = BatchNormalization(axis=-1)(hidden)
    else:
        x_shape = X[0].shape
        input_shape = [x_shape[1], x_shape[2]]
        input = Input(input_shape)
        hidden = BatchNormalization(axis=-1)(input)

    hidden = LSTM(neurons)(hidden)

    if dense_layer:
        input_3 = Input([n_features, ])
        hidden_dense = Dense(neurons_dense, activation=activation_setting)(input_3)
        hidden = Concatenate(axis=1)([hidden, hidden_dense])

    if additional_dense_layer:
        hidden = Dense(neurons)(hidden)
        hidden = BatchNormalization(axis=-1)(hidden)
        hidden = Activation(activation_setting)(hidden)
        hidden = Dropout(dropout)(hidden)

    output = Dense(1, activation='sigmoid')(hidden)

    # define model & input dimensions
    if embedding_layer:
        if use_event_attributes:
            if dense_layer:
                model = Model([input, input_2, input_3], output)
            else:
                model = Model([input, input_2], output)
        else:
            if dense_layer:
                model = Model(input, input_3, output)
            else:
                model = Model(input, output)
    elif dense_layer:
        model = Model([input, input_3], output)
    else:
        model = Model(input, output)
    return model


if __name__ == '__main__':
    main()
