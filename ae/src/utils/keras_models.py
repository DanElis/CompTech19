from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import Input, Dense, Flatten, Reshape, Permute, BatchNormalization, Dropout, concatenate, merge, Activation
from keras.optimizers import RMSprop
from keras.models import model_from_json, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ELU
from keras.layers import Lambda
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy
import numpy as np
from .losses import *
import keras


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        # calc
        loss = y_true * keras.backend.log(y_pred) * weights
        loss = -keras.backend.sum(loss, -1)
        return loss

    return loss


def load_keras_model(name):
    path = "{}.json".format(name)
    with open(path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    path = "{}.h5".format(name)
    model.load_weights(path)
    return model


def save_keras_model(model, name):
    path = "{}.json".format(name)
    model_json = model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

    path = "{}.h5".format(name)
    model.save_weights(path)
    return

def conv2d_block(
    layer, 
    filters, 
    kernel_size, 
    activation,
    dropout, 
    batch_norm, 
    pooling
):
    x = Conv2D(filters, 
               kernel_size=kernel_size,
               kernel_initializer="he_normal",
               activation=None, 
               padding='same'
              )(layer)
    
    x = Activation(activation)(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout)(x) if dropout else x
    x = MaxPooling2D(pooling)(x) if pooling else x

    return x

def create_conv2d(
    no_of_classes=2, 
    depth=5, 
    skip_connection=False,
    filters=32,
    kernel_size=(3, 3),
    pooling=None,
    activation='relu',
    activation_out='softmax',
    dropout=.3, 
    batch_norm=True, 
    shape=(None, None, 1),
    weight_true=1,
    weight_false=1,
    lr=.001
):

    input_img = Input(shape=shape)
    
    x = input_img
    
    for i in range(depth):
        
        x_prev = x
        
        if skip_connection & (i > 1):
            x = concatenate([x, x_prev])
        x = conv2d_block(
            x, 
            filters, 
            kernel_size, 
            activation,
            dropout, 
            batch_norm, 
            pooling,
        )
        
        
    x = Conv2D(
        no_of_classes, 
        kernel_size=kernel_size,
        activation=activation_out, 
        padding='same',
        name='classification',
    )(x)
    

    model = Model(input_img, x, name="classification")

    optimizer = keras.optimizers.Adam(lr=lr)
    
    loss = weighted_categorical_crossentropy(np.array([weight_false, weight_true]))
    
    model.compile(
        optimizer=optimizer,
        loss=loss, 
        metrics=['accuracy']
    )

    return model


def create_conv2d_ae():
    dropout=0.3
    x, input_img = [Input(shape=(None, None, 1))]*2

    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout*0.5)(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    encoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    x, input_encoded = [Input(shape=(8, 8, 1))]*2
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout)(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout)(x)
    decoded1 = Conv2D(2, (3, 3), activation='softmax', padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder1 = Model(input_encoded, decoded1, name="decoder")
    x = [Input(shape=(8, 8, 1))]
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout)(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(dropout)(x)
    decoded2 = Conv2D(2, (3, 3), activation='softmax', padding='same')(x)
    decoder2 = Model(input_encoded, decoded2, name="decoder")
    
    autoencoder = Model(input_img, decoder1(encoder(input_img)), name="autoencoder")
    
    optimizer = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    )
    
    loss = weighted_categorical_crossentropy(np.array([.1, 1]))
#     loss='categorical_crossentropy'
    autoencoder.compile(optimizer=optimizer,
                  loss=loss, metrics=['accuracy'])
    return autoencoder