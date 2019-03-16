from keras.models import Input
from keras.engine.training import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from keras.optimizers import Adam
import numpy as np

IMAGE_SIZE = (256,256,1)

def model(weights_input=None):

    inputs = Input(IMAGE_SIZE)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate(axis=3)([drop4,up6])
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate(axis=3)([conv3,up7])
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis=3)([conv2,up8])
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis=3)([conv1,up9])
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)

    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    if weights_input:
        model.load_weights(weights_input)

    return model

def prepare_input(image):
    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)
    image = np.clip(image, 0, 255)
    return np.divide(image, 255)

def prepare_output(image):
    image = image[:,:,0]
    image = np.clip(image, 0, 1)
    return np.multiply(image, 255)