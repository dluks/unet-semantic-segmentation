#!/usr/bin/env python3
import glob
import os
import datetime

import numpy as np
import tifffile as tiff
from patchify import patchify
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPool2D,
)
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Config
BASE_LOGS_DIR = "logs"
DATA_DIR = "../../data/"
SET_NAMES = ["strict", "loose"]

# Hyperparams
N_FOLDS = 10
EPOCHS = 75
ETA = 1e-2
BATCH_SIZE = 16

tf.get_logger().setLevel("ERROR")


def patch_train_label(raster, labels, img_size, channels=False, merge_channel=False):
    assert len(raster) > 0, "Raster list is empty."
    samp_rast = tiff.imread(raster[0])
    img_base_size = samp_rast.shape[0]
    n = len(raster)
    m = (img_base_size // img_size) ** 2

    if not channels:
        channels = samp_rast.shape[-1]

    if merge_channel:
        channels += tiff.imread(merge_channel[0]).shape[-1]

    data_train = np.zeros((n * m, img_size, img_size, channels))
    data_label = np.zeros((n * m, img_size, img_size))

    for k in range(n):
        if merge_channel:
            r = np.concatenate(
                (tiff.imread(raster[k]), tiff.imread(merge_channel[k])), axis=-1
            )
        else:
            r = tiff.imread(raster[k])[..., :channels]

        # Only read in the specified number of channels from input raster
        patches_train = patchify(
            r,
            (img_size, img_size, channels),
            step=img_size,
        )
        patches_label = patchify(
            tiff.imread(labels[k]), (img_size, img_size), step=img_size
        )
        data_train[k * m : (k + 1) * m, :, :, :] = patches_train.reshape(
            -1, img_size, img_size, channels
        )
        data_label[k * m : (k + 1) * m, :, :] = patches_label.reshape(
            -1, img_size, img_size
        )

    data_label = (data_label > 0).astype("int")
    data_label = np.expand_dims(data_label, axis=-1)
    data_train = data_train.astype("float") / 255

    print(
        f"\nPatched data sizes:\ndata_train: {data_train.shape}\ndata_label: {data_label.shape}"
    )

    return data_train, data_label


# Construct the U-Net
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def data_augmentation():
    augs = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.5),
            keras.layers.RandomContrast(0.1),
            keras.layers.RandomBrightness(0.1),
        ]
    )
    return augs


def build_unet(input_shape, aug=False):
    inputs = Input(input_shape)
    if aug:
        inputs = data_augmentation()(inputs)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model


def train_unet(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size=16,
    epochs=100,
    eta=1e-2,
    cb=None,
):
    input_shape = X_train.shape[1:]
    model = build_unet(input_shape)
    batch_size = batch_size
    epochs = epochs
    model.compile(
        optimizer=Adam(learning_rate=eta),
        loss="binary_crossentropy",
        metrics=[BinaryIoU(target_class_ids=[1], threshold=0.5)],
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=cb,
    )
    return model, history


def callbacks(name):
    cb = []
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOGS_DIR = os.path.join(BASE_LOGS_DIR, f"{now}_{name}")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1,
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(LOGS_DIR, f"{name}_best_wg.h5"),
        save_best_only=True,
        mode="min",
        monitor="val_loss",
        save_weights_only=True,
        verbose=1,
    )
    lr_decay = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=1,
        min_lr=0.00001,
    )
    cb.append(tensorboard)
    cb.append(checkpoint)
    cb.append(lr_decay)
    return cb


def train_set(set_name):
    # DATASET
    # Patchify hand-labeled data PLUS NIR data
    HAND_DIR = os.path.join(DATA_DIR, "hand")
    HAND_RGB_DIR = os.path.join(HAND_DIR, "rgb")
    HAND_LABEL_DIR = os.path.join(HAND_DIR, "label")

    patch_rgb = glob.glob(os.path.join(HAND_RGB_DIR, "*.tif"))
    patch_label = glob.glob(os.path.join(HAND_LABEL_DIR, "*.tif"))
    patch_rgb.sort()
    patch_label.sort()

    data_train, data_label = patch_train_label(patch_rgb, patch_label, 128)

    X_train, X_test, y_train, y_test = train_test_split(
        data_train, data_label, test_size=0.33, shuffle=True, random_state=157
    )

    print(
        f"\nSizes with only hand-labeled data:\n\
    X_train: {X_train.shape}\n\
    y_train: {y_train.shape}\n\
    X_test: {X_test.shape}\n\
    y_test: {y_test.shape}"
    )

    # Patchify watershed data (pre-patchified)
    WS_DIR = os.path.join(DATA_DIR, "watershed")
    WS_RGBI_DIR = os.path.join(WS_DIR, f"rgbi/{set_name}/512")
    WS_LABEL_DIR = os.path.join(WS_DIR, f"labels/{set_name}/512")

    ws_rgbi = glob.glob(os.path.join(WS_RGBI_DIR, "*.tif"))
    ws_labels = glob.glob(os.path.join(WS_LABEL_DIR, "*.tif"))
    ws_rgbi.sort()
    ws_labels.sort()

    data_train, data_label = patch_train_label(ws_rgbi, ws_labels, 128, channels=3)

    # Always use the hand-labeled test split as final test (outside KF CV) because
    # we know it is higher quality
    X_train = np.concatenate((X_train, data_train), axis=0)
    y_train = np.concatenate((y_train, data_label), axis=0)

    print(
        f"\nSizes after adding watershed data:\n\
    X_train: {X_train.shape}\n\
    y_train: {y_train.shape}\n\
    X_test: {X_test.shape}\n\
    y_test: {y_test.shape}"
    )

    # MODEL

    # Data structure
    stats = ["mean_biou", "tree_biou", "bg_biou"]
    data = np.zeros((N_FOLDS, len(stats)))

    # Initialize the KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=7)

    for i, (itrain, itest) in enumerate(
        tqdm(
            kf.split(
                X_train,
                y_train,
            ),
            desc="K-Folds",
            position=0,
            leave=True,
            total=N_FOLDS,
        )
    ):
        X_train_fold = X_train[itrain]
        y_train_fold = y_train[itrain]
        X_test_fold = X_train[itest]
        y_test_fold = y_train[itest]

        # Set callbacks each iteration so that logs are stored in new
        # directory
        cb = callbacks(set_name)
        model, _ = train_unet(
            X_train_fold,
            y_train_fold,
            X_test_fold,
            y_test_fold,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            eta=ETA,
            cb=cb,
        )

        # # Loss and accuracies from each epoch
        # loss = history.history["loss"]
        # val_loss = history.history["val_loss"]
        # iou = history.history[list(history.history.keys())[1]]
        # val_iou = history.history[list(history.history.keys())[3]]

        # Test the model on the preserved test data
        y_pred = model.predict(X_test)

        # Get the IoU for the test data
        biou = BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
        biou.update_state(y_pred=y_pred, y_true=y_test)
        pred_biou = biou.result().numpy()

        # only for trees
        tree_biou = BinaryIoU(target_class_ids=[1], threshold=0.5)
        tree_biou.update_state(y_pred=y_pred, y_true=y_test)
        pred_tree_biou = tree_biou.result().numpy()

        # only for non-tree pixel (background)
        bg_biou = BinaryIoU(target_class_ids=[0], threshold=0.5)
        bg_biou.update_state(y_pred=y_pred, y_true=y_test)
        pred_bg_biou = bg_biou.result().numpy()

        # Log the five stats according to their K-Fold and parameter iteration
        stats = [
            pred_biou,
            pred_tree_biou,
            pred_bg_biou,
        ]

        for s, stat in enumerate(stats):
            data[i, s] = stat

        # Save the updated array each iteration
        np.save(f"stats/{N_FOLDS}folds_CV_{set_name}.npy", data)


for set_name in tqdm(SET_NAMES, desc="Set", total=len(SET_NAMES)):
    train_set(set_name)
