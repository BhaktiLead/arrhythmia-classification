# coding: utf-8
# Implementation of AlexNet-style model (TF 2.x compatible)
# Reads data, builds model, trains, and saves best weights.

import os
import cv2
import numpy as np
import pandas as pd
from collections import deque
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                     Flatten, BatchNormalization, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

import directory_structure  # your helper module

# -------------------- Config --------------------
CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']
NUMBER_OF_CLASSES = len(CLASSES_TO_CHECK)
IMAGES_TO_TRAIN = 2450
IMG_H, IMG_W, IMG_C = 224, 224, 3
TEST_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 1
SEED = 816

# Reduce TF verbosity (e.g., AVX warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.utils.set_random_seed(SEED)
# ------------------------------------------------

def save_metrics_and_weights(score, model):
    """
    Save (best) accuracy metric and weights under testing/{model_weights,accuracy_metrics}.
    """
    loss, current_acc = float(score[0]), float(score[1])

    directory_structure.getWriteDirectory('testing', None)
    weights_path = directory_structure.getWriteDirectory('testing', 'model_weights')
    metrics_path = directory_structure.getWriteDirectory('testing', 'accuracy_metrics')

    metrics_file = os.path.join(metrics_path, 'metrics.npy')
    weights_file = os.path.join(weights_path, 'my_model.h5')

    if not os.path.exists(metrics_file):
        # first time
        np.save(metrics_file, [0.0])
        model.save(weights_file)
    else:
        highest_acc = float(np.load(metrics_file)[0])
        if current_acc > highest_acc:
            np.save(metrics_file, [current_acc])
            model.save(weights_file)
            print('\nAccuracy Increase: {:.2f}%'.format((current_acc - highest_acc) * 100.0))


def get_signal_dataframe():
    """
    Read signal images from beat_write_dir into a dataframe with columns:
    [Signal ID, Signal (np.array HxWxC), Type].
    """
    signal_path = directory_structure.getWriteDirectory('beat_write_dir', None)

    df = pd.DataFrame(columns=['Signal ID', 'Signal', 'Type'])
    arrhythmia_classes = directory_structure.getAllSubfoldersOfFolder(signal_path)

    image_paths, image_ids, class_types = deque(), deque(), deque()
    images = []

    for classification in arrhythmia_classes:
        classification_path = os.path.join(signal_path, classification)
        image_list = directory_structure.filesInDirectory('.png', classification_path)
        for beat_id in image_list:
            image_ids.append(directory_structure.removeFileExtension(beat_id))
            class_types.append(classification)
            image_paths.append(os.path.join(classification_path, beat_id))

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            # Skip unreadable files
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        images.append(img)

    df['Signal ID'] = list(image_ids)[:len(images)]
    df['Type'] = list(class_types)[:len(images)]
    df['Signal'] = images
    return df


def normalize_data(X_train, X_test, y_train_idx, y_test_idx):
    """
    Normalize images to [0,1] float32 and one-hot the labels.
    """
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0

    y_train = to_categorical(y_train_idx, NUMBER_OF_CLASSES)
    y_test  = to_categorical(y_test_idx, NUMBER_OF_CLASSES)
    return X_train, X_test, y_train, y_test


def train_test_split_balanced(df, test_size=0.15):
    """
    Select up to IMAGES_TO_TRAIN per class from CLASSES_TO_CHECK, then split.
    """
    X, y = [], []
    counts = {c: 0 for c in CLASSES_TO_CHECK}

    # Shuffle rows for randomness
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    for _, row in df.iterrows():
        cls = row['Type']
        if cls in CLASSES_TO_CHECK and counts[cls] < IMAGES_TO_TRAIN:
            X.append(row['Signal'])
            y.append(CLASSES_TO_CHECK.index(cls))
            counts[cls] += 1
        # stop early if all classes reached target
        if all(counts[c] >= IMAGES_TO_TRAIN or (df['Type'].value_counts().get(c, 0) <= counts[c])
               for c in CLASSES_TO_CHECK):
            # continue to absorb if not enough available, but we already shuffled
            pass

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
    )

    X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def build_model(model_name='Alexnet'):
    """
    Build AlexNet-like (or 'Novelnet') model. Input: 224x224x3, output: NUMBER_OF_CLASSES.
    """
    model = models.Sequential(name=model_name)

    if model_name == 'Alexnet':
        # 1
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4),
                         padding='valid', input_shape=(IMG_H, IMG_W, IMG_C)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 2
        model.add(Conv2D(256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 3
        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 4
        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 5
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # Dense head
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))

    elif model_name == 'Novelnet':
        # 1
        model.add(Conv2D(96, kernel_size=(13, 13), strides=(4, 4),
                         padding='valid', input_shape=(IMG_H, IMG_W, IMG_C)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 2
        model.add(Conv2D(256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 3
        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 4
        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 5
        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 6
        model.add(Conv2D(500, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # Dense head
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.6))
        model.add(BatchNormalization())

        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model


def main():
    # (1) Load data
    df = get_signal_dataframe()
    X_train, X_test, y_train, y_test = train_test_split_balanced(df, TEST_SIZE)

    # (2) Optional multi-GPU with MirroredStrategy
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = build_model('Alexnet')
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    # (3) Train
    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(
            filepath=os.path.join(directory_structure.getWriteDirectory('testing', 'model_weights'),
                                  'checkpoint_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # (4) Evaluate & save
    score = model.evaluate(X_test, y_test, verbose=0)
    print("\nTest loss:     {:.6f}".format(score[0]))
    print("Test accuracy: {:.4f}".format(score[1]))
    save_metrics_and_weights(score, model)


if __name__ == '__main__':
    main()
