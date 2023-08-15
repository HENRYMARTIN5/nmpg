import argparse
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
import numpy as np
from betterlib import logging
import sys, os
import json
from tqdm import tqdm
from keras.utils import plot_model
import keras.backend as K

log = logging.Logger("./logs/train.log", "train")

def closeenough_metric(threshold=0.5):
    """
    The "Close Enough" metric. Returns the mean of the number of samples where the model's output is within the threshold (0.5 by default) of the actual value.
    """
    def metric(y_true, y_pred):
        distance = K.abs(y_true - y_pred)
        return K.mean(distance <= threshold)
    return metric

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def fix_oob(percentage):
    # fix out of bounds percentages
    if percentage < 0:
        return 0
    elif percentage > 100:
        return 100
    return percentage

def convert_weight_to_percentage(weight):
    """
    Converts a weight to a percentage.
    """
    return fix_oob(translate(-(translate(weight, 1.75, 17.60, -1, 1)), -1, 1, 0, 1)) # not 0 to 100 because of sigmoid activation

def train(epochs, metric_threshold, output, datapath, test_samples, batch_size):
    """
    Trains the model.
    """
    log.info("Loading data...")
    dirs = os.listdir(datapath)
    filepaths = []
    filepaths_test = []
    i = 0
    for dir in dirs:
        if i < test_samples:
            filepaths_test.append(os.path.join(datapath, dir, "processed_drain.json"))
        else:
            filepaths.append(os.path.join(datapath, dir, "processed_drain.json"))
        i += 1
    data_train = []
    data_test = []
    for filepath in tqdm(filepaths):
        with open(filepath, "r") as f:
            dat = json.load(f)
            for frf in dat:
                data_train.append([frf[1], convert_weight_to_percentage(frf[2])])
    for filepath in tqdm(filepaths_test):
        with open(filepath, "r") as f:
            dat = json.load(f)
            for frf in dat:
                data_test.append([frf[1], convert_weight_to_percentage(frf[2])])
    log.info("Loaded data.")

    log.info("Normalizing data...")
    # pad with zeros
    max_len = max([len(x[0]) for x in data_train])
    maxlen_test = max([len(x[0]) for x in data_test])
    if maxlen_test > max_len:
        max_len = maxlen_test
    for i in tqdm(range(len(data_train))):
        data_train[i][0] += [0] * (max_len - len(data_train[i][0]))
    for i in tqdm(range(len(data_test))):
        data_test[i][0] += [0] * (max_len - len(data_test[i][0]))
    x_train = np.array([x[0] for x in data_train])
    y_train = np.array([x[1] for x in data_train])
    x_test = np.array([x[0] for x in data_test])
    y_test = np.array([x[1] for x in data_test])
    log.info("Done.")

    log.info("Creating model...")
    model = Sequential()
    model.add(Input(shape=(max_len,))) # input layer
    for i in range(5):
        model.add(Dense(max_len/2**i, activation='relu'))
    model.add(Dense(100, activation='relu')) # relu works quite well here, but there are other options to try later
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu')) # I try to avoid sharp decreases in the number of neurons between layers, that generally doesn't help the model
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # sigmoid because we want a percentage 0-1

    # build model and print layer info
    model.build()
    model.summary()
    log.info("Done.")

    log.info("Compiling model...")
    model.compile(optimizer=Adam(), loss='mse', metrics=[closeenough_metric(threshold=metric_threshold)])
    log.info("Done.")

    log.info("Training model...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    log.info("Done.")

    log.info("Saving model...")
    model.save(output)
    with open(output.split(".")[0] + ".json", "w") as f:
        f.write("""{
    "max_len":  """ + str(max_len) + """,
    "metric_threshold": """ + str(metric_threshold) + """
}""")
    log.info("Script done.")

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='Path to the data directory.')
    parser.add_argument('--epochs', type=int, default=100) # 
    parser.add_argument('--metric-threshold', type=float, default=0.2)
    parser.add_argument('--output', type=str, default='nmpg-model.h5')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-samples', type=int, default=2)
    parsed = parser.parse_args(args)
    train(parsed.epochs, parsed.metric_threshold, parsed.output, parsed.datapath, parsed.test_samples, parsed.batch_size)
