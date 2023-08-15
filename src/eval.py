import argparse
import numpy as np
from betterlib import logging
import sys, os
import json
from tqdm import tqdm
import keras.backend as K
from keras.models import load_model


log = logging.Logger("./logs/train.log", "train")

def closeenough_metric(threshold=0.5):
    """
    The "Close Enough" metric. Returns the mean of the number of samples where the model's output is within the threshold (0.5 by default) of the actual value.
    """
    def metric(y_true, y_pred):
        distance = K.abs(y_true - y_pred)
        return K.mean(distance <= threshold)
    return metric

def evaluate_distance(y_true, y_pred):
    """
    Returns the mean distance between the actual and predicted values.
    """
    return K.abs(y_true - y_pred)

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

def eval(datapath, test_samples, modelpath, threshold, limit):
    log.info("Loading data...")
    infopath = modelpath.replace(".h5", ".json")
    with open(infopath, "r") as f:
        info = json.load(f)
    max_len = info["max_len"]
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
    log.info("Normalizing data...")
    for i in tqdm(range(len(data_train))):
        data_train[i][0] += [0] * (max_len - len(data_train[i][0]))
    for i in tqdm(range(len(data_test))):
        data_test[i][0] += [0] * (max_len - len(data_test[i][0]))
    x_train = np.array([x[0] for x in data_train])
    y_train = np.array([x[1] for x in data_train])
    x_test = np.array([x[0] for x in data_test])
    y_test = np.array([x[1] for x in data_test])
    log.info("Loading model...")
    model = load_model(modelpath, custom_objects={"metric": closeenough_metric})
    log.info("Evaluating model...")
    # Loop through each sample and evaluate it
    overall_accuracy = 0
    average_distance = 0
    num_samples = len(x_test) if limit == 0 else limit
    for i in tqdm(range(num_samples)):
        sample = x_test[i]
        actual = y_test[i]
        predicted = model.predict(np.array([sample]))[0]
        overall_accuracy += closeenough_metric(threshold=threshold)(actual, predicted)
        average_distance += evaluate_distance(actual, predicted)
        log.debug(f"Sample {i}: Predicted {predicted}, Actual {actual}, Distance: {evaluate_distance(actual, predicted)} Close Enough: {closeenough_metric(threshold=threshold)(actual, predicted)}")

    log.info(f"Overall accuracy: {overall_accuracy / len(x_test) * 100}% on {len(x_test)} samples. (Threshold: {threshold})")
    log.info(f"Average distance: {average_distance / len(x_test)}.")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath', type=str, help='Path to the model.')
    parser.add_argument('datapath', type=str, help='Path to the data directory.')
    parser.add_argument('test_samples', type=int, help='Number of samples to use for testing.')
    parser.add_argument('--limit', type=int, default=0, help='Place a hard limit on how many samples will be run.')
    parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for the "Close Enough" metric.')
    parsed = parser.parse_args(args)
    eval(parsed.datapath, parsed.test_samples, parsed.modelpath, parsed.threshold, parsed.limit)
