import keras.backend as K

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
    return fix_oob(translate(-(translate(weight, 1.75, 17.60, -1, 1)), -1, 1, 0, 100))
