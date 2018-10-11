import numpy as np
import math
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def root_mean_square_error(predictions, labels):
    errors = []
    for x in range(np.shape(predictions)[0]):
        errors.append(math.sqrt(np.sum(np.square(np.subtract(predictions[x], labels[x]))) / np.shape(predictions[1])))
    return errors

def fastDTWDistance_v1(predictions, labels):
    errors = []
    for x in range(np.shape(predictions)[0]):
        distance, path = fastdtw(predictions[x], labels[x], dist=euclidean)
        errors.append(distance)
    return errors

def fastDTWDistance_v2(predictions, labels):
    pred = predictions.tolist()
    lab = labels.tolist()
    num_predictions = len(pred)
    errors = []
    sigma = 10.0 ** -3
    for x in range(num_predictions - 1, -1, -1):
        while abs(pred[x][-1]) < sigma:
            pred[x].pop()
        errors.append(fastdtw(pred[x], lab[x], dist=euclidean))
    return errors

def diversity(dataset):
    # average pairwise distance between items in the list
    # or aggregate
    distance = []
    for x in range(len(dataset)):
        for y in range(x + 1, len(dataset)):
            distance.append(euclidean(dataset[x], dataset[y]))

    return np.mean(distance)

