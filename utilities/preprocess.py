import arff
import numpy as np

def load_dataset(file_path):
    with open("dataset/weather.arff") as f:
        file_ = arff.load(f, encode_nominal= True)
        f.close()
    raw_dataset = file_['data']

    expected = []
    for data in raw_dataset:
        expected.append([data[-1]])
        del data[-1]

    dataset = np.array(raw_dataset)
    expected = np.array(expected)
    return dataset, expected

def normalise(arr):
    return arr / arr.max(axis=0)