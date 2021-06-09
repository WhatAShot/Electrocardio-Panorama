import numpy as np


def read_ecg(path):
    with open(path, 'r') as f:
        raw_data = f.read().splitlines()
    data = []
    for index, line in enumerate(raw_data):
        if index == 0:
            continue
        line_data = line.split(' ')
        line_data = list(map(int, line_data))
        data.append(line_data)
    data = np.array(data)
    return data
