import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = ["Coffee", "50words", "Beef", "Gun_Point"]

BASE_PATH = ""
WEIGHT = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 0), (0, 1, 2), (2, 1, 3), (3, 1, 2)]
COLUMNS = ["file_name", "weight", "accuracy"]
VERSION = "v2.1"
excel_file_path = f"result_{VERSION}.csv"


def to_float(str_num):
    return float(str_num)


def prepare_data(file_name):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open(f"{BASE_PATH}{file_name}_TRAIN.txt", "r") as file_train:
        for line in file_train:
            data_list = (line.split())[:]
            x_train.append(list(map(to_float, data_list[1:])))
            y_train.append(float(data_list[0]))
    with open(f"{BASE_PATH}{file_name}_TEST.txt", "r") as file_test:
        for line in file_test:
            data_list = (line.split())[:]
            x_test.append(list(map(to_float, data_list[1:])))
            y_test.append(float(data_list[0]))
    return x_train, y_train, x_test, y_test


def euclidian(a, b):
    return abs(a - b)


def calc_shape_avg(q, c, weight):
    path = np.zeros((len(q), len(c)))
    g = np.zeros((len(q), len(c)))
    g[0, 0] = weight[1] * euclidian(q[0], c[0])
    for i in range(1, len(q)):
        g[i, 0] = weight[0] * euclidian(q[i], c[0]) + g[i - 1, 0]
        path[i, 0] = 0
    for j in range(1, len(c)):
        g[0, j] = weight[2] * euclidian(q[0], c[j]) + g[0, j - 1]
        path[0, j] = 2
    for i in range(1, len(q)):
        for j in range(1, len(c)):
            weight_path_0 = weight[0] * euclidian(q[i], c[j]) + g[i - 1, j]
            weight_path_1 = weight[1] * euclidian(q[i], c[j]) + g[i - 1][j - 1]
            weight_path_2 = weight[2] * euclidian(q[i], c[j]) + g[i][j - 1]
            if weight_path_0 <= weight_path_1 and weight_path_0 <= weight_path_2:
                g[i, j] = weight_path_0
                path[i, j] = 0
            if weight_path_1 <= weight_path_0 and weight_path_1 <= weight_path_2:
                g[i, j] = weight_path_1
                path[i, j] = 1
            if weight_path_2 <= weight_path_0 and weight_path_2 <= weight_path_1:
                g[i, j] = weight_path_2
                path[i, j] = 2
    i = len(q) - 1
    j = len(c) - 1
    prev = path[i][j]
    shape_avg = [0]
    cnt = 0
    while i > 0 or j > 0:
        shape_avg[-1] += q[i] + c[j] / 2
        cnt += 1

        if path[i][j] == 0:
            if path[i][j] != prev:
                shape_avg[-1] /= cnt
                cnt = 0
                shape_avg.append(0)
            i -= 1
        if path[i][j] == 2:
            if path[i][j] != prev:
                shape_avg[-1] /= cnt
                cnt = 0
                shape_avg.append(0)
            j -= 1
        if path[i][j] == 1:
            shape_avg[-1] /= cnt
            cnt = 0
            shape_avg.append(0)
            i -= 1
            j -= 1
        cnt += 1
    return shape_avg.reverse()[:-1]


def process_dataset(file_name):
    # def process_dataset(file_name, df):
    # def process_dataset(file_name):
    x_train, y_train, x_test, y_test = prepare_data(file_name)
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)
    # print(len(x_train[0]))
    weight = (1, 1, 1)
    acc = calc_shape_avg(x_train, y_train, x_test, y_test, weight)
    print(f"{file_name}{weight}:{acc}")
    df = pd.DataFrame({"file_name": [file_name], "weight": [weight], "accuracy": [acc]})
    df.to_csv(excel_file_path, mode="a", index=False, header=False)
    # df = df.append(
    #     {"file_name": file_name, "weight": weight, "acc": acc},
    #     ignore_index=True,
    # )
    # return df


def main():
    df = pd.DataFrame(columns=COLUMNS)
    df.to_csv(excel_file_path, mode="a", index=False)
    for file_name in FILE_NAME:
        # df = process_dataset(file_name, df)
        process_dataset(file_name)
    # df.to_csv(excel_file_path, index=False)
