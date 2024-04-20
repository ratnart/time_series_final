import numpy as np
import pandas as pd

FILE_NAME = ["Coffee", "50words", "Beef", "Gun_Point"]

BASE_PATH = ""
WEIGHT = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 0), (0, 1, 2), (2, 1, 3), (3, 1, 2)]
COLUMNS = ["file_name", "weight", "acc"]
VERSION = "v1"
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
    return (a - b) ** 2


def distance(q, c, weight):

    g = np.zeros((len(q), len(c)))
    for i in range(len(q)):
        for j in range(len(c)):
            if not i and not j:
                g[i, j] = weight[1] * euclidian(q[i], c[j])
            elif not j:
                g[i, j] = weight[0] * euclidian(q[i], c[j]) + g[i - 1, j]
            elif not i:
                g[i, j] = weight[1] * euclidian(q[i], c[j]) + g[i][j - 1]
            else:
                g[i, j] = min(
                    weight[0] * euclidian(q[i], c[j]) + g[i - 1, j],
                    min(
                        weight[1] * euclidian(q[i], c[j]) + g[i][j - 1],
                        weight[2] * euclidian(q[i], c[j]) + g[i - 1][j - 1],
                    ),
                )
    return g[len(q) - 1, len(c) - 1]


def one_n_n(x_train, y_train, x, weight):
    best_so_far = 1e9
    cls = -1
    for i, x_t in enumerate(x_train):
        D = distance(x, x_t, weight)
        if D < best_so_far:
            best_so_far = D
            cls = y_train[i]
    return cls


def cal_acc(x_train, y_train, x_test, y_test, weight):
    correct = 0
    for i in range(len(x_test)):
        cls = one_n_n(x_train, y_train, x_test[i], weight)
        print(i)
        if cls == y_test[i]:
            correct += 1
    return correct / len(x_test)


def process_dataset(file_name, df):
    x_train, y_train, x_test, y_test = prepare_data(file_name)
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)
    # print(len(x_train[0]))
    for weight in WEIGHT:
        acc = cal_acc(x_train, y_train, x_test, y_test, weight)
        print(f"{file_name}{weight}:{acc}")
        df = df._append(
            {"file_name": file_name, "weight": weight, "acc": acc}, ignore_index=True
        )
    return df


def main():
    df = pd.DataFrame(columns=COLUMNS)
    for file_name in FILE_NAME:
        df = process_dataset(file_name, df)
    df.to_csv(excel_file_path, index=False)


if __name__ == "__main__":
    main()
