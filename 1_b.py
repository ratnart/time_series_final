import numpy as np
import pandas as pd

FILE_NAME = ["Coffee", "50words", "Beef", "Gun_Point"]
# FILE_NAME = ["Coffee"]

BASE_PATH = ""
P = [0, 0.5, 1, 1.5, 2]
COLUMNS = ["file_name", "P", "acc"]
VERSION = "v1"
excel_file_path = f"result_1b_{VERSION}.csv"


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


def distance0(q, c):

    g = np.zeros((len(q), len(c)))
    g[0, 0] = 2* euclidian(q[0], c[0])
    for i in range(1, len(q)):
        g[i, 0] = euclidian(q[i], c[0]) + g[i - 1, 0]
    
    for j in range(1, len(c)):
        g[0, j] = euclidian(q[0], c[j]) + g[0, j - 1]

 
    for i in range(1, len(q)):
        for j in range(1, len(c)):
            g[i, j] = min(
                euclidian(q[i], c[j]) + g[i - 1, j],
                min(
                    euclidian(q[i], c[j]) + g[i][j - 1],
                    2*euclidian(q[i], c[j]) + g[i - 1][j - 1],
                )
            )
    return g[len(q) - 1, len(c) - 1]

def distance0_5(q, c):

    g = np.zeros((len(q), len(c)))
    g[0, 0] = 2* euclidian(q[0], c[0])
    for i in range(1, len(q)):
        g[i, 0] = euclidian(q[i], c[0]) + g[i - 1, 0]
    
    for j in range(1, len(c)):
        g[0, j] = euclidian(q[0], c[j]) + g[0, j - 1]

 
    for i in range(1, len(q)):
        for j in range(1, len(c)):
            minD = 100000000
            if i-1 >= 0 and j-3 >= 0:
                minD = min(minD, g[i-1][j-3]+2*euclidian(q[i], c[j-2])+euclidian(q[i], c[j-1])+euclidian(q[i], c[j]))
            if i-1 >= 0 and j-2 >= 0:
                minD = min(minD, g[i-1][j-2]+2*euclidian(q[i], c[j-1])+euclidian(q[i], c[j]))
            if i-1 >= 0 and j-1 >= 0:
                minD = min(minD, g[i-1][j-1]+2*euclidian(q[i], c[j]))
            if i-2 >= 0 and j-1 >= 0:
                minD = min(minD, g[i-2][j-1]+2*euclidian(q[i-1], c[j])+euclidian(q[i], c[j]))
            if i-3 >= 0 and j-1 >= 0:
                minD = min(minD, g[i-3][j-1]+2*euclidian(q[i-2], c[j])+euclidian(q[i-1], c[j])+euclidian(q[i], c[j]))
            g[i, j] = minD
    return g[len(q) - 1, len(c) - 1]

def distance1(q, c):

    g = np.zeros((len(q), len(c)))
    g[0, 0] = 2* euclidian(q[0], c[0])
    for i in range(1, len(q)):
        g[i, 0] = euclidian(q[i], c[0]) + g[i - 1, 0]
    
    for j in range(1, len(c)):
        g[0, j] = euclidian(q[0], c[j]) + g[0, j - 1]

 
    for i in range(1, len(q)):
        for j in range(1, len(c)):
            minD = 100000000
            if i-1 >=0 and j-2 >=0:
                minD = min(minD, g[i-1][j-2]+2*euclidian(q[i], c[j-1])+euclidian(q[i], c[j]))
            if i-1 >=0 and j-1 >=0:
                minD = min(minD, g[i-1][j-1]+2*euclidian(q[i], c[j]))
            if i-2 >=0 and j-1 >=0:
                minD = min(minD, g[i-2][j-1]+2*euclidian(q[i-1], c[j])+euclidian(q[i], c[j]))
            g[i, j] = minD
    return g[len(q) - 1, len(c) - 1]

def distance1_5(q, c):

    g = np.zeros((len(q), len(c)))
    g[0, 0] = 2* euclidian(q[0], c[0])
    for i in range(1, len(q)):
        g[i, 0] = euclidian(q[i], c[0]) + g[i - 1, 0]
    
    for j in range(1, len(c)):
        g[0, j] = euclidian(q[0], c[j]) + g[0, j - 1]

 
    for i in range(1, len(q)):
        for j in range(1, len(c)):
            minD = 100000000
            if i-5 >= 0 and j-3 >= 0:
                minD = min(minD, g[i-5][j-3]+3*euclidian(q[i-4], c[j-2])+2*euclidian(q[i-3], c[j-1])+euclidian(q[i-2], c[j-1])+euclidian(q[i-1], c[j])+euclidian(q[i], c[j]))
            if i-2 >= 0 and j-1 >=0:
                minD = min(minD, g[i-2][j-1]+2*euclidian(q[i-1], c[j])+euclidian(q[i], c[j]))
            if i-1 >= 0 and j-1 >=0:
                minD = min(minD, g[i-1][j-1]+2*euclidian(q[i], c[j]))
            if i-1 >= 0 and j-2 >=0:
                minD = min(minD, g[i-1][j-2]+2*euclidian(q[i], c[j-1])+euclidian(q[i], c[j]))
            if i-3 >= 0 and j-5 >=0:
                minD = min(minD, g[i-3][j-5]+3*euclidian(q[i-2], c[j-4])+2*euclidian(q[i-1], c[j-3])+euclidian(q[i-1], c[j-2])+euclidian(q[i], c[j-1])+euclidian(q[i], c[j]))
            g[i, j] = minD
    return g[len(q) - 1, len(c) - 1]

def distance2(q, c):

    g = np.zeros((len(q), len(c)))
    g[0, 0] = 2* euclidian(q[0], c[0])
    for i in range(1, len(q)):
        g[i, 0] = euclidian(q[i], c[0]) + g[i - 1, 0]
    
    for j in range(1, len(c)):
        g[0, j] = euclidian(q[0], c[j]) + g[0, j - 1]

 
    for i in range(1, len(q)):
        for j in range(1, len(c)):
            minD = 100000000
            if i-1 >=0 and j-1 >=0:
                minD = min(minD, g[i-1][j-1]+2*euclidian(q[i], c[j]))
            if i-2 >=0 and j-3 >= 0:
                minD = min(minD, g[i-2][j-3]+2*euclidian(q[i-1], c[j-2])+2*euclidian(q[i], c[j-1])+euclidian(q[i], c[j]))
            if i-3 >=0 and j-2 >=0:
                minD = min(minD, g[i-3][j-2]+2*euclidian(q[i-2], c[j-1])+2*euclidian(q[i-1], c[j])+euclidian(q[i], c[j]))
    return g[len(q) - 1, len(c) - 1]

def one_n_n(x_train, y_train, x, p):
    best_so_far = 1e9
    cls = -1
    for i, x_t in enumerate(x_train):
        if p==0:
            D = distance0(x, x_t)
        elif p==0.5:
            D = distance0_5(x, x_t)
        elif p==1:
            D = distance1(x, x_t)
        elif p==1.5:
            D = distance1_5(x, x_t)
        elif p==2:
            D = distance2(x, x_t)

        if D < best_so_far:
            best_so_far = D
            cls = y_train[i]
    return cls


def cal_acc(x_train, y_train, x_test, y_test, p):
    correct = 0
    for i in range(len(x_test)):
        cls = one_n_n(x_train, y_train, x_test[i], p)
        print(i)
        if cls == y_test[i]:
            correct += 1
    return correct / len(x_test)


def process_dataset(file_name):
    x_train, y_train, x_test, y_test = prepare_data(file_name)
    for p in P:
        acc = cal_acc(x_train, y_train, x_test, y_test, p)
        print(f"{file_name} {p}:{acc}")
        df = pd.DataFrame(
            {"file_name": [file_name], "P": [p], "accuracy": [acc]}
        )
        df.to_csv(excel_file_path, mode="a", index=False, header=False)

def main():
    df = pd.DataFrame(columns=COLUMNS)
    df.to_csv(excel_file_path, mode="a", index=False)
    for file_name in FILE_NAME:
        process_dataset(file_name)


if __name__ == "__main__":
    main()
