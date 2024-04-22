import numpy as np
import matplotlib.pyplot as plt


BASE_PATH = ""


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

x_train, y_train, x_test, y_test = prepare_data("Beef")

# x_train = [[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0]]
# y_train = [0, 0]

# x_test = [[0,0,0,0,0,1,1,1,1,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0,1,1,1,1,0,0]]

def distance(a, b):
    return abs(a-b)

def shape_average(q, c):
    n = len(q)
    m = len(c)

    path = np.zeros((n, m))
    acc_dist = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                acc_dist[i][j] = distance(q[i], c[j])
            elif i == 0:
                acc_dist[i][j] = acc_dist[i][j - 1] + distance(q[i], c[j])
                path[i][j] = 1
            elif j == 0:
                acc_dist[i][j] = acc_dist[i - 1][j] + distance(q[i], c[j])
                path[i][j] = 2
            else:
                if acc_dist[i - 1][j] < acc_dist[i][j - 1]:
                    if acc_dist[i - 1][j] < acc_dist[i - 1][j - 1]:
                        acc_dist[i][j] = acc_dist[i - 1][j] + distance(q[i], c[j])
                        path[i][j] = 2
                    else:
                        acc_dist[i][j] = acc_dist[i - 1][j - 1] + distance(q[i], c[j])
                        path[i][j] = 3
                else:
                    if acc_dist[i][j - 1] < acc_dist[i - 1][j - 1]:
                        acc_dist[i][j] = acc_dist[i][j - 1] + distance(q[i], c[j])
                        path[i][j] = 1
                    else:
                        acc_dist[i][j] = acc_dist[i - 1][j - 1] + distance(q[i], c[j])
                        path[i][j] = 3
    return acc_dist[n-1][m-1], path

def process_path(path, q, c):
    shape_avg = np.zeros(2*len(q))

    i = len(q) - 1
    j = len(c) - 1

    shape_avg[0] = q[0] + c[0]

    while i > 0 and j > 0:
        if path[i][j] == 1:
            shape_avg[i+j] = c[j]
            j -= 1
        elif path[i][j] == 2:
            shape_avg[i+j] = q[i]
            i -= 1
        else:
            shape_avg[i+j+1] = (q[i] + c[j])
            i -= 1
            j -= 1
    return shape_avg


def get_shape_avg(x_train):
    base = np.zeros(len(x_train[0]*2))

    for i in range(len(x_train)):
        for j in range(i+1, len(x_train)):
            _, path = shape_average(x_train[i], x_train[j])
            base += process_path(path, x_train[i], x_train[j])


    base = base / (len(x_train)*(len(x_train)-1)/2)
        
    base = base.reshape(len(x_train[0]), 2)

    base = np.average(base, axis=1)

    return base


cluster = {}

for i in range(len(y_train)):
    if y_train[i] not in cluster.keys():
        cluster[y_train[i]] = []
    cluster[y_train[i]].append(x_train[i])

representatives = {}

for key in cluster.keys():
    print(key)
    # plt.plot(cluster[key])
    # plt.show()
    representatives[key] = get_shape_avg(cluster[key])

correct = 0

for i in range(len(x_test)):
    best_so_far = 1e9
    cls = -1
    for key in representatives.keys():
        dist, _ = shape_average(x_test[i], representatives[key])
        if dist < best_so_far:
            best_so_far = dist
            cls = key

    if cls == y_test[i]:
        correct += 1

print("Accuracy=", correct / len(x_test))
    

def plot(cluster, representatives):
    for key in cluster.keys():
        plt.plot(representatives[key], color='blue', linestyle='-')
        for series in cluster[key]:
            plt.plot(series, color='red', linestyle='--')
        plt.show()

plot(cluster, representatives)
