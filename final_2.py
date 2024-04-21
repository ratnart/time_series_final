arr = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
m = {}
m["0"] = [
    "dataset/0/36.jpg",
    "dataset/0/45.jpg",
    "dataset/0/12580.jpg",
    "dataset/0/13793.jpg",
]
m["1"] = [
    "dataset/1/10107.jpg",
    "dataset/1/16692.jpg",
    "dataset/1/2961.jpg",
    "dataset/1/4951.jpg",
]
m["2"] = [
    "dataset/2/4169.jpg",
    "dataset/2/6642.jpg",
    "dataset/2/3575.jpg",
    "dataset/2/7708.jpg",
]
m["3"] = [
    "dataset/3/26615.jpg",
    "dataset/3/16479.jpg",
    "dataset/3/20612.jpg",
    "dataset/3/20736.jpg",
]
m["4"] = [
    "dataset/4/40588.jpg",
    "dataset/4/59135.jpg",
    "dataset/4/39045.jpg",
    "dataset/4/38870.jpg",
]
m["5"] = [
    "dataset/5/20073.jpg",
    "dataset/5/28645.jpg",
    "dataset/5/29275.jpg",
    "dataset/5/29588.jpg",
]
m["6"] = [
    "dataset/6/17502.jpg",
    "dataset/6/17521.jpg",
    "dataset/6/17941.jpg",
    "dataset/6/12341.jpg",
]
m["7"] = [
    "dataset/7/5540.jpg",
    "dataset/7/1520.jpg",
    "dataset/7/1577.jpg",
    "dataset/7/2352.jpg",
]
m["8"] = [
    "dataset/8/14889.jpg",
    "dataset/8/21084.jpg",
    "dataset/8/21594.jpg",
    "dataset/8/28015.jpg",
]
m["9"] = [
    "dataset/9/20336.jpg",
    "dataset/9/19449.jpg",
    "dataset/9/20890.jpg",
    "dataset/9/23582.jpg",
]
import cv2
import numpy as np
import matplotlib.pyplot as plt

time_series_data = []
labels = []

for i in arr:

    print(f"processing {i}")
    for j in m[i]:
        img = cv2.imread(j)
        img = cv2.resize(img, (128, 128))
        img = cv2.bitwise_not(img)
        sum = np.sum(img[:, :, 0], axis=0)
        sum = sum / (128 * 255)
        time_series_data.append(sum)
        labels.append(i)


def distance(a, b):
    return (a - b) ** 2


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

    shape_avg = []

    i = n - 1
    j = m - 1
    while i > 0 and j > 0:
        if path[i][j] == 1:
            j -= 1
        elif path[i][j] == 2:
            i -= 1
        else:
            shape_average[(i + j) // 2] = (q[i] + c[j]) / 2
            shape_avg.append((q[i] + c[j]) / 2)
            i -= 1
            j -= 1
    shape_avg.reverse()
    return np.array(shape_avg)


x = np.array([1, 2, 3, 4, 5])

for i in range(10):
    current = time_series_data[i * 4]
    for j in range(1, 4):
        current = shape_average(current, time_series_data[i * 4 + j])
    ax, fig = plt.subplots(2, 3)

    fig[0, 0].plot(time_series_data[i * 4])
    fig[0, 0].set_title(f"Time Series Data for number {i}")
    fig[0, 0].set_yticks([])
    fig[0, 0].set_xticks([])

    fig[0, 1].plot(time_series_data[i * 4 + 1])
    fig[0, 1].set_title(f"Time Series Data for number {i}")
    fig[0, 1].set_yticks([])
    fig[0, 1].set_xticks([])

    fig[0, 2].plot(time_series_data[i * 4 + 2])
    fig[0, 2].set_title(f"Time Series Data for number {i}")
    fig[0, 2].set_yticks([])
    fig[0, 2].set_xticks([])

    fig[1, 0].plot(time_series_data[i * 4 + 3])
    fig[1, 0].set_title(f"Time Series Data for number {i}")
    fig[1, 0].set_yticks([])
    fig[1, 0].set_xticks([])

    fig[1, 1].plot(current)
    fig[1, 1].set_title(f"Shape Average for number {i}")
    fig[1, 1].set_yticks([])
    fig[1, 1].set_xticks([])

    plt.savefig(f"shape_base_{i}.png")
