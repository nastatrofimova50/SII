import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_func(x1, x2):
    return 2 * np.pow(x1, 2) + 3 * np.pow(x2, 2)


def create_csv(path, size):
    x1 = np.linspace(0.1, 5, size)
    x2 = np.linspace(5, 10, size)
    y = calculate_func(x1, x2)
    pd.DataFrame({"y": y, "x1": x1, "x2": x2}).to_csv(path, index=False)


def draw_plot_2d(x, y, xlabel, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color="red", edgecolors="black", linewidths=0.5)
    plt.plot(x, y, color="blue")
    plt.xlabel(xlabel)
    plt.ylabel("y")
    plt.title(title)
    plt.grid()
    plt.show()


def main():
    create_csv("data2.csv", 500)
    data = pd.read_csv("data2.csv")
    x1 = data["x1"]
    x2 = data["x2"].iloc[len(data) // 2]
    y = calculate_func(x1, x2)
    draw_plot_2d(x1, y, "x1", f"y(x1) and x2 = const ({x2})")

    x1 = data["x1"].iloc[len(data) // 2]
    x2 = data["x2"]
    y = calculate_func(x1, x2)
    draw_plot_2d(x2, y, "x2", f"y(x2) and x1 = const ({x1})")

    mean_x1 = data["x1"].mean()
    mean_x2 = data["x2"].mean()

    print(f"\nMean: \nx1: {mean_x1}, x2: {mean_x2}, y: {round(data["y"].mean(), 3)}\n")
    print(f"Min: \nx1: {data["x1"].min()}, x2: {data["x2"].min()}, y: {round(data["y"].min(), 3)}\n")
    print(f"Max: \nx1: {data["x1"].max()}, x2: {data["x2"].max()}, y: {round(data["y"].max(), 3)}\n")

    data_by_condition = data[(data["x1"] < mean_x1) | (data["x2"] < mean_x2)]
    data_by_condition.to_csv("data_by_condition.csv", index=False)

    x1_3d = np.array(data["x1"])
    x2_3d = np.array(data["x2"])
    x1_3d, x2_3d = np.meshgrid(x1_3d, x2_3d)
    y_3d = calculate_func(x1_3d, x2_3d)

    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x1_3d, x2_3d, y_3d, cmap='viridis')
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == "__main__":
    main()