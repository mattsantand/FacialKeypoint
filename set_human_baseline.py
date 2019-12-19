import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

IMG_SHAPE = (96, 96, 1)


def RMSE(yhat, y):
    return np.sqrt(np.nanmean((yhat - y) ** 2))


def visualise_element(df, idx):
    """
    Function to visualise a single row of the dataset

    :param df: dataset
    :param idx: row index
    :return:
    """
    fig = None
    if idx < df.shape[0]:
        X = np.array([float(point) for point in data.loc[idx, "Image"].split(" ")])
        X = X.reshape(*IMG_SHAPE[:-1])
        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.imshow(X, cmap="Greys_r")
    return fig


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(ix, iy)
    plt.close("all")


if __name__ == '__main__':
    # loading the data

    data = pd.read_csv("updated_train.csv")

    features = data.columns.values
    print(features)

    ITER = 100
    answers = np.zeros((ITER, 4))
    iter = 0
    while iter < ITER:
        idx = np.random.randint(0, data.shape[0])
        elemidx = np.random.randint(0, (data.shape[1] - 1) // 2)
        if not np.isnan(data.iloc[idx, 2 * elemidx]) and not np.isnan(data.iloc[idx, 2 * elemidx + 1]):
            fig = visualise_element(data, idx)
            ix, iy = np.nan, np.nan
            plt.title(features[2 * elemidx])
            plt.show()
            answers[iter, 0] = data.iloc[idx, 2 * elemidx]
            answers[iter, 1] = data.iloc[idx, 2 * elemidx + 1]
            answers[iter, 2] = ix
            answers[iter, 3] = iy
            print(answers[iter, :])
            fig = visualise_element(data, idx)
            plt.plot(answers[iter, 0], answers[iter, 1], "or")
            plt.plot(answers[iter, 2], answers[iter, 3], "xb")
            plt.show()
            iter += 1
    print(RMSE(answers[:, :2], answers[:, 2:]))
