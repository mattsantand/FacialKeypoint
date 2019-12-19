import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

IMG_SHAPE = (96, 96, 1)


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
        plt.imshow(X, cmap="Greys_r")
    Y = df.iloc[idx, :-1]
    plt.plot(Y[::2], Y[1::2], "or")
    print(Y)
    return fig


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(ix, iy)
    plt.close("all")


if __name__ == '__main__':
    # loading the data

    data = pd.read_csv("updated_train.csv")
    print(data.loc[3, :])
    # visualise an example
    visualise_element(data, 3)
    plt.show()

    features = data.columns.values
    print(features)

    for idx in range(data.shape[0]):
        # for i in range(1):
        if data.loc[idx, :].isnull().sum() > 20:
            # if there are empty values
            for elemidx, elem in enumerate(data.iloc[idx, :-1:2]):
                fig = visualise_element(data, idx)
                fig.canvas.mpl_connect('button_press_event', onclick)
                ix, iy = data.iloc[idx, 2 * elemidx], data.iloc[idx, 2 * elemidx + 1]
                plt.title(features[2 * elemidx])
                plt.show()
                data.iloc[idx, 2 * elemidx] = ix
                data.iloc[idx, 2 * elemidx + 1] = iy
                print("\n" * 3, data.iloc[idx, 2 * elemidx:2 * elemidx + 2])
            data.to_csv("updated_train.csv", index_label=False)
