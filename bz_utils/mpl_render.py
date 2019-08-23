import numpy as np
import matplotlib.pyplot as plt


def to_numpy(fig=None):
    fig = fig or plt.gcf()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    return np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
