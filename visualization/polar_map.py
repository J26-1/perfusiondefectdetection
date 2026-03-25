import numpy as np
import matplotlib.pyplot as plt

def generate_polar_map(slice, save_path=None):

    slice = np.squeeze(slice)

    if slice.ndim != 2:
        raise ValueError("Polar map expects 2D image")

    h, w = slice.shape
    center = w // 2

    r_profile = slice[:, center]
    theta = np.linspace(0, 2 * np.pi, len(r_profile))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    ax.plot(theta, r_profile)
    ax.set_title("Myocardial Perfusion Polar Map")

    if save_path:
        plt.savefig(save_path)

    plt.close()

