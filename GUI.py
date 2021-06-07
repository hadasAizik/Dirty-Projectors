import matplotlib.pyplot as plt
import numpy as np


def show_result(solver, solver_name):
    ans = solver.get_answer().get_matrix()
    title = f"{solver_name}, dim:{ans.shape[0]}, {solver.get_num_iters()} iterations, black pixels:{np.count_nonzero(ans != 0)} "
    print_3d_matrix(ans, title, ans.shape[0]).show()


def get_3d_plot(three_d_matrix, ax, title, length):
    """
    updates a plt.ax object with a plot of a given 3d matrix
    """
    x, y, z = np.where(three_d_matrix != 0)
    ax.scatter(x, y, z, c='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.set_title(title)


def print_3d_matrix(three_d_matrix, title, length):
    x, y, z = np.where(three_d_matrix != 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.set_title(title)
    return fig


def print_2d_projections(three_d_matrix, length):
    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
    for idx, mat in enumerate(extract_pics(three_d_matrix)):
        x, y = np.where(mat)
        axes[idx].scatter(x, y)

    fig.set_tight_layout(True)
    fig.suptitle("Solution Projections")
    return fig


def extract_pics(mat):
    return (np.sum(mat, axis=0) > 0), (np.sum(mat, axis=1) > 0), (np.sum(mat, axis=2) > 0)
