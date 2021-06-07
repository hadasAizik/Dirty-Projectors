from solver import *
from state import *
from statistics import Statistics
from GUI import get_3d_plot, print_2d_projections
import matplotlib.pyplot as plt
from mesh import *


class HeuristicSolver(Solver):
    def __init__(self, fitness_func, fitness_name, successor_func, generate_initial_func, num_iteration=0):
        self._get_fitness = fitness_func
        self._fitness_name = fitness_name
        self._get_successor = successor_func
        self._initial_state_generator = generate_initial_func
        self._current_state = generate_initial_func()
        self._current_state.save_score(fitness_func(self._current_state.get_matrix()))
        self._num_iterations = num_iteration
        self._statistics = Statistics(self._current_state)
        self._run_time = 0
        self._solver_name = self.__class__.__name__

    def set_num_iters(self, num):
        self._num_iterations = num

    def get_num_iters(self):
        return self._num_iterations

    def set_loss_func(self, func):
        self._get_fitness = func

    def set_init_func(self, func):
        self._initial_state_generator = func

    def set_successor_func(self, func):
        self._get_successor = func

    def set_current_state(self, matrix):
        self._current_state = State(matrix)
        self._current_state.save_score(self._get_fitness(matrix))

    def solve(self):
        pass

    def get_answer(self, export_mesh=False):
        """
        1. plots 2 figures - the first figure includes plot of 3d object of the best
            solution found after algorithm termination, and statistics about fitness and num of
            black pixels along iterations.
            the second figure includes projections of the input pictures of the 3d solution.
        2. The function also outputs the mesh represenation of the solution into a stl file.
        @param export_mesh:
        @return:
        """
        solution = self._current_state.get_matrix()
        # ==== first plot=====
        fig = plt.figure(figsize=(18, 5))
        ax_3d = fig.add_subplot(131, projection='3d')
        ax_fitness = fig.add_subplot(132)
        ax_blacks = fig.add_subplot(133)
        fig.subplots_adjust(top=0.8, bottom=0.095, hspace=0, wspace=0.25, left=0.08, right=0.94)
        get_3d_plot(solution, ax_3d, f"{np.sum(solution)} black pixels", solution.shape[0])
        plot_statistics(self._statistics, ax_fitness, ax_blacks, solution.shape[0] ** 3)
        configuration = f"dim={solution.shape[0]}^3, Target_func={self._fitness_name}"
        title = f"{self._solver_name} - runtime={round(self._run_time / 60, 3)} min. " + configuration
        fig.suptitle(title)

        # ==== second plot =====
        print_2d_projections(solution, solution.shape[0])
        plt.show()

        if export_mesh:
            export_mesh_to_stl(smooth_mesh_laplacian(matrix_to_mesh(solution)), title)


def get_answer_loss(self):
    pass


def get_avg_array(self):
    return self._statistics.get_avg()


def get_std_array(self):
    return self._statistics.get_std()


def get_scores(self):
    return self._statistics.get_scores()


def get_num_blacks(self):
    return self._statistics.get_num_blacks()


def get_volume_ratio(self):
    return self._statistics.get_volume_ratio()


def get_extra_data_dict(self):
    return self._statistics.get_extra_data_dict()


def plot_statistics(stat_obj, ax_fitness, ax_blacks, dim):
    fit_values = stat_obj.get_scores()
    num_iters = np.arange(1, len(fit_values) + 1)
    line_1 = ax_fitness.plot(num_iters, fit_values, 'g*--')

    ax_fitness.set_xlabel("Iteration")
    ax_fitness.set_ylabel("Score")
    ax_fitness.set_title("Fitness along generations")

    blacks_num = stat_obj.get_num_blacks()
    blacks_ratio = np.round_(np.divide(blacks_num, dim), 2)
    line1 = ax_blacks.plot(num_iters, blacks_ratio, 'k--', label="Black pixels ratio")
    ax_blacks.set_ylabel("Black pixels ratio")

    avg_ax = ax_blacks.twinx()
    line2 = avg_ax.plot(num_iters, blacks_num, 'g-', label="Num Black pixels")
    avg_ax.set_ylabel("Num Black pixels")
    for tl in avg_ax.get_yticklabels():
        tl.set_color("g")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax_blacks.legend(lns, labs, loc="upper right")
    ax_blacks.set_title("Black pixels rate")
