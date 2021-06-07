from deap import creator, tools, base, algorithms
from solver import Solver
from specified_genetic_operators import *
import matplotlib.pyplot as plt
import time
from GUI import print_2d_projections, get_3d_plot
from mesh import *


class GeneticSolver(Solver):
    def __init__(self, shape, objective_func, objective_name, params):
        self._individual_shape = shape
        self._objective = objective_func, objective_name
        self._num_generations = params['num_generations']
        self._mupb = params['mutation_probability']
        self._cxpb = params['crossover_probability']
        # creates the base classes uses - alias, base class, and attributes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", init_3d_individual, creator.Individual, shape)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)

        # baseline operators for the 3d algorithm
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.register("mate", cx_3d_single_cut, icls=creator.Individual)
        self.toolbox.register("mutate", mut_flip_random_pixel, icls=creator.Individual,
                              num_pixels=shape[0])

        self.population = self.toolbox.population(params['population_size'])
        self.best = None
        self.log = None

        fit_stat = tools.Statistics(key=lambda ind: ind.fitness.values)
        black_stat = tools.Statistics(key=lambda ind: np.sum(ind))
        self.mstats = tools.MultiStatistics(fitness=fit_stat, black_pixels=black_stat)
        self.mstats.register("avg", np.average)
        self.mstats.register("std", np.std, axis=0)
        self.mstats.register("min", np.min, axis=0)

        self.run_time = 0

    def evaluate(self, individual):
        eval = self._objective[0](individual)
        # if eval < 1500:
        #     print_3d_matrix(individual, f"{eval}", 32)
        return eval,

    def solve(self):
        start = time.time()
        self.population, self.log = algorithms.eaMuCommaLambda(self.population,
                                                               self.toolbox,
                                                               cxpb=self._cxpb,
                                                               mutpb=self._mupb,
                                                               mu=100, lambda_=100,
                                                               ngen=self._num_generations,
                                                               stats=self.mstats,
                                                               halloffame=tools.HallOfFame(1,
                                                                                           similar=np.array_equal),
                                                               verbose=False)
        self.run_time = time.time() - start

        self.best = self.population[np.argmax([ind.fitness for ind in self.population])]

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
        if self.best is not None:
            # ==== first plot=====
            fig = plt.figure(figsize=(18, 5))
            ax_3d = fig.add_subplot(131, projection='3d')
            ax_fitness = fig.add_subplot(132)
            ax_blacks = fig.add_subplot(133)
            fig.subplots_adjust(top=0.8, bottom=0.095, hspace=0, wspace=0.25, left=0.08,
                                right=0.94)
            get_3d_plot(self.best, ax_3d, f"{np.sum(self.best)} black pixels",
                        self._individual_shape[0])
            plot_statistics(self.log, ax_fitness, ax_blacks, self._individual_shape[0] ** 3)
            configuration = f"dim={self._individual_shape[0]}^3, objective={self._objective[1]}," \
                            f"pop_size={len(self.population)}, " \
                            f"cxpb={self._cxpb}, mupb={self._mupb}"
            title = f"Genetic - runtime={round(self.run_time / 60, 3)} min. " + configuration
            fig.suptitle(title)

            # ==== second plot =====
            print_2d_projections(self.best, self._individual_shape[0])
            plt.show()

            if export_mesh:
                export_mesh_to_stl(smooth_mesh_laplacian(matrix_to_mesh(self.best)), title)

    def get_scores(self):
        """
        Retrieve an array of fitness values of the best solution in every population (iteration).
        """
        min_fit = self.log.chapters['fitness'].select("min")
        min_fit = [val[0] for val in min_fit]
        return np.array(min_fit)


def plot_statistics(logbook, ax_fitness, ax_blacks, dim):
    gen = logbook.select("gen")
    fit_avg = logbook.chapters['fitness'].select("avg")
    fit_std = logbook.chapters['fitness'].select("std")
    fit_std = [val[0] for val in fit_std]
    min_fit = logbook.chapters['fitness'].select("min")
    min_fit = [val[0] for val in min_fit]

    line_1 = ax_fitness.plot(gen, min_fit, 'g*--', label="Minimum Fitness")
    line_2 = ax_fitness.errorbar(x=gen, y=fit_avg, yerr=fit_std, label="Average Fitness and STD",
                                 fmt='b-', ecolor='r')

    ax_fitness.set_xlabel("Generation")
    ax_fitness.set_ylabel("Fitness")

    lns = [line_1[0], line_2]
    labs = [l.get_label() for l in lns]
    ax_fitness.legend(lns, labs, loc="upper right")
    ax_fitness.set_title("Fitness along generations")

    blacks_avg = logbook.chapters['black_pixels'].select("avg")
    blacks_ratio = np.round_(np.divide(blacks_avg, dim), 2)
    line1 = ax_blacks.plot(gen, blacks_ratio, 'k--', label="Average Black pixels ratio")
    ax_blacks.set_ylabel("Black pixels ratio")

    avg_ax = ax_blacks.twinx()
    line2 = avg_ax.plot(gen, blacks_avg, 'g-', label="Average Black pixels")
    avg_ax.set_ylabel("num Black pixels")
    for tl in avg_ax.get_yticklabels():
        tl.set_color("g")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax_blacks.legend(lns, labs, loc="upper right")
    ax_blacks.set_title("Average Black pixels ratio")
