from heuristic_solver import *
import functools
import operator
from tqdm import tqdm


class LocalBeamSearch(HeuristicSolver):
    """ this class implements the local beam search algorithm.
    it keeps track of K search processes simultaneously. generate their successors and chooses the K best states out
    of all states.
    """

    def __init__(self, fitness_func, fitness_name, successor_func, generate_initial_func,
                 num_iteration, k=5):
        super().__init__(fitness_func, fitness_name, successor_func, generate_initial_func, num_iteration)
        self._k = k
        self._states_list = [self._initial_state_generator() for i in range(self._k)]
        scores = [self._get_fitness(father.get_matrix()) for father in self._states_list]
        best = scores.index(min(scores))
        for i in range(self._k):
            self._states_list[i].save_score(scores[i])
        self._current_state = self._states_list[best]

    def solve(self):
        """after all the iterations it will choose the best state"""
        self._statistics.add_extra_data("k states avg", np.array(np.average([father.get_score() for father in
                                                                             self._states_list])))

        progress_bar_iterator = tqdm(
            iterable=range(self._num_iterations),
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
            desc='local beam solver')
        for i in progress_bar_iterator:
            # creating new states from the current states
            children = [self._get_successor(father) for father in self._states_list]
            children = functools.reduce(operator.iconcat, children, [])

            for child in children:
                child.save_score(self._get_fitness(child.get_matrix()))

            self._states_list = self._states_list + children
            self._states_list = sorted(self._states_list, key=lambda state: state.get_score())

            del self._states_list[self._k:]

            # statistics:
            # add the neighbors data
            self._statistics.add_neighbors_data([child.get_score() for child in children])
            # add the best state data out of the k states
            self._statistics.add_current_state_data(self._states_list[0])
            # add the k states average score
            self._statistics.add_extra_data("k states avg", np.append(self._statistics.get_from_extra_data("k states "
                                                                                                           "avg"),
                                                                      np.average([father.get_score() for father in
                                                                                  self._states_list])))

            progress_bar_iterator.set_postfix_str('loss=%.2f' % self._states_list[0].get_score())

        self._current_state = self._states_list[0]

    def get_answer(self):
        return self._current_state

    def get_answer_loss(self):
        return self._current_state.get_score()

    def set_current_state(self, matrix):
        self._states_list = [State(matrix) for i in range(self._k)]
        for state in self._states_list:
            state.save_score(self._get_fitness(state.get_matrix()))
