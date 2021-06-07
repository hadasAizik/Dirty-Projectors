from heuristic_solver import HeuristicSolver
import copy
from tqdm import tqdm


class HillClimbingSolver(HeuristicSolver):
    """hill climbing with restarts:
    continuously moves in the direction of increasing value. when it have no better ways to go to, it will restart"""

    def solve(self, num_of_trys=3):
        trys = 0
        best_state = copy.deepcopy(self._current_state)
        self._statistics.add_extra_data("num of resets", 0)

        progress_bar_iterator = tqdm(
            iterable=range(self._num_iterations),
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
            desc='hill climber')

        for i in progress_bar_iterator:
            neighbors = self._get_successor(self._current_state)
            neighbors_score = [self._get_fitness(neighbor.get_matrix()) for neighbor in
                               neighbors]

            best_neighbor_idx = neighbors_score.index(min(neighbors_score))
            best_neighbor = neighbors[best_neighbor_idx]

            if neighbors_score[best_neighbor_idx] >= self._current_state.get_score():
                " if we are at a local minimum and we passed the number of tries"
                if trys == num_of_trys:
                    "keep best state so far and restart"
                    if self._current_state.get_score() < best_state.get_score():
                        best_state = copy.deepcopy(self._current_state)

                    " updates number of restart"
                    self._statistics.add_extra_data("num of resets", self._statistics.get_from_extra_data("num of "
                                                                                                          "resets") + 1)
                    self._current_state = self._initial_state_generator()
                    self._current_state.save_score(self._get_fitness(self._current_state.get_matrix()))
                    trys = 0
                else:
                    trys += 1
            else:
                self._current_state = copy.deepcopy(best_neighbor)
                self._current_state.save_score(neighbors_score[best_neighbor_idx])

            # statistics:
            # add current state data
            self._statistics.add_current_state_data(self._current_state)
            # add neighbors data
            self._statistics.add_neighbors_data(neighbors_score)

            progress_bar_iterator.set_postfix_str('loss=%.2f' % self._current_state.get_score())

        if best_state.get_score() < self._current_state.get_score():
            self._current_state = best_state

    def get_answer_loss(self):
        return self._current_state.get_score()

    def get_answer(self):
        return self._current_state
