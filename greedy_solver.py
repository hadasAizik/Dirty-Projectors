from heuristic_solver import HeuristicSolver
from tqdm import tqdm
import copy


# the greedy solver progresses by choosing the best change out of few changes possible
class GreedySolver(HeuristicSolver):
    """
    generate successors and continuously move towards the best state.
    """

    def solve(self):
        progress_bar_iterator = tqdm(
            iterable=range(self._num_iterations),
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
            desc='greedy solver'
        )
        for i in progress_bar_iterator:
            # neighbors - a list of neighbors states
            neighbors = self._get_successor(self._current_state)
            neighbors_score = [self._get_fitness(neighbor.get_matrix()) for neighbor in neighbors]
            best_idx = neighbors_score.index(min(neighbors_score))

            if neighbors_score[best_idx] < self._current_state.get_score():
                self._current_state = copy.deepcopy(neighbors[best_idx])
                self._current_state.save_score(neighbors_score[best_idx])
            progress_bar_iterator.set_postfix_str('loss=%.2f' % self._current_state.get_score())

            # statistics:
            # add the current state data
            self._statistics.add_current_state_data(self._current_state)
            # add neighbors data
            self._statistics.add_neighbors_data(neighbors_score)

    def get_answer_loss(self):
        return self._current_state.get_score()

    def get_answer(self):
        return self._current_state
