import numpy as np
from ox import OX


class OX_minimax(OX):
    def __init__(self):
        super().__init__()

    def ai_move(self):
        if np.all(self.board == ' '):
            super().ai_move()
        else:
            score, loc = self.minimax()
            print(f'score = {score}')
            self.move(loc, self.ai)

    def minimax(self, is_max=True):
        scores = {self.ai: 1, self.player: -1, 'No one': 0}
        best_loc = None
        winner = self.check_end()
        if winner is not None:
            return scores[winner] * np.sum(self.board == ' '), best_loc
        best_score = -np.inf if is_max else np.inf
        for loc in self.available_moves():
            self.board[loc] = self.ai if is_max else self.player
            score, _ = self.minimax(not is_max)
            # reverse the move
            self.board[loc] = ' '
            if is_max:
                if score > best_score:
                    best_score = score
                    best_loc = loc
            else:
                if score < best_score:
                    best_score = score
                    best_loc = loc

        return best_score, best_loc


if __name__ == '__main__':
    g = OX_minimax()
    g.show()
    while g.check_end() is None:
        g.move(eval(input('move:')))
        g.show()
