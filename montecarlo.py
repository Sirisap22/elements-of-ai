import numpy as np
from ox import OX
from copy import deepcopy


class OX_montecarlo(OX):
    def __init__(self):
        super().__init__()

    def montecarlo(self, n=500):
        P = np.zeros(self.board.shape)
        locs = self.available_moves()
        if len(locs) == 1:
            return P, locs[0]
        for loc in locs:
            for _ in range(n):
                obj = deepcopy(self)
                while True:
                    if obj.board[loc] == ' ':
                        move = loc
                    else:
                        moves = obj.available_moves()
                        move = moves[np.random.choice(len(moves))]
                    obj.board[move] = obj.turn
                    winner = obj.check_end()
                    if winner == obj.ai:
                        P[loc] += 1
                    if winner is not None:
                        break
                    obj.toggle()
        return P / n, np.unravel_index(P.argmax(), P.shape)

    def ai_move(self):
        score, loc = self.montecarlo()
        print(f'score = {score}')
        self.move(loc, self.ai)


if __name__ == '__main__':
    g = OX_montecarlo()
    g.show()
    while g.check_end() is None:
        g.move(eval(input('move:')))
        g.show()
