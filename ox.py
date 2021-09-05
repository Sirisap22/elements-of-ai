import numpy as np


class OX:
    def __init__(self, player='O', player_first=False):
        self.player = player
        self.ai = 'X' if player == 'O' else 'O'
        self.turn = player if player_first else self.ai
        self.board = np.full((3, 3), ' ')

        if not player_first:
            self.ai_move()

    def show(self):
        print(self.board)

    def move(self, loc, turn=None):
        turn = self.player if turn is None else turn
        if loc in self.available_moves():
            self.board[loc] = turn
            winner = self.check_end()
            if winner is not None:
                print(winner, 'wins')
                return True
            self.toggle()
            if self.turn == self.ai:
                self.ai_move()
            return True
        else:
            print('Invalid move')
            return False

    def available_moves(self):
        moves = np.where(self.board == ' ')
        moves = np.vstack((moves[0], moves[1])).T
        return tuple(map(tuple, moves))

    def ai_move(self):
        moves = self.available_moves()
        # random move
        i = np.random.choice(len(moves))
        self.move(moves[i], self.ai)

    def check_end(self):
        # return None for not end game, or
        #        'X' or 'O' for winner, or
        #        'No one' for draw.

        # Horizontal check
        for r in range(self.board.shape[0]):
            temp = np.unique(self.board[r])
            if len(temp) == 1 and temp != ' ':
                return temp[0]

        # Horizontal check
        for c in range(self.board.shape[1]):
            temp = np.unique(self.board[:, c])
            if len(temp) == 1 and temp != ' ':
                return temp[0]

        # Diagonal check
        temp = np.unique(np.diag(self.board))
        if len(temp) == 1 and temp != ' ':
            return temp[0]
        temp = np.unique(np.diag(self.board[:, ::-1]))
        if len(temp) == 1 and temp != ' ':
            return temp[0]

        if np.any(self.board == ' '):
            return None  # Not end game
        else:
            return 'No one'  # Draw

    def toggle(self):
        self.turn = self.ai if self.turn == self.player else self.player


if __name__ == '__main__':
    g = OX()
    g.show()
    while g.check_end() is None:
        g.move(eval(input('move:')))
        g.show()



