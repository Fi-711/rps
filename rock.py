import random
import numpy as np

# so matrices are printed to 3.dp when testing
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

"""
Markov chains implemented using numpy arrays. Essentially, you look at the 
current state to determine what to do in the future. You multiply where you 
are now by what the possible states are using matrix multiplication. This 
alone is not good enough as you need to factor in long runs, slow time for 
matrix to correct itself and random number generators.

Below is a fairly basic markov implementation with a few factors to tweak 
such as history length, losing streak size, matrix fluidity and when to 
reset if you play the same move too many times.

The strategy is to try and get a lead (e.g. 30) and then play random for 
rest of the game, this strategy has a high success chance but the difficulty 
is getting the necessary lead. Against RNG generators it has a 55% win 
chance, I'm not sure how well it will be against players.

Feel free to tweak/ modify/ change anything or make your own solution. We 
should submit multiple variations to increase our chances of winning.

I've heavily commented everything so it shouldn't be too bad to follow.

Just remember, if you get a 'Shebang' you'll likely win. You'll know what 
that means by running the code.
"""


class RPS:
    """
    Rock, Paper, Scissors class. Analyzes game data and generates an
    appropriate move.
    """
    def __init__(self):
        self.op_moves = []              # op move list
        self.my_moves = []              # my move list
        self.tm = self.create_tm()      # initial transition matrix (TM)
        self.score_count = 0            # adds or subtracts 1, +ve = winning
        self.losing_streak = 0          # games lost in a row
        self.last_outcome = 0           # result of last point => +1, -1, 0
        self.reset = False              # whether matrix was reset

    @staticmethod
    def create_tm():
        """
        Creates initial transition matrix:
            R       P       S
            R->R     R->P     R->S
            P->R     P->P     P->S
            S->R     S->P     S->S
        :return: array
        """
        return np.array([
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3],
        ])

    # updates transition matrix depending on game factors
    def update_tm(self, my_moves, op_moves, decay, inf, reset_tm=-3, reps=3):
        # my_moves and opponent's moves - supplied by tournament file
        self.my_moves = my_moves
        self.op_moves = op_moves

        # dictionary to keep track of moves
        moves_dictionary = {
            "R": 0,
            "P": 0,
            "S": 0
        }

        # updates the moves dictionary with latest opp moves
        for index in range(len(op_moves)-1, -1, -1):
            if index == 0:
                moves_dictionary[op_moves[index]] += 1
                break

            if index / (len(op_moves)-1) >= decay:
                moves_dictionary[op_moves[index]] += 1
            else:
                break

        # updates transition matrix with data from moves_dictionary
        # j==0 is rock row, j==1 is paper row, j==2 is scissors row
        for i in range(3):
            for j in range(3):
                if j == 0:
                    self.tm[i][j] = (inf * self.tm[i][j] + moves_dictionary[
                        "R"] / sum(moves_dictionary.values())) / (inf + 1)
                elif j == 1:
                    self.tm[i][j] = (inf * self.tm[i][j] + moves_dictionary[
                        "P"] / sum(moves_dictionary.values())) / (inf + 1)
                else:
                    self.tm[i][j] = (inf * self.tm[i][j] + moves_dictionary[
                        "S"] / sum(moves_dictionary.values())) / (inf + 1)

        # Adds one to losing streak
        if self.last_outcome == -1 and self.losing_streak <= 0:
            self.losing_streak -= 1
            # resets transition matrix if reset_tm amount triggered
            if self.losing_streak == reset_tm:
                self.losing_streak = 0
                self.reset = True
                self.tm = self.create_tm()
        else:
            # reset losing streak if not lost last turn as looking for streaks
            self.losing_streak = 0

        # reset TM if used same move x times in a row AND lost - note,
        # you must lose as you don't want to reset if winning!
        if len(self.my_moves) >= 3 and self.my_moves[-1*reps-1:-1].count(
                self.my_moves[-1]) >= reps and self.last_outcome == -1:
            self.reset = True
            self.tm = self.create_tm()

    # private method to update current state - these matrices are multiplied
    # by the transition matrix
    def _update_state(self):
        if self.op_moves[-1] == "R":
            return [1, 0, 0] @ self.tm
        elif self.op_moves[-1] == "P":
            return [0, 1, 0] @ self.tm
        else:
            return [0, 0, 1] @ self.tm

    # choose move based on current state
    def choose_move(self, start=20, lead=30):
        # finds index of move likely to be played next - R=0, P=1, S=2
        choice = np.argmax(self._update_state())

        # first 20 moves play random to build a 'history' - also play random
        # when ahead by more than lead or after a reset of the matrix
        if len(self.op_moves) <= start or (self.score_count >= lead and
                                           self.last_outcome < 0) or self.reset:
            # turn off reset
            self.reset = False
            return random.choice(['R', 'P', 'S'])

        # plays winning move from choice e.g. 0 == rock, so P played
        return "P" if choice == 0 else "S" if choice == 1 else "R"

    # keeps track of losing streaks
    def score_counter(self, my_move, op_move):
        result = {"R": {"R": 0, "P": -1, "S": 1},
                  "P": {"R": 1, "P": 0, "S": -1},
                  "S": {"R": -1, "P": 1, "S": 0}}

        self.last_outcome = result[my_move][op_move]

        self.score_count += self.last_outcome

    # reset class - for testing purposes
    def reset_class(self):
        self.op_moves = []              # op move list
        self.my_moves = []              # my move list
        self.tm = self.create_tm()      # initial transition matrix (TM)
        self.score_count = 0            # adds or subtracts 1, +ve = winning
        self.losing_streak = 0          # games lost in a row
        self.last_outcome = 0           # result of last point => +1, -1, 0
        self.reset = False              # whether matrix was reset


# instantiate rps class
rps = RPS()


# play moves based on what rps class decided - lots of variables to tweak
def move(my_moves, op_moves):
    # play random first move
    if len(op_moves) <= 0:
        return random.choice(['R', 'P', 'S'])
    # rps class will make rest of moves
    else:
        # provides info to score counter which tracks streaks
        rps.score_counter(my_moves[-1], op_moves[-1])
        # if you get a shebang then 99/100 you win
        if rps.score_count >= 30:
            print("Shebang")

        """
        tweak all variables here: 
        decay = 0-1: percentage of history when updating the matrix => 0 means look at ALL previous moves, 1 means just last move - 
        It's in reverse because you are going back through the index values
        
        inf = influence, 0-10, how much existing matrix score should play when updating the matrix
        
        decay and inf can work together e.g. inf 1 and decay 0.9 => short history and fluid matrix so score change much more rapidly.
        
        reset_tm = how many losses in a row before resetting matrix
        
        reps = how many repeats before resetting matrix - it will only reset if you lose on a long repeat e.g. repeat = 10, 
        it will not reset if you are drawing or winning, only if you lose so don't worry about missing streaks.
        Remember, when matrix resets, the next outcome is random
        """
        rps.update_tm(my_moves, op_moves, decay=0.8, inf=2, reset_tm=-3,
                      reps=5)
        """
        start = how many plays at start to be random before using algorithm
        lead = switch to random when lead is this value - from my 
        experience, 30 has very high chance of winning.
        """
        return rps.choose_move(start=20, lead=30)


def name():
    return "Rock"
