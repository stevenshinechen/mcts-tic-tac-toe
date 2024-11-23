"""
An example implementation of the abstract Node class for use in MCTS

A tic-tac-toe self is represented as a tuple of 9 values, each either
Piece.EMPTY, Piece.X, or Piece.O

The self is indexed by row:
0 1 2
3 4 5
6 7 8

For example, this game self
O - X
O X -
X - -
corrresponds to this tuple:
(Piece.O, Piece.EMPTY, Piece.X, Piece.O, Piece.X, Piece.EMPTY, Piece.X, Piece.EMPTY, Piece.EMPTY)
"""

from enum import Enum
from random import choice
from typing import NamedTuple, Optional
from mcts import MCTSNode

LOSS_REWARD = 0
TIE_REWARD = 0.5
WIN_REWARD = 1

BOARD_SIZE = 3

class Piece(Enum):
  EMPTY = 0
  X = 1
  O = 2

  def __str__(self):
    if self is Piece.EMPTY:
      return "_"
    return self.name
  
  def opponent(self):
    if self is Piece.EMPTY:
      return Piece.EMPTY
    return Piece.X if self is Piece.O else Piece.O

class _TicTacToeBoard(NamedTuple):
  tup: tuple[Piece, Piece, Piece, Piece, Piece, Piece, Piece, Piece, Piece]
  turn: Piece
  winner: Piece
  terminal: bool

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(_TicTacToeBoard, MCTSNode):
  "A MCTS Node for a Tic Tac Toe Board"
  @classmethod
  def new(cls) -> "TicTacToeBoard":
    return cls(tup=(Piece.EMPTY,) * BOARD_SIZE * BOARD_SIZE, turn=Piece.X, winner=Piece.EMPTY, terminal=False)

  def find_children(self) -> set["TicTacToeBoard"]:
    if self.terminal:
      # If the game is finished then no moves can be made
      return set()
    # Otherwise, you can make a move in each of the empty spots
    return {
      self.make_move(i) for i, value in enumerate(self.tup) if value is Piece.EMPTY
    }
  
  def find_random_child(self) -> Optional["TicTacToeBoard"]:
    if self.terminal:
      # If the game is finished then no moves can be made
      return None
    
    empty_spots = [i for i, value in enumerate(self.tup) if value is Piece.EMPTY]
    return self.make_move(choice(empty_spots))
  
  def reward(self) -> float:
    if not self.terminal:
      raise RuntimeError(f"reward called on nonterminal board {self}")
    
    if self.winner is self.turn:
      # It's your turn and you've already won. Should be impossible.
      raise RuntimeError(f"reward called on unreachable board {self}")
    
    if self.turn is self.winner.opponent():
      # Your opponent has just won. Bad.
      return LOSS_REWARD
    
    if self.winner is Piece.EMPTY:
      # self is a tie
      return TIE_REWARD
    
    # The winner is neither Piece.X, Piece.O, nor Piece.EMPTY
    raise RuntimeError(f"self has unknown winner type {self.winner}, {self.turn=}")
  
  def is_terminal(self) -> bool:
    return self.terminal
  
  def make_move(self, index: int) -> "TicTacToeBoard":
    tup = self.tup[:index] + (self.turn,) + self.tup[index + 1:]
    turn = self.turn.opponent()
    winner = _find_winner(tup)
    is_terminal = (winner is not Piece.EMPTY) or not any(v is Piece.EMPTY for v in tup)
    return TicTacToeBoard(tup, turn, winner, is_terminal)

  def __str__(self):
    rows = [[self.tup[row_col_to_index(row, col)] for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)]
    indices = [str(i) for i in range(1, BOARD_SIZE + 1)]
    return (
      f"\n  {" ".join(indices)}\n"
      + "\n".join(str(i + 1) + " " + " ".join(str(cell) for cell in row) for i, row in enumerate(rows))
      + "\n"
    )
  
def row_col_to_index(row: int, col: int) -> int:
  return row * BOARD_SIZE + col
    
def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(tup) -> Piece:
    "Returns Piece.EMPTY if no winner, Piece.X if X wins, Piece.O if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if Piece.O is v1 is v2 is v3:
            return Piece.O
        if Piece.X is v1 is v2 is v3:
            return Piece.X
    return Piece.EMPTY
