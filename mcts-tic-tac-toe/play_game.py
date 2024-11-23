from mcts import MCTS
from tictactoe import Piece, TicTacToeBoard, row_col_to_index

def play_game(num_rollouts: int = 200):
  tree = MCTS()
  board = TicTacToeBoard.new()

  print(board)

  while not board.is_terminal():
    row_col = input("enter row,col: ")
    row, col = map(int, row_col.split(","))
    index = row_col_to_index(row - 1, col - 1) 

    if board.tup[index] is not Piece.EMPTY:
      print("Invalid move, spot already taken")
      continue
    
    board = board.make_move(index)
    print(board)

    if board.is_terminal():
      break
    
    # You can train as you go, or only at the beginning.
    # Here, we train as we go, doing `num_rollouts` rollouts each turn.
    for _ in range(num_rollouts):
      tree.rollout(board)
    
    board = tree.choose(board)
    print(board)
  
  print("Game over")
  if board.winner is Piece.EMPTY:
    print("Tie")
  else:
    print(f"{board.winner} wins!")

if __name__ == "__main__":
  play_game()
