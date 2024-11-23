from collections import defaultdict
from abc import ABC, abstractmethod
import math

class MCTSNode(ABC):
  @abstractmethod
  def find_children(self) -> set['MCTSNode']:
    "Return a set of all possible successors of this node"
    pass
  
  @abstractmethod
  def find_random_child(self) -> 'MCTSNode':
    "Return a random successor of this node"
    pass

  @abstractmethod
  def is_terminal(self) -> bool:
    "Returns True if the node has no children"
    pass

  @abstractmethod
  def reward(self) -> float:
    "Return the reward for this node. Assumes that the node is terminal"
    pass

  @abstractmethod
  def __hash__(self):
    "Nodes must be hashable"
    pass

  @abstractmethod
  def __eq__(self, other):
    "Nodes must be comparable"
    pass


class MCTS:
  def __init__(self):
    # total reward of each node
    self.Q: dict[MCTSNode, float] = defaultdict(float)
    # total visit count for each node
    self.N: dict[MCTSNode, int] = defaultdict(int)
    # children of each node
    self.children: dict[MCTSNode, set[MCTSNode]] = dict()
  
  def choose(self, node: MCTSNode) -> MCTSNode:
    "Choose the best successor of the node. (Choose a move in the game)"
    if node.is_terminal():
      raise RuntimeError(f"choose called on terminal node {node}")
    
    if node not in self.children: # if the node has not been explored
      return node.find_random_child() # choose a random child
    
    def score(n: MCTSNode) -> float:
      "Return the average reward of the node"
      if self.N[n] == 0:
        return float("-inf") # avoid unseen moves
      
      return self.Q[n] / self.N[n] # average reward
     
    return max(self.children[node], key=score) # choose the child with the highest score
  
  def rollout(self, node: MCTSNode) -> None:
    path = self._select(node) # select a path to an unexplored leaf node
    leaf = path[-1]
    self._expand(leaf) # expand the leaf node
    reward = self._simulate(leaf) # simulate a random game from the leaf node
    self._backpropagate(path, reward) # backpropagate the reward to the ancestors of the leaf node

  def _select(self, node: MCTSNode) -> list[MCTSNode]:
    "Find an unexplored descendent of the node"
    path = []
    while True:
      path.append(node)

      if node not in self.children or not self.children[node]:
        # node is either unexplored or terminal
        return path
      
      unexplored = self.children[node] - self.children.keys()
      if unexplored:
        unexplored_node = unexplored.pop()
        path.append(unexplored_node)
        return path
      
      node = self._uct_select(node) # descend a layer deeper
  
  def _expand(self, node: MCTSNode) -> None:
    "Update the children dict with the children of the `node`"
    if node in self.children:
      return # already expanded
    self.children[node] = node.find_children()
  
  def _simulate(self, node: MCTSNode) -> float:
    """Returns the reward for a random simulation (to completion) of the `node`
    The reward is 1 if the node is a win, 0 if it is a loss
    """
    invert_reward = True
    while not node.is_terminal():
      node = node.find_random_child()
      invert_reward = not invert_reward
    
    reward = node.reward()
    return 1 - reward if invert_reward else reward
  
  def _backpropagate(self, path: list[MCTSNode], reward: float) -> None:
    "Send the reward back up to the ancestors of the leaf"
    for node in reversed(path):
      self.N[node] += 1
      self.Q[node] += reward
      reward = 1 - reward # 1 if the node is a win, 0 if it is a loss

  def _all_expanded(self, node: MCTSNode) -> bool:
    "Return True if all children of the node are already expanded"
    return all(n in self.children for n in self.children[node])
    
  def _uct_select(self, node: MCTSNode) -> MCTSNode:
    "Select a child of node, balancing exploration & exploitation"
    # All children of node should already be expanded
    assert(self._all_expanded(node))

    log_N_parent = math.log(self.N[node])

    def uct(n: MCTSNode) -> float:
      "Upper confidence bound for trees"
      return self.Q[n] / self.N[n] + math.sqrt(log_N_parent / self.N[n])

    # choose the child with the highest UCT value
    return max(self.children[node], key=uct)
  
