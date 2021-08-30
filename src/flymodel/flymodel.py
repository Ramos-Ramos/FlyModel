import numpy as np
import cupy as cp
from einops import reduce
# from sklearn.preprocessing import minmax_scale

from typing import Union, Optional, Dict


try:
  cp_array_class = cp.core.core.ndarray
except:
  cp_array_class = cp._core.core.ndarray
Array = Union[np.ndarray, cp_array_class]


class FlyModel():

  def __init__(self, input_size: int, hidden_size: int, output_size: int,
               top_activations: int, lr: float, decay: float,
               input_density: Optional[float] = None,
               input_connections: Optional[float] = None):
    """Fruit fly model as described in `Algorithmic insights on continual 
    learning from fruit flies`
    (arXiv:2107.07617)

    Args:
      input_size: number of input features
      hidden_size: number of hidden features
      output_size: number of classes
      top_activations: number of top hidden activations to keep
      lr: learning rate
      decay: forgetting term
      input_density: percentage of inputs to connect to each hidden neuron.
                     Leave as `None` if `input_connections` is not `None`.
      input_connections: number of inputs to connect to each hidden neuron.
                         Leave as `None` if `input_density` is not `None`.

    """

    assert input_density or input_connections, "`input_density` or `input_connections` must not be `None`"
    assert not (input_density and input_connections), "`input_density` or `input_connections` cannot both be not `None`"
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.top_activations = top_activations
    self.lr = lr
    self.decay = decay
    self.input_connections = input_connections if input_connections else input_size * input_density

    self.weights0 = (np.random.rand(hidden_size, input_size).argsort(axis=1) < self.input_connections).astype(np.int)
    self.weights1 = np.random.rand(output_size, hidden_size)

    self.training = True
    self.xp = np

  def __call__(self, x: Array, labels: Array = None) -> None:
    """Computes class scores. If `self.training` is `True` and `labels` is not
    `None`, updates class projection weights.
    
    Args:
      x: input of shape batch x input_features
      labels: class labels as an array of integers of length batch
    """
    b, _ = x.shape
    
    hidden = x.dot(self.weights0.T)
    # self.xp.put_along_axis(hidden, hidden.argsort()[:, :-self.top_activations], 0, axis=1)
    hidden[self.xp.indices((b, self.hidden_size - self.top_activations))[0], hidden.argsort()[:, :-self.top_activations]] = 0
    # hidden = minmax_scale(hidden, axis=1) # no gpu support for sklearn
    hidden = (hidden - hidden.min(axis=1, keepdims=True))/(hidden.max(axis=1, keepdims=True) - hidden.min(axis=1, keepdims=True))

    out = hidden.dot(self.weights1.T)

    if self.training:
      self._backward(hidden, labels)

    return out

  def _backward(self, hidden: Array, labels: Array) -> None:
    """Updates class projection weights.
    
    Args:
      hidden: hidden activations
      labels: class labels as an array of integers of length batch
    """
    
    assert self.training, "Cannot update weights in eval mode"
    assert labels is not None, "Cannot updates weights without labels"
    self.weights1 *= (1 - self.decay)
    for label in labels:
      self.weights1[label] += reduce(self.lr * hidden, 'b m -> m', 'sum')
    self.weights1 = self.xp.clip(self.weights1, 0, 1)

  def state_dict(self) -> Dict[str, Array]:
    """Returns dictionary of weights"""

    return {'weights0': cp.asnumpy(self.weights0).copy(), 'weights1': cp.asnumpy(self.weights1).copy()}

  def load_state_dict(self, state_dict: Dict[str, Array]) -> None:
    """Loads weights
    
    Args:
      state_dict: dictionary of weights
    """
    for i in range(2):
      curr_shape, new_shape = getattr(self, f'weights{i}').shape, state_dict[f'weights{i}'].shape
      assert curr_shape == new_shape, f"Incorrect size for `weights{i}`. Expected {curr_shape}, got {new_shape}."
    self.weights0 = state_dict['weights0']
    self.weights1 = state_dict['weights1']
    self.to('cpu' if self.xp==np else 'gpu')

  def eval(self) -> None:
    """Turns off training mode"""

    self.training = False

  def train(self) -> None:
    """Turns on training mode"""

    self.training = True

  def to(self, device: str) -> None:
    """Moves weight array to device
    
    Args:
      device: device to move weights to; must be "cpu" or "gpu"
    """
    
    if device == 'cpu':
      self.weights0 = cp.asnumpy(self.weights0)
      self.weights1 = cp.asnumpy(self.weights1)
    elif device == 'gpu':
      self.weights0 = cp.asarray(self.weights0)
      self.weights1 = cp.asarray(self.weights1)
    else:
      raise ValueError("`device` must be either 'cpu' or 'gpu'")
    self.xp = cp.get_array_module(self.weights0)
