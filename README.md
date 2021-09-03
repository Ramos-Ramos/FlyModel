# FlyModel 🍄

Unofficial Python implementation of [Algorithmic insights on continual learning from fruit flies](https://arxiv.org/abs/2107.07617) with a PyTorch flavored API.

## Installation

```
pip install git+https://github.com/Ramos-Ramos/flymodel
```

## Usage

```python
import numpy as xp
from flymodel import FlyModel

model = FlyModel(
  input_size=84,        # input dimension size (no. of projection neurons)
  hidden_size=3200,     # hidden dimension size (no. of Kenyon cells)
  output_size=20,       # output dimension size (no. of mushroom body output neurons)
  top_activations=160,   # no. of top cells to be left active in hidden layer
  lr=1e-2,              # learning rate (learning is performed internally)
  decay=0,              # forgetting term
  input_connections=10  # number of inputs to connect to for each hidden neuron; alternativey, `input_density` can be specified
)
x = xp.random.randn(2, 84)
output = model(x)
```

Learning is performed internally as long as the model is in train mode and labels are provided. No need to call `.backward()` or instantiate optimizers. To set the mode, use `.train()` and `.eval()`.

```python
x, labels = xp.random.randn(2, 84), xp.random.randint(0, 20, (2,))
model.train() # will update weights on forward pass
output = model(x, labels)
model.eval()  # will not update weights on forward pass
```

To enable gpu learning, move the model to the gpu via `.to` and use cupy instead of numpy.

```python
import cupy as xp

model = FlyModel(
  input_size=84,
  hidden_size=3200,
  output_size=20,
  top_activations=16,
  lr=1e-4,
  decay=0,
  input_connections=10
)
model.to('gpu')
```

## Citation
```bibtex
@misc{shen2021algorithmic,
      title={Algorithmic insights on continual learning from fruit flies}, 
      author={Yang Shen and Sanjoy Dasgupta and Saket Navlakha},
      year={2021},
      eprint={2107.07617},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
