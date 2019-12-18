# sam_cnn

Script for training hexagonal convolutional neural networks on 6x6 SAM patch patterns

## Getting started

The easiest way to get all of the dependencies is to use the Anaconda Python Distribution (w/ Python 3.6.x)

You'll also need [PyTorch](https://github.com/pytorch/pytorch) and [HexagDLy](https://github.com/ai4iacts/hexagdly). HexagDLy provides support for convolutional and pooling operations for inputs arranged on hexagonal grids.

### Installing with conda and pip

Make sure you're using the version of Pip installed with the Anaconda distribution

PyTorch:

with conda:
```
conda install pytorch torchvision -c pytorch
```

with pip:
```
pip3 install torch torchvision
```

HexagDLy:

```
pip install hexagdly
```


### Quick training 

All of the training data (i.e, patch patterns and their free energies) are contained in *sam_pattern_data.dat.npz*. 

You can train a model using *model_fnot.py*

For a list of training options, run:
```
python model_fnot.py -h
```

For instance, to train a CNN with 6 convolutional filters, 1 hidden layer, and 12 hidden nodes:

```
python model_fnot.py --augment-data --n-layers 1 --n-hidden 12 --do-conv --n-out-channels 6
```

## Authors

Nick Rego ([nrego@pennmedicine.upenn.edu])
