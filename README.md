# Multilayer-Perceptron


## Description

In this repo, a rudimentary MLP is built to do the following classification task:
- Training data (X, Y): Training data contains N1 = 100,000 points in 2-dimentional space and
are followed by the uniform radius between 0 and 5 and its label is 1 is it is inside circle of
radius 5, otherwise it is 0 (see below).
- Validation data (X, Y):
Validation data contains N2 = 20,000 points in 2-dimentional space and are followed by the
uniform radius between 0 and 5 and its label is 1 is it is inside the circle, otherwise it is 0
- Testing data (X, Y):Testing data contains N2 = 20,000 points in 2-dimentional space and are followed by the
uniform radius between 0 and 5 and its label is 1 is it is inside the circle, otherwise it is 0

## Getting Started

### Dependencies
Install conda and run the following block:
```
pip install -r requirements.txt
```

### Running the code
* One layer MLP
```
python train_mlp.py --num_units 3 --loss 'L2' --optimizer 'sgd' --epochs 10
python train_mlp.py --num_units 3 --loss 'L2' --optimizer 'sgd' --epochs 500 --batch_size 500
python train_mlp.py --num_units 128 --loss 'CE' --optimizer 'adam' --epochs 100
```

* Multilayer MLP
```
python train_multiple_mlp.py --loss 'CE' --optimizer 'adam' --epochs 100
python train_multiple_mlp.py --loss 'CE' --optimizer 'adam' –drop_out 0.2 --epochs 100
```

#### List of Arguments accepted
```--lr``` Learning rate (Default = 3) <br>
```--epochs``` Num of epochs <br>
```--optimizer``` Optimizer <br>
```--loss``` Loss function <br>
```--batch_size``` Batch size <br>
```--num_workers``` Number of workers <br>


# Experimental Results
The results and comments can be found at [Wandb](https://wandb.ai/mediaeval-sport/BB_Homework_DL/reports/Homework-1-Exercise-6--VmlldzoxNDc0NTY3?accessToken=arnuynijwczemfk4m0nlk3wzanry65tvp5muvmi6vi7llk0v5qvf6gbin36kr5hf)
