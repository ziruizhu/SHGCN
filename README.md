# Inhomogeneous Social Recommendation with Hypergraph Convolutional Networks

This is our Pytorch implementation for the paper:

>Inhomogeneous Social Recommendation with Hypergraph Convolutional Networks.

Author: Anonymous authors.

## Introduction
Social HyperGraph Convolutionl Network (SHGCN) is a new recommendation framework based on hypergraph convolution, effectively utilizing the triple social relations.


## Environment Requirement
The code has been tested under Python 3.6.10. The required packages are as follows:
* Pytorch == 1.6.0
* numpy == 1.19.1
* scipy == 1.5.2
* pandas == 1.1.1 

## Example to Run the Codes
* Beidian dataset
```
python main.py --dataset Beidian --model SHGCN --gpu 0 --emb_dim 32 --num_layer 2 --epoch 500 --batch_size 4096
```

* Beibei dataset
```
python main.py --dataset Beibei --model SHGCN --gpu 0 --emb_dim 32 --num_layer 2 --epoch 500 --batch_size 4096
```

## Dataset
There are two datasets released here. Each contains four files.
* `data.train`
  * Training set.
  * Each line contains a purchase log, which can be represented as:
    * (user ID, item ID)

* `data.val`
  * Validation set.
  * Each line contains a purchase log, which can be represented as:
    * (user ID, item ID)
  
* `data.test`
  * Testing set.
  * Each line contains a purchase log, which can be represented as:
    * (user ID, item ID)
  
* `social.share`
  * Social interactions logs.
  * Each line contains a triplet. It can be represented uniformly as
    * (user1 ID, user2 ID, item ID)
  * In the Beidian dataset, each triplet represents a social sharing behavior.
  * In the Beibei dataset, each triplet represents a group buying behavior.

