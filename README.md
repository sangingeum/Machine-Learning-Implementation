# Machine Learning Implementation

## Description

This is a library of machine learning implementations for quick reference.

## Implementations
### - Tabular Classification -
#### Binary Classification
* MLP
* Bagging and Voting
#### Multi-class Classification
* MLP 
* Bagging and Voting
#### Multi-label Classification
* MLP
### - Tabular Regression -
* MLP
### - Image Classification -
#### Multi-class Classification
* CNN
### - Image Generation -
#### MNIST
* WGAN
### - Text Classification -
#### Binary Classification 
* Pretrained model + RNN
### - Graph Classification -
#### Node Classification 
* GCN
### - Reinforcement Learning -
#### FrozenLake
* Value Iteration
* Policy Iteration
* Monte Carlo
#### CartPole
* DQN (Double DQN, Dueling DQN, PER, n-step reward)
#### LunarLander
* DQN (Double DQN, Dueling DQN, PER, n-step reward)
    
## Installation
To install the necessary packages, run the following command in your terminal:

    pip3 install -r requirements.txt
We recommend installing the CUDA-enabled version of PyTorch, which can be found [here](https://pytorch.org/get-started/locally/)
## Usage
To use a specific implementation, navigate to the main folder and run the corresponding [filename]_main.py file like this:
    
    python3 [filename]_main.py
## Example
To run the GCN example on the Cora dataset, run the following command:

    python3 graph_convolutional_network_cora_main.py
