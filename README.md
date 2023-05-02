# Machine Learning Library

## Description

This is a library of machine learning implementations for quick reference.

## Features
* ### Tabular Classification
    * #### Binary Classification
    * #### Multi-class Classification
    * #### Multi-label Classification
* ### Tabular Regression
* ### Image Classification
    * #### Multi-class Classification
* ### Text Classification
    * #### Binary Classification
      * ##### Sentiment Analysis
* ### Reinforcement Learning
    * #### FrozenLake
      * ##### Value Iteration
      * ##### Policy Iteration
      * ##### Monte Carlo
    * #### CartPole
      * ##### DQN (Double DQN, Dueling DQN, PER, n-step reward)
    * #### LunarLander
      * ##### DQN (Double DQN, Dueling DQN, PER, n-step reward)
    
## Installation
To install the necessary packages, run the following command in your terminal:

    pip3 install -r requirements.txt
We recommend installing the CUDA-enabled version of PyTorch, which can be found [here](https://pytorch.org/get-started/locally/)
## Usage
To use a specific implementation, run the corresponding filename_main.py file in your terminal using Python3, like this:
    
    python3 [filename]_main.py
## Example
To run the neural network regression example, run the following command:

    python3 neural_network_regression_main.py
