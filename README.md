# Overview

This repository contains 3 Neural Networks projects.

# Project 1 - Digit recognition - No ML specific libraries
My implementation (Perceptron.py) trains 10 perceptrons to identify the digit in an image. Essentially, we have one perceptron for each digit, which will learn to differentiate between that particular digit and the rest. By assembling these, we can deduce which digit is in a specific image.
A mini-batch was used. 

I use the MNIST dataset, which contains tens of thousands of examples of such images (the input), along with the digit (the desired output of the assembly). The obtained accuracy for validation, test and training set was over 85%.

<img width="871" alt="image" src="https://github.com/alupu0607/NN/assets/100222484/ae22cd1f-1421-47cf-9299-4948e9c115e1">

# Project 2 - Digit recognition - Backpropagation using Pytorch
My project (Backpropagation-PyTorch.py) involves training a Multi-Layer Perceptron (MLP) neural network with at least one hidden layer to identify the digit in an image using PyTorch. It utilizes the MNIST dataset, which comprises tens of thousands of examples of such images (the input), along with the digit (the desired output).
In my project, I will calculate both the overall accuracy and the F1 score for each class on both the training set and the evaluation set. Additionally, the project incorporates several key features:

- **Use of ReLU and softmax activations**: These activation functions are crucial for the neural network's ability to learn non-linear patterns effectively and for classifying the outputs in a probabilistic manner, respectively.

- **Use of cross-entropy as the loss function**: This choice is appropriate for classification problems, as it measures the performance of the model whose output is a probability value between 0 and 1.

- **Implementation of mini-batch processing**: This approach helps in speeding up the training process, making it more computationally efficient without sacrificing the model's ability to learn from the data.

Furthermore, achieving an **accuracy of over 95% on the evaluation set** is considered sufficient, and that's exactly what I aim to accomplish.

# Project 3 - FlappyBird Q-Learning DQN
In my project, I implement and train a neural network using the Q-learning algorithm to control an agent in the Flappy Bird game. I used pygame for the environment (flappy_bird.py)
Aditionally, I utilize Convolutional Neural Networks (CNNs) alongside the Q-learning algorithm to control an agent in the Flappy Bird game. This approach leverages the spatial hierarchy of images, allowing the network to efficiently recognize and act upon patterns within the game environment.
References: https://github.com/yenchenlin/DeepLearningFlappyBird

