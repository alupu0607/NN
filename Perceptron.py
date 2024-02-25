import gzip
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests

# IMPORT DATA
url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)

with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, validation_set, test_set = pickle.load(fd, encoding="latin")

# Initialize 10 perceptrons (784 features)
def initialize_perceptrons():
    perceptrons = [np.random.rand(train_set[0].shape[1]) for _ in range(10)]
    return perceptrons

# Shuffle the data
def shuffle_data(train_set):
    train_x, train_y = train_set
    num_samples = len(train_x)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    shuffled_x = train_x[indices]
    shuffled_y = train_y[indices]
    return shuffled_x, shuffled_y


# Train the perceptrons using the training set
def positive(value):
    return value>0

def train(trainset, w, beta, alpha=0.0075):
    delta = np.zeros_like(w)
    B = 0

    for sample in trainset:
        input_data, label = sample[0], sample[1]
        dot_product = np.dot(input_data, w) + beta
        classified = positive(label * dot_product)
        if classified:
            continue
        delta += alpha * label * input_data
        B += alpha * label

    return (delta, B)

def mini_batch_training(train_set, num_perceptrons=10, max_epochs=50, batch_size=5000, alpha=0.0075):
    train_x, train_y = shuffle_data(train_set)
    w = initialize_perceptrons()
    beta = np.random.rand()

    training_set = list(zip(train_x, train_y))
    batches = [training_set[i:i + batch_size] for i in range(0, len(training_set), batch_size)]

    for epoch in range(max_epochs):
        for batch in batches:
            for sample in batch:
                input_data, label = sample[0], sample[1]
                for perceptron_id in range(num_perceptrons):
                    if perceptron_id == label:
                        desired_label = 1
                    else:
                        desired_label = -1
                    delta, B = train([(input_data, desired_label)], w[perceptron_id], beta, alpha)
                    w[perceptron_id] += delta
                    beta += B

    return w, beta


def calculate_accuracy(data_x, data_y, w, beta):
    correct = 0
    total = len(data_x)
    actual_labels = []
    predicted_labels = []

    for i in range(total):
        input_data, label = data_x[i], data_y[i]
        max_score = -float("inf")
        predicted_label = -1

        for digit, weights in enumerate(w):
            score = np.dot(input_data, weights) + beta
            if score > max_score:
                max_score = score
                predicted_label = digit

        actual_labels.append(label)
        predicted_labels.append(predicted_label)

        if predicted_label == label:
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy,actual_labels[:50], predicted_labels[:50]


if __name__ == "__main__":
    # Train the perceptrons using mini-batch
    w, beta = mini_batch_training(train_set)

    # Calculate accuracy on the validation set
    validation_accuracy, actual_labels, predicted_label = calculate_accuracy(validation_set[0], validation_set[1], w, beta)
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")
    print(actual_labels)
    print(predicted_label)

    # Calculate accuracy on the evaluation set
    evaluation_accuracy, _, _ = calculate_accuracy(test_set[0], test_set[1], w, beta)
    print(f"Evaluation Accuracy: {evaluation_accuracy:.2f}%")

    training_accuracy, _, _ = calculate_accuracy(train_set[0], train_set[1], w, beta)
    print(f"Training Accuracy: {training_accuracy:.2f}%")
