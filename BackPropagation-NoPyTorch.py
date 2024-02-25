import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Ex 1. Shuffle the data + 80% Training Set + 20% Testing Set
def load_data(file_path=r'./seeds_dataset.txt'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    cr = csv.reader(lines, delimiter='\t')

    # Skip the header if it exists
    next(cr, None)

    data = list(cr)

    np.random.shuffle(data)
    return data


def split_data(data, train_ratio=0.8):
    split_index = int(train_ratio * len(data))
    training_set = data[:split_index]
    training_set = [[float(element) if element else float('nan') for element in row] for row in training_set]
    training_set = [list(filter(lambda x: not np.isnan(x), row)) for row in training_set]

    testing_set = data[split_index:]
    testing_set = [[float(element) if element else float('nan') for element in row] for row in testing_set]
    testing_set = [list(filter(lambda x: not np.isnan(x), row)) for row in testing_set]

    print("Training set: ", training_set)
    print("Testing set: ", testing_set)
    return training_set, testing_set


# Ex 2. Initialize parameters
def initialize_biases(hidden_neurons_per_layers, output_size):
    biases = []

    # Hidden layers biases
    hidden_biases = [np.random.uniform(low=-1, high=1, size=(1, neurons)) for neurons in hidden_neurons_per_layers]
    biases.extend(hidden_biases)

    # Output layer bias
    output_biases = np.random.uniform(low=-1, high=1, size=(1, output_size))
    biases.append(output_biases)
    print("Biases\n", biases)
    return biases


def initialize_weights(input_size, hidden_neurons_per_layer, output_size):
    np.random.seed(42)
    weights = []

    weights.append(np.random.uniform(low=-1, high=1, size=(hidden_neurons_per_layer[0], input_size)))

    for i in range(1, len(hidden_neurons_per_layer)):
        weights.append(
            np.random.uniform(low=-1, high=1, size=(hidden_neurons_per_layer[i], hidden_neurons_per_layer[i - 1])))

    weights.append(np.random.uniform(low=-1, high=1, size=(output_size, hidden_neurons_per_layer[-1])))
    # print("Biases\n", biases)
    print("Weights\n", weights)
    return weights


# Ex3 Functia de activare + derivata, functia de eroare
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mean_squared_error(y_pred, y_true):
    return (1 / 2) * np.mean((y_pred - y_true) ** 2)


# Ex4 forward propagation
def compute_output(input, weight, bias):
    matrix = np.matmul(weight, input) + bias # matrix => z
    result_matrix = np.vectorize(sigmoid)(matrix)
    return result_matrix, matrix


def evaluate(input, layers, biases):
    activations = [input]
    pre_activations = []
    for i in range(len(layers)):
        bias = get_bias_matrix(biases[i])
        output, pre_activation = compute_output(input, layers[i], bias)
        input = output

        activations.append(output)
        pre_activations.append(pre_activation)
        # print("ACTIVATIONS: ", activations)
    return output, activations, pre_activations


def get_input_matrix(row):
    input_data = row[:-1]
    input_matrix = np.array(input_data).reshape(-1, 1)
    return input_matrix


def get_bias_matrix(row):
    input_matrix = np.array(row).reshape(-1, 1)
    return input_matrix

def one_hot_encode(target, num_classes):
    encoded = np.zeros((num_classes, 1))
    encoded[int(target) - 1] = 1  # Assuming targets start from 1
    return encoded.astype(int)  # Convert to integer type


def evaluate_and_train(training_set, weights, biases, alpha, max_epochs):

    for epoch in range(max_epochs):
        for entry in training_set:

            input_matrix = get_input_matrix(entry)

            # Feed forward
            output, activations, pre_activations = evaluate(input_matrix, weights, biases)
            target_output = one_hot_encode(entry[-1], output_size)

            # delta[l][i] => delta pt neuronul 'i' de pe layer-ul 'l'
            delta_b = [np.zeros_like(b) for b in biases]
            delta_w = [np.zeros_like(w) for w in weights]

            # Calculate delta for the output layer
            delta_output = activations[-1] - target_output
            delta_w[-1] = np.dot(delta_output, activations[-2].T)
            delta_b[-1] = delta_output

            for layer_index in range(len(weights) - 2, -1, -1):
                delta_hidden = np.dot(weights[layer_index + 1].T, delta_output) * sigmoid_derivative(
                    pre_activations[layer_index])
                delta_w[layer_index] = np.dot(delta_hidden, activations[layer_index].T)
                delta_b[layer_index] = delta_hidden  # Ensure that you are updating the correct element here

                delta_output = delta_hidden


            # Update weights and biases
            for i in range(len(weights)):
                # print("Delta_bias: ", delta_b[i].shape)
                weights[i] = weights[i] - alpha * delta_w[i]
                biases[i] = biases[i] - alpha * delta_b[i].T

    return weights, biases, output




if __name__ == "__main__":
    data = load_data()
    training_set, testing_set = split_data(data, train_ratio=0.8)
    input_size = len(training_set[0]) - 1
    output_size = len(set(row[-1] for row in data))
    alpha = 0.01
    max_epochs = 2500
    hidden_neurons_per_layers = [5, 5]  # 2 hidden layers, 5 per each layer
    biases = initialize_biases(hidden_neurons_per_layers, output_size)
    weights = initialize_weights(input_size, hidden_neurons_per_layers, output_size)

    for w in weights:
        print("Weights.shape: ", w.shape)

    for b in biases:
        print("Bias.shape: ", b.shape)


    trained_weights, trained_biases, output = evaluate_and_train(training_set, weights.copy(), biases.copy(), alpha, max_epochs)


    # Predict on the test set
    predictions = []
    true_labels = []

    for entry in testing_set:
        input_matrix = get_input_matrix(entry)
        output, _, _ = evaluate(input_matrix, trained_weights, trained_biases)

        # Convert one-hot encoded output to class label
        predicted_label = np.argmax(output) + 1  # Assuming classes start from 1
        true_label = int(entry[-1])

        predictions.append(predicted_label)
        true_labels.append(true_label)

    # Print performance metrics
    accuracy = accuracy_score(true_labels, predictions)


    print("Accuracy: {:.2f}%".format(accuracy * 100))
