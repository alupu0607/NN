import gzip
import pickle
import requests
import torch
import torch.nn as nn
import numpy as np


# IMPORT DATA
url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)

with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, validation_set, test_set = pickle.load(fd, encoding="latin")
# Shuffle the data

class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims,  output_dim):
        super(MulticlassLogisticRegression, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

        # Add hidden layers with ReLU activation
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())

        # Add the output layer with softmax activation
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.LogSoftmax(dim=1))
        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = self.model(x)
        return x

def shuffle_data(train_set):
    train_x, train_y = train_set
    num_samples = len(train_x)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    shuffled_x = train_x[indices]
    shuffled_y = train_y[indices]
    return shuffled_x, shuffled_y

def calculate_class_metrics(model, dataset):
    model.eval()
    inputs, labels = dataset
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    class_correct = [0] * 10
    class_total = [0] * 10
    true_positive = [0] * 10
    false_positive = [0] * 10
    false_negative = [0] * 10

    for i in range(len(predicted)):
        label = labels[i]
        class_correct[label] += (predicted[i] == label).item()
        class_total[label] += 1

        if predicted[i] == label:
            true_positive[label] += 1
        else:
            false_positive[predicted[i]] += 1
            false_negative[label] += 1

    class_accuracy = [correct / total if total > 0 else 0.0 for correct, total in zip(class_correct, class_total)]
    precision = [tp / (tp + fp) if (tp + fp) > 0 else 0.0 for tp, fp in zip(true_positive, false_positive)]
    recall = [tp / (tp + fn) if (tp + fn) > 0 else 0.0 for tp, fn in zip(true_positive, false_negative)]
    f1_score = [2 * (p * r) / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(precision, recall)]

    return class_accuracy, f1_score
def train_model(model, criterion, optimizer, train_set, batch_size=500, max_epochs=150):
    train_x, train_y = shuffle_data(train_set)
    train_set = list(zip(train_x, train_y))
    batches = [train_set[i:i + batch_size] for i in range(0, len(train_set), batch_size)]

    for epoch in range(max_epochs):
        running_loss = 0.0
        for batch in batches:
            inputs, labels = zip(*batch)
            inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
            labels = torch.tensor(np.array(labels), dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(batches)))

    print('Finished Training')

# construct model instance
simplenn = MulticlassLogisticRegression(
    input_dim=28 * 28,
    hidden_dims=[128, 64],
    output_dim=10
)

# loss function
criterion = nn.CrossEntropyLoss()

# set up optimizer
optimizer = torch.optim.SGD(simplenn.parameters(), lr=0.015)
train_model(simplenn, criterion, optimizer, train_set)


# TRAIN SET METRICS
training_class_accuracy, training_f1_score = calculate_class_metrics(simplenn, train_set)
print("Class Metrics - Training Set:")
for digit in range(10):
    print(f'Digit {digit}: Accuracy={training_class_accuracy[digit] * 100:.2f}%, ' +
          f' F1 Score={training_f1_score[digit]:.2f}')

# EVAL SET METRICS
eval_class_accuracy , eval_f1_score= calculate_class_metrics(simplenn, test_set)
print("Class Metrics - Evaluation Set:")
for digit in range(10):
    print(f'Digit {digit}: Accuracy={eval_class_accuracy[digit] * 100:.2f}%, ' +
          f'F1 Score={eval_f1_score[digit]:.2f}')
average_accuracy = sum(eval_class_accuracy) / len(eval_class_accuracy)

print(f'Average Accuracy: {average_accuracy:.2f}')