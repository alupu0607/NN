import os
import random
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState
from neural_network import NeuralNetwork


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def get_initial_state(game_state, model):
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1  # "Do nothing" action
    image_data, _, _ = game_state.frame_step(action)

    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    return state


def select_action_greedy(model, state, epsilon):
    output = model(state)[0]
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    random_action = random.random() <= epsilon
    if random_action:
        action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
    else:
        action_index = torch.argmax(output)
    action[action_index] = 1
    return action, action_index, output


def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    game_state = GameState()

    iteration = 0
    replay_memory = []

    state = get_initial_state(game_state, model)

    epsilon = model.initial_epsilon
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)


    while iteration < model.number_of_iterations:

        action, action_index, output = select_action_greedy(model, state, epsilon)
        epsilon = epsilon_decrements[iteration]

        # Frame step interacts with the game environment
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)

        # Create the new state by stacking the last three frames with the new processed image
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_1, terminal))
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)


        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(data[0] for data in minibatch))
        action_batch = torch.cat(tuple(data[1] for data in minibatch))
        reward_batch = torch.cat(tuple(data[2] for data in minibatch))
        state_1_batch = torch.cat(tuple(data[3] for data in minibatch))

        output_1_batch = model(state_1_batch)

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)

        loss.backward()
        optimizer.step()

        state = state_1
        iteration += 1

        if iteration % 50000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(),"Q max:",
              np.max(output.cpu().detach().numpy()), "Reward:", reward.numpy()[0][0])


def test(model):
    game_state = GameState()
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1 # do nothing
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    with open("pretrained_model/current_model.txt", "w") as file:
        pass

    while True:
        output = model(state)[0]

        output_str = str(output.detach().numpy())

        with open("pretrained_model/current_model.txt", "a") as file:
            file.write(output_str + '\n')

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        action_index = torch.argmax(output)
        action[action_index] = 1

        image_data_1, reward, terminal = game_state.frame_step(action)

        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1


def main(mode):
    if mode == 'test':
        model = torch.load('pretrained_model/current_model_2000000.pth', map_location=torch.device('cpu')).eval()
        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        model.apply(init_weights)
        start = time.time()

        train(model, start)



if __name__ == "__main__":
    main(sys.argv[1])
