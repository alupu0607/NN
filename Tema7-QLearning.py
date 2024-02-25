import numpy as np
import matplotlib.pyplot as plt

environment_rows = 7
environment_columns = 10
q_values = np.zeros((environment_rows, environment_columns, 4)) #(0,1,LEFT)

actions = ['up', 'right', 'down', 'left']

rewards = np.full((environment_rows, environment_columns), -1)
rewards[3, 7] = 999999


def is_terminal_state(current_row_index, current_column_index):
      if rewards[current_row_index, current_column_index] == -1:
        return False
      else:
        return True


def get_next_action(current_row_index, current_column_index, epsilon):
      if np.random.random() > epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
      else:
        return np.random.randint(4)

def get_next_location(current_row_index, current_column_index, action_index):
      new_row_index = current_row_index
      new_column_index = current_column_index
      if new_column_index == 0 or new_column_index == 1 or new_column_index == 2 or new_column_index == 9:
          wind = 0
      elif new_column_index == 6 or new_column_index == 7:
          wind = 2
      else:
          wind = 1

      if actions[action_index] == 'up' and current_row_index > 0:
            new_row_index = new_row_index - 1 - wind

      elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
            new_row_index = new_row_index + 1 - wind


      elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
                new_column_index += 1
                new_row_index -= wind

      elif actions[action_index] == 'left' and current_column_index > 0:
                new_column_index -= 1
                new_row_index -= wind

      if new_row_index < 0:
          new_row_index = 0
      return new_row_index, new_column_index



#training parameters
epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn

episode_rewards = []
for episode in range(1000):
    row_index, column_index = 3,0
    total_reward = 0

    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)
        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        reward = rewards[row_index, column_index]
        total_reward += reward

        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value
    episode_rewards.append(total_reward)


print('Training complete!')


def show_learned_policy(q_values):
    rows, columns, _ = q_values.shape
    learned_policy = np.zeros((rows, columns), dtype=str)

    for i in range(rows):
        for j in range(columns):
            if is_terminal_state(i, j):
                learned_policy[i, j] = 'T'  # Mark terminal states
            else:
                action_index = np.argmax(q_values[i, j])
                learned_policy[i, j] = actions[action_index][0].upper()

    print("Learned Policy:")
    print(learned_policy)


# After training
show_learned_policy(q_values)

print("Max Total Reward:", np.max(episode_rewards))
print("Min Total Reward:", np.min(episode_rewards))
print("Mean Total Reward:", np.mean(episode_rewards))


plt.figure(figsize=(12, 6))

# Plot the total reward over episodes
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)

# Set labels and title
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Convergence Check')

# Show the plot
plt.show()

# def visualize_optimal_path(q_values, start_row, start_column):
#     current_row, current_column = start_row, start_column
#     path = [(current_row, current_column)]
#
#     while not is_terminal_state(current_row, current_column):
#         action_index = np.argmax(q_values[current_row, current_column])
#         current_row, current_column = get_next_location(current_row, current_column, action_index)
#         path.append((current_row, current_column))
#
#     return path


# After training
#optimal_path = visualize_optimal_path(q_values, start_row=3, start_column=0)
#print("Optimal Path:")
#print(optimal_path)

