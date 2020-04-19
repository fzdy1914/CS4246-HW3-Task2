import collections
import random

import numpy as np
import math
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from gym_grid_driving.envs.grid_driving import AgentState, Point

from models import ConvDQN
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit = 8196
batch_size = 128
max_episodes  = 10000
t_max         = 1000
min_buffer = 1024
target_update = 10 # episode(s)
train_steps   = 10
max_epsilon   = 0.2
min_epsilon   = 0.05
epsilon_decay = 1000
print_interval= 20
curriculum_num= 10
use_epsilon   = 0.05

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def compute_loss(model, target, states, actions, rewards, next_states, dones):
    model_values = model.forward(states)
    model_values = model_values.gather(1,actions)
    model_values = model_values.squeeze()
    with torch.no_grad():
        target_values = target.forward(next_states)
        target_values = target_values.max(1)[0]
        target_values -= torch.mul(dones.squeeze(dim=1), target_values)
        target_values = target_values * gamma + rewards.squeeze(dim=1)
    loss = F.smooth_l1_loss(model_values, target_values)
    return loss


def optimize(model, target, success_memory, fail_memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch_success = success_memory.sample(batch_size // 2)
    batch_fail = fail_memory.sample(batch_size // 2)

    batch = (
        torch.cat([batch_success[0], batch_fail[0]]),
        torch.cat([batch_success[1], batch_fail[1]]),
        torch.cat([batch_success[2], batch_fail[2]]),
        torch.cat([batch_success[3], batch_fail[3]]),
        torch.cat([batch_success[4], batch_fail[4]])
    )
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon


def train(model, env, train_type=0, model_class=ConvDQN):
    # Initialize model and target network
    f = open('record.txt', 'a')
    if not model:
        model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    success_memory = ReplayBuffer(buffer_limit)
    fail_memory = ReplayBuffer(buffer_limit)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        if not use_epsilon:
            epsilon = compute_epsilon(episode)
        else:
            epsilon = use_epsilon

        if train_type == 0:
            env.agent_pos_init = Point(random.randint(1, 5 * curriculum_num - 1), random.randint(0, curriculum_num - 1))
        elif train_type == 1:
            env.agent_pos_init = Point(random.randint(5 * curriculum_num - 6, 5 * curriculum_num - 1), random.randint(curriculum_num - 2, curriculum_num - 1))
        elif train_type == 2:
            env.agent_pos_init = Point(5 * curriculum_num - 1, curriculum_num - 1)
        elif train_type == 3:
            pass

        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            action = model.act(state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            if env.world.agent_state == AgentState.crashed:
                reward = -2

            if reward > 0:
                success_memory.push(Transition(state, [action], [reward], next_state, [done]))
            elif reward < 0:
                fail_memory.push(Transition(state, [action], [reward], next_state, [done]))
            else:
                if t % 2 == 0:
                    success_memory.push(Transition(state, [action], [reward], next_state, [done]))
                else:
                    fail_memory.push(Transition(state, [action], [reward], next_state, [done]))

            # Save transition to replay buffer
            state = next_state
            episode_rewards += reward
            if done or train_type == 4 and t > 40:
                break
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(success_memory) > min_buffer and len(fail_memory) > min_buffer:
            # if np.mean(rewards[print_interval:]) < -60:
            #     print('Bad initialization. Please restart the training.')
            #     exit()
            for i in range(train_steps):
                loss = optimize(model, target, success_memory, fail_memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Curriculum {} Type {} Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            curriculum_num, train_type, episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(success_memory) + len(fail_memory), epsilon * 100))
            f.write("[Curriculum {} Type {} Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}% \n".format(
                            curriculum_num, train_type, episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(success_memory) + len(fail_memory), epsilon * 100))
            f.flush()

        if episode % 1000 == 0:
            save_model_with_path(model, './model_' + str(curriculum_num) + '_' + str(train_type) + '_' + str(episode) + '.pt')
    f.close()

    return model


def test(model, env, max_episodes=600):
    '''
    Test the `model` on the environment `env` (GridDrivingEnv) for `max_episodes` (`int`) times.

    Output: `avg_rewards` (`float`): the average rewards
    '''
    rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(t_max):
            action = model.act(state)
            state, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
    avg_rewards = np.mean(rewards)
    print("{} episodes avg rewards : {:.1f}".format(max_episodes, avg_rewards))
    return avg_rewards


def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`. 
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)


def save_model_with_path(model, model_path):
    '''
    Save `model` to disk. Location is specified in `model_path`.
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)


if __name__ == '__main__':
    import argparse
    from env import construct_task2_env

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    parser.add_argument('--curriculum', dest='curriculum', action='store_true', help='curriculum train the agent')
    parser.add_argument('--deep', dest='deep', action='store_true', help='deep train the agent')
    parser.add_argument('--limit', dest='limit', action='store_true', help='deep train the agent to limit to 40 step')
    args = parser.parse_args()

    env = construct_task2_env()
    if args.train:
        curriculum_num = 1
        model = train(None, env, model_class=ConvDQN)
        save_model(model)
    elif args.curriculum:
        model = get_model()
        for i in range(1, 11):
            curriculum_num = i
            use_epsilon = None
            max_epsilon = 1
            min_epsilon = 0.05
            if i < 5:
                max_epsilon = 0.5
                max_episodes = 6020
            elif i < 9:
                max_epsilon = 0.3
                max_episodes = 10020
            else:
                max_epsilon = 0.2
                max_episodes = 14020
            train(model, env, train_type=0)
            train(model, env, train_type=1)
            use_epsilon = 0.05
            train(model, env, train_type=2)
            save_model(model)
    elif args.deep:
        model = get_model()
        curriculum_num = 10
        use_epsilon = 0.05
        max_episodes = 50020
        learning_rate = 0.00001
        buffer_limit = 16384
        batch_size = 512
        print_interval = 50
        train(model, env, train_type=3)
        save_model(model)
    elif args.limit:
        curriculum_num = 10
        use_epsilon = 0.05
        max_episodes = 50020
        learning_rate = 0.00001
        buffer_limit = 16384
        batch_size = 512
        print_interval = 50
        model = get_model()
        train(model, env, train_type=4)
        save_model(model)
    else:
        model = get_model()
    test(model, env, max_episodes=600)
