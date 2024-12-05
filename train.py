import gym
import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.network import ActorCriticContinuous
from utils.ppo_utils import random_alpha, shufffle_trajectory, compute_gae
from utils.logging import init_neptune
import random

import argparse
from datetime import datetime
import os
import pickle



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



# Hyperparameters





    
def collect_trajectories(env, model, alpha, n_steps):

    states, actions, rewards, log_ps, state_values, dones = [], [], [], [], [], []

    state, _ = env.reset()
    

    total_reward = 0
    step_count = 0

    for _ in range(n_steps):
        state = torch.FloatTensor(state).to(device)
        action, log_p, state_value, entropy = model(state, alpha)
        next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_ps.append(log_p)
        state_values.append(state_value)
        dones.append(done)

        state = next_state
        total_reward += reward
        step_count += 1

        if done:
            state, _ = env.reset()
            
    next_value = model.critic(torch.FloatTensor(next_state).unsqueeze(0).to(device)).squeeze(0).cpu()
    
    states = torch.stack(states)
    
    actions = torch.LongTensor(actions) if action.shape == 1 else torch.stack(actions)
    rewards = torch.LongTensor(rewards)
    state_values = torch.FloatTensor(state_values)
    dones = torch.LongTensor(dones)
    next_state_values = torch.concatenate((state_values[1:], next_value))
    
    
    advantages, returns = compute_gae(dones, rewards, state_values, next_state_values)
         
    # Normalize advantages
    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(state_values)
            
    trajectories = {
        "states" : states.detach(),
        "actions" : actions.detach(),
        "rewards" : rewards.detach(),
        "dones" : dones.detach(),
        "log_ps" : torch.stack(log_ps).detach(),
        "state_values": state_values.detach(),
        "next_state_values": next_state_values.detach(),
        "returns" : returns.detach(),
        "advantages" : advantages.detach(),
    }
    
    return trajectories


def ppo_optimization(args, trajectories, model, alpha, optimizer, epochs, batch_size):
    
    model.train()
    
    traj_states = trajectories["states"]
    traj_actions = trajectories["actions"]
    traj_log_ps = trajectories["log_ps"]
    traj_returns = trajectories["returns"]  
    traj_advantages = trajectories["advantages"]


    len_trajectory = traj_states.shape[0]

    for epoch in range(1, epochs+1):
        for i in range(len_trajectory // batch_size):
            state = traj_states[batch_size*i:batch_size*(i+1)].to(device)
            action = traj_actions[batch_size*i:batch_size*(i+1)].to(device)
            log_p = traj_log_ps[batch_size*i:batch_size*(i+1)].to(device)
            return_ = traj_returns[batch_size*i:batch_size*(i+1)].to(device)
            advantage = traj_advantages[batch_size*i:batch_size*(i+1)].to(device)
            
            new_action, new_log_p, new_state_value, entropy = model(state, alpha, action)
            assert(new_action == action).all()
            
            
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            new_log_p, log_p, advantage = new_log_p.reshape(-1), log_p.reshape(-1), advantage.reshape(-1)
            
            ratio = torch.exp(new_log_p - log_p.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantage
            policy_loss = - torch.min(surr1, surr2).mean()
            
            # print(policy_loss)
            
            
            return_, new_state_value = return_.reshape(-1), new_state_value.reshape(-1)
            critic_loss = ((return_ - new_state_value)**2).mean()
            
            penalty = 0
            if model.n_anchors > 1:
                j,k = random.sample(range(model.n_anchors),2)
                penalty = model.cosine_similarity(j,k)

            loss = policy_loss - 2e-7*entropy.mean() + 0.5*critic_loss + args.beta*penalty

            optimizer.zero_grad()
            loss.backward()
            clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    
    return entropy.mean().item(), critic_loss.item(), penalty


def evaluate(env, model, alpha):
    model.eval()

    state, _ = env.reset()
        
    total_reward = 0
    step_count = 0

    trajectory = []
    info = {
        'ffoot_touch_ground': [],
        'bfoot_touch_ground': [],
    }
    

    while True:
        trajectory.append(state)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action, log_p, state_value, entropy = model(state, alpha)
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

        info['bfoot_touch_ground'].append(int(5 in env.data.contact.geom2))
        info['ffoot_touch_ground'].append(int(8 in env.data.contact.geom2))
        
        done = terminated or truncated

        state = next_state
        total_reward += reward
        step_count += 1
        
        
        if done:
            break
        
    return total_reward, trajectory, info


def get_reward_from_trajectory(trajectory):
    save_ = []
    acc_reward = 0
    for i in range(len(trajectory['rewards'])):
        if trajectory['dones'][i] == True:
            save_.append(acc_reward)
            acc_reward = 0
        
        acc_reward += trajectory['rewards'][i]
    return np.median(save_)


def get_descriptor(info):
    return np.mean(info['bfoot_touch_ground']), np.mean(info['ffoot_touch_ground'])


def get_descriptors(args, env, model):
    descriptors = []
    rewards = []

    for _ in tqdm(range(args.n_population), desc='Constructing map-elite'):
        reward, states, info = evaluate(env, model, random_alpha(args.n_anchors).to(device))
        descriptors.append(get_descriptor(info))
        rewards.append(reward)
        
    return descriptors, rewards


def get_map_elite(args, descriptors, rewards):

    x1_min, x1_max, x2_min, x2_max = 0, 1, 0, 1
    N = 30

    x1_bins = np.linspace(x1_min, x1_max, N)
    x2_bins = np.linspace(x2_min, x2_max, N)

    map_grid = np.full((N, N), np.nan)


    def get_bin(value, bins):
        return np.digitize(value, bins) - 1


    for descriptor, reward in zip(descriptors, rewards):
        feature1, feature2 = descriptor[0], descriptor[1]
        f1_bin = get_bin(feature1, x1_bins)
        f2_bin = get_bin(feature2, x2_bins)
        
        if np.isnan(map_grid[f1_bin, f2_bin]):
            map_grid[f1_bin][f2_bin] = reward
        else:
            map_grid[f1_bin][f2_bin] = max(map_grid[f1_bin][f2_bin], reward) 



    fig, axs = plt.subplots(1, 1, figsize=(5,5))


    cax = axs.matshow(map_grid, cmap="viridis", vmin=-1000, vmax=3000)
    axs.set_title(f'n={args.n_anchors} b={args.beta}', loc='center')

    axs.set_xlabel('bfoot touch ground')
    axs.set_ylabel('ffoot touch ground')
    axs.set_xticks(np.linspace(0, N-1, num=11))
    axs.set_xticklabels(np.round(np.linspace(0, 1, num=11), 2))
    axs.set_yticks(np.linspace(0, N-1, num=11))
    axs.set_yticklabels(np.round(np.linspace(0, 1, num=11), 2))

    fig.colorbar(cax, ax=axs, orientation='vertical', label='Reward')
    
    return fig










def main(args):
    
    
    now = datetime.now()
    formatted_time = now.strftime("%d.%m.%Y-%H:%M:%S")
    save_folder = f'./models/halfcheetah/{formatted_time}'
    os.mkdir(save_folder)
    
    neptune_run = init_neptune(
        mode=args.mode,
        tags=[
            formatted_time, 
            f'n_anchors={args.n_anchors}',
            f'beta={args.beta}'
        ]
    )
    
    print(save_folder)
    print(args)
    
    
    env = gym.make('HalfCheetah-v4', render_mode = "rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCriticContinuous(
        args.n_anchors,
        state_dim,
        action_dim,
        same_init=False,
        actor_hidden_layers=args.actor_hidden_layers,
        critic_hidden_layers=args.critic_hidden_layers,
        action_std=args.action_std
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    rewards = []

    tqdm_epochs = tqdm(range(0, args.epochs + 1))
    for epoch in tqdm_epochs:
        
        alpha = random_alpha(args.n_anchors).to(device)
        
        trajectory = collect_trajectories(env, model, alpha, n_steps=args.len_trajectory)
        shuffled_trajectory = shufffle_trajectory(trajectory)
        entropy, critic_loss, cosine_similarity = ppo_optimization(args, shuffled_trajectory, model, alpha, optimizer, epochs=8, batch_size=args.batch_size)
        
        final_reward = get_reward_from_trajectory(trajectory)
        rewards.append(final_reward)
        
        tqdm_epochs.set_description(f'Reward: {final_reward}')
        
        neptune_run["reward"].append(final_reward)
        neptune_run["entropy"].append(entropy)
        neptune_run["critic_loss"].append(critic_loss)
        neptune_run["cosine_similarity"].append(cosine_similarity)
        

        if epoch % 100:
            torch.save(model.state_dict(), f'{save_folder}/model.pt')
    
    
    torch.save(model.state_dict(), f'{save_folder}/model.pt')
    
    
    descriptors, map_rewards = get_descriptors(args, env, model)
    with open(f'{save_folder}/descriptors_map_rewards.pkl', 'wb') as f:
        pickle.dump((descriptors, map_rewards), f)
    
    fig = get_map_elite(args, descriptors, map_rewards)    
    neptune_run["map-elite"].upload(fig)
    
    
    neptune_run.stop()
    
            
            
            
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Argument Parser for Training Configuration')
    
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs for training')
    parser.add_argument('--n_anchors', type=int, default=2, help='Number of anchor models of the subspace') 
    parser.add_argument('--n_population', type=int, default=1000, help='') 
    parser.add_argument('--len_trajectory', type=int, default=1024, help='Length of each trajectory to collect')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--action_std', type=float, default=0.5, help='Standard deviation of action distribution')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for the optimizer')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta parameter for PPO (e.g., for clipping)')
    parser.add_argument('--actor_hidden_layers', type=int, nargs='+', default=[256, 256], help='Sizes of hidden layers for the actor network')
    parser.add_argument('--critic_hidden_layers', type=int, nargs='+', default=[256, 256], help='Sizes of hidden layers for the critic network')
    
    parser.add_argument('--mode', type=str, default="debug", help='Neptune mode')
    

    args = parser.parse_args()
    
    main(args)
