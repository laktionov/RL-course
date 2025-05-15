import gymnasium as gym
from tqdm import tqdm
from visualisation_utils import visualise


def train_agent(agent, env, n_episodes, epsilon_decay):
    record_env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    for episode in tqdm(range(1, n_episodes + 1), position=0, leave=True):
        state, _ = record_env.reset()
        done = False
        agent.on_episode_start()
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = record_env.step(action)
            agent.update(state, action, reward, terminated, next_state)

            done = terminated or truncated
            state = next_state

        if not episode % 1000:
            agent.epsilon *= epsilon_decay

        if not episode % 10000:
            visualise(record_env, agent)

    return record_env


def evaluate_agent(env, agent, num_episodes=1000):
    win_count = 0
    lose_count = 0
    draw_count = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.get_best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated

        if reward == 1:
            win_count += 1
        elif reward == -1:
            lose_count += 1
        elif reward == 0:
            draw_count += 1

    return {
        "win": win_count / num_episodes,
        "draw": draw_count / num_episodes,
        "lose": lose_count / num_episodes,
    }
