from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import clear_output
from matplotlib.patches import Patch


def moving_average(values, window):
    return (
        np.convolve(np.array(values).flatten(), np.ones(window), mode="valid") / window
    )


def visualise(env, agent):
    window = 500
    clear_output(True)
    # Create subplots
    fig, axs = plt.subplots(ncols=3, figsize=(16, 6))

    # Episode rewards plot
    axs[0].set_title("Episode Rewards", fontsize=14)
    reward_moving_average = moving_average(env.return_queue, window)
    axs[0].plot(
        range(len(reward_moving_average)), reward_moving_average, color="tab:blue"
    )
    axs[0].set_xlabel("Episodes", fontsize=12)
    axs[0].set_ylabel("Reward", fontsize=12)
    axs[0].grid(True)

    # Episode lengths plot
    axs[1].set_title("Episode Lengths", fontsize=14)
    length_moving_average = moving_average(env.length_queue, window)
    axs[1].plot(
        range(len(length_moving_average)), length_moving_average, color="tab:orange"
    )
    axs[1].set_xlabel("Episodes", fontsize=12)
    axs[1].set_ylabel("Length", fontsize=12)
    axs[1].grid(True)

    # Training Error plot
    axs[2].set_title("Bellman equation MSE", fontsize=14)
    mse_moving_average = moving_average(agent.mse, window)
    axs[2].plot(range(len(mse_moving_average)), mse_moving_average, color="tab:red")
    axs[2].set_xlabel("Episodes", fontsize=12)
    axs[2].set_ylabel("MSE", fontsize=12)
    axs[2].grid(True)

    # Adjust layout to avoid overlapping labels and titles
    plt.tight_layout()

    # Show the plot
    plt.show()


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


def visualise_strategy(agent):
    # state values & policy with usable ace (ace counts as 11)
    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    plt.show()

    # state values & policy without usable ace (ace counts as 1)
    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
    plt.show()
