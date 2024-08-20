import numpy as np
from PIL import Image
import random

size = 10

value_func = np.zeros((size, size))

action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def policy(state: tuple[int, int], epsilon=0) -> tuple[int, int]:
    # find the best action to get the desired state using the action-value function
    action_val_d = action_value(state)
    highest = max(action_val_d.values())
    max_actions = [k for k, v in action_val_d.items() if v == highest]
    # choose at random between all max actions or all actions with epsilon probability
    action = random.choice(action_space) if random.random() < epsilon else random.choice(max_actions)
    return action


def observe(state: tuple[int, int], action: tuple[int, int]):
    # get next state based on the action
    new_state = (state[0] + action[0], state[1] + action[1])
    if max(new_state) >= size or min(new_state) < 0:
        return state
    return new_state


def action_value(state: tuple[int, int]) -> dict[tuple[int, int], float]:
    action_val_d = {action: -1.0 for action in action_space}
    for action in action_space:
        new_state = observe(state, action)
        action_val_d[action] = value_func[new_state[0]][new_state[1]]

    return action_val_d


def reward(state: tuple[int, int]):
    if state == (size - 1, size - 1):
        return 100

    return -1


def sarsa(episodes, steps, discount, learn_rate, lambda_sarsa=0):
    eligibility = np.zeros((size, size))
    for episode in range(1, episodes):
        state = (0, 0)
        for _ in range(steps):
            if state == (size - 1, size - 1):
                value_func[state[0], state[1]] = reward(state)
                break
            action = policy(state, 1 / episode)
            new_state = observe(state, action)

            # todo: this should be an action-value function
            v_cur = value_func[state[0], state[1]]
            v_next = value_func[new_state[0], new_state[1]]

            delta = reward(state) + discount * v_next - v_cur
            eligibility[state[0], state[1]] += 1
            for i in range(size):
                for j in range(size):
                    value_func[i, j] += learn_rate * delta * eligibility[i, j]
                    eligibility[i, j] *= discount * lambda_sarsa
            state = new_state


def mc(episodes, steps, discount):
    visit_count = np.zeros((size, size))
    for episode in range(1, episodes):
        curstate = (0, 0)
        sequence = []
        sequence.append(curstate)
        for step in range(steps):
            action = policy(curstate, 1 / episode)
            new_state = observe(curstate, action)
            sequence.append(new_state)
            if new_state == (size - 1, size - 1):
                break
            curstate = new_state

        prev_return = 0
        episode_visit_count = np.zeros((size, size))
        for state in reversed(sequence):
            if (episode_visit_count[state[0]][state[1]]) > 0:
                continue
            episode_visit_count[state[0]][state[1]] += 1
            visit_count[state[0]][state[1]] += 1
            prev_return = reward(state) + discount * prev_return
            state_val = value_func[state[0], state[1]]
            new_val = state_val + (1 / visit_count[state[0]][state[1]]) * (prev_return - state_val)
            value_func[state[0]][state[1]] = new_val


if __name__ == "__main__":
    import pandas as pd

    sarsa(episodes=1000, steps=1000, discount=1, learn_rate=0.4, lambda_sarsa=0.5)
    print(pd.DataFrame(value_func))
