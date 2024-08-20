import numpy as np
from PIL import Image
import numpy.random as random

size = 15

value_func = np.zeros((size, size))
q_action_value = np.zeros((size, size, 4))
action_space = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}


def policy(state: tuple[int, int], epsilon=0) -> int:
    if state == (size - 1, size - 1):
        return 0
    # find the best action to get the desired state using the action-value function
    highest = max([action_value(state, action) for action in action_space.keys()])
    max_actions = [action for action in action_space.keys() if action_value(state, action) == highest]
    # choose at random between all max actions or all actions with epsilon probability
    action = random.choice(list(action_space.keys())) if random.random() < epsilon else random.choice(max_actions)
    return action


def observe(state: tuple[int, int], action: int):
    # get next state based on the action
    b_near_edge = state[0] < size / 5 or state[1] < size / 5 or state[0] > size - size / 5 or state[1] > size - size / 5
    b_near_center = state[0] > size / 5 and state[1] > size / 5 and state[0] < size - size / 5 and state[1] < size - size / 5
    p_sway = 0.05 if b_near_edge else 0
    sway = random.choice([-1, 0, 1], p=[p_sway, 1 - 2 * p_sway, p_sway])
    action = (action + sway) % 4
    new_state = (state[0] + action_space[action][0], state[1] + action_space[action][1])
    if max(new_state) >= size or min(new_state) < 0:
        return state
    return new_state


def action_value(state: tuple[int, int], action: int) -> float:
    return q_action_value[state[0], state[1], action]


def reward(state: tuple[int, int]):
    if state == (size - 1, size - 1):
        return 100

    return -1


def sarsa(episodes, steps, discount, learn_rate, lambda_sarsa=0):
    eligibility = np.zeros((size, size, 4))
    for episode in range(1, episodes):
        state = (0, 0)
        for step in range(steps):
            print((episode, step), end="\r")
            if state == (size - 1, size - 1):
                value_func[state[0], state[1]] = reward(state)
                break
            action = policy(state, 1 / episode)
            new_state = observe(state, action)

            # todo: this should be an action-value function
            v_cur = action_value(state, action)
            v_next = action_value(new_state, policy(new_state))

            delta = reward(state) + discount * v_next - v_cur
            eligibility[state[0], state[1], action] += 1
            for i in range(size):
                for j in range(size):
                    for act in range(4):
                        q_action_value[i, j, act] += learn_rate * delta * eligibility[i, j, act]
                        eligibility[i, j, act] *= discount * lambda_sarsa
            state = new_state


def mc(episodes, steps, discount):
    visit_count = np.zeros((size, size))
    for episode in range(1, episodes):
        curstate = (0, 0)
        sequence = []
        for step in range(steps):
            print((episode, step), end="\r")

            action = policy(curstate, 1 / episode)
            new_state = observe(curstate, action)
            sequence.append((curstate, action))
            if new_state == (size - 1, size - 1):
                # sequence.append((new_state, 0))
                break
            curstate = new_state

        prev_return = 0
        episode_visit_count = np.zeros((size, size))
        for state, action in reversed(sequence):
            print((episode, state, action, "updating..."), end="\r")

            if (episode_visit_count[state[0]][state[1]]) > 0:
                continue

            episode_visit_count[state[0]][state[1]] += 1
            visit_count[state[0]][state[1]] += 1
            prev_return = reward(state) + discount * prev_return
            state_val = q_action_value[state[0], state[1], action]
            new_val = state_val + (1 / visit_count[state[0]][state[1]]) * (prev_return - state_val)
            q_action_value[state[0], state[1], action] = new_val


if __name__ == "__main__":
    import pandas as pd

    sarsa(episodes=1000, steps=300, discount=1, learn_rate=0.4, lambda_sarsa=0.5)
    # mc(episodes=100, steps=3000, discount=1)

    # draw path
    path = []
    state = (0, 0)
    print("\n")
    while state != (size - 1, size - 1):
        path.append(state)
        action = policy(state)
        print(state, action, end="\r")
        state = observe(state, action)

    img = Image.new("RGB", (size, size), "white")
    pixels = img.load()
    # for i in range(size):
    #     for j in range(size):
    #         pixels[i, j] = (int(value_func[i, j] * 255), int(value_func[i, j] * 255), int(value_func[i, j] * 255))

    for state in path:
        pixels[state[0], state[1]] = (255, 0, 0)
    img.show()
    # print(pd.DataFrame(value_func))
