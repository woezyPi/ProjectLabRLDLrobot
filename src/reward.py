def compute_reward(
    state,
    next_state,
    action,
    max_range,
    terminal_reason=None
):

    # Pénalités terminales
    if terminal_reason == "collision":
        return -20.0

    if terminal_reason == "stuck":
        return -10.0

    if terminal_reason == "timeout":
        return 0.0

    distance = float(
        next_state[-12:].min() * max_range
    )

    reward = -0.01

    # Récompense liée à l'action
    action_rewards = {
        0: 1.0,
        1: 0.2,
        2: 0.2,
        3: -0.2,
        4: -0.1
    }

    reward += action_rewards[action]

    # Pénalité de proximité
    if distance < 0.18:
        reward -= 3.0

    elif distance < 0.30:
        reward -= 1.0

    return float(reward)