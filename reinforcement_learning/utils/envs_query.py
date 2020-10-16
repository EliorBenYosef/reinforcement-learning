from gym import envs


class NullE:
    def __init__(self):
        self.observation_space = self.action_space = self.reward_range = "N/A"


def print_all_envs_specs(envs_specs):
    table = "|Id|nonDet|obsSpace|actSpace|maxEpSecs|maxEpSteps|rRange|rThresh|tStepLim|Trials|LocalOnly|kwargs|\n"
    table += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"

    for env_spec in envs_specs:
        try:
            env = env_spec.make()
        except:
            env = NullE()
            continue  # Skip these for now

        table += '|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|\n'.format(
            env_spec.id, env_spec.nondeterministic, env.observation_space, env.action_space,
            env_spec.max_episode_seconds, env_spec.max_episode_steps,
            env.reward_range, env_spec.reward_threshold,
            env_spec.timestep_limit, env_spec.trials,
            getattr(env_spec, '_local_only', 'not defined'), getattr(env_spec, '_kwargs', '')
        )

    print(table)


if __name__ == '__main__':
    # envs_specs = envs.registry.all()

    envs_ids = [
        'FrozenLake-v0', 'Taxi-v2', 'Blackjack-v0',
        'MountainCar-v0', 'MountainCarContinuous-v0', 'CartPole-v0', 'CartPole-v1', 'Pendulum-v0', 'Acrobot-v1',
        'LunarLander-v2', 'LunarLanderContinuous-v2', 'BipedalWalker-v2',
        'Breakout-v0', 'SpaceInvaders-v0'
    ]
    envs_specs = []
    for env_id in envs_ids:
        envs_specs.append(envs.registry.spec(env_id))

    print_all_envs_specs(envs_specs)
