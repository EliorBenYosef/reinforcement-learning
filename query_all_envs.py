from gym import envs


class NullE:
    def __init__(self):
        self.observation_space = self.action_space = self.reward_range = "N/A"


envall = envs.registry.all()

table = "|Environment Id|Observation Space|Action Space|Reward Range|tStepL|Trials|rThresh|\n"  # Local|nonDet|kwargs|
table += "|---|---|---|---|---|---|---|\n"  # ---|---|---|

for e in envall:
    try:
        env = e.make()
    except:
        env = NullE()
        continue  # Skip these for now
    table += '|{}|{}|{}|{}|{}|{}|{}|\n'.format(  # {}|{}|{}|
        e.id, env.observation_space, env.action_space, env.reward_range, e.timestep_limit, e.trials, e.reward_threshold
        # , getattr(e, 'local_only', -1), e.nondeterministic, getattr(e, 'kwargs', "")
    )

print(table)
