import numpy as np


def rollout(env, policy, max_path_length=np.inf,
            animated=False, ignore_done=False,
            num_rollouts=1, adapt_batch_size=None):
    ''' get wrapped env '''
    wrapped_env = env
    while hasattr(wrapped_env, '_wrapped_env'):
        wrapped_env = wrapped_env._wrapped_env

    paths = []
    a_bs = adapt_batch_size
    for i in range(num_rollouts):
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o = env.reset()
        policy.reset()
        path_length = 0

        while path_length < max_path_length:
            if a_bs is not None and len(observations) > a_bs + 1:
                adapt_obs = observations[-a_bs - 1:-1]
                adapt_act = actions[-a_bs - 1:-1]
                adapt_next_obs = observations[-a_bs:]
                policy.dynamics_model.switch_to_pre_adapt()
                policy.dynamics_model.adapt([np.array(adapt_obs)], [np.array(adapt_act)],
                                            [np.array(adapt_next_obs)])
            a, agent_info = policy.get_action(o)
            next_o, r, d, env_info = env.step(a)
            observations.append(o)
            rewards.append(r)
            actions.append(a[0])
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d and not ignore_done: # and not animated:
                break
            o = next_o

            if animated:
                env.render()

        paths.append(dict(
            observations=observations,
            actons=actions,
            rewards=rewards,
            agent_infos=agent_infos,
            env_infos=env_infos
        ))

    return paths
