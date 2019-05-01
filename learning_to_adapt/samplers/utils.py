import numpy as np
import time
import moviepy.editor as mpy


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=True,
            video_filename='sim_out.mp4', ignore_done=False, stochastic=False, num_rollouts=1):
    ''' get wrapped env '''
    wrapped_env = env
    while hasattr(wrapped_env, '_wrapped_env'):
        wrapped_env = wrapped_env._wrapped_env

    assert hasattr(wrapped_env, 'dt'), 'environment must have dt attribute that specifies the timestep'
    timestep = wrapped_env.dt
    images = []
    paths = []
    if animated:
        mode = 'human'
    else:
        mode = 'rgb_array'

    render = animated or save_video

    for i in range(num_rollouts):
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o = env.reset()
        agent.reset()
        path_length = 0
        if render:
            _ = env.render(mode)
            env.viewer.cam.distance = wrapped_env.model.stat.extent * 0.5
            env.viewer.cam.trackbodyid = 0
            env.viewer.cam.type = 1

        while path_length < max_path_length:
            a, agent_info = agent.get_action(o)
            if not stochastic:
                a = agent_info['mean']
            next_o, r, d, env_info = env.step(a)
            observations.append(o)
            rewards.append(r)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d and not ignore_done: # and not animated:
                break
            o = next_o

            if animated:
                env.render(mode)
                time.sleep(timestep/speedup)

            if save_video:
                image = env.render(mode)
                images.append(image)

        paths.append(dict(
                observations=observations,
                actons=actions,
                rewards=rewards,
                agent_infos=agent_infos,
                env_infos=env_infos
            ))
    if save_video:
        fps = int(speedup/timestep)
        clip = mpy.ImageSequenceClip(images, fps=fps)
        if video_filename[-3:] == 'gif':
            clip.write_gif(video_filename, fps=fps)
        else:
            clip.write_videofile(video_filename, fps=fps)
        print("Video saved at %s" % video_filename)

    return paths
