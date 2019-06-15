from learning_to_adapt.dynamics.rnn_dynamics import RNNDynamicsModel
from learning_to_adapt.trainers.mb_trainer import Trainer
from learning_to_adapt.policies.rnn_mpc_controller import RNNMPCController
from learning_to_adapt.samplers.sampler import Sampler
from learning_to_adapt.logger import logger
from learning_to_adapt.envs.normalized_env import normalize
from learning_to_adapt.utils.utils import ClassEncoder
from learning_to_adapt.samplers.model_sample_processor import ModelSampleProcessor
from learning_to_adapt.envs import *

import json
import os

EXP_NAME = 'rebal'


def run_experiment(config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(config['env'](reset_every_episode=True, task=config['task']))

    dynamics_model = RNNDynamicsModel(
        name="dyn_model",
        env=env,
        hidden_sizes=config['hidden_sizes'],
        learning_rate=config['learning_rate'],
        backprop_steps=config['backprop_steps'],
        cell_type=config['cell_type'],
        batch_size=config['batch_size'],
    )

    policy = RNNMPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config['discount'],
        n_candidates=config['n_candidates'],
        horizon=config['horizon'],
        use_cem=config['use_cem'],
        num_cem_iters=config['num_cem_iters'],
    )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
        n_parallel=config['n_parallel'],
    )

    sample_processor = ModelSampleProcessor(recurrent=True)

    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        initial_random_samples=config['initial_random_samples'],
        dynamics_model_max_epochs=config['dynamic_model_epochs'],
    )
    algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
            # Environment
            'env': HalfCheetahEnv,
            'task': None,

            # Policy
            'n_candidates': 500,
            'horizon': 10,
            'use_cem': False,
            'num_cem_iters': 5,
            'discount': 1.,

            # Sampling
            'max_path_length': 1000,
            'num_rollouts': 5,
            'initial_random_samples': True,

            # Training
            'n_itr': 50,
            'learning_rate': 1e-2,
            'batch_size': 10,
            'backprop_steps': 100,
            'dynamic_model_epochs': 50,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Dynamics Model
            'cell_type': 'lstm',
            'hidden_sizes': (256,),

            #  Other
            'n_parallel': 5,
            }

    run_experiment(config)