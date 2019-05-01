import sys
import os
import argparse
import itertools

from experiment_utils import config
from experiment_utils.utils import query_yes_no

import doodad as dd
import doodad.mount as mount
import doodad.easy_sweep.launcher as launcher
from doodad.easy_sweep.hyper_sweep import run_sweep_doodad
import multiprocessing
import random
from doodad.easy_sweep.hyper_sweep import Sweeper

import time


def run_sweep(run_experiment, sweep_params, exp_name, instance_type='c4.xlarge'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')

    parser.add_argument('--num_gpu', '-g', type=int, default=1,
                        help='Number of GPUs to use for running the experiments')

    parser.add_argument('--exps_per_gpu', '-e', type=int, default=1,
                        help='Number of experiments per GPU simultaneously')

    parser.add_argument('--num_cpu', '-c', type=int, default=multiprocessing.cpu_count(),
                        help='Number of threads to use for running experiments')

    args = parser.parse_args(sys.argv[1:])

    local_mount = mount.MountLocal(local_dir=config.BASE_DIR, pythonpath=True)

    docker_mount_point = os.path.join(config.DOCKER_MOUNT_DIR, exp_name)

    sweeper = launcher.DoodadSweeper([local_mount], docker_img=config.DOCKER_IMAGE,
                                     docker_output_dir=docker_mount_point,
                                     local_output_dir=os.path.join(config.DATA_DIR, 'local', exp_name))
    sweeper.mount_out_s3 = mount.MountS3(s3_path='', mount_point=docker_mount_point, output=True)

    if args.mode == 'ec2':
        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_name, len(
            list(itertools.product(*[value for value in sweep_params.values()])))))

        if query_yes_no("Continue?"):
            sweeper.run_sweep_ec2(run_experiment, sweep_params, bucket_name=config.S3_BUCKET_NAME,
                                  instance_type=instance_type,
                                  region='us-west-2', s3_log_name=exp_name, add_date_to_logname=False)

    elif args.mode == 'local_docker':
        mode_docker = dd.mode.LocalDocker(
            image=sweeper.image,
        )
        run_sweep_doodad(run_experiment, sweep_params, run_mode=mode_docker,
                         mounts=sweeper.mounts)

    elif args.mode == 'local':
        sweeper.run_sweep_serial(run_experiment, sweep_params)

    elif args.mode == 'local_par':
        sweeper.run_sweep_parallel(run_experiment, sweep_params)

    elif args.mode == 'multi_gpu':
        run_sweep_multi_gpu(run_experiment, sweep_params, num_gpu=args.num_gpu, exps_per_gpu=args.exps_per_gpu)

    elif args.mode == 'local_singularity':
        mode_singularity = dd.mode.LocalSingularity(
            image='~/maml_zoo.simg')
        run_sweep_doodad(run_experiment, sweep_params, run_mode=mode_singularity,
                         mounts=sweeper.mounts)
    else:
        raise NotImplementedError


def run_sweep_multi_gpu(run_method, params, repeat=1, num_cpu=multiprocessing.cpu_count(), num_gpu=2, exps_per_gpu=2):
    sweeper = Sweeper(params, repeat, include_name=True)
    gpu_frac = 0.9 / exps_per_gpu
    num_runs = num_gpu * exps_per_gpu
    cpu_per_gpu = num_cpu / num_gpu
    exp_args = []
    for config in sweeper:
        exp_args.append((config, run_method))
    random.shuffle(exp_args)
    processes = [None] * num_runs
    run_info = [(i, (i * cpu_per_gpu, (i + 1) * cpu_per_gpu)) for i in range(num_gpu)] * exps_per_gpu
    for kwarg, run in exp_args:
        launched = False
        while not launched:
            for idx in range(num_runs):
                if processes[idx] is None or not processes[idx].is_alive():
                    kwarg['gpu_frac'] = gpu_frac
                    p = multiprocessing.Process(target=run, kwargs=kwarg)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % run_info[idx][0]
                    os.system("taskset -p -c %d-%d %d" % (run_info[idx][1] + (os.getpid(),)))
                    p.start()
                    processes[idx] = p
                    launched = True
                    break
            if not launched:
                time.sleep(10)
