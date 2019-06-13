# Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning

Implementation of [Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347).
The code is written in Python 3 and builds on Tensorflow. The environments require the Mujoco131 physics engine.

## Getting Started
### A. Docker
If not installed yet, [set up](https://docs.docker.com/install/) docker on your machine.
Pull our docker container iclavera/learning_to_adapt from docker-hub:

```docker pull iclavera/learning_to_adapt ```
All the necessary dependencies are already installed inside the docker container.

### B. Anaconda
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions).

For Ubuntu you can install MPI through the package manager:
```sudo apt-get install libopenmpi-dev ```

If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux)

``` conda env create -f docker/environment.yml ```

For running the environments, the Mujoco physics engine version 131 is needed.
For setting up [Mujoco](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py)


## Usage
The run scripts are located in the folder ``` run_scripts```.
In order to run experiments with GrBAL, run the following command:

```python run_scripts/run_grbal.py ```

If instead, you want to run ReBAL:

``` python run_scripts/run_rebal.py ```

We have also implement a non-adaptive model-based method that uses random shooting or cross-entropy for planning. You
can run this baseline by executing the command:

``` python run_scripts/run_mb_mpc.py ```


When running experiments, the data will be stored in ``` data/$EXPERIMENT_NAME ```. You can visualize the learning process
by using the visualization kit:

``` python viskit/frontend.py data/$EXPERIMENT_NAME ```

In order to visualize and test a learned policy run:

``` python experiment_utils/sim_policy data/$EXPERIMENT_NAME```

## Acknowledgements
This repository is partly based on [Duan et al., 2016](https://arxiv.org/abs/1611.02779).