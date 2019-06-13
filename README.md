# Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning

Implementation of [Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347).
The code is written in Python 3 and builds on Tensorflow. The environments require the Mujoco131 physics engine.

## Getting Started
### A. Docker
If not installed yet, [set up](https://docs.docker.com/install/) docker on your machine.
Pull our docker container iclavera/learning_to_adapt from docker-hub:

docker pull iclavera/learning_to_adapt
All the necessary dependencies are already installed inside the docker container.

### B. Anaconda
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions).

For Ubuntu you can install MPI through the package manager:
```sudo apt-get install libopenmpi-dev ```

If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux)

``` conda env create -f docker/environment.yml ```

For running the environments, the Mujoco physics engine version 131 is needed.
For setting up [Mujoco](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py)

## Acknowledgements
This repository is partly based on [Duan et al., 2016](https://arxiv.org/abs/1611.02779).