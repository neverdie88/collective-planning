# Introduction


This repository implements NIPS 2017 Policy Gradient With Value Function Approximation
For Collective Multiagent Planning (Nguyen et al.) in Tensorflow.


```
@inproceedings{nguyen2017policy,
  title={Policy gradient with value function approximation for collective multiagent planning},
  author={Nguyen, Duc Thien and Kumar, Akshat and Lau, Hoong Chuin},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4322--4332},
  year={2017}
}
```


Our code is based on a2c implementation from the OpenAI Baselines.

## Prerequisites

Our code is implemented and tested on python 3.6.1, tensforflow 1.2.1 and gym 0.9.2
To install these prerequisites, one can use the following command using anaconda and pip

```bash
conda install python==3.6.1
pip install tensorflow==1.2.1
pip install gym==0.9.2
```
## Installation

After downloading the code to <yourdirector> in your local machine, you can run the code by setting python path

```sh
export PYTHONPATH=<yourdirectory>/collective-planning
```


### Grid Navigation

The example code to run grid navigation problem (in our NIPS17 paper) can be found in

- [Vanilla AC](baselines/grid_navigation/grid_AC.py)
- [fAfC](baselines/grid_navigation/grid_fAfC.py)


To train neural network policy with (32x32) hidden layers using Vanilla AC or fAfC, you can enter these commands

```sh
export PYTHONPATH=<yourdirectory>/collective-planning
cd <yourdirectory>/collective-planning-master/baselines/grid_navigation/
python grid_AC.py --hidden-unit-num 32 --layer-norm --grid-size 5 --population-size 10 --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
python grid_fAfC.py --hidden-unit-num 32 --layer-norm --grid-size 5 --population-size 10 --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
```

### Grid Patrolling
The example code to run grid patrolling problem (in our NIPS18 paper) can be found in

- [Vanilla AC](baselines/grid_patrolling/grid_patrol_AC.py)
- [CCAC](baselines/grid_patrolling/grid_patrol_CCAC.py)
- [MCAC](baselines/grid_patrolling/grid_patrol_MCAC.py)


To train neural network policy with (32x32) hidden layers using Vanilla AC, CCAC or MCAC, you can enter these commands

```sh
export PYTHONPATH=<yourdirectory>/collective-planning
cd <yourdirectory>/collective-planning-master/baselines/grid_navigation/
python grid_patrol_AC.py --hidden-unit-num 32 --layer-norm --vf-normalization --grid-size 5 --population-size 10 --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
python grid_patrol_CCAC.py --hidden-unit-num 32 --layer-norm --vf-normalization --grid-size 5 --population-size 10 --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
python grid_patrol_MCAC.py --hidden-unit-num 32 --layer-norm --vf-normalization --grid-size 5 --population-size 10 --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
```

### Taxi Navigation
The example code to run taxi navigation problem (in our NIPS18 paper) can be found in

- [Vanilla AC](baselines/taxi_navigation/taxi_AC.py)
- [CCAC](baselines/taxi_navigation/taxi_CCAC.py)
- [MCAC](baselines/taxi_navigation/taxi_MCAC.py)


To train neural network policy with (18x18) hidden layers using Vanilla AC, CCAC or MCAC, you can enter these commands

```sh
export PYTHONPATH=<yourdirectory>/collective-planning
cd <yourdirectory>/collective-planning-master/baselines/grid_navigation/
python taxi_AC.py --vf-normalization --layer-norm --hidden-unit-num 18 --critic-lr 1e-4 --actor-lr 1e-5 --critic-training-type 2 --satisfied-percentage 0.95 --max-var 10.0 --penalty-weight 3.0
python taxi_CCAC.py --vf-normalization --layer-norm --hidden-unit-num 18 --critic-lr 1e-4 --actor-lr 1e-5 --critic-training-type 2 --satisfied-percentage 0.95 --max-var 10.0 --penalty-weight 3.0
python taxi_MCAC.py --vf-normalization --layer-norm --hidden-unit-num 18 --critic-lr 1e-4 --actor-lr 1e-5 --critic-training-type 2 --satisfied-percentage 0.95 --max-var 10.0 --penalty-weight 3.0
```

### Police Patrolling
The example code to run police patrolling problem (in our NIPS18 paper) can be found in

- [Vanilla AC](baselines/police_patrolling/police_AC.py)
- [CCAC](baselines/police_patrolling/police_CCAC.py)
- [MCAC](baselines/police_patrolling/police_MCAC.py)


To train neural network policy with (32x32) hidden layers using Vanilla AC, CCAC or MCAC, you can enter these commands

```sh
export PYTHONPATH=<yourdirectory>/collective-planning
cd <yourdirectory>/collective-planning-master/baselines/grid_navigation/
python grid_patrol_AC.py --shift 1 --layer-norm --vf-normalization --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
python grid_patrol_CCAC.py --shift 1 --layer-norm --vf-normalization --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
python grid_patrol_MCAC.py --shift 1 --layer-norm --vf-normalization --num-cpu 1 --actor-lr 1e-4 --critic-lr 1e-3
```



=======
```
>>>>>>> origin/master
