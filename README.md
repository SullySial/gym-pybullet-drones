# Drone Reinforcement Learning Environment

This repository is a fork of the original Gym PyBullet Drones, updated with additional scripts and a new environment for reinforcement learning with drones. This code is designed to simulate multiple drones performing tasks synchronously using PID controls and reinforcement learning.

## Overview

The original code was designed for a drone to hover at a given Z coordinate utilising reinforcement learning. This fork allows for additional movement of the drone - moving in X, Y, Z locations. It also simplifies some aspects to focus primarily on reinforcement learning.

## Features

- **Forked from Gym PyBullet Drones**: Updated with an additional script and a new environment.
- **Simplified Environment**: Adjusted for reinforcement learning, removing extraneous functionalities like different drone types, physics options, recording, etc.
- **Target Position**: Allows setting a target position as an array with XYZ values.
- **Episode Length**: Configurable episode length.
- **Properties and Constants**: Loaded from the drone's URDF and includes mass, geometry, and other physical properties.

## Installation

### Requirements

- **Intel x64**: Ubuntu 22.04
- **Apple Silicon**: macOS 14.1

### Steps

1. Clone the repository:

```bash
git clone git@github.com:SullySial/gym-pybullet-drones.git 
cd gym-pybullet-drones/
```

2. Create and activate a Conda environment on conda terminal:

```bash
conda create -n drones python=3.10
conda activate drones
```

3. From VS code Install dependencies:

```bash
pip3 install --upgrade pip
pip3 install -e . # If needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`
```

## Usage

### Running the Environment

To run the environment, use the `simplefly.py` script. This script initializes the environment, sets the target position, and runs the reinforcement learning algorithm. In this example, it will run using preloaded weights given by the .pkl file in the same directory.

```bash
cd gym-pybullet-drones/gym_pybullet_drones/examples/
python simplefly.py
```

## Files

- **`gym_pybullet_drones/envs/SimpleBase.py`**: Base class for the environment, handling the initialization, action processing, and physics simulation of the drone.
- **`gym_pybullet_drones/examples/simplefly.py`**: Main script for reinforcement learning with the drone. It uses the `SimpleBase` environment and implements a Deep Q-Network (DQN) solver for training the drone.

### Utilities

- **Logger**: Utility for logging simulation data.
- **Goal**: Visualization of the goal position in the simulation.

## Key Components

### SimpleBase Class

The `SimpleBase` class in `gym_pybullet_drones/envs/SimpleBase.py` is a base class for the drone environment. Key functionalities include:

- **Initialization**: Sets up the drone model, physics, GUI, and other parameters.
- **Action Processing**: Converts actions into motor RPMs or direct forces depending on the gravity setting.
- **Physics Simulation**: Applies forces and torques to the drone based on the calculated RPMs or direct forces.
- **Reward Calculation**: Computes rewards based on the drone's distance to the target and its tilt angle.
- **Termination Check**: Determines if the episode should terminate based on the drone's position relative to the target.

### SimpleFly Script

The `simplefly.py` script in `gym_pybullet_drones/examples/simplefly.py` is the main script for running reinforcement learning with the drone. It includes:

- **DQN Solver**: Implements a Deep Q-Network for training the drone to reach a target position.
- **Training Loop**: Runs the training episodes, collecting experiences and updating the DQN.
- **Testing**: Tests the trained model by running the environment and evaluating the drone's performance.

### DQN Solver

The DQN solver in `simplefly.py` includes:

- **Replay Buffer**: Stores experiences for training the neural network.
- **Neural Network**: A simple multi-layer perceptron with two hidden layers.
- **Training Loop**: Samples experiences from the replay buffer and updates the neural network.

### Hyperparameters

The following hyperparameters are used in the DQN algorithm:

- **EPISODES**: Number of training episodes (default: 12000)
- **LEARNING_RATE**: Learning rate for the neural network optimizer (default: 0.00025)
- **MEM_SIZE**: Maximum size of the replay memory (default: 50000)
- **REPLAY_START_SIZE**: Number of samples to fill the replay memory before training starts (default: 10000)
- **BATCH_SIZE**: Number of samples used for training in each iteration (default: 32)
- **GAMMA**: Discount factor for future rewards (default: 0.99)
- **EPS_START**: Initial epsilon value for epsilon-greedy action selection (default: 0.1)
- **EPS_END**: Final epsilon value (default: 0.0001)
- **EPS_DECAY**: Rate at which epsilon decays (default: 4 * MEM_SIZE)
- **NETWORK_UPDATE_ITERS**: Frequency of updating the target network (default: 5000)

## Notes

- **Gravity**: Disabling gravity improves the performance of the reinforcement learning algorithm, as the drone does not have to counteract a constant force.
- **Reward Function**: The reward function may require further fine-tuning to improve the learning performance.

## Future Improvements

- **Dynamic Principles**: Integrating dynamic principles with gravity for more realistic simulations.
- **Reward Function**: Adjustments to the reward function to better incentivize desired behaviors.

## References

- **Original Repository**: git clone https://github.com/utiasDSL/gym-pybullet-drones.git
- **Original Repositiory References**: 
    * Carlos Luis and Jeroome Le Ny (2016) Design of a Trajectory Tracking Controller for a Nanoquadcopter
    * Nathan Michael, Daniel Mellinger, Quentin Lindsey, Vijay Kumar (2010) The GRASP Multiple Micro UAV Testbed
    * Benoit Landry (2014) Planning and Control for Quadrotor Flight through Cluttered Environments
    * Julian Forster (2015) System Identification of the Crazyflie 2.0 Nano Quadrocopter
    * Antonin Raffin, Ashley Hill, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, and Noah Dormann (2019) Stable Baselines3
    * Guanya Shi, Xichen Shi, Michael O’Connell, Rose Yu, Kamyar Azizzadenesheli, Animashree Anandkumar, Yisong Yue, and Soon-Jo Chung (2019) Neural Lander: Stable Drone Landing Control Using Learned Dynamics
    * C. Karen Liu and Dan Negrut (2020) The Role of Physics-Based Simulators in Robotics
    * Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza (2020) Flightmare: A Flexible Quadrotor Simulator
