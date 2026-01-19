# The Matrix 🧬

A neural network-based evolutionary simulation where organisms evolve to survive by finding water resources in a 2D environment.

## Overview

This project simulates artificial life using **genetic algorithms** and **neural networks**. Organisms with neural network "brains" navigate a 2D world, and those closer to water sources (oases) have higher fitness, making them more likely to reproduce and pass on their neural network weights to the next generation.

## Features

- **Neural Network Brains**: Each organism has a feedforward neural network that decides its movement (up, down, left, right)
- **Genetic Algorithm**: Evolution through selection, crossover, and mutation
- **Fitness-Based Selection**: Organisms closer to water sources are more likely to reproduce
- **Real-Time Visualization**: Pygame-based rendering of the simulation
- **GPU Acceleration**: PyTorch with CUDA support for neural network computations

## Project Structure

```
The_Matrix/
├── entity.py          # Defines all entities 
├── environment.py     # Environment class managing population and resources
├── evolution.py       # Genetic algorithm: selection, crossover, mutation, fitness
├── simulator.py       # Pygame renderer for visualization
├── simulation.py      # Main entry point to run the simulation     
└── .env               # Configuration file (see below)
```

## Requirements

- Python 3.10+
- PyTorch
- Pygame
- python-dotenv

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/trabelsiaziz/The_Matrix.git
   cd The_Matrix
   ```

2. Install dependencies:

   ```bash
   pip install torch pygame python-dotenv
   ```

3. Create a `.env` file with the following configuration:
   ```env
   SCREEN_WIDTH=800
   SCREEN_HEIGHT=600
   POPULATION_SIZE=20
   WATER_SOURCE=1
   EVOLUTION_RATE=100
   MUTATION_RATE=0.1
   CROSSOVER_RATE=0.7
   ```

## Usage

Run the simulation:

```bash
python simulation.py
```

A Pygame window will open showing:

- **Brown circles**: Organisms navigating the environment
- **Blue circles**: Water sources (oases)
- **Yellow background**: The environment

## How It Works

### Neural Network (Brain)

Each organism has a 3-layer neural network:

- **Input**: Environment dimensions, resource positions, own position, and other organisms' positions
- **Hidden layers**: 64 → 32 neurons with ReLU activation
- **Output**: 4 actions (up, down, left, right) via softmax

### Evolution Cycle

1. **Simulation**: Organisms move based on their neural network decisions
2. **Fitness Evaluation**: Distance to nearest water source determines fitness
3. **Selection**: Roulette wheel selection based on fitness
4. **Crossover**: Single-point crossover of neural network weights
5. **Mutation**: Small random changes to weights

### Configuration Parameters

| Parameter         | Description                              |
| ----------------- | ---------------------------------------- |
| `SCREEN_WIDTH`    | Width of the simulation window           |
| `SCREEN_HEIGHT`   | Height of the simulation window          |
| `POPULATION_SIZE` | Number of organisms in the simulation    |
| `WATER_SOURCE`    | Number of water sources (oases)          |
| `EVOLUTION_RATE`  | Steps between evolution cycles           |
| `MUTATION_RATE`   | Probability of gene mutation (0-1)       |
| `CROSSOVER_RATE`  | Probability of crossover occurring (0-1) |


## Author

Created by [trabelsiaziz](https://github.com/trabelsiaziz)
