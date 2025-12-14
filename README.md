# RL Snake Game

A Snake game where the snake is controlled by a Reinforcement Learning (RL) agent that learns to play better over time.

## Features

- Classic Snake game implementation
- Deep Q-Network (DQN) RL agent that learns through experience
- Training visualization with progress tracking
- Play mode to watch the trained AI play

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

Train the agent (headless, faster):
```bash
python main.py --mode train --episodes 1000
```

Train with visualization (slower but you can watch):
```bash
python main.py --mode train --episodes 1000 --render
```

Or use the training script directly:
```bash
python train.py
```

### Playing with Trained Agent

After training, watch the AI play:
```bash
python main.py --mode play --model model_final.npz --speed 10
```

The `--speed` parameter controls game speed (higher = faster).

## How It Works

- **Game (`game.py`)**: Implements Snake game logic with state representation for RL
- **Agent (`agent.py`)**: DQN agent with 2-layer neural network that learns Q-values
- **Training (`train.py`)**: Runs episodes, collects experiences, and trains the agent
- **Main (`main.py`)**: Entry point for training or playing

The agent uses:
- **State**: 20 features (danger detection, vision rays in 3 directions, direction encoding, snake length)
- **Actions**: 0=straight, 1=right turn, 2=left turn
- **Rewards**: +100 for eating food, -10 for death, +0.5/-0.3 for moving closer/farther from food, step penalty based on length

The agent gradually improves through exploration (epsilon-greedy) and learns from experience replay.

