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
python main.py --mode train --episodes 3000
```

Train with visualization (slower but you can watch):
```bash
python main.py --mode train --episodes 3000 --render
```

Or use the training script directly (defaults to 3000 episodes):
```bash
python train.py
```

**Note**: The model is automatically saved every 50 episodes to `models/model_episode_{episode}.npz`, and a final model is saved at the end of training.

### Playing with Trained Agent

After training, watch the AI play:
```bash
python main.py --mode play --model model_final.npz --speed 10
```

The `--speed` parameter controls game speed (higher = faster).

## How It Works

- **Game (`game.py`)**: Implements Snake game logic with state representation for RL
- **Agent (`agent.py`)**: DQN agent with 3-layer neural network (256 hidden units per layer) that learns Q-values
- **Training (`train.py`)**: Runs episodes, collects experiences, and trains the agent
- **Main (`main.py`)**: Entry point for training or playing

### Agent Architecture

The agent uses a **deep Q-network** with:
- **Network**: 3-layer neural network (20 → 256 → 256 → 3)
- **State**: 20 features (danger detection, vision rays in 3 directions, direction encoding, snake length)
- **Actions**: 0=straight, 1=right turn, 2=left turn
- **Rewards**: +100 for eating food, -10 for death, +0.5/-0.3 for moving closer/farther from food, step penalty based on length

### Training Improvements

The training process includes several optimizations for better learning:
- **Frequent training**: Agent trains every 2 steps during episodes, plus 3 training steps at episode end
- **Larger batches**: Uses batch size of 64 for more stable learning
- **Better exploration**: Linear epsilon decay for first 1000 steps, then exponential decay
- **Larger memory**: 20,000 experience replay buffer
- **Faster updates**: Target network updates every 50 steps (instead of 100)
- **Higher learning rate**: 0.001 for faster convergence

The agent gradually improves through exploration (epsilon-greedy) and learns from experience replay with a target network for stability.

