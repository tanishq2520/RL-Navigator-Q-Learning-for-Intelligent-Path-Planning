# 🤖 RL Navigator: Q-Learning for Intelligent Path Planning

## 📌 Overview

This project implements a **Q-learning based reinforcement learning agent** for obstacle-aware path planning in a grid-based environment.
The agent learns to navigate from a start position to a goal while avoiding obstacles and optimizing path efficiency.

---

## 🚀 Key Features

* 🧠 Q-learning based autonomous agent
* 📍 Grid-world environment with obstacles
* 🎯 Reward system encouraging shortest safe paths
* 📊 Training visualization and convergence analysis
* ⚡ Lightweight and easily extendable

---

## 🧠 How It Works

The agent interacts with the environment and learns an optimal policy using Q-learning:

* **State**: Current position in grid
* **Actions**: Move (Up, Down, Left, Right)
* **Reward**:

  * Positive reward for reaching goal
  * Negative penalty for collisions
  * Small penalty per step to encourage efficiency

Over time, the agent updates its Q-table and converges to an optimal path.

---

## 🏗️ Project Structure

```
├── train_qlearning.py     # Training script
├── environment.py         # Grid environment
├── utils/                 # Helper functions
├── results/               # Plots and outputs
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/rl-navigator.git
cd rl-navigator
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python train_qlearning.py
```

---

## 📈 Results

* Agent successfully learns optimal paths avoiding obstacles
* Convergence observed through decreasing episode cost
* Visualization demonstrates policy improvement over time

---

## 🛠️ Tech Stack

* Python 🐍
* Gymnasium
* NumPy
* Matplotlib

---

## 🔮 Future Improvements

* Deep Q-Network (DQN) implementation
* Dynamic obstacles
* Multi-agent path planning
* Real-time visualization

---

## 📌 Author

Tanishq
B.Tech CSE | Reinforcement Learning Enthusiast

---
