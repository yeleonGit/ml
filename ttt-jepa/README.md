# Tic-Tac-Toe JEPA

A **Joint Embedding Predictive Architecture (JEPA)** that learns to play Tic-Tac-Toe by predicting future latent states — no reward signal needed. The model builds an internal "imagination" of how the game evolves.

## Key Result

**100% win rate** against a depth-limited Minimax expert over 20 games.

## How It Works

Instead of learning from rewards (like RL) or reconstructing pixels, JEPA learns a latent space where it predicts the next board state. Given a sequence of moves, the encoder maps each board to a latent representation, and the predictor learns to forecast the next latent state.

### Components

- **Encoder:** Maps board states to latent embeddings
- **Predictor:** Predicts the next latent state from the current one and an action
- **Loss:** VicReg-based — invariance (similar states close), variance (prevent collapse), covariance (decorrelate latent dimensions)

## Features

- Self-play data generation for training
- Interactive `ipywidgets` GUI to play against the JEPA agent
- PCA visualization of learned latent space
- Nearest-neighbor decoding to "visualize" predictions

## Results

| Opponent | JEPA Wins | Draws | Losses |
|---|---|---|---|
| Depth-limited Minimax Expert | 20 (100%) | 0 | 0 |

## Libraries

`PyTorch`, `scikit-learn` (PCA), `matplotlib`, `ipywidgets`

## Usage

Open `ttt_jepa.ipynb` in Jupyter and run all cells. The interactive GUI is at the end of the notebook.
