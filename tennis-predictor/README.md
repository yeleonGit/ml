# Wimbledon Match Prediction

Predicts tennis match outcomes using both classical ML and deep learning, trained on 25 years of ATP match data (2000–2025).

## Data

- **Source:** Historical men's tennis match data from yearly Excel files (`.xls`/`.xlsx`), 2000–2025
- **Size:** ~66k matches, transformed into 132k player-centric rows
- **Features:** Ranks, betting odds (from multiple bookmakers), surface, tournament, and engineered features like rank difference/ratio and odds ratios

## Models

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Random Forest (baseline) | 82.33% | 82.69% | 81.98% |
| **Random Forest (tuned)** | **82.99%** | **83.03%** | **83.11%** |
| Deep Neural Network | 82.42% | 80.96% | 84.97% |
| **Hybrid Stacking (RF + DNN)** | **82.98%** | **83.02%** | **83.11%** |

Betting odds were the most influential features across all models.

## Approach

1. Load and clean 25 years of match data
2. Transform into player-centric format (one row per player per match)
3. Engineer features (rank difference/ratio, odds ratios)
4. Train and tune Random Forest via `GridSearchCV`
5. Build a Deep Neural Network (TensorFlow/Keras)
6. Combine both into a hybrid stacking model with Logistic Regression meta-learner
7. Evaluate with accuracy, precision, recall, ROC curves, confusion matrices, and feature importance

## Libraries

`pandas`, `numpy`, `scikit-learn`, `tensorflow`/`keras`, `matplotlib`, `seaborn`

## Usage

Open `wimbledon-project.ipynb` in Jupyter and run all cells.
