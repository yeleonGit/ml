import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# --- Data ---
DATA_PATH = "data/dataset.txt"         # Path to your processed training text
TRAIN_SPLIT = 0.9                      # 90% train, 10% validation

# --- Tokenizer ---
# Character-level tokenizer — no external libraries needed
# Vocab size is set automatically from your dataset in tokenizer.py

# --- Model Architecture ---
BLOCK_SIZE = 256        # How many characters the model sees at once (context window)
N_EMBD = 384            # Size of each token's embedding vector
N_HEAD = 6              # Number of attention heads (N_EMBD must be divisible by N_HEAD)
N_LAYER = 6             # Number of transformer blocks stacked
DROPOUT = 0.2           # Randomly zeroes 20% of activations to prevent overfitting

# --- Training ---
BATCH_SIZE = 32         # Number of sequences processed in parallel
MAX_ITERS = 5000        # Total training steps (increase for better results)
EVAL_INTERVAL = 500     # How often to print train/val loss
EVAL_ITERS = 200        # How many batches to average for loss estimation
LEARNING_RATE = 3e-4    # Adam optimizer learning rate

# --- Generation ---
MAX_NEW_TOKENS = 300    # Max tokens to generate per prompt
TEMPERATURE = 0.8       # Higher = more creative, Lower = more conservative
TOP_K = 50              # Only sample from the top K most likely next tokens

# --- Checkpointing ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = "checkpoints/model.pt"
