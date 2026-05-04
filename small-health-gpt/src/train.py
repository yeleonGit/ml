import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PATH, DEVICE, MAX_ITERS, EVAL_INTERVAL,
    EVAL_ITERS, LEARNING_RATE, CHECKPOINT_PATH, CHECKPOINT_DIR
)
from tokenizer import CharTokenizer
from dataset import load_data, get_batch
from transformer import TinyGPT

@torch.no_grad()

def estimate_loss(model, train_data, val_data):
    model.eval()
    losses = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train()
    return losses

def main():
    print(f"Small Health GPT - Training")
    print(f"   Device: {DEVICE}")
    print(f"   Max iterations: {MAX_ITERS:,}")
    print()

    # --- 1. Load and tokenize data ---
    tokenizer = CharTokenizer()
    train_data, val_data = load_data(DATA_PATH, tokenizer)
    tokenizer.save()   # Save vocab for use during generation

    # --- 2. Initialize model ---
    model = TinyGPT(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model.count_parameters()
    print()

    # --- 3. Set up optimizer ---
    # AdamW is Adam with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training
    print("Starting training...\n")
    start_time = time.time()

    for iteration in range(MAX_ITERS):

        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start_time
            print(
                f"Step {iteration:5d} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"time: {elapsed:.0f}s"
            )

        x, y = get_batch("train", train_data, val_data)

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        optimizer.step()

    # --- 5. Save checkpoint ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "iterations": MAX_ITERS,
        "final_train_loss": losses["train"],
        "final_val_loss": losses["val"],
    }, CHECKPOINT_PATH)

    print(f"Training complete!")
    print(f"   Final train loss: {losses['train']:.4f}")
    print(f"   Final val loss:   {losses['val']:.4f}")
    print(f"   Checkpoint saved: {CHECKPOINT_PATH}")
    print(f"\nNext step: python src/generate.py")


if __name__ == "__main__":
    main()