import torch
from config import BLOCK_SIZE, BATCH_SIZE, DEVICE, TRAIN_SPLIT

def load_data(data_path: str, tokenizer):
    """
    Read the dataset file, encode it to integers, and split into train/val sets.

    Returns:
        train_data: torch.Tensor of encoded training text
        val_data:   torch.Tensor of encoded validation text
    """
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Dataset loaded: {len(text):,} characters")

    # Build vocabulary from this text
    tokenizer.build_vocab(text)

    # Encode entire dataset as a 1D tensor of integers
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Split: first 90% is training, last 10% is validation
    split_idx = int(TRAIN_SPLIT * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    return train_data, val_data


def get_batch(split: str, train_data, val_data):
    """
    Sample a random batch of (input, target) pairs from the dataset.

    Args:
        split: "train" or "val"
        train_data, val_data: tensors from load_data()

    Returns:
        x: input tensor of shape  (BATCH_SIZE, BLOCK_SIZE)
        y: target tensor of shape (BATCH_SIZE, BLOCK_SIZE)
           y is x shifted right by 1 position
    """
    data = train_data if split == "train" else val_data

    start_positions = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    # Stack into tensors
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in start_positions])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in start_positions])

    # Move to the right device (MPS/CUDA/CPU)
    x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y
