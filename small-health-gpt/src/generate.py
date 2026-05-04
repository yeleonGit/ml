import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, CHECKPOINT_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_K
from tokenizer import CharTokenizer
from transformer import TinyGPT


def top_k_sample(logits, k: int, temperature: float):
    """
    Apply temperature scaling and top-k filtering, then sample.

    Args:
        logits:      (vocab_size,) raw scores from model
        k:           number of top tokens to consider
        temperature: controls randomness

    Returns:
        sampled token index (integer)
    """
    # Scale logits by temperature
    logits = logits / temperature

    # Zero out all logits except the top-k
    if k > 0:
        top_k_values, _ = torch.topk(logits, min(k, logits.size(-1)))
        threshold = top_k_values[-1]   # The k-th largest value
        logits[logits < threshold] = float("-inf")

    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int):
    """
    Generate text from a prompt using the trained model.

    Args:
        model:          trained TinyGPT instance
        tokenizer:      fitted CharTokenizer instance
        prompt:         starting text string
        max_new_tokens: how many characters to generate
        temperature:    sampling temperature
        top_k:          top-k filtering value

    Returns:
        generated string (prompt + continuation)
    """
    from config import BLOCK_SIZE

    model.eval()

    # Encode prompt to integer tensor
    encoded = tokenizer.encode(prompt)
    context = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)

    generated = list(encoded)   # Start with prompt tokens

    for _ in range(max_new_tokens):
        # Crop context to block_size (model can't see more than this at once)
        context_crop = context[:, -BLOCK_SIZE:]

        # Forward pass — only need logits for the LAST position
        logits, _ = model(context_crop)
        next_logits = logits[0, -1, :]   # (vocab_size,)

        # Sample next token
        next_token = top_k_sample(next_logits, k=top_k, temperature=temperature)

        # Append to sequence
        generated.append(next_token)
        context = torch.cat([
            context,
            torch.tensor([[next_token]], device=DEVICE)
        ], dim=1)

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text with TinyGPT Medical")
    parser.add_argument("--prompt", type=str, default="Question: What is hypertension?\nAnswer:",
                        help="Starting prompt for generation")
    parser.add_argument("--tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Sampling temperature (0.1-1.5)")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help="Top-k sampling filter")
    args = parser.parse_args()

    print(f"\n🏥 TinyGPT Medical — Generation")
    print(f"   Device: {DEVICE}")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print()

    # Load tokenizer
    tokenizer = CharTokenizer()
    tokenizer.load()

    # Load model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = TinyGPT(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded — trained for {checkpoint['iterations']:,} iterations")
    print(f"Final val loss: {checkpoint['final_val_loss']:.4f}")
    print()
    print("=" * 60)
    print("PROMPT:", args.prompt)
    print("=" * 60)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(output)
    print("=" * 60)
    print(f"\n⚠️  Disclaimer: This model is for learning purposes only.")
    print("   Do not use for actual medical advice.")


if __name__ == "__main__":
    main()
