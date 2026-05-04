import json
import os

class CharTokenizer:
    def __init__(self):
        self.char_to_int = {}   # e.g. {'a': 0, 'b': 1, ...}
        self.int_to_char = {}   # e.g. {0: 'a', 1: 'b', ...}
        self.vocab_size = 0

    def build_vocab(self, text: str):
        """
        Scan all unique characters in the text and assign each an integer ID.
        Called once during training setup.
        """
        unique_chars = sorted(set(text))   # Sorted for determinism
        self.char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
        self.int_to_char = {i: ch for ch, i in self.char_to_int.items()}
        self.vocab_size = len(unique_chars)
        print(f"Vocabulary size: {self.vocab_size} unique characters")

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integers."""
        return [self.char_to_int[ch] for ch in text if ch in self.char_to_int]

    def decode(self, indices: list[int]) -> str:
        """Convert a list of integers back into a string."""
        return "".join([self.int_to_char[i] for i in indices if i in self.int_to_char])

    def save(self, path: str = "checkpoints/tokenizer.json"):
        """Save the vocabulary to disk so generation works without retraining."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"char_to_int": self.char_to_int}, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str = "checkpoints/tokenizer.json"):
        """Load a saved vocabulary from disk."""
        with open(path, "r") as f:
            data = json.load(f)
        self.char_to_int = data["char_to_int"]
        self.int_to_char = {int(v): k for k, v in self.char_to_int.items()}
        self.vocab_size = len(self.char_to_int)
        print(f"Tokenizer loaded — vocab size: {self.vocab_size}")
