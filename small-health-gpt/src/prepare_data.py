import os
import json

def download_and_prepare():
    print("(This requires: pip install datasets)")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found.")
        print("Run: pip install datasets")
        return

    dataset = load_dataset("lavita/MedQuAD", split="train")

    print(f"Loaded {len(dataset)} Q&A pairs from MedQuAD")
    print("Processing and formatting...")

    os.makedirs("data", exist_ok=True)

    output_lines = []
    skipped = 0

    for item in dataset:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()

        if not question or not answer or len(answer) < 30:
            skipped += 1
            continue

        block = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"---\n"
        )
        output_lines.append(block)

    full_text = "\n".join(output_lines)

    with open("data/dataset.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    size_mb = os.path.getsize("data/dataset.txt") / (1024 * 1024)
    print(f"Done!")
    print(f"   Saved {len(output_lines)} Q&A pairs ({skipped} skipped)")
    print(f"   File size: {size_mb:.1f} MB")
    print(f"   Output: data/dataset.txt")

if __name__ == "__main__":
    download_and_prepare()