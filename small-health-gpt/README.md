# Small Health GPT

A local character-level GPT-style transformer trained on MedQuAD (medical Q&A data), served through a FastAPI backend with a static frontend.

Built to understand how transformer language models work from the ground up — inspired by Andrej Karpathy's transformer lectures.

## Architecture

- **Model:** Small character-level transformer (custom PyTorch implementation)
- **Backend:** FastAPI (`src/api.py`) — loads the trained checkpoint and serves predictions
- **Frontend:** Static HTML/JS (`frontend/index.html`) — chat-style interface

## Quick Start

From the project root:

```bash
bash run.sh
```

This single command will:
- Create a Python virtual environment (if missing)
- Install dependencies from `requirements.txt`
- Start the backend at `http://127.0.0.1:8000`
- Start the frontend at `http://127.0.0.1:5173`

Open `http://127.0.0.1:5173` in your browser.

Press `Ctrl+C` once to stop both servers.

## Project Structure

```
small-health-gpt/
├── src/
│   ├── api.py              # FastAPI server
│   ├── model.py            # Transformer model definition
│   ├── generate.py         # Text generation logic
│   ├── tokenizer.py        # Character-level tokenizer
│   └── checkpoints/
│       └── model.pt        # Trained model weights
├── frontend/
│   └── index.html          # Chat interface
├── requirements.txt
├── run.sh
└── README.md
```

## Disclaimer

For educational purposes only. Not a substitute for professional medical advice.
