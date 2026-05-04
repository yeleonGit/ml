## small-health-gpt

Local character-level GPT trained on MedQuAD, with:
- FastAPI backend (`src/api.py`)
- static frontend (`frontend/index.html`)
- checkpoint inference (`src/checkpoints/model.pt`)

From the project root:

```bash
bash run.sh
```

This single command will:
- create `.venv` if missing
- install `requirements.txt`
- start backend at `http://127.0.0.1:8000`
- start frontend at `http://127.0.0.1:5173`

Open `http://127.0.0.1:5173` in your browser.

Press `Ctrl+C` once to stop both servers.