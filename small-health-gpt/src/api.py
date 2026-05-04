import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from config import DEVICE, TEMPERATURE, TOP_K
from tokenizer import CharTokenizer
from transformer import TinyGPT
from generate import generate as run_generate


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_SRC_DIR, "checkpoints", "model.pt")
TOKENIZER_PATH = os.path.join(_SRC_DIR, "checkpoints", "tokenizer.json")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=160, ge=1, le=2048)
    temperature: float = Field(default=TEMPERATURE, ge=0.05, le=2.0)
    top_k: int = Field(default=TOP_K, ge=0, le=512)


tokenizer = CharTokenizer()
tokenizer.load(TOKENIZER_PATH)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model = TinyGPT(vocab_size=tokenizer.vocab_size).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


@app.get("/")
def home():
    return {
        "message": "Small Health GPT API is running",
        "checkpoint": CHECKPOINT_PATH,
        "device": DEVICE,
    }


@app.post("/generate")
def generate_endpoint(request: GenerateRequest):
    prompt = request.prompt.strip()
    if not prompt:
        return {"prompt": prompt, "response": ""}

    with torch.no_grad():
        output = run_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

    return {"prompt": prompt, "response": output}
