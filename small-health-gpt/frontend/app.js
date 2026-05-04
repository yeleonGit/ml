const promptBox = document.getElementById("prompt");
const generateBtn = document.getElementById("generateBtn");
const responseBox = document.getElementById("response");
const statusText = document.getElementById("status");

generateBtn.addEventListener("click", async () => {
  const prompt = promptBox.value.trim();

  if (!prompt) {
    responseBox.textContent = "Please enter a question first.";
    return;
  }

  generateBtn.disabled = true;
  statusText.textContent = "Generating...";
  responseBox.textContent = "";

  try {
    const res = await fetch("http://127.0.0.1:8000/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        prompt: prompt,
        max_new_tokens: 140
      })
    });

    if (!res.ok) {
      throw new Error("The local model server returned an error.");
    }

    const data = await res.json();

    responseBox.textContent = data.response;
    statusText.textContent = "Done";
  } catch (error) {
    responseBox.textContent =
      "Could not connect to the local model server. Make sure the backend is running with: python -m uvicorn src.api:app --reload";
    statusText.textContent = "Error";
  }

  generateBtn.disabled = false;
});