import os, httpx

def complete(model: str, prompt: str, timeout=20) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "usage": {"prompt_tokens": usage.get("prompt_tokens",0), "completion_tokens": usage.get("completion_tokens",0)}}
