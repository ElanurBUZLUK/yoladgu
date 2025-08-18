import os, httpx

def complete(model: str, prompt: str, timeout=20) -> dict:
    url = os.getenv("VLLM_URL", "http://vllm:8001")
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{url}/generate", json={"model": model, "prompt": prompt})
        r.raise_for_status()
        data = r.json()
        return {"text": data.get("text", ""), "usage": data.get("usage", {"prompt_tokens":0, "completion_tokens":0})}