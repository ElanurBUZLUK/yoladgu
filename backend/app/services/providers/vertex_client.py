import os, httpx

def complete(model: str, prompt: str, timeout=20) -> dict:
    api_url = os.getenv("VERTEX_PROXY_URL")
    api_key = os.getenv("VERTEX_API_KEY")
    if not api_url or not api_key:
        raise RuntimeError("VERTEX config missing")
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{api_url}/chat", json={"model": model, "prompt": prompt}, headers={"X-API-Key": api_key})
        r.raise_for_status()
        data = r.json()
        return {"text": data["text"], "usage": data.get("usage", {"prompt_tokens":0, "completion_tokens":0})}