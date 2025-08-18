import os, httpx

def complete(model: str, prompt: str, timeout=20) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    url = "https://api.anthropic.com/v1/messages"
    payload = {"model": model, "max_tokens": 512, "messages": [{"role":"user","content": prompt}]}
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        text = "".join([b.get("text", "") for b in data.get("content", [])])
        return {"text": text, "usage": {"prompt_tokens": usage.get("input_tokens",0), "completion_tokens": usage.get("output_tokens",0)}}