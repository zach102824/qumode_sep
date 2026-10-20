"""Minimal OpenAI-compatible chat-completions adapter -> Cursor Agent CLI (print mode).
Key: read by cursor-agent from env CURSOR_API_KEY (never logged). Bind 127.0.0.1 only.
Run: python3.13 cursor_adapter.py --port 8765 --model composer-2.5
"""
import argparse, asyncio, os, shutil, tempfile, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--model", default="composer-2.5")
ap.add_argument("--conc", type=int, default=2); ap.add_argument("--timeout", type=int, default=600)
A = ap.parse_args()
BIN = shutil.which("cursor-agent") or os.path.expanduser("~/.local/bin/cursor-agent")
SEM = asyncio.Semaphore(A.conc); app = FastAPI()

def render(msgs):
    out = []
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, list): c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append(f"### {m.get('role','user').upper()}\n{c}")
    out.append("### INSTRUCTION\nAnswer the conversation above directly as plain text in your reply. Do NOT use tools, do NOT read or edit files, do NOT run commands.")
    return "\n\n".join(out)

async def run_agent(prompt, model):
    d = tempfile.mkdtemp(prefix="ca_")   # empty sandbox cwd: agent sees no repo files
    try:
        p = await asyncio.create_subprocess_exec(BIN, "-p", "--trust", "--model", model, "--output-format", "text", prompt,
                                                 cwd=d, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=os.environ.copy())
        o, e = await asyncio.wait_for(p.communicate(), A.timeout)
        if p.returncode != 0: raise RuntimeError(f"cursor-agent rc={p.returncode}: {e.decode()[-300:]}")
        return o.decode()
    finally: shutil.rmtree(d, ignore_errors=True)

@app.get("/v1/models")
async def models(): return {"object": "list", "data": [{"id": A.model, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    b = await req.json(); model = b.get("model") or A.model
    if "fable" in model: print(f"[adapter] remap {model} -> claude-opus-5-5-high (Opus-only)", flush=True); model = "claude-opus-5-5-high"
    if model.startswith("gpt-") and model not in ("gpt-5.2",) and "codex" not in model and "5.6" not in model: model = A.model  # map unknown openai names
    t = time.time()
    async with SEM:
        try: txt = await run_agent(render(b.get("messages", [])), model)
        except Exception as ex: return JSONResponse({"error": {"message": str(ex)}}, status_code=500)
    print(f"[adapter] {model} {time.time()-t:.1f}s out={len(txt)}ch", flush=True)
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=A.port, log_level="warning")
