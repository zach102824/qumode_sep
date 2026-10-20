"""Minimal OpenAI-compatible chat-completions adapter -> Cursor Agent CLI (print mode).
Key: read by cursor-agent from env CURSOR_API_KEY (never logged). Bind 127.0.0.1 only.
Run: python3.13 cursor_adapter.py --port 8765 --model claude-opus-5-5-high
"""
import argparse, asyncio, os, re, shutil, tempfile, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--model", default="cursor-grok-4.6-high")
ap.add_argument("--idea_model", default="claude-opus-5-5-medium"); ap.add_argument("--code_model", default="cursor-grok-4.6-high"); ap.add_argument("--conc", type=int, default=2); ap.add_argument("--timeout", type=int, default=600)
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

def recent_ideas(k=5):
    try:
        L_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ideas_raw.log")).read().strip().splitlines()[-k:]
        return "\n".join("- " + l.split("\t", 2)[-1][:400] for l in L_) or "- (none yet)"
    except Exception: return "- (none yet)"

def compact(msgs, cap_sys=2500, cap_user=6000):
    """Stage-1 prompt: goal (system, truncated) + user prompt with code blocks replaced by a def-signature summary."""
    sysm = "".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in msgs if m.get("role") == "system")[:cap_sys]
    user = "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in msgs if m.get("role") != "system")
    def summ(mo):
        code = mo.group(0); defs = re.findall(r"^\s*def [^\n]*", code, re.M)
        return "[code omitted; defs: " + "; ".join(d.strip()[:90] for d in defs[:20]) + "]"
    user = re.sub(r"```.*?```", summ, user, flags=re.S)[-cap_user:]
    return ("### GOAL\n" + sysm + "\n\n### CONTEXT (parent metrics, top programs, history; code omitted)\n" + user +
            "\n\n### ALREADY FAILED - BANNED (do not propose these or close variants)\n"
            "- additive dS/dB SNAP/bus correction tables (post-hoc or trained jointly)\n"
            "- free SNAP/bus tables that REPLACE the projected cost\n- residual Walsh terms on the compiler support\n"
            "- clamping / two-round pinning of variables\n- KD / distillation from the global-cost teacher\n"
            "- assignment-head / placement selectors\n- minor tweaks of width, depth, learning rate or step count\n"
            "\n### RECENT IDEAS (do NOT repeat; yours must be structurally different from these and from the parent)\n" + recent_ideas() +
            "\n\n### PICK ONE CATEGORY\nnew features / formula encodings; layer-wise angle schedules; per-clause weights chosen to compensate "
            "the dropped 3-cavity / non-neighbour terms; placement-aware outputs; loss shaping on the hard tail; multi-placement training objectives; "
            "curriculum over n; equivariant architectures; new output parameterizations of gamma/beta/w/p_v.\n"
            "Rules: frozen gate sequence/compiler/simulator untouched; no classical solving; p_v must not encode a solution.\n"
            "\n### TASK\nPropose ONE concrete, structurally new change to the network/training. Start with 'CATEGORY: <name>'. "
            "Then 3-6 short lines, plain text, no code, max ~150 words. Do NOT use tools.")

@app.get("/v1/models")
async def models(): return {"object": "list", "data": [{"id": A.model, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    b = await req.json(); model = b.get("model") or A.model
    if "fable" in model: print(f"[adapter] remap {model} -> opus-idea+grok46-code", flush=True); model = "claude-opus-5-5-high"
    t = time.time(); msgs = b.get("messages", [])
    async with SEM:
        try:
            if "opus" in model:   # two-stage: Opus idea (compact) -> grok-4.6 code (full prompt + idea)
                cp = compact(msgs); idea = (await run_agent(cp, A.idea_model)).strip()[:4000]; t1 = time.time()
                print(f"[adapter] stage1 {A.idea_model} {t1-t:.1f}s in={len(cp)}ch out={len(idea)}ch (~{(len(cp)+len(idea))//4} tok)", flush=True)
                try:
                    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ideas_raw.log"), "a") as f_:
                        f_.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{uuid.uuid4().hex[:8]}\t" + idea.replace("\n", " | ") + "\n")
                except Exception: pass
                full = render(msgs + [{"role": "user", "content": "IMPLEMENT THIS IDEA (from a senior reviewer) in your answer, following the exact output format requested above:\n" + idea}])
                txt = await run_agent(full, A.code_model)
                print(f"[adapter] stage2 {A.code_model} {time.time()-t1:.1f}s in={len(full)}ch out={len(txt)}ch (~{(len(full)+len(txt))//4} tok)", flush=True)
                model = "opus-idea+grok46-code"
            else:
                model = A.code_model; full = render(msgs); txt = await run_agent(full, model)
                print(f"[adapter] {model} {time.time()-t:.1f}s in={len(full)}ch out={len(txt)}ch", flush=True)
        except Exception as ex: return JSONResponse({"error": {"message": str(ex)}}, status_code=500)
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=A.port, log_level="warning")
