"""Minimal OpenAI-compatible chat-completions adapter -> Cursor Agent CLI (print mode).
Key: read by cursor-agent from env CURSOR_API_KEY (never logged). Bind 127.0.0.1 only.
Run: python3.13 cursor_adapter.py --port 8765 --model claude-opus-5-5-high
"""
import argparse, asyncio, os, re, shutil, tempfile, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--model", default="claude-opus-5-5-high")
ap.add_argument("--idea_model", default="claude-opus-5-5-medium"); ap.add_argument("--code_model", default="cursor-grok-4.6-high-fast"); ap.add_argument("--idea_every", type=int, default=5); ap.add_argument("--run_dir", default="runs/net16r"); ap.add_argument("--conc", type=int, default=2); ap.add_argument("--timeout", type=int, default=600)
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

def recent_ideas(k=8):
    try:
        L_ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ideas_raw.log")).read().strip().splitlines()[-k:]
        return "\n".join("- " + l.split("\t", 2)[-1][:400] for l in L_) or "- (none yet)"
    except Exception: return "- (none yet)"

CATS = ["new features / formula encodings", "layer-wise angle schedules", "lost-fraction clause weights (compensate dropped 3-cavity / non-neighbour terms)",
        "placement-aware outputs", "loss shaping on the hard tail", "multi-placement training objectives", "curriculum over n",
        "sign-equivariant / literal-graph architectures", "attention / transformer over clauses",
        "physics-inspired priors (adiabatic / QAOA-limit / mean-field)", "per-cavity SNAP-level structure via Gray code",
        "meta-learning / hypernetworks", "ensembles / averaging over seeds"]
_CNT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "idea_counter.txt")
def forced_category():
    try: n = int(open(_CNT).read().strip())
    except Exception: n = 0
    open(_CNT, "w").write(str(n + 1))
    if n % 4 == 3:
        return ("### FORCED CATEGORY: WILDCARD\nPropose an idea that fits NONE of these categories: " + "; ".join(CATS) +
                ". Write 'CATEGORY: wildcard - <your own name>'.")
    c = CATS[(n - n // 4) % len(CATS)]
    return "### FORCED CATEGORY (round-robin; you MUST use this one)\n" + c + ". Write 'CATEGORY: " + c + "'."

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
            "\n\n" + forced_category() +
            "\nRules: frozen gate sequence/compiler/simulator untouched; no classical solving; p_v must not encode a solution.\n"
            "\n### TASK\nPropose ONE concrete, structurally new change to the network/training. Be bold and divergent: prefer an unusual "
            "mechanism over a safe tweak. Start with 'CATEGORY: <name>'. "
            "Then 3-6 short lines, plain text, no code, max ~150 words. Do NOT use tools.")


# ---------------- net16r mode (model name "net16r"): Grok 4.6 Fast code edits; Opus idea every --idea_every calls; repair edits ----------
HERE_ = os.path.dirname(os.path.abspath(__file__))
TECH = ["QAOA parameter transfer / concentration (angles transferred across n, fixed-angle conjecture)", "warm-start QAOA (biased initial/mixer from a relaxation, without encoding a solution)",
        "counterdiabatic / approximate adiabatic gauge potential corrections (CD-QAOA, ADAPT-style terms on the allowed gates)",
        "Walsh/Fourier compensation of the dropped non-local terms (re-weight the representable low-order terms so their effect matches)",
        "recursive / layerwise parameter fixing (INTERP / layerwise-learned schedules)", "Trotterised annealing schedule priors (linear-ramp QAOA)",
        "mean-field / product-state approximation (mean-field AOA) to set per-variable fields", "graph neural network parameter prediction for QAOA (GNN-predicted angles, edge-aware)",
        "CVaR / Gibbs objective / tail-focused losses (CVaR-QAOA, Gibbs objective)", "meta-learning of variational parameters (MAML / LSTM optimiser / hypernetwork)",
        "symmetry-aware / equivariant parameterisation (literal-flip and relabelling equivariance)", "threshold / grover-mixer style phase shaping (Th-QAOA) within the fixed gate set",
        "curriculum learning / self-paced learning on hard instances", "multi-angle QAOA (ma-QAOA: per-clause / per-term angles)"]
FAMS = ["loss", "architecture", "features", "curriculum", "placement"]
_ST = os.path.join(HERE_, "logs", "net16r_adapter_state.json")
def _st():
    try: import json as _j; return _j.load(open(_ST))
    except Exception: return {"calls": 0, "tech": 0, "repairs": {}, "batch": []}
def _sv(st):
    import json as _j; tmp = _ST + ".tmp"; _j.dump(st, open(tmp, "w")); os.replace(tmp, _ST)
def _txt(msgs): return "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in msgs)
def _best_s2():
    try: import json as _j; return float(_j.load(open(os.path.join(HERE_, A.run_dir, "best_s2.json")))["combined_score"])
    except Exception: return None
def net16r_plan(msgs):
    """returns (kind, extra_text_for_code_model, idea_prompt_or_None)."""
    import hashlib
    st = _st(); st["calls"] += 1; T = _txt(msgs)
    fam = (re.search(r"# FAMILY: (\w+)", T) or [None, "base"])[1]
    cur = re.search(r"# Current Program\s*```\w*\n(.*?)```", T, re.S); h = hashlib.sha1((cur.group(1) if cur else T[-3000:]).encode()).hexdigest()[:12]
    m2 = re.search(r"STAGE-2 n=16 selection: score (-?[\d.]+)", T); best = _best_s2()
    tgt = (re.search(r"CHECKABLE TARGET: [^\n]*", T) or [""])[0]
    if m2 and best is not None and float(m2.group(1)) < best - 0.05 and st["repairs"].get(h, 0) < 2:
        st["repairs"][h] = st["repairs"].get(h, 0) + 1; _sv(st)
        return ("repair", "REPAIR EDIT (%d/2): the current program passed the n=12 screen but lost on the n=16 selection set (score %s vs best %.3f). "
                "Do NOT add a new idea. Read its DIAGNOSTICS above (chain vs global-cost p, worst formulas, dropped-cost fraction) and make the smallest "
                "change that fixes why it loses at n=16 (transfer from n=12 to n=16, hard tail, stability). %s" % (st["repairs"][h], m2.group(1), best, tgt), None)
    if st["calls"] % A.idea_every == 0:
        forced = st.get("batch") or []
        tech = forced.pop(0) if forced else TECH[st["tech"] % len(TECH)]
        st["tech"] += 0 if forced else 1; st["batch"] = forced; _sv(st)
        ip = compact16r(msgs, fam, tech, tgt); return ("idea", None, ip)
    _sv(st)
    return ("code", "Island family of this program: %s (stay within this family). %s" % (fam, tgt), None)
def compact16r(msgs, fam, tech, tgt, cap_sys=2500, cap_user=6000):
    sysm = "".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in msgs if m.get("role") == "system")[:cap_sys]
    user = "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in msgs if m.get("role") != "system")
    user = re.sub(r"```.*?```", lambda mo: "[code omitted; defs: " + "; ".join(d.strip()[:90] for d in re.findall(r"^\s*def [^\n]*", mo.group(0), re.M)[:20]) + "]", user, flags=re.S)[-cap_user:]
    return ("### GOAL\n" + sysm + "\n\n### CONTEXT (parent metrics + DIAGNOSTICS, top programs; code omitted)\n" + user +
            "\n\n### RECENT IDEAS (do NOT repeat)\n" + recent_ideas() +
            f"\n\n### ISLAND FAMILY (fixed by the human): {fam}\n### NAMED TECHNIQUE TO ADAPT: {tech}\n"
            "Adapt THIS known technique (as published) to our fixed native cavity chain + network, inside the island family. Say which paper/method "
            "it comes from and exactly what changes in the network/training.\n"
            f"### CHECKABLE SUBGOAL\n{tgt or 'raise chain p on the 5 worst n=12 screen formulas listed in DIAGNOSTICS without lowering the median.'}\n"
            "Rules: frozen gate sequence/compiler/simulator untouched; no classical solving; p_v must not encode a solution; the global-cost pull "
            "(LAM_GLOB) is optional and never part of fitness.\n### TASK\nStart with 'TECHNIQUE: <name>'. Then 3-6 short lines, plain text, no code, max ~150 words. Do NOT use tools.")
async def ideas_batch(k=5):
    """stagnation: ask Opus for k DISTINCT named techniques; stored as the next forced techniques."""
    q = ("List %d DISTINCT, published, named techniques (not in this list: %s) that could help a network choose phases/angles for a FIXED "
         "QAOA-like circuit whose cost phase is a chain-projected (lossy) version of the global cost, to maximise ground-state probability. "
         "One per line, format '<name> - <one-line how to adapt>'. No tools." % (k, "; ".join(TECH)))
    out = await run_agent(q, A.idea_model); L_ = [l.strip("-* ").strip() for l in out.splitlines() if l.strip()][:k]
    st = _st(); st["batch"] = L_; _sv(st); return L_
@app.post("/v1/ideas_batch")
async def ideas_batch_ep(): return {"techniques": await ideas_batch()}

@app.get("/v1/models")
async def models(): return {"object": "list", "data": [{"id": A.model, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    b = await req.json(); model = b.get("model") or A.model
    if "fable" in model: print(f"[adapter] remap {model} -> opus-idea+grok46-code", flush=True); model = "claude-opus-5-5-high"
    t = time.time(); msgs = b.get("messages", [])
    async with SEM:
        try:
            if model == "net16r":
                kind, extra, ip = net16r_plan(msgs)
                if kind == "idea":
                    idea = (await run_agent(ip, A.idea_model)).strip()[:4000]; t1 = time.time()
                    print(f"[adapter] net16r idea {A.idea_model} {t1-t:.1f}s", flush=True)
                    try:
                        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ideas_raw.log"), "a") as f_:
                            f_.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{uuid.uuid4().hex[:8]}\t" + idea.replace("\n", " | ") + "\n")
                    except Exception: pass
                    extra = "IMPLEMENT THIS IDEA (from a senior reviewer) in your answer, following the exact output format requested above:\n" + idea
                full = render(msgs + [{"role": "user", "content": extra}]); txt = await run_agent(full, A.code_model)
                print(f"[adapter] net16r {kind} {A.code_model} {time.time()-t:.1f}s in={len(full)}ch out={len(txt)}ch", flush=True)
                model = f"net16r-{kind}"
            elif "opus" in model:   # two-stage: Opus idea (compact) -> grok-4.6 code (full prompt + idea)
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
