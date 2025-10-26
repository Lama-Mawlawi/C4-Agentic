"""
LangGraph ReAct Agent for OrbitCoach — auto‑ingests `about_business.pdf` + `summary.txt`
--------------------------------------------------------------------------------------
This single file:
  • Loads your business summary and PDF on startup (no hardcoding).
  • Builds a ReAct loop in LangGraph with two personas.
  • Adds a `kb_search` tool that retrieves snippets from your PDF/TXT so answers stay on‑brand.
  • Includes a tiny experiment harness.

Run:
  pip install langgraph langchain-openai pydantic rich python-dotenv pypdf
  export OPENAI_API_KEY=sk-...
  python app.py --persona friendly
  # or
  python app.py --run-experiment

Place files next to this script:
  ./summary.txt
  ./about_business.pdf
"""
from __future__ import annotations
import argparse
import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# LangGraph / LLM
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# Load environment variables from .env (if present)
load_dotenv()

# -----------------------------
# Debug/streaming controls for ReAct
# -----------------------------
STREAM_REACT: bool = False

def _dbg(msg: str):
    if STREAM_REACT:
        # Use plain stdout to avoid buffering issues; flush to show immediately
        print(f"[ReAct] {msg}", flush=True)

# -----------------------------
# Load business docs
# -----------------------------

def load_summary(path: str = "summary.txt") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "(No summary.txt found — add your executive summary here.)"


def load_pdf_text(path: str = "about_business.pdf") -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception:
        return "(No about_business.pdf found or failed to read.)"

BUSINESS_SUMMARY = load_summary()
BUSINESS_PDF_TEXT = load_pdf_text()

# Try to guess some facts from the files (fallback defaults)
DEFAULT_FACTS = {
    "name": "OrbitCoach",
    "mission": "Personalized, AI‑assisted tutoring that blends human expertise with data‑driven planning.",
    "services": [
        "1‑to‑1 tutoring in Math, Physics, CS",
        "Micro‑sessions (10–20 min) for quick boosts",
        "Exam prep packs and progress dashboards",
    ],
    "policies": {
        "privacy": "Student privacy first; only necessary data retained.",
        "cancellation": "Cancel ≥12h in advance for full refund.",
    },
    "contact": {"email": "lamamawlawi9@gmail.com", "phone": "+961 71751108", "location": "Hamra, Beirut"},
}

# -----------------------------
# Simple retrieval over your PDF/TXT
# -----------------------------
class ToolResult(BaseModel):
    name: str
    content: str


def chunk_text(text: str, max_chars: int = 700) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    count = 0
    for para in (text or "").splitlines():
        para = para.strip()
        if not para:
            continue
        if count + len(para) > max_chars and buf:
            parts.append(" ".join(buf))
            buf, count = [], 0
        buf.append(para)
        count += len(para) + 1
    if buf:
        parts.append(" ".join(buf))
    return parts or [text]


KB_CHUNKS = chunk_text(BUSINESS_SUMMARY + "\n\n" + BUSINESS_PDF_TEXT, max_chars=700)


def tool_kb_search(query: str, k: int = 3) -> ToolResult:
    q = (query or "").lower()
    scored: List[Tuple[int, str]] = []
    for ch in KB_CHUNKS:
        score = sum(ch.lower().count(tok) for tok in q.split())
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:k] if s > 0] or [KB_CHUNKS[0]]
    content = "\n---\n".join(top)
    return ToolResult(name="kb_search", content=content)


# Additional mock business tools you can extend later

def tool_price_estimate(hours: float, modality: Literal["online", "in-person"] = "online") -> ToolResult:
    base = 18.0 if modality == "online" else 22.0
    total = round(base * max(1.0, float(hours)), 2)
    return ToolResult(name="price_estimate", content=f"Estimated price for {hours}h {modality}: ${total} (base ${base}/h)")


def tool_schedule_lookup(subject: str, when: Optional[str] = None) -> ToolResult:
    slots = {
        "Math": ["Mon 17:00-18:00", "Wed 18:00-19:00", "Sat 11:00-12:00"],
        "Physics": ["Tue 16:00-17:00", "Thu 19:00-20:00"],
        "CS": ["Mon 19:00-20:00", "Fri 17:00-18:00"],
    }
    s = subject.strip().title()
    options = slots.get(s, [])
    if when:
        options = [x for x in options if when.lower() in x.lower()]
    msg = f"Open slots for {s}: " + (", ".join(options) if options else "No slots match.")
    return ToolResult(name="schedule_lookup", content=msg)


TOOLS = {
    "kb_search": tool_kb_search,
    "price_estimate": tool_price_estimate,
    "schedule_lookup": tool_schedule_lookup,
}

# -----------------------------
# ReAct parsing utilities
# -----------------------------
ACTION_RE = re.compile(r"Action\s*:\s*(?P<tool>[a-zA-Z_]+)\s*\nAction Input\s*:\s*(?P<input>.+)")


def extract_action(text: str) -> Optional[Tuple[str, str]]:
    m = ACTION_RE.search(text)
    if m:
        return m.group("tool"), m.group("input").strip()
    return None


# -----------------------------
# LangGraph State
# -----------------------------
class AgentState(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    scratchpad: str = ""
    done: bool = False
    # Structured trace of each ReAct cycle
    trace: List[Dict[str, Any]] = Field(default_factory=list)


def _ensure_agent_state(s: Any) -> AgentState:
    if isinstance(s, AgentState):
        return s
    if isinstance(s, dict):
        # Be tolerant to missing keys
        return AgentState(
            messages=s.get("messages", []),
            scratchpad=s.get("scratchpad", ""),
            done=bool(s.get("done", False)),
        )
    # Fallback to empty state if unexpected type
    return AgentState()


# -----------------------------
# System prompts / personas
# -----------------------------
BASE_SYSTEM = f"""
You are OrbitCoach’s official assistant.
Use the company’s real documentation when answering.

Company Summary (from summary.txt):
{BUSINESS_SUMMARY}

You can retrieve details from the docs using the kb_search tool.
If a user asks about offerings, policies, pricing, or differentiators, prefer kb_search first.

Available tools and signatures:
- kb_search(query: str, k?: int)
- price_estimate(hours: float, modality: "online"|"in-person")
- schedule_lookup(subject: str, when?: str)

Follow **ReAct** strictly:
1) Think briefly.
2) When you need a tool, emit exactly:
   Action: <tool_name>\nAction Input: <JSON or plain text>
3) Wait for Observation, then continue reasoning.
4) Finish with a clear, student-friendly answer grounded in the docs.
"""

PERSONAS = {
    "friendly": "Tone: Warm, encouraging, student-first, explain simply, add next steps.",
    "strict": "Tone: Concise, rigorous, prioritize accuracy, state constraints and assumptions.",
}

FEW_SHOT_PROMPT = """
You are OrbitCoach's assistant. Use the ReAct pattern exactly as shown.

EXAMPLE 1
User: What’s your privacy policy?
Thought: I should look that up.
Action: kb_search
Action Input: {"query":"privacy policy"}
Observation: (doc snippet)
Answer: (short grounded summary)

EXAMPLE 2
User: Estimate price for 2h in-person Physics.
Thought: I should check the price.
Action: price_estimate
Action Input: {"hours":2, "modality":"in-person"}
Observation: Estimated price for 2.0h in-person: $44.0 (base $22.0/h)
Answer: It costs about $44 for 2h in-person.
""".strip()

# -----------------------------
# LLM
# -----------------------------

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


# -----------------------------
# Graph nodes
# -----------------------------

def node_reason(state: AgentState, llm: ChatOpenAI, persona: str) -> AgentState:
    msgs = [
        {"role": "system", "content": BASE_SYSTEM + "\n" + PERSONAS.get(persona, "")},
        *state.messages,
    ]
    if state.scratchpad:
        msgs.append({"role": "system", "content": f"Scratchpad so far:\n{state.scratchpad}"})
    _dbg("reason: calling LLM…")
    resp = llm.invoke(msgs)
    content = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
    _dbg(f"reason: model output -> {content[:300]}{'…' if len(content) > 300 else ''}")
    state.scratchpad += f"\n{content}\n"
    # Record model output as a new ReAct step
    state.trace.append({"model": content})
    if "Action:" not in content:
        state.messages.append({"role": "assistant", "content": content})
        state.done = True
        _dbg("reason: no Action detected → finalizing")
    return state


def node_maybe_act(state: AgentState) -> AgentState:
    if state.done:
        return state
    # Look at the latest scratchpad chunk for an action
    chunk = "\n".join(state.scratchpad.splitlines()[-12:])
    parsed = extract_action(chunk)
    if not parsed:
        state.done = True
        _dbg("maybe_act: no Action parsed → done")
        return state
    tool_name, tool_input = parsed
    _dbg(f"maybe_act: Action parsed → {tool_name} | input: {tool_input}")
    tool = TOOLS.get(tool_name)
    if not tool:
        observation = f"Tool '{tool_name}' not found."
    else:
        # Try JSON input first
        try:
            data = json.loads(tool_input)
            if isinstance(data, dict):
                observation = tool(**data).content  # type: ignore
            else:
                observation = tool(data).content  # type: ignore
        except Exception:
            # Fallback: positional/plain input
            try:
                observation = tool(tool_input).content  # type: ignore
            except Exception as e:
                observation = f"Error executing tool: {e}"
    _dbg(f"maybe_act: Observation -> {observation[:300]}{'…' if len(observation) > 300 else ''}")
    # Attach action and observation to the most recent trace step
    if state.trace:
        state.trace[-1].update({
            "action": tool_name,
            "input": tool_input,
            "observation": observation,
        })
    else:
        state.trace.append({
            "action": tool_name,
            "input": tool_input,
            "observation": observation,
        })
    # Use a simple system message to carry tool output.
    # Avoid LangChain ToolMessage which requires a tool_call_id.
    state.messages.append({"role": "system", "content": f"Observation ({tool_name}): {observation}"})
    # Nudge the model to finalize instead of looping with more tools
    state.messages.append({
        "role": "system",
        "content": "You now have the necessary information from tools. Do not call any tools or emit 'Action:' again. Provide the final, concise answer to the user."
    })
    state.scratchpad += f"\nObservation: {observation}\n"
    return state


# -----------------------------
# Build the graph
# -----------------------------

def build_graph(llm: ChatOpenAI, persona: str = "friendly"):
    graph = StateGraph(AgentState)
    graph.add_node("reason", lambda s: node_reason(s, llm=llm, persona=persona))
    graph.add_node("maybe_act", node_maybe_act)
    graph.set_entry_point("reason")
    graph.add_edge("reason", "maybe_act")
    graph.add_conditional_edges("maybe_act", lambda s: END if s.done else "reason")
    return graph.compile()


# -----------------------------
# CLI / Experiment harness
# -----------------------------
@dataclass
class RunConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    persona: str = "friendly"
    max_turns: int = 6
    fewshot: bool = False



def run_interactive(cfg: RunConfig):
    console = Console()
    llm = build_llm(cfg.model, cfg.temperature)
    app = build_graph(llm, persona=cfg.persona)
    state = AgentState(messages=[])
    console.print(f"[bold]OrbitCoach ReAct (persona={cfg.persona}, model={cfg.model}, T={cfg.temperature})[/bold]")
    console.print("Type 'quit' to exit.\n")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        #state.messages.append({"role": "user", "content": user})
        if cfg.fewshot and not state.messages:
             # Inject your examples before the task, as one single user message
            user = FEW_SHOT_PROMPT + "\n\nTASK: " + user
        state.messages.append({"role": "user", "content": user})

        steps = 0
        while not state.done and steps < cfg.max_turns:
            out = app.invoke(state)
            state = _ensure_agent_state(out)
            steps += 1
        # If streaming is enabled, print a structured ReAct trace before final answer
        if STREAM_REACT and state.trace:
            print("\n=== ReAct Trace ===", flush=True)
            for idx, step in enumerate(state.trace, start=1):
                print(f"Step {idx}", flush=True)
                if (model_out := step.get("model")):
                    preview = model_out if len(model_out) <= 800 else model_out[:800] + "…"
                    print(f"  Thought/Model: {preview}", flush=True)
                if (act := step.get("action")):
                    print(f"  Action: {act}", flush=True)
                    print(f"  Action Input: {step.get('input')}", flush=True)
                if (obs := step.get("observation")):
                    obs_prev = obs if len(obs) <= 800 else obs[:800] + "…"
                    print(f"  Observation: {obs_prev}", flush=True)
            print("=== End Trace ===\n", flush=True)

        last_assistant = next((m for m in reversed(state.messages) if m["role"]=="assistant"), None)
        if last_assistant:
            console.print(f"[green]\nAssistant:[/green] {last_assistant['content']}\n")
        # reset for next question
        state.done = False
        state.scratchpad = ""
        state.trace = []
        state.messages = [m for m in state.messages if m["role"] == "assistant"][-6:]


def run_experiment():
    prompts = [
        "Give me a 2-week Algebra catch-up plan grounded in OrbitCoach offerings.",
        "What is your cancellation policy?",
        "How are you different from typical tutoring centers?",
        "Estimate price for 1.5h in-person CS, and suggest a cheaper option.",
    ]
    configs = [
        RunConfig(persona="friendly", model="gpt-4o-mini", temperature=0.7),
        RunConfig(persona="strict", model="gpt-4o-mini", temperature=0.2),
    ]
    rows: List[Tuple[str, str, str]] = []
    for cfg in configs:
        llm = build_llm(cfg.model, cfg.temperature)
        app = build_graph(llm, persona=cfg.persona)
        for p in prompts:
            state = AgentState(messages=[{"role": "user", "content": p}])
            turns = 0
            while not state.done and turns < cfg.max_turns:
                out = app.invoke(state)
                state = _ensure_agent_state(out)
                turns += 1
            answer = next((m["content"] for m in reversed(state.messages) if m["role"]=="assistant"), "(no answer)")
            rows.append((f"{cfg.persona}", f"T={cfg.temperature}", textwrap.shorten(answer, width=160)))

    console = Console()
    table = Table(title="Experiment: Persona x Temperature")
    table.add_column("Persona")
    table.add_column("Config")
    table.add_column("Answer (truncated)")
    for r in rows:
        table.add_row(*r)
    console.print(table)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=["friendly", "strict"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--stream-react", action="store_true", help="Print ReAct steps (model, action, observation)")
    parser.add_argument("--run-experiment", action="store_true")
    parser.add_argument("--fewshot", action="store_true", help="Prepend few-shot ReAct examples to the first user turn")

    args = parser.parse_args()

    if args.run_experiment:
        run_experiment()
        return

    global STREAM_REACT
    STREAM_REACT = bool(args.stream_react or os.getenv("STREAM_REACT"))

    persona = args.persona or "friendly"
    cfg = RunConfig(model=args.model, temperature=args.temperature, persona=persona, fewshot=bool(args.fewshot))
    run_interactive(cfg)



if __name__ == "__main__":
    main()
