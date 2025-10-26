# OrbitCoach ReAct Agent – C4 Assignment

## 📘 Overview
This project implements a **ReAct (Reasoning + Acting)** agent for **OrbitCoach Tutors**,  
a personalized tutoring platform that blends human expertise with AI.  
The agent is built using **LangGraph**, integrating multiple tools and distinct personas  
to simulate reasoning, tool usage, and adaptive tone.

The system automatically loads company data from:
- `about_business.pdf` – detailed company profile  
- `summary.txt` – executive summary  

This ensures answers are grounded in real OrbitCoach information.

---

## 🧠 Core Features

| Feature | Description |
|----------|-------------|
| **Framework** | Built with [LangGraph](https://github.com/langchain-ai/langgraph) for explicit ReAct control. |
| **ReAct Logic** | Manual reasoning–action–observation loop implemented with states and conditional edges. |
| **Personas** | Two modes: `friendly` (warm, motivational) and `strict` (concise, precise). |
| **Tools** | `kb_search`, `price_estimate`, and `schedule_lookup` for retrieval, pricing, and scheduling. |
| **Document Grounding** | The `kb_search` tool retrieves snippets from company documents (PDF/TXT). |
| **Experiments** | CLI harness for temperature, persona, and prompt-style testing. |

---

## ⚙️ How It Works
1. **Reasoning Step:** The model “thinks” about what it needs to answer.  
2. **Action Step:** If a tool is required, it emits  
3. **Observation Step:** The tool executes, returns results, and feeds them back to the model.  
4. **Answer Step:** The model finalizes its grounded answer.

This flow repeats until the agent marks itself `done = True`.

---

## 🧩 Tools

| Tool | Purpose |
|------|----------|
| `kb_search(query)` | Retrieves relevant chunks from company PDF/TXT for factual grounding. |
| `price_estimate(hours, modality)` | Calculates cost estimates for tutoring sessions. |
| `schedule_lookup(subject, when)` | Finds next available session slots. |

---

## 🧪 How to Run

### 1️⃣ Setup
```bash
git clone https://github.com/Lama-Mawlawi/C4-Agentic
cd C4-Agentic
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -U pip
pip install langgraph langchain-openai pydantic rich python-dotenv pypdf

## Run the Agent
python app.py --persona friendly
python app.py --persona strict

##Trace React 
python app.py --persona strict --stream-react

 ## Example ReAct Trace
[ReAct] reason: calling LLM…
Thought: I need to check the tutoring services.
Action: kb_search
Action Input: {"query":"top services"}
Observation: OrbitCoach Tutors blends human expertise with AI to deliver fast, effective tutoring...
Answer: OrbitCoach offers 1-to-1 tutoring, micro-sessions, personalized practice, and progress tracking.

