"""One-off diagnostic: does RAGAS faithfulness measure grounding, or verb mood?

Rewrites the 9 past-tense proposed_fix answers into imperative mood WITHOUT
changing any factual content, so faithfulness can be rescored on a paired
before/after basis with contexts held identical. If scores jump, the metric was
measuring mood; if they don't, 0.698 reflects a real grounding gap.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.llm import call_llm_with_tools

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "eval" / "results" / "ragas_inputs.json"
OUT = ROOT / "eval" / "results" / "mood_normalized_answers.json"

SCHEMA = [{"type": "function", "function": {
    "name": "record_rewrite",
    "description": "Record the mood-normalised rewrite.",
    "parameters": {"type": "object", "properties": {
        "rewritten": {"type": "string", "description": "The answer in imperative mood."}},
        "required": ["rewritten"]}}}]

PROMPT = """Rewrite the following remediation text from past-tense report form into imperative instruction form.

STRICT RULES:
- Change ONLY grammatical mood/tense. Preserve every fact, number, unit, threshold, setting name, and ordering exactly.
- Do not add, remove, generalise, or reorder any step.
- Do not add commentary.
Example: "We rotated the logs and set retention to 5 files." -> "Rotate the logs and set retention to 5 files."

Text:
{answer}

Call record_rewrite with the rewritten text."""

def is_past(a):
    a = a.strip()
    return a.lower().startswith("we ") or " we " in a[:80].lower()

def main():
    rows = json.loads(INPUTS.read_text(encoding="utf-8"))
    targets = [r for r in rows if is_past(r["answer"])]
    print(f"{len(targets)} past-tense answers to normalise")
    out = []
    for r in targets:
        resp = call_llm_with_tools(
            [{"role": "user", "content": PROMPT.format(answer=r["answer"])}], SCHEMA,
            tool_choice={"type": "function", "function": {"name": "record_rewrite"}})
        if isinstance(resp, dict):
            print(f"  {r['ticket_id']}: ERROR {resp}"); continue
        tc = resp.choices[0].message.tool_calls
        if not tc:
            print(f"  {r['ticket_id']}: no tool call"); continue
        new = json.loads(tc[0].function.arguments)["rewritten"]
        out.append({**r, "answer_original": r["answer"], "answer": new})
        print(f"  {r['ticket_id']}: {len(r['answer'])} -> {len(new)} chars")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} rows)")

if __name__ == "__main__":
    main()
