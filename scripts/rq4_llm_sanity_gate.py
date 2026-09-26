#!/usr/bin/env python
"""RQ4 pre-experiment LLM sanity gate.

Neither RQ4 paper's own LLM is locally hostable: arXiv:2403.17706 (2025)
used GPT-3.5-turbo (temperature=0), and 2026.findings-acl.482 (2026) used
GPT-4o (default params) - both closed APIs. Any open, Colab-hostable model
is therefore a documented substitution, and per instruction it must be
selected ONLY by instruction-following correctness on this fixed test set
(never by downstream C_V/NMI/etc, which would be circular).

Uses the SAME Fig. 3 prompt template as run_rq4_topic_refinement.py,
unmodified, against:
  - the DASFAA 2025 paper's own Fig. 2 worked example ("wonder" must be
    flagged as the intruder in a space topic, and "orbit" must be an
    accepted replacement)
  - 9 additional constructed cases: 5 obvious intruders (one word
    unrelated to an otherwise coherent topic) and 4 coherent topics with
    no intruder (every word genuinely fits) - both directions matter,
    since a model that always says "No" would pass the intruder cases
    trivially while failing every real experiment by over-replacing.

A configuration PASSES the gate only if: (a) every response parses as
valid JSON with the required fields, and (b) the Answer is correct on
every one of the 10 cases (strict - this is a pre-registered instruction-
following test, not a downstream metric to optimize against).

Usage:
    python scripts/rq4_llm_sanity_gate.py --llm-model mistralai/Mistral-7B-Instruct-v0.3 --quantization 8bit
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PROMPT_TEMPLATE = """Please analyze the following tasks and provide your answer in the specified format.

1. Determine the common topic shared by these words: [{topic_words}].
2. Assess whether the word "{word}" aligns with the same common topic as the words listed above.

Respond with:
- "Yes", if the given word shares the common topic.
- If "No", suggest 10 single-word alternatives that are commonly used and closely related to this topic. These words should be easily recognizable and distinct from the ones in the provided list.

Format your response in JSON, including the fields "Topic", "Answer", and "Alternative words" (only if the answer is "No")."""

# (topic_words, word, expected_answer, accepted_replacements_or_None)
TEST_CASES = [
    # The paper's own Fig. 2 worked example.
    (["space", "nasa", "launch", "air", "bird", "flight", "station", "shuttle", "rocket"],
     "wonder", "no", {"orbit", "cosmos", "satellite", "mission", "astronaut", "galaxy", "lunar"}),
    # 5 additional obvious-intruder cases.
    (["apple", "banana", "orange", "grape", "mango", "pear", "peach", "cherry", "plum"],
     "car", "no", None),
    (["guitar", "piano", "violin", "drums", "trumpet", "flute", "cello", "harp", "saxophone"],
     "bicycle", "no", None),
    (["doctor", "nurse", "hospital", "surgery", "patient", "clinic", "medicine", "diagnosis"],
     "spaceship", "no", None),
    (["football", "soccer", "basketball", "tennis", "baseball", "hockey", "rugby", "volleyball"],
     "algebra", "no", None),
    (["ocean", "river", "lake", "waterfall", "stream", "ice", "pond", "ocean tide"],
     "keyboard", "no", None),
    # 4 coherent-topic (no intruder) cases - every word genuinely fits.
    (["king", "queen", "prince", "princess", "throne", "castle", "crown", "royal"],
     "duke", "yes", None),
    (["python", "java", "javascript", "ruby", "programming", "code", "compiler", "software"],
     "algorithm", "yes", None),
    (["mountain", "hill", "peak", "summit", "cliff", "valley", "ridge", "slope"],
     "canyon", "yes", None),
    (["teacher", "student", "classroom", "school", "lesson", "homework", "exam", "textbook"],
     "university", "yes", None),
]


def parse_json_response(response: str) -> dict | None:
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if not match:
        return None
    blob = match.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None


def run_gate(llm_client) -> dict:
    results = []
    n_pass = 0
    for topic_words, word, expected, accepted_repl in TEST_CASES:
        prompt = PROMPT_TEMPLATE.format(topic_words=", ".join(topic_words), word=word)
        raw = llm_client.generate(prompt).text
        parsed = parse_json_response(raw)
        json_ok = parsed is not None and "Answer" in parsed
        answer = str(parsed.get("Answer", "")).strip().lower() if parsed else None
        answer_ok = answer is not None and answer.startswith(expected)

        repl_ok = True
        chosen_repl = None
        if expected == "no" and answer_ok and accepted_repl:
            alts = [a for a in (parsed.get("Alternative words") or []) if isinstance(a, str)]
            chosen_repl = next((a for a in alts if a.lower() in accepted_repl), None)
            repl_ok = chosen_repl is not None

        case_pass = json_ok and answer_ok and repl_ok
        n_pass += int(case_pass)
        results.append({
            "topic_words": topic_words, "word": word, "expected": expected,
            "got_answer": answer, "json_ok": json_ok, "answer_ok": answer_ok,
            "repl_ok": repl_ok, "chosen_repl": chosen_repl, "pass": case_pass,
            "raw_response": raw,
        })
    return {"n_pass": n_pass, "n_total": len(TEST_CASES), "passed_gate": n_pass == len(TEST_CASES), "cases": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-model", required=True)
    p.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", default="auto", help="Passed straight to LLMClient (e.g. 'auto' or 'cuda:0').")
    args = p.parse_args()

    from vaebm_benchmark.llm.client import LLMClient

    client = LLMClient(model_name=args.llm_model, quantization=args.quantization,
                        max_new_tokens=args.max_new_tokens, device=args.device)
    report = run_gate(client)

    print(f"=== Sanity gate: {args.llm_model} ({args.quantization}) ===")
    for c in report["cases"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"[{status}] word={c['word']!r} expected={c['expected']} got={c['got_answer']} "
              f"json_ok={c['json_ok']} repl_ok={c['repl_ok']} chosen_repl={c['chosen_repl']!r}")
    print(f"\nScore: {report['n_pass']}/{report['n_total']}  PASSED_GATE={report['passed_gate']}")

    out_path = REPO_ROOT / "results" / "rq4_llm" / f"sanity_gate_{args.llm_model.replace('/', '_')}_{args.quantization}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.llm_model, "quantization": args.quantization, **report}, indent=2))
    print(f"Full report: {out_path}")


if __name__ == "__main__":
    main()
