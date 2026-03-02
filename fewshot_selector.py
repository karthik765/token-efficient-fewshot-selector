#!/usr/bin/env python3
"""Select the most relevant few-shot examples under a token budget.

No external dependencies. Uses simple bag-of-words cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def estimate_tokens(text: str) -> int:
    # Fast rough estimate suitable for budgeting.
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Example:
    id: str
    input: str
    output: str

    @property
    def text(self) -> str:
        return f"Input: {self.input}\nOutput: {self.output}"

    @property
    def cost(self) -> int:
        return estimate_tokens(self.text)


def load_examples(path: Path) -> List[Example]:
    examples = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        examples.append(
            Example(
                id=str(obj.get("id", i)),
                input=obj["input"],
                output=obj["output"],
            )
        )
    return examples


def vec(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in (a.keys() & b.keys()))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def select_examples(query: str, examples: Iterable[Example], max_tokens: int, max_k: int) -> List[Example]:
    qv = vec(query)
    scored = sorted(
        ((cosine(qv, vec(ex.input)), ex) for ex in examples),
        key=lambda x: x[0],
        reverse=True,
    )

    chosen: List[Example] = []
    used = 0
    for _, ex in scored:
        if len(chosen) >= max_k:
            break
        if used + ex.cost > max_tokens:
            continue
        chosen.append(ex)
        used += ex.cost
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Token-efficient few-shot selector")
    p.add_argument("--examples", required=True, help="Path to JSONL examples")
    p.add_argument("--query", required=True, help="Current user query")
    p.add_argument("--max-tokens", type=int, default=240, help="Budget for selected examples")
    p.add_argument("--max-k", type=int, default=4, help="Maximum number of examples")
    args = p.parse_args()

    examples = load_examples(Path(args.examples))
    selected = select_examples(args.query, examples, max_tokens=args.max_tokens, max_k=args.max_k)

    print(json.dumps({
        "query": args.query,
        "selected": [
            {
                "id": ex.id,
                "input": ex.input,
                "output": ex.output,
                "estimated_tokens": ex.cost,
            }
            for ex in selected
        ],
        "total_estimated_tokens": sum(ex.cost for ex in selected),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
