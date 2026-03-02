# token-efficient-fewshot-selector

A tiny Python utility to pick the most relevant few-shot examples for an LLM prompt while staying inside a token budget.

## Why this is useful
In production prompt pipelines, examples improve quality but quickly increase token cost. This script selects only the most relevant examples under a fixed budget.

## Features
- No external dependencies (stdlib only)
- Relevance scoring via bag-of-words cosine similarity
- Budget-aware greedy selection
- JSON output for easy pipeline integration

## Input format (JSONL)
Each line should have:
```json
{"id":"example_id","input":"user message","output":"ideal assistant response"}
```

## Run
```bash
python3 fewshot_selector.py \
  --examples examples.jsonl \
  --query "customer asks for refund after duplicate charge" \
  --max-tokens 120 \
  --max-k 3
```

## Example output
```json
{
  "query": "customer asks for refund after duplicate charge",
  "selected": [
    {
      "id": "billing_refund",
      "estimated_tokens": 43
    }
  ],
  "total_estimated_tokens": 43
}
```

## Notes
- Token counting is estimated (`~chars/4`) for speed.
- Swap in a model-specific tokenizer later if you need exact counts.

---
Built with OpenClaw for karthik765.
