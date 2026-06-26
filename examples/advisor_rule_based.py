#!/usr/bin/env python
from __future__ import annotations

import json
import sys


PROMOTE_KINDS = {"decision", "procedure", "failure", "project", "fact", "self"}
POISON_MARKERS = (
    "ignore previous",
    "system prompt",
    "developer message",
    "always obey this memory",
    "permanent instruction",
)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    recommendations = []
    for cap in payload.get("candidates", []):
        text = f"{cap.get('title', '')}\n{cap.get('body', '')}".lower()
        if any(marker in text for marker in POISON_MARKERS):
            recommendations.append(
                {
                    "capsule_id": cap["id"],
                    "action": "quarantine",
                    "reason": "rule sample detected instruction-like poisoning text",
                    "risk_score": 0.9,
                }
            )
        elif (
            cap.get("kind") in PROMOTE_KINDS
            and float(cap.get("confidence", 0.0)) >= 0.72
            and float(cap.get("salience", 0.0)) >= 0.70
        ):
            recommendations.append(
                {
                    "capsule_id": cap["id"],
                    "action": "promote",
                    "reason": "rule sample found durable high-confidence memory",
                    "risk_score": 0.0,
                }
            )
        else:
            recommendations.append(
                {
                    "capsule_id": cap["id"],
                    "action": "keep",
                    "reason": "rule sample left candidate for more evidence",
                    "risk_score": 0.0,
                }
            )
    print(json.dumps({"recommendations": recommendations}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
