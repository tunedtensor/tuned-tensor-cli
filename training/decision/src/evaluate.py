"""Predict one typed decision per input row.

The config names only opaque row IDs and inputs. Reference labels stay in Node,
which joins these predictions to trusted expected outputs before scoring.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from common import QUESTION_ID, load_agent, load_config, option_labels, pick_device, read_jsonl, write_json, write_jsonl


def answer_distribution(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, float]:
    labels = option_labels(question)
    if question["type"] == "noul":
        p_true = float(answer["noul"])
        return {"false": 1.0 - p_true, "true": p_true}
    # Laya keys choice probabilities by label and score probabilities by level index.
    return {label: float(answer["probabilities"][label]) for label in labels}


def predict_rows(agent, question: dict[str, Any], rows: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        began = time.perf_counter()
        results = agent.predict_batch([row["input"] for row in batch], {QUESTION_ID: question}, batch_size=batch_size)
        latency_ms = int(round((time.perf_counter() - began) * 1000 / max(1, len(batch))))
        for row, result in zip(batch, results):
            distribution = answer_distribution(question, result["answers"][QUESTION_ID])
            labels = list(distribution)
            values = np.array([distribution[label] for label in labels])
            predictions.append({
                "id": row["id"],
                "prediction": labels[int(values.argmax())],
                "probabilities": {label: round(float(value), 6) for label, value in distribution.items()},
                "confidence": round(float(values.max()), 6),
                "latency_ms": latency_ms,
            })
    return predictions


def main() -> None:
    config = load_config()
    device = pick_device(config.get("device"))
    rows = read_jsonl(config["inputs_path"])
    for row in rows:
        if set(row) != {"id", "input"}:
            raise ValueError("Decision evaluation rows must contain only id and input.")
    began = time.perf_counter()
    agent, checkpoint = load_agent(config["model"], config.get("revision"), device)
    load_seconds = time.perf_counter() - began
    predictions = predict_rows(agent, config["question"], rows, int(config.get("batch_size", 16)))
    write_jsonl(config["output_path"], predictions)
    write_json(config["metrics_path"], {
        "ok": True,
        "model": config["model"],
        "revision": config.get("revision"),
        "checkpoint_dir": str(checkpoint),
        "device": str(agent.device),
        "rows": len(predictions),
        "load_seconds": round(load_seconds, 3),
    })


if __name__ == "__main__":
    main()
