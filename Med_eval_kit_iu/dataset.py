from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


def _expand_path(path: str, base_dir: str | None) -> str:
    if path.startswith(("http://", "https://", "file://")):
        return path
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


@dataclass
class ReportSample:
    id: str
    image: str
    instruction: str
    reference: str

    def to_model_input(self, image_base_dir: str | None = None) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image": _expand_path(self.image, image_base_dir),
            "instruction": self.instruction,
            "reference": self.reference,
            "image_base_dir": image_base_dir,
        }


class FullModelReportDataset(Sequence[ReportSample]):
    """
    Minimal loader for the medical-report evaluation set. Expected keys per sample:
      - "id"
      - "image" / "image_path"
      - "instruction" / "prompt" / "question" / "input"
      - "reference" / "report" / "answer" / "gt"
    """

    def __init__(self, path: str, image_base_dir: str | None = None):
        assert os.path.exists(path), f"Dataset file not found: {path}"
        self.path = path
        self.image_base_dir = image_base_dir
        self.samples: List[ReportSample] = self._load()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReportSample:
        return self.samples[idx]

    # ------------------------------------------------------------------ #
    def _load(self) -> List[ReportSample]:
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".jsonl":
            raw = [
                json.loads(line)
                for line in open(self.path, "r", encoding="utf-8")
                if line.strip()
            ]
        else:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = raw.get("data") or raw.get("samples") or raw.get("annotations")
        assert isinstance(raw, Iterable), f"Unsupported dataset structure in {self.path}"

        parsed: List[ReportSample] = []
        for idx, item in enumerate(raw):
            if item is None:
                continue
            sample_id = str(item.get("id", idx))
            image_path = item.get("image") or item.get("image_path")
            assert image_path, f"Sample {sample_id} missing image path"

            instruction = (
                item.get("instruction")
                or item.get("prompt")
                or item.get("question")
                or item.get("input")
            )
            assert instruction, f"Sample {sample_id} missing instruction/prompt"

            reference = (
                item.get("reference")
                or item.get("report")
                or item.get("answer")
                or item.get("gt")
            )
            assert reference, f"Sample {sample_id} missing reference report"

            parsed.append(
                ReportSample(
                    id=sample_id,
                    image=image_path,
                    instruction=instruction,
                    reference=reference,
                )
            )
        return parsed

    def to_model_inputs(self) -> List[Dict[str, Any]]:
        return [sample.to_model_input(self.image_base_dir) for sample in self.samples]
