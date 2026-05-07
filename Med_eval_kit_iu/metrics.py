from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


def _prepare(preds: List[str], refs: List[str]) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    gts = defaultdict(list)
    res = defaultdict(list)
    for idx, (pred, ref) in enumerate(zip(preds, refs)):
        res[idx].append(pred or "")
        gts[idx].append(ref or "")
    return res, gts


def compute_nlg_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    返回 BLEU-1/2/3/4、ROUGE-L、CIDEr。
    （可按需扩展 METEOR，当前默认关闭以避免 Java 依赖）
    """
    assert len(preds) == len(refs), "predictions and references must align"
    res, gts = _prepare(preds, refs)

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)

    rouge_scorer = Rouge()
    rouge_l, _ = rouge_scorer.compute_score(gts, res)

    cider_scorer = Cider()
    cider, _ = cider_scorer.compute_score(gts, res)

    return {
        "BLEU-1": bleu_scores[0],
        "BLEU-2": bleu_scores[1],
        "BLEU-3": bleu_scores[2],
        "BLEU-4": bleu_scores[3],
        "ROUGE-L": rouge_l,
        "CIDEr": cider,
    }

