"""Evaluation: run classifiers on samples, compute accuracy metrics.

Latency profiling is intentionally excluded — it will be a separate pass
with batch_size=1 on a representative dataset. This module only measures
accuracy (F1, precision, recall, FPR) and records wall-clock throughput.
"""
import time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Result:
    model: str
    params: str
    dataset: str
    device: str
    batch_size: int
    samples: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    wall_time_s: float = 0.0
    timestamp: str = ""

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else 0

    @property
    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d else 0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0

    @property
    def fpr(self):
        d = self.fp + self.tn
        return self.fp / d if d else 0

    @property
    def accuracy(self):
        return (self.tp + self.tn) / self.samples if self.samples else 0

    def to_dict(self):
        throughput = round(self.samples / self.wall_time_s, 1) if self.wall_time_s > 0 else 0
        return {
            "model": self.model,
            "params": self.params,
            "dataset": self.dataset,
            "device": self.device,
            "batch_size": self.batch_size,
            "timestamp": self.timestamp,
            "samples": self.samples,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "throughput_samples_per_s": throughput,
            "wall_time_s": round(self.wall_time_s, 2),
        }


def run_classifier(name: str, classifier_fn, samples: list,
                   dataset_name: str, params: str,
                   device: str = "unknown", batch_size: int = 1) -> Result:
    """Run a classifier on samples and return accuracy metrics.

    Uses batched inference when available for speed.
    Records total wall time for throughput estimation (not per-sample latency).
    """
    r = Result(model=name, params=params, dataset=dataset_name,
               device=device, batch_size=batch_size)

    texts = [s["text"] for s in samples]
    labels = [s["is_injection"] for s in samples]

    batch_fn = getattr(classifier_fn, "batch", None)

    # Warmup (3 samples)
    warmup_n = min(3, len(texts))
    if batch_fn:
        batch_fn(texts[:warmup_n])
    else:
        for t in texts[:warmup_n]:
            classifier_fn(t)

    # Inference
    t0 = time.perf_counter()
    if batch_fn and len(samples) > 10:
        predictions = batch_fn(texts)
    else:
        predictions = [classifier_fn(t)[0] for t in texts]
    r.wall_time_s = time.perf_counter() - t0

    # Confusion matrix
    for pred, expected in zip(predictions, labels):
        r.samples += 1
        if expected and pred:
            r.tp += 1
        elif expected and not pred:
            r.fn += 1
        elif not expected and pred:
            r.fp += 1
        else:
            r.tn += 1

    r.timestamp = datetime.now(timezone.utc).isoformat()

    throughput = r.samples / r.wall_time_s if r.wall_time_s > 0 else 0
    print(f"  {name} x {dataset_name}: F1={r.f1:.3f} P={r.precision:.3f} "
          f"R={r.recall:.3f} FPR={r.fpr:.3f} ({r.samples} samples, "
          f"{throughput:.0f} samples/s, {r.wall_time_s:.1f}s)")

    return r
