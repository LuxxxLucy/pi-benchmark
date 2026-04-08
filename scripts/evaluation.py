"""Evaluation: run classifiers on samples, compute metrics."""
from dataclasses import dataclass, field

from .classifiers import ClassifierFn


@dataclass
class Result:
    model: str
    params: str
    dataset: str
    samples: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies_ms: list = field(default_factory=list)

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
        s = sorted(self.latencies_ms) if self.latencies_ms else [0]
        n = len(s)
        return {
            "model": self.model, "params": self.params, "dataset": self.dataset,
            "samples": self.samples,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "latency_p50_ms": round(s[n // 2], 2),
            "latency_p95_ms": round(s[int(n * 0.95)], 2),
            "latency_mean_ms": round(sum(s) / n, 2),
            "errors": 0,
        }


def run_classifier(name: str, classifier_fn: ClassifierFn, samples: list,
                   dataset_name: str, params: str, warmup: int = 3) -> Result:
    """Run a classifier on samples and return metrics.

    Runs `warmup` samples first (not counted) to ensure model is loaded
    and caches are warm before measuring latency.
    """
    print(f"\n  {name} on {dataset_name} ({len(samples)} samples)...", end="", flush=True)

    # Warmup: run a few samples to ensure model is loaded and JIT-compiled
    for i in range(min(warmup, len(samples))):
        classifier_fn(samples[i]["text"])

    r = Result(model=name, params=params, dataset=dataset_name)

    for sample in samples:
        is_inj, lat = classifier_fn(sample["text"])
        r.latencies_ms.append(lat)
        r.samples += 1
        if sample["is_injection"] and is_inj:
            r.tp += 1
        elif sample["is_injection"] and not is_inj:
            r.fn += 1
        elif not sample["is_injection"] and is_inj:
            r.fp += 1
        else:
            r.tn += 1

    print(f" F1={r.f1:.3f} P={r.precision:.3f} R={r.recall:.3f} lat={r.to_dict()['latency_p50_ms']:.1f}ms")
    return r
