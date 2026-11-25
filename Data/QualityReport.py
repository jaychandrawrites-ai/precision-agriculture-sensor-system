"""Coverage and chronological eligibility for prepared six-feature series."""
import numpy as np

from Data.DataQuality import invalid_rows
from Data.Dataset import build_dataset


def quality_report(features, flags=None, training=None):
    values = features.to_numpy(dtype=float)
    timestamps = features.index.to_numpy(dtype="datetime64[ns]")
    valid = ~invalid_rows(values)
    report = {
        "hours": len(features),
        "first_utc": features.index[0].isoformat() if len(features) else None,
        "last_utc": features.index[-1].isoformat() if len(features) else None,
        "valid_feature_rows": int(valid.sum()),
        "missing_feature_counts": dict(zip(features.columns, (~np.isfinite(values)).sum(axis=0).tolist())),
    }
    if flags is not None:
        report["flags"] = {name: int(flags[name].sum()) for name in flags}
    kwargs = ({"train_fraction": training.train_fraction,
               "validation_fraction": training.validation_fraction} if training else {})
    try:
        dataset = build_dataset(values, timestamps, **kwargs)
    except ValueError as exc:
        report["chronology"] = {"unavailable": str(exc)}
    else:
        report["chronology"] = {}
        for split in ("train", "validation", "test"):
            rows = dataset.target_indices[getattr(dataset, split)]
            report["chronology"][split] = {
                "target_count": len(rows), "first_target_utc": features.index[rows[0]].isoformat(),
                "last_target_utc": features.index[rows[-1]].isoformat(),
            }
    return report
