"""Mechanical forecast-horizon checks for portable Jupiter artifacts."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from config import FEATURES
from Data.ForecastHistory import require_history
from Physics.Constraints import project
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import decode_json


SUPPORTED_LEADS = tuple(range(1, 73))


class InferenceArtifact:
    def __init__(self, path, device="cpu"):
        path = Path(path)
        raw = path.read_bytes()
        record = decode_json(raw)
        self.checksum = sha256(raw).hexdigest()
        marker = record.get("format", "")
        self._direct = marker.startswith("jupiter-direct-")
        self.kind = "direct" if self._direct else "recursive"
        self.context = 3
        if marker == "jupiter-quantile-v1":
            from Temporal.Quantile import QuantileForecaster

            self.kind = "quantile"
            self.predictor = QuantileForecaster.load(path, device)
            self.limit = self.predictor.horizon
        elif marker == "jupiter-seasonal-residual-v1":
            from Temporal.SeasonalCheckpoint import load_seasonal

            self.kind = "seasonal_ensemble"
            self.predictor = load_seasonal(path, device)
            self.limit = self.predictor.horizon
        else:
            self.predictor = load_forecaster(path, device)
            self.context = self.predictor.history_hours
            if marker == "jupiter-spectral-v1":
                self.kind = "spectral"
                self.limit = self.predictor.maximum_forecast_hours
            elif self._direct:
                architecture = record.get("architecture", "")
                family = architecture.get("family") if isinstance(architecture, dict) else architecture
                from Temporal.SequenceForecast import DENSE_FAMILIES, FAMILIES

                if self.context != 3 or (family in FAMILIES and family not in DENSE_FAMILIES):
                    self.kind = "sequence"
                self.limit = self.predictor.direct.model.horizon
            else:
                self.limit = record.get("training_config", {}).get("rollout_steps")
                if type(self.limit) is not int or self.limit < 1:
                    raise ValueError("Recursive checkpoint has no declared trained horizon")
        self.limit = min(72, self.limit)

    def forecast(self, values, times, steps):
        if type(steps) is not int or not 1 <= steps <= self.limit:
            raise ValueError(f"{self.kind} checkpoint supports 1–{self.limit} hours in this verified workflow")
        history = require_history(values, times, hours=self.context)
        if self.kind == "quantile":
            q = self.predictor.predict(history[None], steps)[0]
            if not np.isfinite(q).all() or (np.diff(q, axis=-1) < 0).any():
                raise ValueError("Quantile checkpoint produced nonfinite or unordered quantiles")
            return project(q[:, :, np.argmin(abs(self.predictor.levels - .5))])
        if self.kind == "seasonal_ensemble":
            forecast, _, _ = self.predictor.predict(values, times, times[-1:], history[None])
            return forecast[0, :steps]
        return self.predictor.forecast(history, steps, origin=times[-1])


class VerificationEnsemble:
    """A manifest-ordered, equal-weight bundle of already loaded artifacts."""
    kind = "checkpoint_ensemble"

    def __init__(self, members):
        self.members = tuple(members)
        if len(self.members) < 2 or len({member.checksum for member in self.members}) != len(self.members):
            raise ValueError("Verification ensemble needs two distinct checkpoint members")
        self.context = max(member.context for member in self.members)
        self.limit = min(member.limit for member in self.members)
        checksum = "".join(member.checksum for member in self.members).encode()
        self.checksum = sha256(checksum).hexdigest()

    def forecast(self, values, times, steps):
        if type(steps) is not int or not 1 <= steps <= self.limit:
            raise ValueError(f"Ensemble members share only leads 1–{self.limit}")
        return np.stack([member.forecast(values, times, steps)
                         for member in self.members]).mean(axis=0)


def verify_leads(artifact, values, times, leads=SUPPORTED_LEADS):
    """Check actual calls and horizon refusal without implying predictive accuracy."""
    leads = tuple(leads)
    if not leads or len(set(leads)) != len(leads) or any(type(n) is not int or n not in SUPPORTED_LEADS for n in leads):
        raise ValueError("Use distinct leads from 1 through 72")
    reports = []
    for lead in leads:
        row = {"lead_hours": lead}
        try:
            forecast = artifact.forecast(values, times, lead)
            complete = (forecast.shape == (lead, len(FEATURES)) and np.isfinite(forecast).all())
        except ValueError as exc:
            row["status"] = "rejected_as_unsupported" if lead > artifact.limit else "failed"
            row["reason"] = str(exc)
        else:
            row["status"] = "verified" if complete and lead <= artifact.limit else "failed"
        reports.append(row)
    return {"checkpoint_sha256": artifact.checksum, "family": artifact.kind,
            "limit": artifact.limit, "history_hours": artifact.context, "leads": reports,
            "interpretation": "Mechanical shape, finiteness and horizon verification; no accuracy claim."}
