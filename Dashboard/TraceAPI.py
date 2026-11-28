"""Loopback API endpoints for survey recording and analysis."""
import base64
from urllib.parse import parse_qs, urlsplit

from Acquisition.TraceImport import import_csv
from Utils.Serialization import decode_json



def _trace_model(handler, identifier):
    for artifact in ("model.json", "direct-model.json", "quantile-model.json", "blend.json", "models/classical_ga.json"):
        try:
            return handler.server.jobs.artifact(identifier, artifact)
        except KeyError:
            continue
    raise ValueError("Choose an available saved forecast model")

def _metadata(status):
    """Keep exact coordinates in explicit local exports, not routine polling."""
    for location in status.get("locations", []):
        location["manual_coordinates_available"] = location.get("latitude_deg") is not None
        location.pop("latitude_deg", None)
        location.pop("longitude_deg", None)
        for segment in location.get("segments", ()):
            gps = segment.get("calibration", {}).get("csv_quality", {}).get("gps")
            if gps is not None:
                gps["stationary_cluster_verified"] = gps.pop("location", None) is not None
    return status


def _reply_status(handler, identifier):
    return handler._json(200, {
        "survey": _metadata(handler.server.trace.status(identifier)),
        "receiver": handler.server.trace_receiver.diagnostics(identifier),
    })


def trace_get(handler):
    request = urlsplit(handler.path)
    suffix = request.path.removeprefix("/api/trace").strip("/")
    store = handler.server.trace
    try:
        if not suffix:
            return handler._json(200, store.list_surveys())
        if "/" not in suffix:
            return _reply_status(handler, suffix)
        identifier, action = suffix.split("/", 1)
        if action == "export":
            query = parse_qs(request.query)
            kind = query.get("format", ["json"])[0]
            content_types = {"json": ("application/json", "survey.json"),
                             "csv": ("text/csv; charset=utf-8", "survey.csv"),
                             "raw": ("application/x-ndjson", "raw-packets.jsonl")}
            if kind not in content_types:
                raise ValueError("Choose a supported survey export")
            payload = store.export(identifier, format=kind,
                                   location_id=query.get("location", [None])[0],
                                   segment_id=query.get("segment", [None])[0])
            content_type, name = content_types[kind]
            return handler._reply(200, payload, content_type, name)
        return handler._json(404, {"error": "Trace route not found"})
    except (KeyError, ValueError, OSError) as exc:
        return handler._json(400, {"error": str(exc)[:200]})


def trace_post(handler, body):
    from Acquisition.TraceAnalysis import ensemble_forecast_maps, observational_map, survey_forecasts

    suffix = urlsplit(handler.path).path.removeprefix("/api/trace").strip("/")
    store, receiver = handler.server.trace, handler.server.trace_receiver
    if not suffix:
        survey = store.create(body["label"])
        return handler._json(201, _metadata(survey))
    if suffix == "restore":
        archive = (decode_json(base64.b64decode(body["archive_bytes"], validate=True))
                   if "archive_bytes" in body else body["archive"])
        survey = store.restore(archive)
        return handler._json(201, _metadata(survey))
    if suffix == "import-csv":
        payload = base64.b64decode(body["csv"], validate=True)
        return handler._json(201, _metadata(import_csv(store, payload, body["label"], body["mapping"])))
    identifier, separator, action = suffix.partition("/")
    if not separator:
        return handler._json(404, {"error": "Trace action not found"})
    if action == "connect":
        receiver.connect(identifier, body["port"], body.get("baud", 9600), body.get("mapping"))
    elif action == "disconnect":
        receiver.disconnect(identifier)
    elif action == "start":
        receiver.start(identifier, label=body.get("label"), location_id=body.get("location_id"),
                       coordinates=body.get("coordinates"), calibration=body.get("calibration"))
    elif action == "stop":
        receiver.stop(identifier)
    elif action == "finish":
        receiver.finish(identifier)
    elif action == "replay":
        receiver.replay(identifier, base64.b64decode(body["archive"], validate=True),
                        label=body.get("label"), location_id=body.get("location_id"), mapping=body.get("mapping"))
    elif action == "readiness":
        model = body.get("model")
        checkpoint = handler.server.jobs.artifact(model, "model.json") if model else None
        return handler._json(200, survey_forecasts(store, identifier, body["configuration"], checkpoint))
    elif action == "forecast":
        steps = body.get("steps", 6)
        if type(steps) is not int or not 1 <= steps <= 72:
            raise ValueError("Choose a forecast horizon from 1 to 72 hours")
        checkpoint = _trace_model(handler, body["model"])
        return handler._json(200, survey_forecasts(store, identifier, body["configuration"], checkpoint,
                                                    steps=steps))
    elif action == "diffusion-forecast":
        from datetime import timedelta
        from Acquisition.Hourly import _aware
        from Acquisition.TraceAnalysis import forecast_map

        lead = body.get("lead", 1)
        if type(lead) is not int or not 1 <= lead <= 72:
            raise ValueError("Choose a forecast lead from 1 to 72 hours")
        checkpoint = _trace_model(handler, body["model"])
        forecast = survey_forecasts(store, identifier, body["configuration"], checkpoint, lead)
        origin = _aware(body["origin"])
        return handler._json(200, forecast_map(
            store, identifier, forecast, body["feature"], origin + timedelta(hours=lead), origin,
            grid_size=body.get("resolution", 25), domain=body.get("domain"), diffusion=body.get("diffusion", {})))
    elif action == "ensemble-map":
        import numpy as np

        steps, lead = body.get("steps", 6), body.get("lead", 1)
        if type(steps) is not int or type(lead) is not int or not 1 <= lead <= steps <= 72:
            raise ValueError("Choose a map lead within the 1–72-hour forecast horizon")
        members = body.get("members", [])
        if not isinstance(members, list) or not 1 <= len(members) <= 5 or not all(
            isinstance(member, str) for member in members
        ):
            raise ValueError("Choose one to five additional saved models")
        checkpoint = _trace_model(handler, body["model"])
        other = [_trace_model(handler, member) for member in members]
        forecast = survey_forecasts(store, identifier, body["configuration"], checkpoint, steps,
                                    ensemble=other)
        adaptive = body.get("adaptive")
        if adaptive is not None and not isinstance(adaptive, dict):
            raise ValueError("Adaptive map configuration must be an object")
        predictive = body.get("predictive")
        if predictive is not None and not isinstance(predictive, dict):
            raise ValueError("Predictive map configuration must be an object")
        arrays, report = ensemble_forecast_maps(store, identifier, forecast, body["origin"],
                                                steps=steps, adaptive=adaptive, predictive=predictive,
                                                grid_size=body.get("resolution", 25), domain=body.get("domain"))
        feature = body["feature"]
        if feature not in report["features"]:
            raise ValueError("Choose a supported forecast feature")
        index = report["features"].index(feature)
        layers = {key: np.where(arrays["grid_inside"], value[lead - 1, :, :, index], None).tolist() for key, value in arrays.items()
                  if key in ("mean", "spread", "lower", "upper")}
        result = {"metadata": report, "feature": feature, "lead_hours": lead, "layers": layers}
        if adaptive is not None:
            result["adaptive"] = {
                "bounds_m": arrays["adaptive_bounds_m"].tolist(),
                "inside": arrays["adaptive_inside"].tolist(),
                **{key: arrays["adaptive_" + key][lead - 1, :, index].tolist()
                   for key in ("mean", "spread", "lower", "upper")},
            }
        if predictive is not None:
            fields = {}
            mask = arrays["predictive_inside"]
            variational = report["predictive"]["method"] in {"variational", "adaptive_variational"}
            if variational:
                result["adaptive"] = {"bounds_m": arrays["variational_bounds_m"].tolist(),
                                      "inside": mask.tolist()}
            for key in ("mean", "median", "std", "lower", "upper", "width"):
                layer = arrays["predictive_" + key][lead - 1, :, index]
                values = np.where(np.isfinite(layer), layer, None)
                fields[key] = values.tolist() if adaptive is not None else values.reshape(len(arrays["north_m"]), len(arrays["east_m"])).tolist()
            result["predictive"] = {**fields, **({"inside": mask.tolist()} if adaptive is not None else {})}
        return handler._json(200, result)
    elif action == "map":
        return handler._json(200, observational_map(store, identifier, body["feature"], body["timestamp"],
                                                    window_minutes=body.get("window_minutes", 60),
                                                    grid_size=body.get("resolution", 25), domain=body.get("domain"),
                                                    diffusion=body.get("diffusion")))
    else:
        return handler._json(404, {"error": "Trace action not found"})
    return _reply_status(handler, identifier)
