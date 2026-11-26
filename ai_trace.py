"""Manage AI Trace surveys, archives and offline records without the dashboard."""
import argparse
import json
from pathlib import Path

from Acquisition.AITrace import TraceStore
from Acquisition.TraceImport import import_csv, replay_segment
from Utils.Serialization import atomic_output, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("create")
    create.add_argument("label")
    commands.add_parser("list")
    for action in ("status", "finish", "export", "replay"):
        sub = commands.add_parser(action)
        sub.add_argument("survey")
        if action == "export":
            sub.add_argument("output", type=Path)
            sub.add_argument("--format", choices=("json", "csv", "raw"), default="json")
            sub.add_argument("--location")
            sub.add_argument("--segment")
        if action == "replay":
            sub.add_argument("source", type=Path)
            location = sub.add_mutually_exclusive_group(required=True)
            location.add_argument("--label")
            location.add_argument("--location")
            sub.add_argument("--mapping", type=Path)
    for action in ("restore", "import-csv"):
        sub = commands.add_parser(action)
        sub.add_argument("source", type=Path)
        if action == "import-csv":
            sub.add_argument("mapping", type=Path)
            sub.add_argument("--label", required=True)
    args = parser.parse_args()
    store = TraceStore(args.workspace, recover=args.action not in {"list", "status", "export"})
    if args.action == "create":
        result = store.create(args.label)
    elif args.action == "list":
        result = store.list_surveys()
    elif args.action == "status":
        result = store.status(args.survey)
    elif args.action == "finish":
        result = store.finish(args.survey)
    elif args.action == "restore":
        result = store.restore(args.source.read_bytes())
    elif args.action == "import-csv":
        result = import_csv(store, args.source.read_bytes(), args.label, read_json(args.mapping))
    elif args.action == "replay":
        result = replay_segment(store, args.survey, args.source, label=args.label, location_id=args.location,
                                schema=read_json(args.mapping) if args.mapping else None)
    else:
        with atomic_output(args.output, binary=True) as handle:
            handle.write(store.export(args.survey, args.format, args.location, args.segment))
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
