"""Import, verify, describe and export immutable local Jupiter datasets."""
import argparse
import json

from Data.Registry import DatasetReference, DatasetRegistry
from Data.Studies import seasonal_drift
from Utils.Serialization import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    command = parser.add_subparsers(dest="action", required=True)
    importing = command.add_parser("import")
    importing.add_argument("source")
    importing.add_argument("mapping")
    command.add_parser("list")
    verifying = command.add_parser("verify")
    verifying.add_argument("reference")
    verifying.add_argument("--source")
    exporting = command.add_parser("export")
    exporting.add_argument("reference")
    exporting.add_argument("destination")
    exporting.add_argument("--start")
    exporting.add_argument("--end")
    comparing = command.add_parser("compare")
    comparing.add_argument("reference")
    comparing.add_argument("target")
    comparing.add_argument("destination")
    args = parser.parse_args()
    registry = DatasetRegistry(args.root)
    if args.action == "list":
        print("\n".join(map(str, registry.references())))
    elif args.action == "import":
        print(registry.import_archive(args.source, read_json(args.mapping)))
    elif args.action == "verify":
        manifest = registry.verify(DatasetReference.parse(args.reference), source=args.source)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.action == "export":
        registry.export(DatasetReference.parse(args.reference), args.destination,
                        start=args.start, end=args.end)
    else:
        a, am = registry.load(DatasetReference.parse(args.reference))
        b, bm = registry.load(DatasetReference.parse(args.target))
        report = seasonal_drift(a, b, reference_timezone=am["timezone"],
                                target_timezone=bm["timezone"])
        write_json(args.destination, {"reference": args.reference, "target": args.target,
                                      "drift": report})


if __name__ == "__main__":
    main()
