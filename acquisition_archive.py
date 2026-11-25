"""Audit, recover, or summarize local Jupiter sensor records."""
import argparse
import json
from pathlib import Path

from Acquisition.Archives import inspect_session
from Acquisition.CalibrationReport import summarize_leaf_calibration
from Acquisition.Coverage import coverage_report
from Acquisition.Recovery import recover_session
from Utils.Serialization import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "recover", "coverage", "calibration"):
        sub = commands.add_parser(name)
        sub.add_argument("source", type=Path)
        if name in ("inspect", "recover"):
            sub.add_argument("--accept-torn-tail", action="store_true")
        if name == "recover":
            sub.add_argument("destination", type=Path)
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument("--packet-schema", type=Path)
            mode.add_argument("--checked-in-text", action="store_true")
        else:
            sub.add_argument("--output", type=Path)
        if name == "calibration":
            sub.add_argument("--raw-mv", type=float)
            sub.add_argument("--reading-standard-uncertainty-mv", type=float)
        if name == "coverage":
            sub.add_argument("--as-of")
            sub.add_argument("--require", nargs="+", default=("air_temperature_c", "relative_humidity_pct"))
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_session(args.source, accept_torn_tail=args.accept_torn_tail)
    elif args.command == "recover":
        mapping = (read_json(args.packet_schema) if args.packet_schema else
                   "checked_in_arduino_text" if args.checked_in_text else None)
        recover_session(args.source, args.destination, packet_schema=mapping,
                        accept_torn_tail=args.accept_torn_tail)
        return
    elif args.command == "calibration":
        report = summarize_leaf_calibration(**read_json(args.source))
        result = report.as_record()
        if (args.raw_mv is None) != (args.reading_standard_uncertainty_mv is None):
            parser.error("Provide raw voltage and input standard uncertainty together")
        if args.raw_mv is not None:
            result["estimate"] = report.estimate(
                args.raw_mv, voltage_standard_uncertainty_mv=args.reading_standard_uncertainty_mv)
    else:
        result = coverage_report(args.source, required=args.require, as_of=args.as_of)
    if args.output is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
