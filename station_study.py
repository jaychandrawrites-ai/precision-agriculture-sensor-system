"""Prepare independent stations and write a manifest with seasonal summaries."""
import argparse
from pathlib import Path

from azmet import prepare
from Data.AZMET import AZMETSchema
from Data.StationAnalysis import compare_reference_et, seasonal_profile
from Utils.Serialization import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--station", nargs=2, action="append", required=True,
                        metavar=("SOURCE", "MAPPING"))
    args = parser.parse_args()
    args.output.mkdir()
    entries = {}
    for path, mapping in args.station:
        config = read_json(mapping)
        name = config.get("station_id")
        if not isinstance(name, str) or not name.isascii() or not name.isalnum() or name in entries:
            parser.error("Each mapping requires a unique alphanumeric station_id")
        features, flags, report = prepare(path, config)
        directory = args.output / name
        directory.mkdir()
        features.to_csv(directory / "hourly.csv")
        flags.to_csv(directory / "flags.csv")
        report["seasonal"] = seasonal_profile(features)
        if reference_column := config.get("station_eto_mm_column"):
            report["station_et_comparison"] = compare_reference_et(
                path, AZMETSchema.from_dict(config["schema"]), features, reference_column
            )
        write_json(directory / "report.json", report)
        entries[name] = {"hourly": f"{name}/hourly.csv", "report": f"{name}/report.json",
                         "source_sha256": report["provenance"]["source_sha256"]}
    write_json(args.output / "manifest.json", {"version": 1, "stations": entries})


if __name__ == "__main__":
    main()
