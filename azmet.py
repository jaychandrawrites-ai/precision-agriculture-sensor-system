"""Inspect or prepare a caller-mapped hourly AZMET export."""
import argparse
import json
from pathlib import Path

from assumptions import Assumptions, CropAssumptions, LeafAssumptions, SiteAssumptions, SoilAssumptions
from Data.AZMET import AZMETSchema, inspect_schema, read_azmet
from Data.HistoricalFeatures import historical_features
from Data.QualityReport import quality_report
from Data.WeatherPhysics import ReferenceET


def prepare(source, metadata):
    assumptions = metadata.get("assumptions", {})
    site = Assumptions(soil=SoilAssumptions(**assumptions.get("soil", {})),
                       site=SiteAssumptions(**assumptions.get("site", {})),
                       crop=CropAssumptions(**assumptions.get("crop", {})),
                       leaf=LeafAssumptions(**assumptions.get("leaf", {})))
    reference = ReferenceET(site.site.latitude_deg, **metadata["reference_et"])
    weather = read_azmet(source, AZMETSchema.from_dict(metadata["schema"]),
                         source_label=metadata["source_label"], synthetic=metadata.get("synthetic", False))
    features, flags, provenance = historical_features(weather, site, reference, **metadata["scenario"])
    return features, flags, {"quality": quality_report(features, flags), "provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--delimiter", default=",")
    prepare_command = subcommands.add_parser("prepare")
    prepare_command.add_argument("source", type=Path)
    prepare_command.add_argument("mapping", type=Path)
    prepare_command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_schema(args.source, args.delimiter), indent=2))
        return
    metadata = json.loads(args.mapping.read_text(encoding="utf-8"))
    features, flags, report = prepare(args.source, metadata)
    args.output.mkdir()
    features.to_csv(args.output / "hourly.csv")
    flags.to_csv(args.output / "flags.csv")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
