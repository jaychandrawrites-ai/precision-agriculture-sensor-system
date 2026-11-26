"""Compare archived hourly FAO-56 and Hargreaves estimates under stated assumptions."""
import argparse
from pathlib import Path

from assumptions import Assumptions
from Data.AZMET import AZMETSchema, read_azmet
from Data.WeatherPhysics import ReferenceET
from Physics.Evapotranspiration.ETComparison import RadiationAssumptions, compare_et
from Utils.Serialization import atomic_output, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--wind-height", required=True, type=float)
    parser.add_argument("--dark-cloud-ratio", required=True, type=float)
    args = parser.parse_args()
    mapping = read_json(args.mapping)
    site = Assumptions.from_dict(mapping.get("assumptions", {})).site
    reference = ReferenceET(site.latitude_deg, **mapping["reference_et"])
    radiation = RadiationAssumptions(site.elevation_m, args.wind_height, args.dark_cloud_ratio)
    weather = read_azmet(args.source, AZMETSchema.from_dict(mapping["schema"]),
                         source_label=mapping["source_label"], synthetic=mapping.get("synthetic", False))
    forecasts, report = compare_et(weather, reference, radiation)
    with atomic_output(args.report.with_suffix(".csv")) as file:
        forecasts.to_csv(file)
    write_json(args.report, report)


if __name__ == "__main__":
    main()
