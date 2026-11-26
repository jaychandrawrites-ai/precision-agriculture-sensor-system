"""Compare modeled hourly physics with a declared independent observation file."""
import argparse
from pathlib import Path

from Physics.PairedValidation import ObservationContract, paired_validation
from Utils.Serialization import atomic_output, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation-from", required=True)
    parser.add_argument("--model-index-definition")
    parser.add_argument("--calibrate-bias", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    rows, report = paired_validation(args.model, args.observations,
                                     ObservationContract(**read_json(args.contract)),
                                     validation_from=args.validation_from,
                                     model_index_definition=args.model_index_definition,
                                     calibrate_bias=args.calibrate_bias, synthetic=args.synthetic)
    args.output.mkdir()
    with atomic_output(args.output / "validation.csv") as stream:
        rows.to_csv(stream)
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
