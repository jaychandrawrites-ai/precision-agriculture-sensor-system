"""Capture the checked-in Arduino text format or replay a saved session."""
import argparse
from pathlib import Path
import signal
from threading import Event

from Acquisition.AcquisitionController import capture
from Acquisition.DataLogger import replay
from Acquisition.SerialReader import serial_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port")
    source.add_argument("--replay", type=Path)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stop = Event()

    def request_stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if args.port:
        capture(serial_lines(args.port, args.baud, stop), args.output)
    else:
        capture(replay(args.replay), args.output, stop)


if __name__ == "__main__":
    main()
