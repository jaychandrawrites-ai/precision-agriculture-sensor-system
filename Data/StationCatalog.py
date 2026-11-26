"""Catalog of verified public weather archive release mappings."""
from pathlib import Path
import re
from urllib.parse import urlsplit

from Data.AZMET import AZMETSchema
from Utils.Serialization import read_json


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "Examples" / "azmet" / "catalog.json"


def _official_url(value):
    parts = urlsplit(value)
    return parts.scheme == "https" and parts.netloc == "azmet.arizona.edu"


DIGEST = re.compile(r"[a-f0-9]{64}\Z")


class StationCatalog:
    def __init__(self, path=DEFAULT_CATALOG):
        file = Path(path).resolve()
        self._directory = file.parent
        record = read_json(file)
        if record.get("version") != 1 or not record.get("stations"):
            raise ValueError("Unsupported or empty station catalog")
        self.stations = record["stations"]

    def entries(self):
        return [{"station": station, "name": self.stations[station]["name"],
                 "metadata_url": self.stations[station]["metadata_url"],
                 "year": int(year), "source_url": self.mapping(station, year)["source_url"]}
                for station in sorted(self.stations)
                for year in sorted(self.stations[station]["archives"])]

    def mapping(self, station, year):
        record = self.stations.get(station, {})
        name = record.get("archives", {}).get(str(year))
        if name is None:
            raise ValueError("No verified mapping for that station and year")
        file = self._directory / name
        if Path(name).name != name or file.resolve().parent != self._directory:
            raise ValueError("Catalog mappings must reside next to the catalog")
        mapping = read_json(file)
        if (mapping.get("station_id") != station or mapping.get("metadata_url") != record["metadata_url"]
                or not _official_url(mapping.get("source_url", ""))
                or not _official_url(mapping.get("metadata_url", ""))
                or not _official_url(mapping.get("schema_reference", ""))
                or not DIGEST.fullmatch(mapping.get("source_sha256", ""))):
            raise ValueError("Catalog and archive metadata disagree")
        expected = f"/azmet/data/{station[2:]}{str(year)[-2:]}rh.txt"
        if urlsplit(mapping["source_url"]).path != expected:
            raise ValueError("Legacy archive path disagrees with catalog station or year")
        AZMETSchema.from_dict(mapping["schema"])
        return mapping
