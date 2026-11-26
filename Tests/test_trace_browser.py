"""Opt-in Playwright regression against the real local server and isolated fixtures."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from Dashboard.Jobs import Job
from Dashboard.Server import DashboardServer
from Examples.SyntheticTrace import run


class BrowserReceiver:
    def __init__(self, frames):
        self.frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read_until(self, size):
        try:
            return self.frames.get(timeout=.015)
        except Empty:
            return b""


@unittest.skipUnless(os.environ.get("JUPITER_TRACE_BROWSER"), "Opt in with JUPITER_TRACE_BROWSER=1")
class TraceBrowserTests(unittest.TestCase):
    def test_desktop_and_narrow_field_workflow(self):
        from playwright.sync_api import expect, sync_playwright

        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            fixture = root / "fixture"
            run(fixture, ensemble=True)
            server = DashboardServer(0, root / "jobs", ports=("synthetic",))
            queue = Queue()
            server.trace_receiver.connector = lambda *_: BrowserReceiver(queue)
            for index, source in enumerate(("synthetic-model.json", "synthetic-model-2.json"), 1):
                identifier = str(index) * 32
                model = server.jobs.root / identifier
                model.mkdir()
                shutil.copyfile(fixture / source, model / "model.json")
                server.jobs.store[identifier] = Job(
                    identifier, "train", model, {"label": "Synthetic model " + str(index)},
                    datetime.now(timezone.utc).isoformat(), "complete",
                )
            runner = Thread(target=server.serve_forever, daemon=True)
            runner.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        executable_path=os.environ.get("JUPITER_CHROMIUM") or shutil.which("chromium"))
                    for width in (1365, 390):
                        with self.subTest(width=width):
                            page = browser.new_page(viewport={"width": width, "height": 950},
                                                    timezone_id="UTC", accept_downloads=True)
                            page.goto("http://" + server.authority + "/trace")
                            page.get_by_label("New survey name").fill(f"Synthetic browser {width}")
                            page.get_by_role("button", name="Create Survey").click()
                            expect(page.locator("#trace-alert")).to_have_text("Survey created.")
                            page.get_by_label("Serial receiver").select_option("synthetic")
                            page.locator("#trace-mapping").set_input_files({
                                "name": "mapping.json", "mimeType": "application/json",
                                "buffer": json.dumps({"fields": {
                                    "air_temperature_c": {"key": "temp", "unit": "C"},
                                    "leaf_mv": {"key": "leaf", "unit": "mV"},
                                }}).encode(),
                            })
                            page.get_by_role("button", name="Connect Receiver").click()
                            expect(page.locator("#trace-diagnostics dd").first).to_have_text("connected")
                            page.get_by_label("Location name", exact=True).fill("Browser test location")
                            page.get_by_label("Calibration metadata (optional)").set_input_files({
                                "name": "calibration.json", "mimeType": "application/json",
                                "buffer": json.dumps({"leaf_calibration": {
                                    "dry_mv": 300, "wet_mv": 600, "source": "SYNTHETIC illustrative fixture",
                                }}).encode(),
                            })
                            page.get_by_role("button", name="Start Recording").click()
                            expect(page.locator("#trace-alert")).to_have_text("Location recording started.")
                            queue.put(b'{"temp": 24, "leaf": 450}\n')
                            expect(page.locator("#trace-recording")).to_contain_text("1 parsed samples")
                            expect(page.locator("#trace-calibration")).to_contain_text("0.5")
                            page.get_by_role("button", name="Stop Location").click()
                            expect(page.locator("#trace-alert")).to_contain_text("Location saved.")
                            page.get_by_label("Location name", exact=True).fill("Second browser test location")
                            page.get_by_role("button", name="Record New Location").click()
                            expect(page.locator("#trace-alert")).to_have_text("Location recording started.")
                            page.get_by_role("button", name="Stop Location").click()
                            page.get_by_label("Earlier location").select_option(label="Browser test location")
                            page.get_by_role("button", name="Revisit Location").click()
                            expect(page.locator("#trace-alert")).to_have_text("Revisit recording started.")
                            page.get_by_role("button", name="Finish Survey").click()
                            expect(page.locator("#trace-alert")).to_have_text("Survey finalized.")
                            archive = root / ("browser-" + str(width) + ".json")
                            with page.expect_download() as download:
                                page.get_by_role("link", name="Export Survey", exact=True).click()
                            download.value.save_as(archive)
                            self.assertEqual(len(json.loads(archive.read_text())["survey"]["locations"]), 2)
                            page.get_by_label("Survey archive").set_input_files(archive)
                            page.get_by_role("button", name="Import Survey").click()
                            expect(page.locator("#trace-alert")).to_contain_text("duplicate identifiers")
                            # For analysis load the explicitly synthetic common-issue archive with hourly coverage.
                            archive_fixture = fixture
                            if width == 390:
                                archive_fixture = root / "narrow-fixture"
                                run(archive_fixture)
                            page.get_by_label("Survey archive").set_input_files(archive_fixture / "survey.json")
                            page.get_by_role("button", name="Import Survey").click()
                            expect(page.locator("#trace-alert")).to_have_text("Survey imported.")
                            page.get_by_label("Window end (local time)").fill("2020-07-02T07:00")
                            page.get_by_role("button", name="Generate Spatial Map").click()
                            expect(page.get_by_role("img", name="Observational map", exact=False)).to_be_visible()
                            page.get_by_label("Field configuration", exact=True).set_input_files(
                                fixture / "field-settings.json")
                            page.get_by_role("button", name="Check Forecast Readiness").click()
                            expect(page.locator("#trace-readiness")).to_contain_text("5 of 5")
                            page.get_by_label("Ensemble field configuration").set_input_files(
                                fixture / "field-settings.json")
                            page.get_by_label("Primary ensemble model").select_option("1" * 32)
                            page.get_by_label("Additional models").select_option("2" * 32)
                            page.get_by_label("Forecast origin (local time)").fill("2020-07-02T07:00")
                            page.get_by_role("button", name="Map Ensemble Forecast").click()
                            expect(page.get_by_role("img", name="Ensemble forecast mean", exact=False)).to_be_visible()
                            expect(page.get_by_role("img", name="Ensemble model spread", exact=False)).to_be_visible()
                            predictive = root / "predictive.json"
                            predictive.write_text(json.dumps({
                                "draws": 8, "feature_scales": [1.] * 6,
                                "covariance_scenarios": [
                                    {"id": "fixture", "weight": 1., "covariance": {"range_m": 50.}}
                                ],
                            }))
                            page.get_by_label("Predictive uncertainty configuration").set_input_files(predictive)
                            page.get_by_role("button", name="Map Ensemble Forecast").click()
                            expect(page.get_by_role("img", name="Predictive scenario mean")).to_be_visible()
                            expect(page.get_by_role("img", name="Predictive scenario interval width")).to_be_visible()
                            self.assertTrue(page.evaluate(
                                "document.documentElement.scrollWidth <= window.innerWidth"))
                            page.close()
                    browser.close()
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=3)
