"""Opt-in Chromium coverage for real research console training and inference."""
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from Dashboard.Server import DashboardServer
from main import synthetic_history


@unittest.skipUnless(os.environ.get("JUPITER_TRACE_BROWSER"), "Opt in with JUPITER_TRACE_BROWSER=1")
class ResearchBrowserTests(unittest.TestCase):
    def test_quantile_training_calibration_and_forecast(self):
        from playwright.sync_api import expect, sync_playwright

        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "synthetic.csv"
            synthetic_history(300).to_csv(data, index=False)
            server = DashboardServer(0, root / "jobs")
            runner = Thread(target=server.serve_forever, daemon=True)
            runner.start()
            try:
                with sync_playwright() as play:
                    browser = play.chromium.launch(
                        executable_path=os.environ.get("JUPITER_CHROMIUM") or shutil.which("chromium"))
                    page = browser.new_page(viewport={"width": 1365, "height": 950})
                    page.goto("http://" + server.authority)
                    page.locator("#kind").select_option("quantile_train")
                    page.locator("#data").set_input_files(data)
                    page.locator("#epochs").fill("1")
                    page.locator("#horizon").fill("4")
                    page.get_by_role("button", name="Queue job").click()
                    expect(page.get_by_role("button", name="Metrics and results")).to_be_visible(timeout=20000)
                    page.get_by_role("button", name="Metrics and results").click()
                    expect(page.locator("#details")).to_be_visible()
                    expect(page.locator("#view")).to_contain_text("Pinball loss")
                    page.locator("#kind").select_option("forecast")
                    page.locator("#data").set_input_files(data)
                    page.locator("#model").select_option(index=1)
                    page.locator("#steps").fill("2")
                    page.get_by_role("button", name="Queue job").click()
                    expect(page.get_by_role("button", name="Forecast chart")).to_be_visible(timeout=20000)
                    page.get_by_role("button", name="Forecast chart").click()
                    expect(page.locator("#plot")).to_be_visible()
                    expect(page.locator("#legend")).to_contain_text("Quantile 0.5")
                    page.set_viewport_size({"width": 390, "height": 950})
                    self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))
                    browser.close()
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=3)
