"""Verify diagnostic artifacts from a restarted loopback server."""
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from main import synthetic_history
from Tests.test_phase5_dashboard import dashboard, encoded, DashboardResearchTests
from Dashboard.Server import DashboardServer


class DiagnosticsTests(unittest.TestCase):
    def test_training_artifacts_survive_server_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weather = encoded(synthetic_history(240).to_csv(index=False).encode())
            with dashboard(root) as request:
                job = DashboardResearchTests._submit(self, request, "train", {"data": weather}, {"epochs": 1})
                DashboardResearchTests._wait(self, request, job)
            with dashboard(root) as request:
                status, content = request(f"/api/artifacts/{job}/diagnostics/training.png")
                self.assertEqual(status, 200)
                self.assertTrue(content.startswith(b"\x89PNG"))
                self.assertEqual(request(f"/api/artifacts/{job}/model.json")[0], 200)
            if os.environ.get("JUPITER_TRACE_BROWSER"):
                from playwright.sync_api import expect, sync_playwright

                server = DashboardServer(0, root)
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with sync_playwright() as play:
                        browser = play.chromium.launch(
                            executable_path=os.environ.get("JUPITER_CHROMIUM") or shutil.which("chromium"))
                        page = browser.new_page()
                        errors = []
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.goto("http://" + server.authority + "/diagnostics")
                        image = page.get_by_role("img", name="training")
                        expect(image).to_be_visible()
                        self.assertTrue(image.evaluate("img => img.complete && img.naturalWidth > 0"))
                        self.assertEqual(errors, [])
                        browser.close()
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)
