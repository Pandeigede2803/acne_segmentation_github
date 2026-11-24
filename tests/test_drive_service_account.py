"""Unit tests for Drive API file synchronization helpers."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.drive_service_account import upload_file


class UploadFileTests(unittest.TestCase):
    def setUp(self):
        self.http_module = types.ModuleType("googleapiclient.http")
        setattr(self.http_module, "MediaFileUpload", MagicMock(return_value="media-body"))
        self.google_module = types.ModuleType("googleapiclient")
        setattr(self.google_module, "http", self.http_module)
        self.modules_patch = patch.dict(sys.modules, {
            "googleapiclient": self.google_module,
            "googleapiclient.http": self.http_module,
        })
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    def test_creates_file_when_destination_has_no_matching_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            path.write_text("result\n", encoding="utf-8")
            service = MagicMock()
            service.files.return_value.list.return_value.execute.return_value = {"files": []}
            service.files.return_value.create.return_value.execute.return_value = {"id": "new-id"}

            self.assertEqual(upload_file(service, path, "parent-id"), "new-id")
            service.files.return_value.create.assert_called_once()
            service.files.return_value.update.assert_not_called()

    def test_updates_file_when_destination_has_matching_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            path.write_text("result\n", encoding="utf-8")
            service = MagicMock()
            service.files.return_value.list.return_value.execute.return_value = {
                "files": [{"id": "existing-id"}]
            }
            service.files.return_value.update.return_value.execute.return_value = {
                "id": "existing-id"
            }

            self.assertEqual(upload_file(service, path, "parent-id"), "existing-id")
            service.files.return_value.update.assert_called_once()
            service.files.return_value.create.assert_not_called()

    def test_rejects_missing_local_file(self):
        with self.assertRaises(FileNotFoundError):
            upload_file(MagicMock(), "/not/a/real/file.csv", "parent-id")


if __name__ == "__main__":
    unittest.main()
