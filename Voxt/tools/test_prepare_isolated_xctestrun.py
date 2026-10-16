"""Verify hosted-test identity and storage isolation before launching the app."""

import plistlib
import tempfile
import unittest
import uuid
from pathlib import Path

from prepare_isolated_xctestrun import prepare_xctestrun


class IsolatedXCTestConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.source_path = self.directory / "Voxt.xctestrun"
        self.isolation_directory = self.directory / "isolation"
        self.bundle_prefix = f"com.voxt.validation.r{uuid.uuid4().hex}"
        self.host_metadata_path = self.directory / "TestDebug/Voxt.app/Contents/Info.plist"
        self.host_metadata_path.parent.mkdir(parents=True)
        self.host_metadata_path.write_bytes(
            plistlib.dumps({"CFBundleIdentifier": f"{self.bundle_prefix}.Voxt.testhost"})
        )
        test_metadata_path = (
            self.host_metadata_path.parent
            / "PlugIns/VoxtTests.xctest/Contents/Info.plist"
        )
        test_metadata_path.parent.mkdir(parents=True)
        test_metadata_path.write_bytes(
            plistlib.dumps({"CFBundleIdentifier": f"{self.bundle_prefix}.VoxtTests.testhost"})
        )
        self.source_path.write_bytes(
            plistlib.dumps(
                {
                    "VoxtTests": {
                        "TestHostBundleIdentifier": "com.voxt.Voxt.testhost",
                        "TestHostPath": "__TESTROOT__/TestDebug/Voxt.app",
                        "TestBundlePath": "__TESTHOST__/Contents/PlugIns/VoxtTests.xctest",
                        "EnvironmentVariables": {
                            "SYNTHETIC_CREDENTIAL": "private-test-value",
                            "XCTestConfigurationFilePath": "generated-xctest-marker",
                            "VOXT_RUN_MODEL_TESTS": "1",
                            "VOXT_LLM_SMOKE_ENHANCEMENT_TEXT": "synthetic input",
                        },
                    }
                }
            )
        )

    def test_preserves_framework_marker_and_original_configuration(self) -> None:
        original_bytes = self.source_path.read_bytes()
        prepared_path = prepare_xctestrun(
            self.source_path, self.isolation_directory, self.bundle_prefix
        )
        target = plistlib.loads(prepared_path.read_bytes())["VoxtTests"]
        self.assertEqual(self.source_path.read_bytes(), original_bytes)
        self.assertEqual(
            target["TestHostBundleIdentifier"], f"{self.bundle_prefix}.Voxt.testhost"
        )
        environment = target["EnvironmentVariables"]
        self.assertEqual(environment["XCTestConfigurationFilePath"], "generated-xctest-marker")
        self.assertEqual(environment["VOXT_RUN_MODEL_TESTS"], "0")
        self.assertEqual(environment["VOXT_LLM_SMOKE_ENHANCEMENT_TEXT"], "")
        self.assertEqual(
            Path(environment["CFFIXED_USER_HOME"]), self.isolation_directory.resolve() / "home"
        )
        self.assertTrue(Path(environment["VOXT_MODEL_STORAGE_ROOT"]).is_dir())
        self.assertTrue((self.isolation_directory / "manifest.json").is_file())
        self.assertNotIn(
            "private-test-value", (self.isolation_directory / "manifest.json").read_text()
        )

    def test_rejects_normal_app_identity_before_creating_storage(self) -> None:
        self.host_metadata_path.write_bytes(
            plistlib.dumps({"CFBundleIdentifier": "com.voxt.Voxt"})
        )
        with self.assertRaises(ValueError):
            prepare_xctestrun(self.source_path, self.isolation_directory, self.bundle_prefix)
        self.assertFalse(self.isolation_directory.exists())

    def test_rejects_nonunique_prefix(self) -> None:
        with self.assertRaises(ValueError):
            prepare_xctestrun(self.source_path, self.isolation_directory, "com.voxt.Voxt")
        self.assertFalse(self.isolation_directory.exists())

    def test_rejects_unexpected_test_host_path(self) -> None:
        configuration = plistlib.loads(self.source_path.read_bytes())
        configuration["VoxtTests"]["TestHostPath"] = "/Applications/Voxt.app"
        self.source_path.write_bytes(plistlib.dumps(configuration))
        with self.assertRaises(ValueError):
            prepare_xctestrun(self.source_path, self.isolation_directory, self.bundle_prefix)
        self.assertFalse(self.isolation_directory.exists())

    def test_refuses_to_reuse_configuration_or_storage(self) -> None:
        prepared_path = prepare_xctestrun(
            self.source_path, self.isolation_directory, self.bundle_prefix
        )
        with self.assertRaises(FileExistsError):
            prepare_xctestrun(self.source_path, self.isolation_directory, self.bundle_prefix)
        prepared_path.unlink()
        with self.assertRaises(FileExistsError):
            prepare_xctestrun(self.source_path, self.isolation_directory, self.bundle_prefix)


if __name__ == "__main__":
    unittest.main()
