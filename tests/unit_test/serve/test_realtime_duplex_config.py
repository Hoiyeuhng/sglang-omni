# SPDX-License-Identifier: Apache-2.0
"""Shared duplex serving options from CLI and Python entry points to runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar, Literal
from unittest.mock import patch

import pytest
import typer
from fastapi.testclient import TestClient
from pydantic import Field
from typer.testing import CliRunner

from sglang_omni.cli.serve import serve
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.placement import StagePlacementPlan
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import ProcessTopologyPlan
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.serve.launcher import PipelineUvicornServer, launch_server, run_server
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from tests.unit_test.serve.test_realtime_duplex_session import ScriptedAdapter


class DuplexPipelineConfig(PipelineConfig):
    """A model-free deployment for testing the public server launch path."""

    realtime_deployment_factory: ClassVar[str] = "test.duplex_deployment"
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="duplex",
                process="duplex",
                factory_path="test.duplex_stage",
                terminal=True,
            )
        ]
    )


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setitem(
        PIPELINE_CONFIG_REGISTRY.configs, "duplex-test", DuplexPipelineConfig
    )
    config_path = tmp_path / "duplex.yaml"
    config_path.write_text(
        "config_cls: DuplexPipelineConfig\nmodel_path: duplex-test\n"
    )
    return config_path


@pytest.fixture
def cli_app() -> typer.Typer:
    application = typer.Typer()
    application.command(
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
    )(serve)
    return application


@pytest.mark.parametrize(
    ("cli_timeouts", "deployment_limits", "expected_timeouts"),
    [
        ([], RuntimeLimits(), (10, 30)),
        ([], RuntimeLimits(admission_timeout_s=15, idle_input_timeout_s=45), (15, 45)),
        (
            [
                "--realtime-admission-timeout-s",
                "20",
                "--realtime-idle-input-timeout-s",
                "300",
            ],
            RuntimeLimits(),
            (20, 300),
        ),
        (
            ["--realtime-idle-input-timeout-s", "600"],
            RuntimeLimits(admission_timeout_s=15),
            (15, 600),
        ),
        (
            ["--realtime-admission-timeout-s", "20"],
            RuntimeLimits(idle_input_timeout_s=45),
            (20, 45),
        ),
    ],
)
def test_timeout_configuration_reaches_served_capabilities(
    config_file: Path,
    cli_app: typer.Typer,
    cli_timeouts: list[str],
    deployment_limits: RuntimeLimits,
    expected_timeouts: tuple[float, float],
) -> None:
    adapter = ScriptedAdapter()
    deployment = RealtimeDeployment(
        capabilities=Capabilities(),
        adapter_factory=lambda: adapter,
        limits=deployment_limits,
    )
    arguments = [
        "--config",
        str(config_file),
        "--enable-realtime",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ]
    arguments.extend(cli_timeouts)
    with patch("sglang_omni.cli.serve.launch_server") as requested_launch:
        result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 0, result.output

    async def check_capabilities(server: PipelineUvicornServer) -> None:
        with TestClient(server.config.app) as client:
            response = client.get("/v1/realtime/capabilities")
        assert response.status_code == 200
        limits = response.json()["limits"]
        assert (
            limits["admission_timeout_s"],
            limits["idle_input_timeout_s"],
        ) == expected_timeouts
        assert limits["max_output_events"] == deployment_limits.max_output_events

    async def wait_for_pipeline_failure() -> None:
        await asyncio.Event().wait()

    with (
        patch(
            "sglang_omni.serve.launcher.MultiProcessPipelineRunner", autospec=True
        ) as runner_factory,
        patch(
            "sglang_omni.serve.launcher.import_string",
            return_value=lambda client: deployment,
        ),
        patch.object(
            PipelineUvicornServer,
            "serve",
            autospec=True,
            side_effect=check_capabilities,
        ) as serve_http,
    ):
        runner = runner_factory.return_value
        runner.prep.placement_plan = StagePlacementPlan(stages={}, gpus={})
        runner.prep.process_plan = ProcessTopologyPlan(
            groups=(), stage_to_process={}, tp_stage_to_processes={}
        )
        runner.stage_control_endpoints = {}
        runner.coordinator.health.return_value = {"running": True}
        runner.wait_failed.side_effect = wait_for_pipeline_failure
        launch_server(
            *requested_launch.call_args.args, **requested_launch.call_args.kwargs
        )
        serve_http.assert_awaited_once()
        runner.start.assert_awaited_once()
        runner.stop.assert_awaited_once()

    assert deployment.limits is deployment_limits


@pytest.mark.parametrize(
    "field_name", ["realtime_admission_timeout_s", "realtime_idle_input_timeout_s"]
)
@pytest.mark.parametrize("invalid_timeout", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_timeouts_before_launch(
    cli_app: typer.Typer, config_file: Path, field_name: str, invalid_timeout: str
) -> None:
    option = "--" + field_name.replace("_", "-")
    with patch("sglang_omni.cli.serve.launch_server") as requested_launch:
        result = CliRunner().invoke(
            cli_app,
            [
                "--config",
                str(config_file),
                "--enable-realtime",
                option,
                invalid_timeout,
            ],
        )
    assert result.exit_code == 2
    assert "finite" in result.output, result.output
    assert "positive" in result.output, result.output
    requested_launch.assert_not_called()


@pytest.mark.parametrize(
    "field_name", ["realtime_admission_timeout_s", "realtime_idle_input_timeout_s"]
)
@pytest.mark.parametrize(
    "invalid_timeout", [0, -1, float("nan"), float("inf"), float("-inf")]
)
@pytest.mark.parametrize("entry_point", ["sync", "async"])
def test_python_rejects_invalid_timeouts_before_pipeline_start(
    field_name: str, invalid_timeout: float, entry_point: Literal["sync", "async"]
) -> None:
    config = DuplexPipelineConfig(model_path="test")
    with patch(
        "sglang_omni.serve.launcher.MultiProcessPipelineRunner"
    ) as runner_factory:
        with pytest.raises(ValueError, match=field_name):
            if entry_point == "sync":
                launch_server(
                    config, enable_realtime=True, **{field_name: invalid_timeout}
                )
            else:
                asyncio.run(
                    run_server(
                        config, enable_realtime=True, **{field_name: invalid_timeout}
                    )
                )
    runner_factory.assert_not_called()


@pytest.mark.parametrize(
    ("enable_realtime", "config_type"),
    [(False, PipelineConfig), (True, PipelineConfig), (False, DuplexPipelineConfig)],
)
def test_unsupported_timeout_overrides_fail_before_pipeline_start(
    enable_realtime: bool,
    config_type: type[PipelineConfig],
) -> None:
    config = config_type(
        model_path="test",
        stages=DuplexPipelineConfig(model_path="test").stages,
    )
    with patch(
        "sglang_omni.serve.launcher.MultiProcessPipelineRunner"
    ) as runner_factory:
        with pytest.raises(ValueError, match="shared duplex"):
            asyncio.run(
                run_server(
                    config,
                    enable_realtime=enable_realtime,
                    realtime_idle_input_timeout_s=300,
                )
            )
    runner_factory.assert_not_called()


@pytest.mark.parametrize(
    "field_name", ["realtime_admission_timeout_s", "realtime_idle_input_timeout_s"]
)
def test_serving_timeouts_are_not_pipeline_yaml_fields(
    config_file: Path, field_name: str
) -> None:
    config_file.write_text(config_file.read_text() + f"{field_name}: 20\n")
    with pytest.raises(ValueError, match=field_name):
        ConfigManager.from_file(str(config_file))


def test_serving_timeouts_appear_in_cli_help(cli_app: typer.Typer) -> None:
    result = CliRunner().invoke(
        cli_app, ["--help"], env={"COLUMNS": "200", "TERM": "dumb"}
    )
    assert result.exit_code == 0
    assert "--realtime-admission-timeout-s" in result.output
    assert "--realtime-idle-input-timeout-s" in result.output
