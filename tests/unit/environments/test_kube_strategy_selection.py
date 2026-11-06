"""Strategy-selection tests: a Helm chart in the task selects the kube strategy.

Provider construction needs the provider SDK; tests skip when it's absent.
The provider-agnostic trigger (kube_chart_present) is tested unconditionally.
"""

import pytest

from harbor.environments.kube_ops import kube_chart_present
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _chart_dir(tmp_path):
    env_dir = tmp_path / "environment"
    (env_dir / "chart").mkdir(parents=True)
    (env_dir / "chart" / "Chart.yaml").write_text(
        "apiVersion: v2\nname: t\nversion: 0.1.0\n"
    )
    return env_dir


def _dockerfile_dir(tmp_path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM alpine\n")
    return env_dir


def _trial_paths(tmp_path):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    tp = TrialPaths(trial_dir=trial_dir)
    tp.mkdir()
    return tp


# --------------------------------------------------------------------------- #
# Provider-agnostic trigger
# --------------------------------------------------------------------------- #
class TestChartDetection:
    def test_chart_present(self, tmp_path):
        assert kube_chart_present(_chart_dir(tmp_path), {}) is True

    def test_chart_absent(self, tmp_path):
        assert kube_chart_present(_dockerfile_dir(tmp_path), {}) is False

    def test_custom_chart_path(self, tmp_path):
        env_dir = tmp_path / "environment"
        (env_dir / "mychart").mkdir(parents=True)
        (env_dir / "mychart" / "Chart.yaml").write_text(
            "apiVersion: v2\nname: t\nversion: 0.1.0\n"
        )
        assert kube_chart_present(env_dir, {"chart_path": "mychart"}) is True
        assert kube_chart_present(env_dir, {}) is False


# --------------------------------------------------------------------------- #
# Daytona (validated sandbox backend)
# --------------------------------------------------------------------------- #
# TestDaytonaSelection removed: _DaytonaKube is out of scope for this
# fork (Modal-only kube support). See TestModalSelection below.

class TestModalSelection:
    def _modal(self, env_dir, tmp_path, **kwargs):
        from harbor.environments.modal import ModalEnvironment

        return ModalEnvironment(
            environment_dir=env_dir,
            environment_name="hello-helm",
            session_id="hello-helm__abc",
            trial_paths=_trial_paths(tmp_path),
            task_env_config=EnvironmentConfig(cpus=2, memory_mb=2048),
            **kwargs,
        )

    def test_chart_selects_kube_strategy(self, tmp_path):
        pytest.importorskip("modal")
        from harbor.environments.modal import _ModalKube

        env = self._modal(_chart_dir(tmp_path), tmp_path, launcher="k3s")
        assert env._kube_mode is True
        assert env._compose_mode is False
        assert isinstance(env._strategy, _ModalKube)
        assert env._strategy._launcher.name == "k3s"
        # k3s needs a real kernel, so kube mode forces vm_runtime on.
        assert env._vm_runtime_enabled is True
        assert env.capabilities.docker_compose is False
        assert env.capabilities.disable_internet is False
        assert env.capabilities.gpus is False

    def test_chart_with_compose_file_still_selects_kube(self, tmp_path):
        pytest.importorskip("modal")
        from harbor.environments.modal import _ModalKube

        env_dir = _chart_dir(tmp_path)
        (env_dir / "docker-compose.yaml").write_text("services: {}\n")
        env = self._modal(env_dir, tmp_path)
        assert isinstance(env._strategy, _ModalKube)

    def test_no_chart_selects_direct(self, tmp_path):
        pytest.importorskip("modal")
        from harbor.environments.modal import _ModalDirect

        env = self._modal(_dockerfile_dir(tmp_path), tmp_path)
        assert env._kube_mode is False
        assert env._vm_runtime_enabled is False
        assert isinstance(env._strategy, _ModalDirect)

    def test_non_k3s_launcher_is_refused_at_start(self, tmp_path):
        pytest.importorskip("modal")
        import asyncio

        env = self._modal(_chart_dir(tmp_path), tmp_path, launcher="kind")
        with pytest.raises(RuntimeError, match="k3s"):
            asyncio.run(env._strategy.start(force_build=False))

    def test_start_creates_vm_sandbox_then_brings_up_cluster(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("modal")
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        import harbor.environments.modal as modal_mod

        env = self._modal(_chart_dir(tmp_path), tmp_path)
        strategy = env._strategy
        order: list[str] = []

        fake_app = object()
        monkeypatch.setattr(
            modal_mod.App, "lookup", MagicMock(aio=AsyncMock(return_value=fake_app))
        )
        monkeypatch.setattr(
            modal_mod.Image, "from_registry", MagicMock(return_value="kube-image")
        )

        async def fake_create_sandbox(**kwargs):
            order.append("create_sandbox")
            assert kwargs["experimental_options"] == {"vm_runtime": True}
            assert kwargs["block_network"] is False
            return MagicMock()

        async def fake_kube_start():
            order.append("kube_start")
            return "main-pod-0"

        async def fake_ensure_dirs(_targets):
            order.append("ensure_dirs")

        monkeypatch.setattr(env, "_create_sandbox", fake_create_sandbox)
        monkeypatch.setattr(strategy, "_kube_start", fake_kube_start)
        monkeypatch.setattr(env, "ensure_dirs", fake_ensure_dirs)

        asyncio.run(strategy.start(force_build=False))

        assert order == ["create_sandbox", "kube_start", "ensure_dirs"]
        assert strategy._main_pod == "main-pod-0"
        assert env._app is fake_app
        modal_mod.Image.from_registry.assert_called_once_with("alpine/k8s:1.30.2")

    def test_stop_tears_down_cluster_before_sandbox(self, tmp_path, monkeypatch):
        pytest.importorskip("modal")
        import asyncio
        from unittest.mock import MagicMock

        env = self._modal(_chart_dir(tmp_path), tmp_path)
        strategy = env._strategy
        env._sandbox = MagicMock()
        order: list[str] = []

        async def fake_kube_stop(*, delete):
            order.append("kube_stop")

        async def fake_teardown():
            order.append("teardown_sandbox")

        monkeypatch.setattr(strategy, "_kube_stop", fake_kube_stop)
        monkeypatch.setattr(strategy, "_teardown_sandbox", fake_teardown)

        asyncio.run(strategy.stop(delete=True))
        assert order == ["kube_stop", "teardown_sandbox"]


class TestHelmFailureDiagnostics:
    def test_failed_install_reports_waiting_pods(self, tmp_path):
        pytest.importorskip("modal")
        import asyncio

        from harbor.environments.base import ExecResult
        from harbor.environments.modal import ModalEnvironment

        env = ModalEnvironment(
            environment_dir=_chart_dir(tmp_path),
            environment_name="hello-helm",
            session_id="hello-helm__abc",
            trial_paths=_trial_paths(tmp_path),
            task_env_config=EnvironmentConfig(cpus=2, memory_mb=2048),
        )
        strategy = env._strategy

        async def fake_host_exec(command, timeout_sec=None):
            if "helm" in command and "install" in command:
                return ExecResult(
                    stdout="", stderr="context deadline exceeded", return_code=1
                )
            if "jsonpath" in command:
                return ExecResult(
                    stdout="db-0 db=:\nminio-0 minio=ImagePullBackOff:pull access denied\n",
                    stderr="",
                    return_code=0,
                )
            return ExecResult(
                stdout="NAME READY\nminio-0 0/1", stderr="", return_code=0
            )

        strategy._host_exec = fake_host_exec
        with pytest.raises(RuntimeError) as err:
            asyncio.run(strategy._helm_install_with_retry(["helm", "install", "x"]))
        message = str(err.value)
        assert "context deadline exceeded" in message
        assert "minio-0 minio=ImagePullBackOff" in message
        assert "db-0" not in message.split("--- waiting ---")[1]
