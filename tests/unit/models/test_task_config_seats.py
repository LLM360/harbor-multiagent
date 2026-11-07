from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task

SEATS_TOML = """
[agent]
timeout_sec = 600

[agents.maya]

[agents.raj]
under_test = false
mcp_servers = [{ name = "notes", transport = "streamable-http", url = "https://notes.example.com/mcp" }]
"""

COMPOSE = """
x-laptop: &laptop
  image: laptop
  command: ["sleep", "infinity"]
services:
  main:
    image: grader
  maya:
    <<: *laptop
    user: maya
  raj:
    <<: *laptop
"""


def _task_dir(
    root: Path, toml: str = SEATS_TOML, compose: str | None = COMPOSE
) -> Path:
    task_dir = root / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (task_dir / "task.toml").write_text(toml)
    if compose is not None:
        (task_dir / "environment" / "docker-compose.yaml").write_text(compose)
    else:
        (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu\n")
    (task_dir / "instructions").mkdir()
    (task_dir / "instructions" / "maya.md").write_text("You are Maya.")
    (task_dir / "instructions" / "raj.md").write_text("You are Raj.")
    return task_dir


def test_seats_parse_with_graded_default_and_mcp():
    config = TaskConfig.model_validate_toml(SEATS_TOML)

    assert list(config.agents) == ["maya", "raj"]
    assert config.agents["maya"].under_test is True
    assert config.agents["raj"].under_test is False
    assert [server.name for server in config.agents["raj"].mcp_servers] == ["notes"]


def test_task_without_seats_dumps_no_agents_table():
    config = TaskConfig.model_validate_toml("[agent]\ntimeout_sec = 10\n")

    assert config.agents == {}
    assert "agents" not in config.model_dump_toml()


def test_seats_survive_toml_round_trip():
    config = TaskConfig.model_validate_toml(SEATS_TOML)

    again = TaskConfig.model_validate_toml(config.model_dump_toml())

    assert again.agents == config.agents


@pytest.mark.parametrize(
    "toml, message",
    [
        ("[agents.maya]\nunder_test = false\n", "at least one agent under test"),
        ("[agents.maya]\nnetwork_mode = 'no-network'\n", "Extra inputs"),
        ("[agent]\nuser = 'maya'\n[agents.maya]\n", "set user: on each seat"),
        ('[agents."bad name"]\n', "Invalid Docker Compose service name"),
        (
            "[agents.maya]\n[[steps]]\nname = 'one'\n",
            "cannot have steps",
        ),
    ],
)
def test_seat_schema_refusals(toml: str, message: str):
    with pytest.raises(ValidationError, match=message):
        TaskConfig.model_validate_toml(toml)


def test_task_reads_each_seat_instruction(tmp_path: Path):
    task = Task(_task_dir(tmp_path))

    assert task.instruction == ""
    assert task.seat_instruction("maya") == "You are Maya."
    assert task.seat_instruction("raj") == "You are Raj."


def test_seat_instruction_gets_extra_instructions(tmp_path: Path):
    task = Task(_task_dir(tmp_path), extra_instructions=["Be brief."])

    assert task.seat_instruction("maya") == "You are Maya.\n\nBe brief."


def test_missing_seat_instruction_is_refused(tmp_path: Path):
    task_dir = _task_dir(tmp_path)
    (task_dir / "instructions" / "raj.md").unlink()

    with pytest.raises(FileNotFoundError, match="expected instructions/raj.md"):
        Task(task_dir)
    assert not Task.is_valid_dir(task_dir)


def test_seats_may_come_from_an_included_compose_file(tmp_path: Path):
    task_dir = _task_dir(tmp_path, compose="include: [seats.yaml]\n")

    assert Task(task_dir).config.agents.keys() == {"maya", "raj"}


def test_seat_task_is_valid_without_instruction_md(tmp_path: Path):
    task_dir = _task_dir(tmp_path)

    assert Task.is_valid_dir(task_dir)
    assert Task.is_valid_dir(task_dir, disable_verification=True)


def test_seat_user_is_accepted():
    # A seat may declare its OS user (used by environments with no per-service
    # user, e.g. a Kubernetes pod shared via the main transport).
    cfg = TaskConfig.model_validate_toml(
        '[agent]\ntimeout_sec = 60\n[agents.maya]\nuser = "agent"\n[agents.raj]\n'
    )
    assert cfg.agents["maya"].user == "agent"
    assert cfg.agents["raj"].user is None
