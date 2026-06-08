from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from . import __version__
from .config import AgentConfig
from .llm import HttpLLMClient
from .monitoring import get_exporter
from .policy import PolicyManager
from .runner import AgentRunner, RunnerDeps
from .sandbox import ExecutionSandbox
from .skills import SkillStore
from .tools.builtins import (
    build_ask_user,
    build_file_tools,
    build_run_python,
    build_search_docs,
)
from .tools.generated import GeneratedToolManager
from .tools.registry import ToolRegistry

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def version() -> None:
    """버전을 출력한다."""
    console.print(__version__)


def _ask(question: str, _choices: list[str] | None = None) -> str:
    # choices is part of the protocol signature but not rendered here
    return Prompt.ask(question)


@app.command()
def chat(docs_dir: str = "demorsc/docs") -> None:
    """대화형 에이전트 세션을 시작한다."""
    load_dotenv()  # read provider settings from a local .env if present
    cfg = AgentConfig.load()
    sandbox = ExecutionSandbox(
        cfg.workspace_dir, cfg.tool_timeout_sec, cfg.max_output_bytes, cfg.network_default
    )
    registry = ToolRegistry()
    for tool in build_file_tools(cfg.workspace_dir):
        registry.register(tool)
    registry.register(build_run_python(sandbox))
    registry.register(build_search_docs(docs_dir))
    registry.register(build_ask_user(lambda q, _c: Prompt.ask(q)))
    deps = RunnerDeps(
        llm=HttpLLMClient(cfg.base_url, cfg.model, cfg.api_key, cfg.llm_timeout_sec),
        registry=registry,
        ask=_ask,
        log_dir=Path(cfg.log_dir),
        max_iterations=cfg.max_iterations,
        max_fix_retries=cfg.max_fix_retries,
        exporter=get_exporter(cfg.monitoring),
    )
    runner = AgentRunner(
        deps,
        generated=GeneratedToolManager(f"{cfg.workspace_dir}/.session", sandbox),
        skills=SkillStore(cfg.skills_dir),
        policy=PolicyManager(ask=lambda q: Prompt.ask(q)),
    )
    console.print("[bold]세션을 시작합니다. 'exit'로 종료.[/bold]")
    while True:
        try:
            request = Prompt.ask("[cyan]you[/cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        if request.strip().lower() in {"exit", "quit"}:
            break
        result = runner.run_turn(request)
        console.print(f"[green]agent[/green]: {result.summary or '작업을 마쳤습니다.'}")


if __name__ == "__main__":
    app()
