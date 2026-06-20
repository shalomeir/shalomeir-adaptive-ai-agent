from __future__ import annotations

from pathlib import Path
import sys
import threading
from typing import Callable

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from . import __version__
from .config import AgentConfig
from .llm import HttpLLMClient
from .monitoring import get_exporter
from .policy import PolicyManager
from .runner import NON_INTERACTIVE_ASK, AgentRunner, RunnerDeps
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
LOADING_FRAMES = ("loading.", "loading..", "loading...", "loading..")
LOADING_INTERVAL_SEC = 0.35
LOADING_STYLE_START = "\033[33m"
LOADING_STYLE_END = "\033[0m"
_active_loading_stop: Callable[[], None] | None = None


@app.command()
def version() -> None:
    """버전을 출력한다."""
    console.print(__version__)


def _ask(question: str, choices: list[str] | None = None) -> str:
    """Render an agent clarification as dialogue instead of a raw prompt."""
    _stop_active_loading()
    console.print(f"[green]agent[/green]: {question}")
    answer = Prompt.ask("[cyan]you[/cyan]", choices=choices)
    if answer.strip().lower() in {"exit", "quit"}:
        raise EOFError
    return answer


def _confirm(question: str) -> str:
    """Render policy confirmations separately from open-ended chat."""
    _stop_active_loading()
    console.print(f"[green]agent[/green]: {question}")
    return Prompt.ask("[cyan]you[/cyan]", default="n")


def _stop_active_loading() -> None:
    if _active_loading_stop is not None:
        _active_loading_stop()


class _LoadingIndicator:
    """Small terminal-only one-line loading indicator."""

    def __init__(self, target_console: Console) -> None:
        self.console = target_console
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.active = False

    def start(self) -> None:
        if not self.console.is_terminal:
            return
        self.active = True
        self._render(LOADING_FRAMES[0])
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.done.set()
        if self.thread is not None and threading.current_thread() is not self.thread:
            self.thread.join(timeout=LOADING_INTERVAL_SEC + 0.1)
        with self.lock:
            if not self.active:
                return
            self.console.file.write("\r\033[K")
            self.console.file.flush()
            self.active = False

    def _animate(self) -> None:
        index = 0
        while not self.done.wait(LOADING_INTERVAL_SEC):
            index = (index + 1) % len(LOADING_FRAMES)
            self._render(LOADING_FRAMES[index])

    def _render(self, frame: str) -> None:
        with self.lock:
            if not self.active:
                return
            self.console.file.write(
                f"\r\033[Kagent: {LOADING_STYLE_START}{frame}{LOADING_STYLE_END}"
            )
            self.console.file.flush()


def _run_turn_with_loading(runner: AgentRunner, request: str, *, enabled: bool = True):
    """Run one interactive turn with a transient terminal loading indicator."""
    if not enabled or not console.is_terminal:
        return runner.run_turn(request)

    indicator = _LoadingIndicator(console)
    stop_loading = indicator.stop
    global _active_loading_stop
    previous_stop = _active_loading_stop
    indicator.start()
    _active_loading_stop = stop_loading
    try:
        return runner.run_turn(request)
    finally:
        stop_loading()
        if _active_loading_stop is stop_loading:
            _active_loading_stop = previous_stop


def _assemble_runner(
    cfg: AgentConfig,
    docs_dir: str,
    *,
    free_ask: Callable[..., str],
    confirm_ask: Callable[[str], str],
    max_iterations: int | None = None,
    non_interactive: bool = False,
) -> AgentRunner:
    """Wire up tools, deps, and the runner.

    ``free_ask`` answers open-ended ask_user questions; ``confirm_ask`` answers
    policy y/n confirmations. Splitting them lets ``run`` auto-approve y/n gates
    without pretending to answer ambiguity questions interactively.
    """
    sandbox = ExecutionSandbox(
        cfg.workspace_dir, cfg.tool_timeout_sec, cfg.max_output_bytes, cfg.network_default
    )
    registry = ToolRegistry()
    for tool in build_file_tools(cfg.workspace_dir):
        registry.register(tool)
    registry.register(build_run_python(sandbox))
    registry.register(build_search_docs(docs_dir))
    registry.register(build_ask_user(free_ask))
    deps = RunnerDeps(
        llm=HttpLLMClient(cfg.base_url, cfg.model, cfg.api_key, cfg.llm_timeout_sec),
        registry=registry,
        ask=free_ask,
        log_dir=Path(cfg.log_dir),
        max_iterations=max_iterations or cfg.max_iterations,
        max_fix_retries=cfg.max_fix_retries,
        compaction_token_threshold=cfg.compaction_token_threshold,
        exporter=get_exporter(cfg.monitoring),
        non_interactive=non_interactive,
    )
    return AgentRunner(
        deps,
        generated=GeneratedToolManager(f"{cfg.workspace_dir}/.session", sandbox),
        skills=SkillStore(cfg.skills_dir),
        policy=PolicyManager(ask=confirm_ask),
    )


@app.command()
def chat(
    docs_dir: str = "demorsc/docs",
    loading: bool = typer.Option(True, "--loading/--no-loading", help="처리 중 loading 표시"),
) -> None:
    """대화형 에이전트 세션을 시작한다."""
    load_dotenv()  # read provider settings from a local .env if present
    cfg = AgentConfig.load()
    runner = _assemble_runner(cfg, docs_dir, free_ask=_ask, confirm_ask=_confirm)
    console.print("[bold]세션을 시작합니다. 'exit'로 종료.[/bold]")
    while True:
        try:
            request = Prompt.ask("[cyan]you[/cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        if not request.strip() and not sys.stdin.isatty():
            break
        if request.strip().lower() in {"exit", "quit"}:
            break
        try:
            result = _run_turn_with_loading(runner, request, enabled=loading)
        except (EOFError, KeyboardInterrupt):
            break
        console.print(f"[green]agent[/green]: {result.summary or '작업을 마쳤습니다.'}")


@app.command()
def run(
    task: str = typer.Argument(..., help="실행할 작업(자연어) 한 건"),
    yes: bool = typer.Option(False, "--yes", "-y", help="부수효과 확인(y/n)을 모두 자동 승인한다"),
    docs_dir: str = typer.Option("demorsc/docs", help="근거 조회용 문서 폴더"),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="최대 반복 횟수 override"
    ),
) -> None:
    """작업 한 건을 비대화형으로 실행하고 결과를 출력한다.

    스크립트·CI·반복 테스트에 쓴다. ``--yes``는 파일 쓰기·도구 저장 같은 y/n 확인을
    모두 승인한다. 자유 형식 ask_user 질문은 비대화형이라 답할 수 없으므로 질문을 출력하고
    HITL 필요 상태로 종료한다.
    """
    load_dotenv()
    cfg = AgentConfig.load()
    # Non-interactive: policy gates get a fixed y/n, but free-form ask_user
    # questions must not be answered with "n" because that corrupts the task.
    answer = "y" if yes else "n"
    runner = _assemble_runner(
        cfg,
        docs_dir,
        free_ask=lambda q, _c=None: NON_INTERACTIVE_ASK,
        confirm_ask=lambda q: answer,
        max_iterations=max_iterations,
        non_interactive=True,
    )
    result = runner.run_turn(task)
    console.print(result.summary or "작업을 마쳤습니다.")


if __name__ == "__main__":
    app()
