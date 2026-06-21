from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEMORSC = ROOT / "demorsc"
DOCS = DEMORSC / "docs"


class DemoFailure(AssertionError):
    pass


def copy_inputs(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEMORSC / "data" / "monsters.json", workspace / "monsters.json")
    shutil.copy(DEMORSC / "data" / "events.csv", workspace / "events.csv")
    shutil.copy(DEMORSC / "data" / "events2.csv", workspace / "events2.csv")
    shutil.copy(DEMORSC / "world" / "world.json", workspace / "world.json")


def make_env(base_env: dict[str, str], workspace: Path, skills: Path, logs: Path, max_iterations: int) -> dict[str, str]:
    env = base_env.copy()
    env.update(
        {
            "AGENT_WORKSPACE_DIR": str(workspace),
            "AGENT_SKILLS_DIR": str(skills),
            "AGENT_LOG_DIR": str(logs),
            "AGENT_MAX_ITERATIONS": str(max_iterations),
            "AGENT_LLM_TIMEOUT_SEC": env.get("AGENT_LLM_TIMEOUT_SEC", "45"),
        }
    )
    return env


def run_agent(
    label: str,
    task: str,
    *,
    env: dict[str, str],
    yes: bool = True,
    max_iterations: int = 30,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "adaptive-agent",
        "run",
        "--docs-dir",
        str(DOCS),
        "--max-iterations",
        str(max_iterations),
    ]
    if yes:
        cmd.append("--yes")
    cmd.append(task)
    print(f"\n== {label} ==")
    print(task)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    output = result.stdout.strip()
    if output:
        print(output)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise DemoFailure(f"{label}: adaptive-agent exited with {result.returncode}")
    return result


def run_chat(
    label: str,
    turns: list[str],
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    cmd = ["adaptive-agent", "chat", "--docs-dir", str(DOCS), "--no-loading"]
    print(f"\n== {label} ==")
    for turn in turns:
        print(turn)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        input="\n".join(turns) + "\n",
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    output = result.stdout.strip()
    if output:
        print(output)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise DemoFailure(f"{label}: adaptive-agent chat exited with {result.returncode}")
    return result


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = tuple(row.items())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoFailure(message)


def verify_csv_clean(path: Path, expected_rows: int) -> None:
    rows = csv_rows(path)
    require(len(rows) == expected_rows, f"{path.name}: expected {expected_rows} rows, got {len(rows)}")
    require(rows == unique_rows(rows), f"{path.name}: duplicate full rows remain")
    dates = [row["date"] for row in rows]
    require(dates == sorted(dates), f"{path.name}: dates are not sorted: {dates}")


def walk_json_objects(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for item in value.values():
            found.extend(walk_json_objects(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_json_objects(item))
    return found


def verify_world(path: Path) -> None:
    data = json.loads(path.read_text())
    objects = walk_json_objects(data)
    entities = [item for item in objects if item.get("type") == "Entity"]
    healths = [item.get("props", {}).get("health") for item in entities]
    ids = {str(item.get("id")) for item in objects}
    require(len(entities) == 3, f"world.json: expected 3 Entity nodes, got {len(entities)}")
    require(all(isinstance(value, int | float) and value >= 100 for value in healths), f"world.json: bad healths {healths}")
    require("e1" not in ids and "rock1" not in ids, "world.json: low-health nodes remain")
    require(abs((sum(healths) / len(healths)) - 190) < 0.01, f"world.json: average mismatch {healths}")


def verify_markdown_hp_table(path: Path) -> None:
    require(path.exists(), f"{path.name}: file was not created")
    text = path.read_text()
    for name, hp in (("Dragon", "300"), ("Orc", "150"), ("Wolf", "110")):
        require(name in text and hp in text, f"{path.name}: missing {name} {hp}: {text}")
    require(
        text.index("Dragon") < text.index("Orc") < text.index("Wolf"),
        f"{path.name}: rows are not hp-descending: {text}",
    )


def verify_d3_output(output: str) -> None:
    lowered = output.lower()
    for marker in ("실패", "중단", "반복 한도", "no_progress", "max_iterations"):
        require(marker not in lowered, f"D3 output reports failure: {output}")
    for token in ("purchase", "2500", "signup", "refund", "-200"):
        require(token in lowered, f"D3 output missing {token!r}: {output}")


def latest_logs(log_dir: Path, before: set[Path]) -> list[Path]:
    return sorted(set(log_dir.glob("session-*.jsonl")) - before)


def log_contains(paths: list[Path], needle: str) -> bool:
    return any(needle in path.read_text(errors="replace") for path in paths)


def log_has_kind(paths: list[Path], kind: str) -> bool:
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == kind:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live adaptive-agent demos with real CLI calls.")
    parser.add_argument("--keep", action="store_true", help="keep the temporary demo directory")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only the named check; can be passed more than once",
    )
    args = parser.parse_args()
    selected = {item.lower() for item in args.only}

    temp = Path(tempfile.mkdtemp(prefix="adaptive-agent-live-demos."))
    workspace = temp / "workspace"
    skills = temp / "skills"
    logs = temp / "logs"
    copy_inputs(workspace)
    skills.mkdir()
    logs.mkdir()

    env = make_env(os.environ.copy(), workspace, skills, logs, args.max_iterations)

    failures: list[str] = []

    def check(label: str, func) -> None:  # type: ignore[no-untyped-def]
        if selected and label.lower() not in selected:
            print(f"[SKIP] {label}")
            return
        try:
            func()
            print(f"[PASS] {label}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"[FAIL] {label}: {exc}")

    def d1() -> None:
        output = run_agent(
            "D1",
            "workspace의 monsters.json에서 hp가 100 이상인 몬스터 이름과 평균 hp를 알려줘.",
            env=env,
            max_iterations=args.max_iterations,
        ).stdout
        for token in ("Orc", "Dragon", "Wolf", "186.67"):
            require(token in output, f"D1 output missing {token!r}: {output}")

    check(
        "D1",
        d1,
    )

    def d2() -> None:
        run_agent(
            "D2",
            "events.csv에서 완전히 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 events-clean.csv로 저장해줘.",
            env=env,
            max_iterations=args.max_iterations,
        )
        verify_csv_clean(workspace / "events-clean.csv", 5)
        require(any(skills.iterdir()), "D2 did not persist any generated skill")

    check("D2", d2)

    def d3() -> None:
        verify_d3_output(
            run_agent(
                "D3",
                "events.csv에서 완전히 중복된 행은 한 번만 세고, amount 합계를 type별로 구해줘.",
                env=env,
                max_iterations=args.max_iterations,
            ).stdout
        )

    check("D3", d3)

    check(
        "D4",
        lambda: require(
            "HITL" in run_agent(
                "D4",
                "데이터 좀 정리해줘.",
                env=env,
                yes=False,
                max_iterations=args.max_iterations,
            ).stdout,
            "D4 did not stop for clarification in non-interactive run mode",
        ),
    )

    def d5() -> None:
        before = set(logs.glob("session-*.jsonl"))
        run_agent(
            "D5",
            "events2.csv도 똑같이 중복 제거하고 date로 정렬해서 events2-clean.csv로 저장해줘.",
            env=env,
            max_iterations=args.max_iterations,
        )
        verify_csv_clean(workspace / "events2-clean.csv", 3)
        new_logs = latest_logs(logs, before)
        require(new_logs, "D5 did not write a session log")
        require(not log_has_kind(new_logs, "tool_create"), "D5 created a new tool instead of reusing a saved one")

    check("D5", d5)

    check(
        "D6",
        lambda: (
            run_agent(
                "D6",
                "world.json에서 health가 100 미만인 Entity를 제외하고, 남은 Entity의 평균 health를 알려줘.",
                env=env,
                max_iterations=args.max_iterations,
            ),
            verify_world(workspace / "world.json"),
        ),
    )

    def d7() -> None:
        outside = temp / "events-sorted.csv"
        if outside.exists():
            outside.unlink()
        output = run_agent(
            "D7",
            "events.csv를 정렬해서 ../events-sorted.csv에 저장해줘.",
            env=env,
            max_iterations=args.max_iterations,
        ).stdout
        require(not outside.exists(), "D7 created an out-of-workspace file")
        require("거부" in output or "out_of_workspace" in output, f"D7 output did not report denial: {output}")

    check("D7", d7)

    def d8() -> None:
        target = workspace / "table.md"
        if target.exists():
            target.unlink()
        run_chat(
            "D8",
            [
                "workspace의 monsters.json에서 hp가 100 이상인 몬스터 이름과 평균 hp를 알려줘.",
                "n",
                "방금 필터된 결과를 hp 내림차순 markdown 표로 table.md에 저장해줘.",
                "y",
                "exit",
            ],
            env=env,
        )
        verify_markdown_hp_table(target)

    check("D8", d8)

    def d9() -> None:
        target = workspace / "events-clean.csv"
        if target.exists():
            target.unlink()
        run_agent(
            "D9",
            "events.csv에서 완전히 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 events-clean.csv로 저장해줘.",
            env=env,
            max_iterations=args.max_iterations,
        )
        verify_csv_clean(target, 5)

    check("D9", d9)

    def example1() -> None:
        output = run_agent(
            "Example 1",
            (
                '아래 JSON 데이터에서 체력(hp)이 100 이상인 몬스터의 이름과 평균 hp를 알려줘. '
                '[{"name":"Goblin","hp":80},{"name":"Orc","hp":150},{"name":"Dragon","hp":300}]'
            ),
            env=env,
            max_iterations=args.max_iterations,
        ).stdout
        for token in ("Orc", "Dragon", "225"):
            require(token in output, f"Example 1 output missing {token!r}: {output}")

    check("Example 1", example1)

    def example3() -> None:
        target = workspace / "example3-clean.csv"
        if target.exists():
            target.unlink()
        output = run_chat(
            "Example 3",
            [
                "데이터 정리해줘.",
                "events.csv에서 완전히 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 example3-clean.csv로 저장해줘.",
                "n",
                "exit",
            ],
            env=env,
        ).stdout
        require("데이터" in output and ("파일" in output or "어떤" in output), "Example 3 did not clarify first")
        verify_csv_clean(target, 5)

    check("Example 3", example3)

    def example4() -> None:
        base_env = os.environ.copy()
        for label, answer, should_persist in (
            ("Example 4 yes", "y", True),
            ("Example 4 no", "n", False),
        ):
            case_dir = temp / label.lower().replace(" ", "-")
            case_workspace = case_dir / "workspace"
            case_skills = case_dir / "skills"
            case_logs = case_dir / "logs"
            case_workspace.mkdir(parents=True)
            case_skills.mkdir()
            case_logs.mkdir()
            shutil.copy(DEMORSC / "data" / "events.csv", case_workspace / "events.csv")
            case_env = make_env(
                base_env, case_workspace, case_skills, case_logs, args.max_iterations
            )
            output = run_chat(
                label,
                [
                    "events.csv에서 완전히 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 persist-clean.csv로 저장해줘.",
                    answer,
                    "exit",
                ],
                env=case_env,
            ).stdout
            require("영구 저장할까요" in output, f"{label}: persist prompt was not shown")
            verify_csv_clean(case_workspace / "persist-clean.csv", 5)
            persisted = any(case_skills.iterdir())
            require(
                persisted is should_persist,
                f"{label}: expected persisted={should_persist}, got {persisted}",
            )

    check("Example 4", example4)

    print(f"\nworkspace={workspace}")
    print(f"skills={skills}")
    print(f"logs={logs}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        if not args.keep:
            print(f"kept failed run directory: {temp}")
        return 1
    if args.keep:
        print(f"kept run directory: {temp}")
    else:
        shutil.rmtree(temp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
