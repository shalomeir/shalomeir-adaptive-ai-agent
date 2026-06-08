from adaptive_agent.sandbox import ExecutionSandbox


def test_runs_and_captures_stdout(tmp_path):
    sb = ExecutionSandbox(workspace=tmp_path, timeout_sec=5, max_output_bytes=1000)
    res = sb.run_code("print('hello')")
    assert res.exit_code == 0
    assert "hello" in res.stdout
    assert res.timed_out is False


def test_timeout(tmp_path):
    sb = ExecutionSandbox(workspace=tmp_path, timeout_sec=1, max_output_bytes=1000)
    res = sb.run_code("import time\ntime.sleep(5)")
    assert res.timed_out is True


def test_output_truncated(tmp_path):
    sb = ExecutionSandbox(workspace=tmp_path, timeout_sec=5, max_output_bytes=50)
    res = sb.run_code("print('x' * 1000)")
    assert res.truncated is True
    assert len(res.stdout) <= 50
