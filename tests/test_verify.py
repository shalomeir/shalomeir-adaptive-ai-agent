from adaptive_agent.verify import verify_file_exists, verify_row_count


def test_file_exists(tmp_path):
    (tmp_path / "out.csv").write_text("a\n1\n")
    v = verify_file_exists(tmp_path / "out.csv", must_contain="a")
    assert v.passed


def test_file_missing(tmp_path):
    v = verify_file_exists(tmp_path / "nope.csv")
    assert not v.passed
    assert "존재" in v.reason


def test_row_count():
    assert verify_row_count(["a", "b", "c"], expected=3).passed
    assert not verify_row_count(["a"], expected=3).passed
