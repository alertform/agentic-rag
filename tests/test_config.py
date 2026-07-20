import os

from agentic_search.config import _load_dotenv


def test_load_dotenv_sets_missing_and_respects_existing(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text(
        "# 注释\nDOTENV_TEST_A=1\nDOTENV_TEST_B = spaced\n\n没有等号的行\n", encoding="utf-8"
    )
    monkeypatch.delenv("DOTENV_TEST_A", raising=False)
    monkeypatch.setenv("DOTENV_TEST_B", "keep")
    _load_dotenv(envfile)
    try:
        assert os.environ["DOTENV_TEST_A"] == "1"
        assert os.environ["DOTENV_TEST_B"] == "keep"  # 真实环境变量优先
    finally:
        os.environ.pop("DOTENV_TEST_A", None)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    _load_dotenv(tmp_path / "nope.env")  # 不抛异常即可
