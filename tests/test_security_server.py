import base64
import io
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp_server_colab_exec import server


def _build_artifact_stdout(zip_entries: list[tuple[str, str]]) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in zip_entries:
            zf.writestr(name, content)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f"{server.CELL_START.format(n=0)}\ncell output\n{server.CELL_END.format(n=0)}\n"
        f"{server.ARTIFACT_B64_START}\n{b64}\n{server.ARTIFACT_B64_END}\n"
    )


def _build_symlink_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("link.txt")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "target")


def _deny_run_on_colab(*args, **kwargs):
    raise AssertionError("_run_on_colab should not be called for denied file paths")


def test_colab_execute_file_allows_workspace_py(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text("print('ok')\n")
    monkeypatch.chdir(tmp_path)

    called = {}

    def fake_run(code: str, accelerator: str, timeout: int):
        called["code"] = code
        called["accelerator"] = accelerator
        called["timeout"] = timeout
        stdout = f"{server.CELL_START.format(n=0)}\nok\n{server.CELL_END.format(n=0)}\n"
        return stdout, "", 0

    monkeypatch.setattr(server, "_run_on_colab", fake_run)
    result = json.loads(server.colab_execute_file(str(script), accelerator="T4", timeout=30))

    assert "error" not in result
    assert result["exit_code"] == 0
    assert "print('ok')" in called["code"]
    assert called["accelerator"] == "T4"
    assert called["timeout"] == 30


def test_colab_execute_file_rejects_non_py(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "_run_on_colab", _deny_run_on_colab)

    result = json.loads(server.colab_execute_file(str(txt_file)))
    assert "error" in result
    assert ".py" in result["error"]


def test_colab_execute_file_rejects_outside_workspace_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(server, "_run_on_colab", _deny_run_on_colab)

    result = json.loads(server.colab_execute_file(str(outside.resolve())))
    assert "error" in result
    assert "workspace root" in result["error"]


def test_colab_execute_file_rejects_traversal_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(server, "_run_on_colab", _deny_run_on_colab)

    result = json.loads(server.colab_execute_file("../outside.py"))
    assert "error" in result
    assert "workspace root" in result["error"]


def test_colab_execute_file_rejects_symlink_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    if os.name == "nt":
        pytest.skip("Symlink behavior varies on Windows and may require admin privileges.")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n")
    link = workspace / "escape.py"
    link.symlink_to(outside)

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(server, "_run_on_colab", _deny_run_on_colab)
    result = json.loads(server.colab_execute_file(str(link)))

    assert "error" in result
    assert "workspace root" in result["error"]


def test_safe_extract_allows_normal_members(tmp_path: Path):
    zip_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("artifact.txt", "data")
        zf.writestr("nested/result.json", '{"ok": true}')

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    extracted = server._safe_extract_zip(str(zip_path), str(out_dir))

    assert extracted == ["artifact.txt", "nested/result.json"]
    assert (out_dir / "artifact.txt").read_text() == "data"
    assert (out_dir / "nested" / "result.json").read_text() == '{"ok": true}'


def test_safe_extract_rejects_dotdot_member(tmp_path: Path):
    zip_path = tmp_path / "unsafe_dotdot.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../evil.txt", "bad")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Unsafe zip member path"):
        server._safe_extract_zip(str(zip_path), str(out_dir))
    assert list(out_dir.rglob("*")) == []


def test_safe_extract_rejects_absolute_member(tmp_path: Path):
    zip_path = tmp_path / "unsafe_absolute.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("/abs.txt", "bad")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Unsafe zip member path"):
        server._safe_extract_zip(str(zip_path), str(out_dir))
    assert list(out_dir.rglob("*")) == []


def test_safe_extract_rejects_symlink_member(tmp_path: Path):
    zip_path = tmp_path / "unsafe_symlink.zip"
    _build_symlink_zip(zip_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Unsafe zip member type"):
        server._safe_extract_zip(str(zip_path), str(out_dir))
    assert list(out_dir.rglob("*")) == []


def test_colab_execute_notebook_safe_zip_populates_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stdout = _build_artifact_stdout([("artifact.txt", "model-bytes")])
    monkeypatch.setattr(server, "_run_on_colab", lambda *args, **kwargs: (stdout, "", 0))

    out_dir = tmp_path / "output"
    result = json.loads(server.colab_execute_notebook("print('x')", str(out_dir)))

    assert result["errors"] == []
    assert result["artifact_files"] == ["artifact.txt"]
    assert (out_dir / "artifact.txt").read_text() == "model-bytes"
    assert (out_dir / "colab_artifacts.zip").exists()


def test_colab_execute_notebook_unsafe_zip_reports_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stdout = _build_artifact_stdout([("../evil.txt", "bad")])
    monkeypatch.setattr(server, "_run_on_colab", lambda *args, **kwargs: (stdout, "", 0))

    out_dir = tmp_path / "output"
    result = json.loads(server.colab_execute_notebook("print('x')", str(out_dir)))

    assert result["artifact_files"] == []
    assert any("artifact_error" in e for e in result["errors"])
    assert not (tmp_path / "evil.txt").exists()
    assert not (out_dir / "evil.txt").exists()
