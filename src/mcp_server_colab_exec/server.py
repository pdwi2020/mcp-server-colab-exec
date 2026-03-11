"""MCP server for executing Python code on Google Colab GPU runtimes.

Exposes three tools via the FastMCP API:
- colab_execute: Run inline Python code on a Colab GPU
- colab_execute_file: Run a local .py file on a Colab GPU
- colab_execute_notebook: Run code and collect generated artifacts
"""

import base64
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

from mcp.server.fastmcp import FastMCP

from .colab_runtime import (
    allocate_runtime,
    create_session,
    execute_code,
    get_credentials,
    start_keepalive,
    unassign_runtime,
)

mcp = FastMCP("colab-exec")

# ── Cell wrapping / parsing ──────────────────────────────────────────────────

CELL_START = "===CELL_START_{n}==="
CELL_END = "===CELL_END_{n}==="
ARTIFACT_B64_START = "ARTIFACT_BASE64_START"
ARTIFACT_B64_END = "ARTIFACT_BASE64_END"


def _wrap_cells(code: str) -> tuple[str, int]:
    """Wrap code in cell-boundary markers so output can be parsed per-cell."""
    raw_cells = re.split(r"\n{2,}", code.strip())
    cells = [c.strip() for c in raw_cells if c.strip()]
    if not cells:
        cells = [code]

    wrapped_parts = []
    for i, cell in enumerate(cells):
        marker_start = CELL_START.format(n=i)
        marker_end = CELL_END.format(n=i)
        wrapped_parts.append(
            f'print("{marker_start}", flush=True)\n'
            f"{cell}\n"
            f'print("{marker_end}", flush=True)'
        )
    return "\n\n".join(wrapped_parts), len(cells)


def _parse_cell_output(stdout: str, num_cells: int) -> list[dict]:
    """Parse cell outputs from stdout using cell boundary markers."""
    cells = []
    for i in range(num_cells):
        start_marker = CELL_START.format(n=i)
        end_marker = CELL_END.format(n=i)
        pattern = re.escape(start_marker) + r"\n?(.*?)\n?" + re.escape(end_marker)
        match = re.search(pattern, stdout, re.DOTALL)
        cell_stdout = match.group(1).strip() if match else ""
        cells.append({
            "cell_num": i,
            "stdout": cell_stdout,
            "status": "ok" if match else "no_output",
        })
    return cells


def _extract_artifact_b64(stdout: str) -> str | None:
    """Extract base64-encoded artifact zip from stdout."""
    pattern = re.escape(ARTIFACT_B64_START) + r"\n(.*?)\n" + re.escape(ARTIFACT_B64_END)
    match = re.search(pattern, stdout, re.DOTALL)
    return match.group(1).strip() if match else None


def _validate_workspace_py_file(file_path: str) -> tuple[Path | None, str | None]:
    """Validate file path for colab_execute_file security policy."""
    expanded = Path(file_path).expanduser()
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError:
        return None, f"File not found: {expanded}"
    except OSError as exc:
        return None, f"Invalid file path: {exc}"

    if not resolved.is_file():
        return None, f"Not a regular file: {resolved}"

    if resolved.suffix.lower() != ".py":
        return None, "file_path must point to a .py file"

    workspace_root = Path.cwd().resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None, f"file_path must be within workspace root: {workspace_root}"

    return resolved, None


def _is_unsafe_zip_entry(entry: zipfile.ZipInfo) -> bool:
    """Return True when entry is symlink or special file type."""
    mode = (entry.external_attr >> 16) & 0xFFFF
    if mode == 0:
        return False
    file_type = stat.S_IFMT(mode)
    if file_type == 0:
        # Some ZIP creators omit unix file type bits; treat as regular.
        return False
    return file_type not in (stat.S_IFREG, stat.S_IFDIR)


def _safe_extract_zip(zip_path: str, output_dir: str) -> list[str]:
    """Safely extract zip while preventing zip-slip and special-file entries."""
    output_root = Path(output_dir).resolve()
    planned_entries: list[tuple[zipfile.ZipInfo, Path]] = []
    extracted_files: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.infolist():
            if not entry.filename:
                raise ValueError("Unsafe zip member: empty filename")
            if "\x00" in entry.filename:
                raise ValueError(f"Unsafe zip member: {entry.filename}")

            entry_path = PurePosixPath(entry.filename)
            if entry_path.is_absolute():
                raise ValueError(f"Unsafe zip member path: {entry.filename}")
            if ".." in entry_path.parts:
                raise ValueError(f"Unsafe zip member path: {entry.filename}")
            if _is_unsafe_zip_entry(entry):
                raise ValueError(f"Unsafe zip member type: {entry.filename}")

            destination = (output_root / Path(*entry_path.parts)).resolve()
            try:
                destination.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path: {entry.filename}") from exc

            planned_entries.append((entry, destination))

        for entry, destination in planned_entries:
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry, "r") as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files.append(entry.filename)

    return extracted_files


# ── Runtime lifecycle helper ─────────────────────────────────────────────────

def _run_on_colab(code: str, accelerator: str, timeout: int) -> tuple[str, str, int]:
    """Allocate a runtime, execute code, release, and return (stdout, stderr, exit_code)."""
    creds = get_credentials()
    access_token = creds.token

    assignment = allocate_runtime(access_token, accelerator)
    stop_event = start_keepalive(access_token, assignment["endpoint"])

    try:
        kernel_id = create_session(assignment["proxy_url"], assignment["proxy_token"])
        stdout, stderr, exit_code = execute_code(
            assignment["proxy_url"],
            assignment["proxy_token"],
            kernel_id,
            code,
            timeout=timeout,
            access_token=access_token,
            endpoint=assignment["endpoint"],
        )
    finally:
        stop_event.set()
        unassign_runtime(access_token, assignment["endpoint"])

    return stdout, stderr, exit_code


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool(annotations={"title": "Execute Python Code on GPU", "readOnlyHint": False, "destructiveHint": False})
def colab_execute(code: str, accelerator: str = "T4", timeout: int = 300) -> str:
    """Execute Python code on a Google Colab GPU runtime.

    Allocates a T4 or L4 GPU, runs the code, and returns structured JSON
    with per-cell output, errors, and stderr.

    Args:
        code: Python code to execute on the Colab GPU runtime.
        accelerator: GPU type — "T4" (free-tier) or "L4" (premium). Default: "T4".
        timeout: Max execution time in seconds. Default: 300.
    """
    wrapped, num_cells = _wrap_cells(code)
    stdout, stderr, rc = _run_on_colab(wrapped, accelerator, timeout)
    cells = _parse_cell_output(stdout, num_cells)
    errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []
    result = {"cells": cells, "errors": errors, "stderr": stderr, "exit_code": rc}
    return json.dumps(result, indent=2)


@mcp.tool(annotations={"title": "Execute Python File on GPU", "readOnlyHint": False, "destructiveHint": False})
def colab_execute_file(file_path: str, accelerator: str = "T4", timeout: int = 300) -> str:
    """Execute a local Python file on a Google Colab GPU runtime.

    Reads the file contents and sends them for execution on a Colab GPU.

    Args:
        file_path: Path to a local .py file to execute on Colab.
        accelerator: GPU type — "T4" (free-tier) or "L4" (premium). Default: "T4".
        timeout: Max execution time in seconds. Default: 300.
    """
    resolved_file_path, error = _validate_workspace_py_file(file_path)
    if error:
        return json.dumps({"error": error})

    with resolved_file_path.open() as f:
        code = f.read()

    wrapped, num_cells = _wrap_cells(code)
    stdout, stderr, rc = _run_on_colab(wrapped, accelerator, timeout)
    cells = _parse_cell_output(stdout, num_cells)
    errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []
    result = {"cells": cells, "errors": errors, "stderr": stderr, "exit_code": rc}
    return json.dumps(result, indent=2)


@mcp.tool(annotations={"title": "Execute Code and Download Artifacts", "readOnlyHint": False, "destructiveHint": False})
def colab_execute_notebook(
    code: str, output_dir: str, accelerator: str = "T4", timeout: int = 300
) -> str:
    """Execute Python code on Colab GPU and collect generated artifacts.

    Runs the code, then scans the runtime for output files (images, CSVs,
    models, etc.), zips them, and downloads to a local directory.

    Args:
        code: Python code to execute on the Colab GPU runtime.
        output_dir: Local directory to save the artifacts zip and extracted files.
        accelerator: GPU type — "T4" (free-tier) or "L4" (premium). Default: "T4".
        timeout: Max execution time in seconds. Default: 300.
    """
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    artifact_code = '''

# --- colab-exec artifact collection ---
import os, zipfile, base64, glob

_artifact_dir = "/tmp/colab_artifacts"
os.makedirs(_artifact_dir, exist_ok=True)

_scan_dirs = ["/tmp", os.getcwd(), "/content"]
_skip_prefixes = ["/tmp/colab_artifacts", "/tmp/."]
_collected = []
for _sd in _scan_dirs:
    if not os.path.isdir(_sd):
        continue
    for _root, _dirs, _files in os.walk(_sd):
        _dirs[:] = [d for d in _dirs if not d.startswith('.')]
        if any(_root.startswith(p) for p in _skip_prefixes):
            continue
        for _f in _files:
            _fp = os.path.join(_root, _f)
            if _f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.csv', '.json',
                           '.txt', '.pt', '.pth', '.h5', '.pkl', '.npy', '.npz',
                           '.onnx', '.mp4', '.wav', '.mp3', '.pdf')):
                try:
                    if os.path.getsize(_fp) < 50 * 1024 * 1024:
                        _collected.append(_fp)
                except OSError:
                    pass

if _collected:
    _zip_path = "/tmp/colab_artifacts.zip"
    with zipfile.ZipFile(_zip_path, 'w', zipfile.ZIP_DEFLATED) as _zf:
        for _fp in _collected:
            _zf.write(_fp, os.path.basename(_fp))
    with open(_zip_path, 'rb') as _zfh:
        _b64 = base64.b64encode(_zfh.read()).decode('ascii')
    print("ARTIFACT_BASE64_START")
    print(_b64)
    print("ARTIFACT_BASE64_END")
    print(f"[colab-exec] Collected {len(_collected)} artifact(s)", flush=True)
else:
    print("[colab-exec] No artifacts found to collect", flush=True)
'''
    full_code = code + artifact_code
    wrapped, num_cells = _wrap_cells(full_code)
    stdout, stderr, rc = _run_on_colab(wrapped, accelerator, timeout)
    cells = _parse_cell_output(stdout, num_cells)
    errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []

    # Extract and save artifacts
    artifact_files = []
    artifacts_zip_path = None
    b64_data = _extract_artifact_b64(stdout)
    if b64_data:
        try:
            zip_bytes = base64.b64decode(b64_data)
            artifacts_zip_path = os.path.join(output_dir, "colab_artifacts.zip")
            with open(artifacts_zip_path, "wb") as f:
                f.write(zip_bytes)
            artifact_files = _safe_extract_zip(artifacts_zip_path, output_dir)
        except Exception as e:
            errors.append({"artifact_error": str(e)})

    result = {
        "cells": cells,
        "errors": errors,
        "artifacts_zip": artifacts_zip_path,
        "artifact_files": artifact_files,
        "stderr": stderr,
        "exit_code": rc,
    }
    return json.dumps(result, indent=2)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
