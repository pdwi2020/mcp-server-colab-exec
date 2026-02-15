# mcp-server-colab-exec

<!-- mcp-name: io.github.pdwi2020/mcp-server-colab-exec -->

MCP server that allocates Google Colab GPU runtimes (T4/L4) and executes Python code on them. Lets any MCP-compatible AI assistant — Claude Code, Claude Desktop, Gemini CLI, Cline, and others — run GPU-accelerated code (CUDA, PyTorch, TensorFlow) without local GPU hardware.

## Prerequisites

- Python 3.10+
- A Google account with access to [Google Colab](https://colab.research.google.com)
- On first run, a browser window opens for OAuth2 consent. The token is cached at `~/.config/colab-exec/token.json` for subsequent runs.

## Installation

```bash
pip install mcp-server-colab-exec
```

Or run directly with `uvx`:

```bash
uvx mcp-server-colab-exec
```

## Configuration

### Claude Code

Add to your project's `.mcp.json` or `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "colab-exec": {
      "command": "mcp-server-colab-exec"
    }
  }
}
```

Or via the CLI:

```bash
claude mcp add colab-exec mcp-server-colab-exec
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "colab-exec": {
      "command": "mcp-server-colab-exec"
    }
  }
}
```

### Gemini CLI

```bash
gemini mcp add colab-exec -- mcp-server-colab-exec
```

## Tools

### `colab_execute`

Execute inline Python code on a Colab GPU runtime.

| Parameter     | Type   | Default | Description                              |
|---------------|--------|---------|------------------------------------------|
| `code`        | string | —       | Python code to execute (required)        |
| `accelerator` | string | `"T4"` | GPU type: `"T4"` (free) or `"L4"` (premium) |
| `timeout`     | int    | `300`   | Max execution time in seconds            |

Returns JSON with per-cell output, errors, and stderr.

### `colab_execute_file`

Execute a local `.py` file on a Colab GPU runtime.

| Parameter     | Type   | Default | Description                              |
|---------------|--------|---------|------------------------------------------|
| `file_path`   | string | —       | Path to a local `.py` file (required)    |
| `accelerator` | string | `"T4"` | GPU type: `"T4"` (free) or `"L4"` (premium) |
| `timeout`     | int    | `300`   | Max execution time in seconds            |

### `colab_execute_notebook`

Execute code and collect all generated artifacts (images, CSVs, models, etc.).

| Parameter     | Type   | Default | Description                              |
|---------------|--------|---------|------------------------------------------|
| `code`        | string | —       | Python code to execute (required)        |
| `output_dir`  | string | —       | Local directory for downloaded artifacts (required) |
| `accelerator` | string | `"T4"` | GPU type: `"T4"` (free) or `"L4"` (premium) |
| `timeout`     | int    | `300`   | Max execution time in seconds            |

Artifacts are downloaded as a zip and extracted into `output_dir`.

## Examples

**Check GPU availability:**
```
colab_execute(code="import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))")
```

**Run nvidia-smi:**
```
colab_execute(code="import subprocess; print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)")
```

**Train a model and download weights:**
```
colab_execute_notebook(
    code="import torch; model = torch.nn.Linear(10, 1); torch.save(model.state_dict(), '/tmp/model.pt')",
    output_dir="./outputs"
)
```

## Authentication

On first use, the server opens a browser window for Google OAuth2 consent. The access token and refresh token are cached at `~/.config/colab-exec/token.json`. Subsequent runs use the cached token and refresh it automatically.

The OAuth2 client credentials are the same ones used by the official Google Colab VS Code extension (`google.colab@0.3.0`). They are intentionally public.

## Troubleshooting

**"GPU quota exceeded"** — Colab has usage limits. Wait and retry, or use a different Google account.

**"Timed out creating kernel session"** — The runtime took too long to start. Retry — Colab sometimes has delays during peak usage.

**"Authentication failed"** — Delete `~/.config/colab-exec/token.json` and re-authenticate.

**OAuth browser window doesn't open** — Ensure you're running in an environment with a browser. For headless servers, authenticate on a machine with a browser first and copy the token file.

## License

MIT
