# Privacy Policy

**mcp-server-colab-exec**
Last updated: February 15, 2026

## Overview

mcp-server-colab-exec is an open-source MCP server that executes Python code on Google Colab GPU runtimes. This policy explains what data the server handles and how.

## Data Collection

This server does **not** collect, store, or transmit any personal data to the developer or any third party. There is no analytics, telemetry, or tracking of any kind.

## Data Handling

### What the server processes

- **Python code** you submit for execution — sent to Google Colab's API over HTTPS
- **Execution results** (stdout, stderr, errors) — returned from Google Colab to your local machine
- **Generated artifacts** (images, files, models) — downloaded from Google Colab to a local directory you specify

### Where data goes

All code and data are transmitted directly between your local machine and Google Colab's API. The server acts as a local bridge and does not route data through any intermediary servers.

### Authentication tokens

- Google OAuth2 tokens are stored locally at `~/.config/colab-exec/token.json`
- Tokens are never transmitted anywhere other than to Google's authentication endpoints
- You can revoke access at any time by deleting this file or revoking the app in your [Google Account settings](https://myaccount.google.com/permissions)

### Data retention

- The server stores no data beyond the current session
- OAuth2 tokens are cached locally until manually deleted or expired
- No execution history, logs, or code is retained by the server

## Third-Party Services

This server connects to **Google Colab** (colab.research.google.com) to allocate GPU runtimes and execute code. Your use of Google Colab is governed by [Google's Terms of Service](https://policies.google.com/terms) and [Privacy Policy](https://policies.google.com/privacy).

The OAuth2 client credentials used are the same public credentials from the official Google Colab VS Code extension.

## Security

- All communication with Google Colab uses HTTPS/TLS encryption
- OAuth2 tokens use refresh token rotation
- No secrets or credentials are hardcoded or transmitted to third parties

## Your Rights

Since no personal data is collected or stored by this server, there is no personal data to access, modify, or delete. You can:

- Delete your local OAuth2 token at any time (`~/.config/colab-exec/token.json`)
- Revoke Google OAuth2 access from your Google Account
- Uninstall the server at any time (`pip uninstall mcp-server-colab-exec`)

## Open Source

This server is fully open source under the MIT License. You can audit the complete source code at:
https://github.com/pdwi2020/mcp-server-colab-exec

## Contact

For privacy questions or concerns, open an issue at:
https://github.com/pdwi2020/mcp-server-colab-exec/issues

Or email: paritoshdwi2019@gmail.com
