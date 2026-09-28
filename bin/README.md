# bin/

Prebuilt llama.cpp binaries are downloaded into this folder by
`scripts/setup.sh` (macOS/Linux) or `scripts/setup.ps1` (Windows).

Expected after setup:
- `llama-server` (or `llama-server.exe`)
- `ggml-rpc-server` (or `ggml-rpc-server.exe`)
- Shared libraries (`.dylib` on macOS, `.so` on Linux, `.dll` on Windows)

This folder is git-ignored; the files come from GitHub Releases.
