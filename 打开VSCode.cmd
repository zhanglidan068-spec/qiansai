@echo off
setlocal
set ELECTRON_RUN_AS_NODE=
start "" "D:\Microsoft VS Code\Code.exe" --user-data-dir "%~dp0.vscode-user" --extensions-dir "%~dp0.vscode-extensions" --new-window "%~dp0."
endlocal
