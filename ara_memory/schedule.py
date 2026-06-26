from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ScheduleScript:
    path: Path
    uninstall_path: Path
    status_path: Path
    log_path: Path
    task_name: str
    command: str
    interval_minutes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "uninstall_path": str(self.uninstall_path),
            "status_path": str(self.status_path),
            "log_path": str(self.log_path),
            "task_name": self.task_name,
            "command": self.command,
            "interval_minutes": self.interval_minutes,
        }


def write_windows_worker_task_script(
    *,
    output: Path,
    repo: Path,
    task_name: str = "AraMemoryWorker",
    interval_minutes: int = 5,
    scope: str = "ara-memory",
    python_executable: Path | None = None,
    regression_manifest: Path | None = None,
    regression_baseline: Path | None = None,
    log_path: Path | None = None,
) -> ScheduleScript:
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")
    python_executable = python_executable or Path(sys.executable)
    regression_manifest = regression_manifest or repo / "examples" / "recall_regression_manifest.json"
    regression_baseline = regression_baseline or repo / ".ara-memory" / "archive" / "recall-regression-baseline.json"
    log_path = log_path or repo / ".ara-memory" / "logs" / "worker-task.log"
    command_parts = [
        str(python_executable),
        "-m",
        "ara_memory",
        "worker-loop",
        "--scope",
        scope,
        "--iterations",
        "1",
        "--interval-seconds",
        "0",
        "--doctor-query",
        "scheduled memory worker health",
        "--regression-manifest",
        str(regression_manifest),
        "--regression-baseline",
        str(regression_baseline),
        "--report-item-limit",
        "3",
    ]
    command = _join_command(command_parts)
    script = _powershell_task_script(
        task_name=task_name,
        repo=repo,
        command=command,
        interval_minutes=interval_minutes,
        log_path=log_path,
    )
    uninstall_script = _powershell_uninstall_script(task_name=task_name)
    status_script = _powershell_status_script(task_name=task_name, log_path=log_path)
    uninstall_path = output.with_name("uninstall-worker-task.ps1")
    status_path = output.with_name("status-worker-task.ps1")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script, encoding="utf-8")
    uninstall_path.write_text(uninstall_script, encoding="utf-8")
    status_path.write_text(status_script, encoding="utf-8")
    return ScheduleScript(
        path=output,
        uninstall_path=uninstall_path,
        status_path=status_path,
        log_path=log_path,
        task_name=task_name,
        command=command,
        interval_minutes=interval_minutes,
    )


def _powershell_task_script(*, task_name: str, repo: Path, command: str, interval_minutes: int, log_path: Path) -> str:
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$TaskName = '{_ps(task_name)}'",
            f"$WorkingDirectory = '{_ps(str(repo))}'",
            f"$LogPath = '{_ps(str(log_path))}'",
            f"$Command = '{_ps(command)}'",
            "$LogDirectory = Split-Path -Parent $LogPath",
            "New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null",
            "$RunLines = @(",
            "    \"Set-Location -LiteralPath '$WorkingDirectory'\",",
            "    \"`$ErrorActionPreference = 'Continue'\",",
            "    \"`$stamp = Get-Date -Format o\",",
            "    \"Add-Content -LiteralPath '$LogPath' -Value \\\"[`$stamp] starting Ara Memory worker\\\"\",",
            "    'try {',",
            "    \"    $Command *>> '$LogPath'\",",
            "    '    $exit = $LASTEXITCODE',",
            "    \"    Add-Content -LiteralPath '$LogPath' -Value \\\"[`$(Get-Date -Format o)] worker exited with code `$exit\\\"\",",
            "    '    exit $exit',",
            "    '} catch {',",
            "    \"    Add-Content -LiteralPath '$LogPath' -Value \\\"[`$(Get-Date -Format o)] worker exception: `$(`$_.Exception.Message)\\\"\",",
            "    '    exit 1',",
            "    '}'",
            ")",
            "$Run = $RunLines -join '; '",
            "$EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Run))",
            "$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument \"-NoProfile -ExecutionPolicy Bypass -EncodedCommand $EncodedCommand\"",
            f"$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes {interval_minutes})",
            "$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew",
            "Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description 'Ara Memory OS background worker loop' -Force",
            "Write-Host \"Registered scheduled task: $TaskName\"",
            "Write-Host \"Worker command: $Command\"",
            "Write-Host \"Worker log: $LogPath\"",
            "",
        ]
    )


def _powershell_uninstall_script(*, task_name: str) -> str:
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$TaskName = '{_ps(task_name)}'",
            "$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue",
            "if ($null -eq $Task) {",
            "    Write-Host \"Scheduled task not found: $TaskName\"",
            "    exit 0",
            "}",
            "Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false",
            "Write-Host \"Unregistered scheduled task: $TaskName\"",
            "",
        ]
    )


def _powershell_status_script(*, task_name: str, log_path: Path) -> str:
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$TaskName = '{_ps(task_name)}'",
            f"$LogPath = '{_ps(str(log_path))}'",
            "$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue",
            "if ($null -eq $Task) {",
            "    Write-Host \"Scheduled task not found: $TaskName\"",
            "    exit 1",
            "}",
            "Get-ScheduledTaskInfo -TaskName $TaskName | Format-List *",
            "Write-Host \"Task state: $($Task.State)\"",
            "Write-Host \"Worker log: $LogPath\"",
            "if (Test-Path -LiteralPath $LogPath) {",
            "    Write-Host \"Recent worker log lines:\"",
            "    Get-Content -LiteralPath $LogPath -Tail 40",
            "} else {",
            "    Write-Host \"Worker log does not exist yet.\"",
            "}",
            "",
        ]
    )


def _join_command(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part or "\\" in part else part for part in parts)


def _ps(value: str) -> str:
    return value.replace("'", "''")
