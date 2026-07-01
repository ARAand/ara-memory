from __future__ import annotations

import json
import re
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


@dataclass(slots=True)
class ScheduleVerification:
    path: Path
    uninstall_path: Path
    status_path: Path
    passed: bool
    issues: list[str]
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "uninstall_path": str(self.uninstall_path),
            "status_path": str(self.status_path),
            "passed": self.passed,
            "issues": self.issues,
            "details": self.details,
        }

    def to_text(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"{status}: worker schedule script readiness",
            "",
            f"Install script: {self.path}",
            f"Status script: {self.status_path}",
            f"Uninstall script: {self.uninstall_path}",
            "",
            "Static guarantees:",
            "- install/status/uninstall scripts exist",
            "- scheduled action uses structured worker arguments",
            "- worker-loop runs as one external-scheduler iteration",
            "- install script runs the same worker command once as a preflight before registration",
            "- overlapping scheduled starts are ignored",
            "- recall-regression manifest and baseline are valid when required",
            f"- repetition interval is <= {self.details.get('max_interval_minutes')} minutes",
            "",
            "Not checked:",
            "- install script has been executed and preflight has passed on this machine",
            "- task installed or enabled in Windows Task Scheduler after installation",
            "- latest worker run passed",
            "- worker log health",
        ]
        if self.issues:
            lines.extend(["", "Issues:", *[f"- {issue}" for issue in self.issues]])
        if "missing WorkerArgs" in " ".join(self.issues):
            lines.extend(
                [
                    "",
                    "Regenerate scripts with worker-schedule, then run worker-schedule-verify again.",
                ]
            )
        return "\n".join(lines)


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
    _validate_scheduler_label(scope, "scope")
    _validate_scheduler_label(task_name, "task_name")
    python_executable = python_executable or Path(sys.executable)
    regression_manifest = regression_manifest or repo / "examples" / "recall_regression_manifest.json"
    regression_baseline = regression_baseline or repo / ".ara-memory" / "archive" / "recall-regression-baseline.json"
    log_path = log_path or repo / ".ara-memory" / "logs" / "worker-task.log"
    worker_args = [
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
        "--regression-baseline-warn-only",
        "--report-item-limit",
        "3",
    ]
    command = _join_command([str(python_executable), *worker_args])
    script = _powershell_task_script(
        task_name=task_name,
        repo=repo,
        python_path=python_executable,
        worker_args=worker_args,
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


def verify_windows_worker_task_script(
    *,
    output: Path,
    scope: str = "ara-memory",
    max_interval_minutes: int = 15,
    require_regression_gates: bool = True,
) -> ScheduleVerification:
    uninstall_path = output.with_name("uninstall-worker-task.ps1")
    status_path = output.with_name("status-worker-task.ps1")
    issues: list[str] = []
    details: dict[str, Any] = {
        "scope": scope,
        "max_interval_minutes": max_interval_minutes,
        "require_regression_gates": require_regression_gates,
    }
    for label, path in (
        ("install", output),
        ("uninstall", uninstall_path),
        ("status", status_path),
    ):
        if not path.exists():
            issues.append(f"missing {label} script: {path}")

    install_text = _read_text(output)
    uninstall_text = _read_text(uninstall_path)
    status_text = _read_text(status_path)
    python_path = _extract_ps_assignment(install_text, "PythonPath")
    worker_args = _extract_ps_array_assignment(install_text, "WorkerArgs")
    command = _extract_ps_assignment(install_text, "Command")
    task_name = _extract_ps_assignment(install_text, "TaskName")
    uninstall_task_name = _extract_ps_assignment(uninstall_text, "TaskName")
    status_task_name = _extract_ps_assignment(status_text, "TaskName")
    log_path = _extract_ps_assignment(install_text, "LogPath")
    status_log_path = _extract_ps_assignment(status_text, "LogPath")
    interval = _extract_interval_minutes(install_text)
    command_tokens = _split_joined_command(command)
    critical_options = (
        "--scope",
        "--iterations",
        "--interval-seconds",
        "--regression-manifest",
        "--regression-baseline",
    )
    option_values = {option: _command_arg_values(worker_args, option) for option in critical_options}
    worker_scope = option_values["--scope"][-1] if option_values["--scope"] else ""
    worker_iterations = option_values["--iterations"][-1] if option_values["--iterations"] else ""
    worker_interval_seconds = (
        option_values["--interval-seconds"][-1] if option_values["--interval-seconds"] else ""
    )
    regression_manifest = option_values["--regression-manifest"][-1] if option_values["--regression-manifest"] else ""
    regression_baseline = option_values["--regression-baseline"][-1] if option_values["--regression-baseline"] else ""
    regression_baseline_warn_only = "--regression-baseline-warn-only" in worker_args
    details.update(
        {
            "task_name": task_name,
            "uninstall_task_name": uninstall_task_name,
            "status_task_name": status_task_name,
            "python_path": python_path,
            "worker_args": worker_args,
            "command": command,
            "command_tokens": command_tokens,
            "log_path": log_path,
            "status_log_path": status_log_path,
            "interval_minutes": interval,
            "worker_scope": worker_scope,
            "worker_iterations": worker_iterations,
            "worker_interval_seconds": worker_interval_seconds,
            "regression_manifest": regression_manifest,
            "regression_baseline": regression_baseline,
            "regression_baseline_warn_only": regression_baseline_warn_only,
        }
    )

    required_install = {
        "scheduled task registration": "Register-ScheduledTask",
        "encoded powershell action": "-EncodedCommand",
        "worker argv splatting": "@WorkerArgs",
        "python path invocation": "& `$PythonPath @WorkerArgs",
        "utf8 worker log append": "Out-File -LiteralPath `$LogPath -Append -Encoding utf8",
        "worker preflight invocation": "& $PythonPath @WorkerArgs",
        "worker preflight failure gate": "Worker preflight failed",
        "baseline drift warn policy": "--regression-baseline-warn-only",
        "overlap prevention": "MultipleInstances IgnoreNew",
        "execution time limit": "ExecutionTimeLimit",
        "missed run catch-up": "StartWhenAvailable",
        "short restart policy": "RestartCount",
    }
    for label, snippet in required_install.items():
        if snippet not in install_text:
            issues.append(f"install script missing {label}")
    for option, values in option_values.items():
        if len(values) > 1:
            issues.append(f"install script has duplicate {option} arguments")
    if not worker_args:
        issues.append("install script missing WorkerArgs array")
    if command_tokens and [python_path, *worker_args] != command_tokens:
        issues.append("install script command preview does not match PythonPath and WorkerArgs")
    if not command_tokens:
        issues.append("install script missing command preview")
    if not regression_baseline_warn_only:
        issues.append("install script missing --regression-baseline-warn-only")
    preflight_index = install_text.find("Worker preflight failed")
    registration_index = install_text.find("Register-ScheduledTask")
    if preflight_index == -1 or registration_index == -1 or preflight_index > registration_index:
        issues.append("install script must run worker preflight before Register-ScheduledTask")
    if len(worker_args) < 3 or worker_args[:3] != ["-m", "ara_memory", "worker-loop"]:
        issues.append("install script must run ara_memory worker-loop")
    if worker_scope != scope:
        issues.append(f"install script scope {worker_scope!r} does not match expected {scope!r}")
    if task_name and uninstall_task_name and task_name != uninstall_task_name:
        issues.append("uninstall script task name does not match install script")
    if task_name and status_task_name and task_name != status_task_name:
        issues.append("status script task name does not match install script")
    if log_path and status_log_path and log_path != status_log_path:
        issues.append("status script log path does not match install script")
    manifest_case_names: set[str] = set()
    if require_regression_gates:
        if not regression_manifest:
            issues.append("install script missing --regression-manifest value")
        else:
            manifest_case_names = _regression_manifest_case_names(Path(regression_manifest))
        if regression_manifest and not manifest_case_names:
            issues.append(f"regression manifest is missing or invalid: {regression_manifest}")
        if not regression_baseline:
            issues.append("install script missing --regression-baseline value")
        elif not _valid_regression_baseline(Path(regression_baseline), manifest_case_names):
            issues.append(f"regression baseline is missing or invalid: {regression_baseline}")
    if worker_iterations != "1":
        issues.append("install script must run worker-loop with --iterations 1")
    if worker_interval_seconds != "0":
        issues.append("install script must run worker-loop with --interval-seconds 0")
    if interval is None:
        issues.append("install script missing repetition interval")
    elif interval > max_interval_minutes:
        issues.append(f"interval {interval} exceeds max {max_interval_minutes} minutes")
    if "Unregister-ScheduledTask" not in uninstall_text:
        issues.append("uninstall script missing Unregister-ScheduledTask")
    if "Get-ScheduledTaskInfo" not in status_text:
        issues.append("status script missing Get-ScheduledTaskInfo")
    if "Get-Content -LiteralPath $LogPath -Tail 40" not in status_text:
        issues.append("status script missing recent log tail")
    return ScheduleVerification(
        path=output,
        uninstall_path=uninstall_path,
        status_path=status_path,
        passed=not issues,
        issues=issues,
        details=details,
    )


def _powershell_task_script(
    *,
    task_name: str,
    repo: Path,
    python_path: Path,
    worker_args: list[str],
    command: str,
    interval_minutes: int,
    log_path: Path,
) -> str:
    worker_arg_lines = [
        f"    '{_ps(arg)}'" + ("," if index < len(worker_args) - 1 else "")
        for index, arg in enumerate(worker_args)
    ]
    args_literal = _ps_array_literal(worker_args)
    lines = [
            "$ErrorActionPreference = 'Stop'",
            f"$TaskName = '{_ps(task_name)}'",
            f"$WorkingDirectory = '{_ps(str(repo))}'",
            f"$LogPath = '{_ps(str(log_path))}'",
            f"$PythonPath = '{_ps(str(python_path))}'",
            "$WorkerArgs = @(",
            *worker_arg_lines,
            ")",
            f"$Command = '{_ps(command)}'",
            "$LogDirectory = Split-Path -Parent $LogPath",
            "New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null",
            "Write-Host \"Running worker preflight before registering scheduled task...\"",
            "& $PythonPath @WorkerArgs",
            "$PreflightExit = $LASTEXITCODE",
            "if ($null -eq $PreflightExit) {",
            "    $PreflightExit = 0",
            "}",
            "if ($PreflightExit -ne 0) {",
            "    throw \"Worker preflight failed with exit code $PreflightExit; fix worker gates before registering scheduled task.\"",
            "}",
            "Write-Host \"Worker preflight passed.\"",
            "$RunLines = @(",
            f"    \"`$WorkingDirectory = '{_ps(str(repo))}'\",",
            f"    \"`$LogPath = '{_ps(str(log_path))}'\",",
            f"    \"`$PythonPath = '{_ps(str(python_path))}'\",",
            f"    \"`$WorkerArgs = {args_literal}\",",
            "    \"Set-Location -LiteralPath `$WorkingDirectory\",",
            "    \"`$ErrorActionPreference = 'Continue'\",",
            "    \"`$stamp = Get-Date -Format o\",",
            "    \"Add-Content -LiteralPath `$LogPath -Value `\"[`$stamp] starting Ara Memory worker`\"\",",
            "    'try {',",
            "    \"    & `$PythonPath @WorkerArgs 2>&1 | Out-File -LiteralPath `$LogPath -Append -Encoding utf8\",",
            "    '    $exit = $LASTEXITCODE',",
            "    \"    Add-Content -LiteralPath `$LogPath -Value `\"[`$(Get-Date -Format o)] worker exited with code `$exit`\"\",",
            "    '    exit $exit',",
            "    '} catch {',",
            "    \"    Add-Content -LiteralPath `$LogPath -Value `\"[`$(Get-Date -Format o)] worker exception: `$(`$_.Exception.Message)`\"\",",
            "    '    exit 1',",
            "    '}'",
            ")",
            "$Run = $RunLines -join '; '",
            "$EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Run))",
            "$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument \"-NoProfile -ExecutionPolicy Bypass -EncodedCommand $EncodedCommand\"",
            f"$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes {interval_minutes})",
            "$Settings = New-ScheduledTaskSettingsSet "
            "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
            "-MultipleInstances IgnoreNew "
            "-ExecutionTimeLimit (New-TimeSpan -Minutes 10) "
            "-StartWhenAvailable -RestartCount 2 "
            "-RestartInterval (New-TimeSpan -Minutes 1)",
            "Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description 'Ara Memory OS background worker loop' -Force",
            "Write-Host \"Registered scheduled task: $TaskName\"",
            "Write-Host \"Worker command: $Command\"",
            "Write-Host \"Worker log: $LogPath\"",
            "",
    ]
    return "\n".join(lines)


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
            "    Get-Content -LiteralPath $LogPath -Tail 40 | ForEach-Object { $_ -replace \"`0\", \"\" }",
            "} else {",
            "    Write-Host \"Worker log does not exist yet.\"",
            "}",
            "",
        ]
    )


def _join_command(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part or "\\" in part else part for part in parts)


def _ps_array_literal(values: list[str]) -> str:
    return "@(" + ", ".join(f"'{_ps(value)}'" for value in values) + ")"


def _ps(value: str) -> str:
    return value.replace("'", "''")


def _validate_scheduler_label(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    if any(char in value for char in "\r\n;&|<>`$\"'"):
        raise ValueError(f"{name} contains unsupported scheduler metacharacters")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _extract_ps_assignment(text: str, name: str) -> str:
    match = re.search(rf"^\${re.escape(name)}\s*=\s*'(.*)'$", text, flags=re.MULTILINE)
    if not match:
        return ""
    return match.group(1).replace("''", "'")


def _extract_ps_array_assignment(text: str, name: str) -> list[str]:
    match = re.search(rf"^\${re.escape(name)}\s*=\s*@\(\s*(.*?)^\)", text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return []
    values: list[str] = []
    for line in match.group(1).splitlines():
        item = line.strip().rstrip(",").strip()
        if not item:
            continue
        if not (item.startswith("'") and item.endswith("'")):
            return []
        values.append(item[1:-1].replace("''", "'"))
    return values


def _extract_interval_minutes(text: str) -> int | None:
    match = re.search(r"New-TimeSpan -Minutes (\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def _split_joined_command(command: str) -> list[str]:
    return [
        match.group(1) if match.group(1) is not None else match.group(2)
        for match in re.finditer(r'"([^"]*)"|(\S+)', command)
    ]


def _command_arg_values(args: list[str], name: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == name:
            if index + 1 < len(args):
                values.append(args[index + 1])
            else:
                values.append("")
            index += 2
        else:
            index += 1
    return values


def _regression_manifest_case_names(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        return set()
    names = {str(item.get("name")) for item in cases if isinstance(item, dict) and item.get("name")}
    return names if len(names) == len(cases) else set()


def _valid_regression_baseline(path: Path, manifest_case_names: set[str]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not manifest_case_names:
        return False
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return False
    baseline_names: set[str] = set()
    for item in cases:
        if not isinstance(item, dict) or not item.get("name"):
            return False
        details = item.get("details")
        if not isinstance(details, dict):
            return False
        if "estimated_tokens" not in details or "selected_capsule_ids" not in details:
            return False
        baseline_names.add(str(item["name"]))
    return manifest_case_names <= baseline_names
