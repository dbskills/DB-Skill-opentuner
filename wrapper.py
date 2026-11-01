#!/usr/bin/env python3
"""Long-lived HTTP server exposing ensemble-search PostgreSQL knob tuning."""

import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import opentuner
import psycopg2
from opentuner.api import TuningRunManager
from opentuner.measurement import MeasurementInterface
from opentuner.resultsdb.models import Result
from opentuner.search.manipulator import (
    ConfigurationManipulator,
    EnumParameter,
    FloatParameter,
    IntegerParameter,
)
from psycopg2.extras import RealDictCursor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_DIR = os.path.join(BASE_DIR, "skill_state")

DB_TYPES = {"web", "oltp", "dw", "desktop", "mixed"}
BUDGET_DEFAULTS = {"time_budget_s": 600.0, "max_rounds": 20, "max_restarts": 0}
DEFAULT_SEED = 1
CONFIG_KEYS = {
    "tuning_scope", "measure_reps", "warmup_passes", "concurrency",
    "duration_s", "restart_command", "restart_wait_s", "seed", "technique",
    "state_dir",
}
DEFAULTS = {
    "tuning_scope": None,
    "measure_reps": 3,
    "warmup_passes": 1,
    "concurrency": 1,
    "duration_s": None,
    "restart_command": None,
    "restart_wait_s": 60,
    "seed": DEFAULT_SEED,
    "technique": "AUCBanditMetaTechniqueA",
    "state_dir": "skill_state",
}
DEFAULT_SCOPE = [
    {"knob": "work_mem", "min": 4096, "max": 2097152},
    {"knob": "random_page_cost", "min": 1.0, "max": 8.0},
    {"knob": "effective_io_concurrency", "min": 1, "max": 200},
    {"knob": "max_parallel_workers_per_gather", "min": 0, "max": 16},
    {"knob": "max_parallel_workers", "min": 1, "max": 64},
    {"knob": "jit", "min": None, "max": None},
]
OBJECTIVES = {"latency", "throughput"}
PENALTY = 1e9
SEMANTIC_PREFIXES = ("lc_", "default_transaction_")
SEMANTIC_KNOBS = {"timezone", "datestyle", "intervalstyle", "search_path"}
BYTES = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3, "tb": 1024 ** 4, "pb": 1024 ** 5}
TIME_MS = {"us": 0.001, "ms": 1, "s": 1000, "min": 60000, "h": 3600000, "d": 86400000}
KNOB_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

STATE_LOCK = threading.RLock()
STATE_CACHE = {}
JOURNAL_LOCK = threading.Lock()
TUNE_LOCK = threading.Lock()
BUSY = threading.Event()
TERMINATING = threading.Event()
SHUTDOWN_FLAG = threading.Event()
SERVER_REF = {"server": None}
LAST_STATE = {"dir": os.path.join(BASE_DIR, "skill_state")}


class BadRequest(Exception):
    pass


class EvaluationError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class Knob:
    def __init__(self, name, row, kind, options=None, low=None, high=None):
        self.name = name
        self.row = row
        self.kind = kind
        self.options = options
        self.low = low
        self.high = high

    @property
    def restart_required(self):
        return self.row.get("context") == "postmaster"

    def parameter(self):
        if self.kind in ("enum", "bool"):
            return EnumParameter(self.name, self.options)
        if self.kind == "integer":
            return IntegerParameter(self.name, int(self.low), int(self.high))
        return FloatParameter(self.name, float(self.low), float(self.high))

    def format(self, value):
        if self.kind in ("enum", "bool"):
            return str(value)
        if self.kind == "integer":
            return str(int(round(float(value))))
        return format(float(value), ".6g")


def _as_dict(value, name):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            raise BadRequest(f"{name} must be a JSON object")
        if not isinstance(parsed, dict):
            raise BadRequest(f"{name} must be a JSON object")
        return parsed
    raise BadRequest(f"{name} must be a JSON object")


def _to_int(value, name, minimum=None):
    if isinstance(value, bool):
        raise BadRequest(f"{name} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise BadRequest(f"{name} must be an integer")
        result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise BadRequest(f"{name} must be an integer")
        result = int(text)
    else:
        raise BadRequest(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise BadRequest(f"{name} must be >= {minimum}")
    return result


def _to_float(value, name, minimum=None):
    if isinstance(value, bool):
        raise BadRequest(f"{name} must be a number")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
            raise BadRequest(f"{name} must be a number")
        result = float(text)
    else:
        raise BadRequest(f"{name} must be a number")
    if minimum is not None and result < minimum:
        raise BadRequest(f"{name} must be >= {minimum}")
    return result


def _parse_scope(value):
    if not isinstance(value, list):
        raise BadRequest("tuning_scope must be a list")
    result = []
    seen = set()
    for item in value:
        if isinstance(item, str):
            knob = item.strip()
            low = None
            high = None
        elif isinstance(item, dict):
            knob = item.get("knob")
            if not isinstance(knob, str):
                raise BadRequest("each tuning_scope object needs a string knob")
            knob = knob.strip()
            low = item.get("min")
            high = item.get("max")
            if low is not None:
                low = _to_float(low, "tuning_scope.min")
            if high is not None:
                high = _to_float(high, "tuning_scope.max")
            if low is not None and high is not None and low > high:
                raise BadRequest("tuning_scope min must not exceed max")
        else:
            raise BadRequest("each tuning_scope item must be a knob name or object")
        if not KNOB_RE.fullmatch(knob):
            raise BadRequest(f"invalid knob name in tuning_scope: {knob!r}")
        key = knob.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append({"knob": key, "min": low, "max": high})
    return result


def _parse_config(raw):
    cfg = dict(DEFAULTS)
    data = _as_dict(raw, "config")
    for key in data:
        if key in BUDGET_DEFAULTS:
            raise BadRequest(f"budget key '{key}' must be supplied via budget")
    for key, value in data.items():
        if key not in CONFIG_KEYS:
            continue
        if key == "tuning_scope":
            cfg[key] = None if value is None else _parse_scope(value)
        elif key in ("measure_reps", "warmup_passes", "restart_wait_s"):
            minimum = 0 if key == "warmup_passes" else 1
            cfg[key] = _to_int(value, key, minimum)
        elif key == "concurrency":
            cfg[key] = _to_int(value, key, 1)
        elif key == "duration_s":
            cfg[key] = None if value is None else _to_float(value, key, 0.001)
        elif key == "restart_command":
            if value is not None and not isinstance(value, str):
                raise BadRequest("restart_command must be a string")
            cfg[key] = value
        elif key == "seed":
            cfg[key] = DEFAULT_SEED if value is None else _to_int(value, key)
        elif key == "technique":
            if not isinstance(value, str) or not value.strip():
                raise BadRequest("technique must be a non-empty string")
            cfg[key] = value.strip()
        elif key == "state_dir":
            if not isinstance(value, str) or not value.strip():
                raise BadRequest("state_dir must be a non-empty string")
            cfg[key] = value.strip()
    return cfg


def _parse_budget(raw):
    budget = dict(BUDGET_DEFAULTS)
    data = _as_dict(raw, "budget")
    for key in data:
        if key in CONFIG_KEYS:
            raise BadRequest(f"config key '{key}' must be supplied via config")
    for key, value in data.items():
        if key not in BUDGET_DEFAULTS:
            continue
        if value is None:
            continue
        if key == "time_budget_s":
            budget[key] = _to_float(value, key, 0.0)
        else:
            budget[key] = _to_int(value, key, 0)
    return budget


def _is_semantic(name):
    return name in SEMANTIC_KNOBS or name.startswith(SEMANTIC_PREFIXES)


def _unit_bytes(unit):
    text = (unit or "").strip()
    match = re.fullmatch(r"(\d+)\s*(kB|MB|GB|TB|PB)", text)
    if match:
        return int(match.group(1)) * BYTES[match.group(2).lower()]
    return BYTES.get(text.lower())


def _setting_number(value, row):
    text = str(value).strip()
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]+)?", text)
    if not match:
        return None
    number = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    unit = (row.get("unit") or "").strip()
    if unit:
        unit_bytes = _unit_bytes(unit)
        if unit_bytes is not None:
            if not suffix:
                return number
            value_bytes = BYTES.get(suffix)
            return None if value_bytes is None else number * value_bytes / unit_bytes
        unit_ms = TIME_MS.get(unit.lower())
        if unit_ms is not None:
            if not suffix:
                return number
            value_ms = TIME_MS.get(suffix)
            return None if value_ms is None else number * value_ms / unit_ms
        return None
    return None if suffix else number


def _same_setting(left, right, row):
    if row.get("vartype") in ("integer", "real"):
        lnum = _setting_number(left, row)
        rnum = _setting_number(right, row)
        if lnum is not None and rnum is not None:
            return abs(lnum - rnum) <= max(1e-9, abs(rnum) * 1e-9)
    return str(left) == str(right)


def _practical_range(row):
    low = _to_float(row.get("min_val"), "min_val") if row.get("min_val") else 0.0
    high = _to_float(row.get("max_val"), "max_val") if row.get("max_val") else 1e12
    current = _setting_number(row.get("setting"), row)
    if current is None:
        current = max(low, 1.0)
    unit = (row.get("unit") or "").strip()
    if unit and _unit_bytes(unit) is not None:
        low = max(low, current / 8.0)
        high = min(high, max(current * 8.0, current + 1024.0))
    else:
        low = max(low, 0.0)
        high = min(high, max(current * 4.0, current + 16.0))
    if high <= low:
        high = low + max(1.0, abs(low))
    return low, high


def _build_space(settings, config, eval_mode, budget):
    scope = config["tuning_scope"] if config["tuning_scope"] is not None else DEFAULT_SCOPE
    scope_names = {item["knob"] for item in scope}
    manipulator = ConfigurationManipulator()
    knobs = {}
    skipped = []
    excluded_restart = []
    restart_allowed = eval_mode == "dedicated" and budget["max_restarts"] > 0 and bool(
        config["restart_command"]
    )
    for item in scope:
        name = item["knob"]
        row = settings.get(name)
        if row is None:
            skipped.append({"knob": name, "reason": "not present on target"})
            continue
        if _is_semantic(name):
            skipped.append({"knob": name, "reason": "result-semantic setting"})
            continue
        restart = row.get("context") == "postmaster"
        if restart and not restart_allowed:
            excluded_restart.append(name)
            continue
        vartype = (row.get("vartype") or "").lower()
        if vartype == "bool":
            knob = Knob(name, row, "bool", options=["off", "on"])
        elif vartype == "enum":
            options = [str(value) for value in (row.get("enumvals") or [])]
            if not options:
                skipped.append({"knob": name, "reason": "enum without values"})
                continue
            knob = Knob(name, row, "enum", options=options)
        elif vartype in ("integer", "real"):
            low = item["min"]
            high = item["max"]
            if low is None or high is None:
                low, high = _practical_range(row)
            kind = "integer" if vartype == "integer" else "real"
            if kind == "integer":
                low = int(math.ceil(low))
                high = int(math.floor(high))
            if high < low:
                skipped.append({"knob": name, "reason": "empty search range"})
                continue
            knob = Knob(name, row, kind, low=low, high=high)
        else:
            skipped.append({"knob": name, "reason": "unsupported setting type"})
            continue
        manipulator.add_parameter(knob.parameter())
        knobs[name] = knob
    return manipulator, knobs, skipped, excluded_restart, scope_names


def _clean_sql(text):
    lines = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith(":") and re.fullmatch(r":[A-Za-z]", stripped):
            continue
        stripped = re.sub(r"--.*$", "", line).rstrip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip().rstrip(";")


class _WorkloadStatement(str):
    def __new__(cls, text, weight=1.0):
        obj = str.__new__(cls, text)
        obj.weight = float(weight)
        return obj


def _normalize_weight(value):
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    return weight if math.isfinite(weight) and weight > 0 else 1.0


def _statement_weight(sql):
    return _normalize_weight(getattr(sql, "weight", 1.0))


def _workload_statements(workload):
    if isinstance(workload, str):
        text = workload.lstrip()
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(workload)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, dict)):
                workload = parsed
    entries = workload if isinstance(workload, list) else [workload]
    order = []
    weights = {}
    for entry in entries:
        weight = 1.0
        if isinstance(entry, dict):
            text = entry.get("sql")
            weight = _normalize_weight(entry.get("weight", entry.get("frequency", 1.0)))
        elif isinstance(entry, str):
            text = entry
        else:
            continue
        if not isinstance(text, str):
            continue
        cleaned = _clean_sql(text)
        if not cleaned:
            continue
        if cleaned in weights:
            weights[cleaned] += weight
        else:
            weights[cleaned] = weight
            order.append(cleaned)
    return [_WorkloadStatement(text, weights[text]) for text in order]


def _payload_key(dsn, eval_dsn, objective, statements, scope_names):
    encoded = [[str(sql), _statement_weight(sql)] for sql in statements]
    if all(weight == 1.0 for _, weight in encoded):
        encoded = [text for text, _ in encoded]
    blob = json.dumps(
        [dsn, eval_dsn or dsn, objective, encoded, sorted(scope_names)],
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


def _resolve_state_dir(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def _connect(dsn, autocommit=True):
    conn = psycopg2.connect(dsn, connect_timeout=5)
    conn.autocommit = autocommit
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 300000")
    return conn


def _read_settings(conn):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT name, setting, unit, context, vartype, min_val, max_val, "
            "enumvals, pending_restart FROM pg_settings WHERE context <> 'internal'"
        )
        return {row["name"]: dict(row) for row in cur.fetchall()}


def _current_settings(conn, names):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT name, setting, unit, context, vartype FROM pg_settings "
            "WHERE name = ANY(%s)",
            (list(names),),
        )
        return {row["name"]: dict(row) for row in cur.fetchall()}


def _snapshot_scope(settings, names):
    snapshot = {}
    for name in names:
        row = settings.get(name)
        if row is None:
            continue
        snapshot[name] = {
            "setting": row.get("setting"),
            "sourcefile": row.get("sourcefile") or "",
            "context": row.get("context"),
        }
    return snapshot


def _reload(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_reload_conf()")


def _wait_effective(conn, expected, timeout=15.0):
    deadline = time.monotonic() + timeout
    names = list(expected)
    while time.monotonic() < deadline:
        current = _current_settings(conn, names)
        if all(
            name in current and _same_setting(current[name].get("setting"), expected[name], current[name])
            for name in names
        ):
            return True
        time.sleep(0.1)
    return False


def _run_restart(command, wait_s, dsn):
    try:
        subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=max(wait_s, 5) + 120,
        )
    except Exception:
        return False
    deadline = time.monotonic() + max(wait_s, 1)
    while time.monotonic() < deadline:
        try:
            conn = psycopg2.connect(dsn, connect_timeout=3)
            conn.close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _apply_config(conn, knobs, cfg, restart_command, restart_wait_s, dsn):
    if TERMINATING.is_set():
        raise EvaluationError("server is terminating")
    current = _current_settings(conn, knobs.keys())
    targets = {}
    changed_restart = []
    for name, knob in knobs.items():
        if TERMINATING.is_set():
            raise EvaluationError("server is terminating")
        target = knob.format(cfg[name])
        targets[name] = target
        if name not in current or not _same_setting(current[name].get("setting"), target, current[name]):
            with conn.cursor() as cur:
                cur.execute("ALTER SYSTEM SET %s = %%s" % name, (target,))
            if knob.restart_required:
                changed_restart.append(name)
    dynamic_targets = {name: value for name, value in targets.items()
                       if not knobs[name].restart_required}
    if targets:
        _reload(conn)
        if dynamic_targets and not _wait_effective(conn, dynamic_targets):
            raise EvaluationError("candidate settings did not become effective after reload")
    restarts = 0
    if changed_restart:
        if not restart_command:
            raise EvaluationError("a restart-requiring knob was selected without restart_command")
        if not _run_restart(restart_command, restart_wait_s, dsn):
            raise EvaluationError("evaluation instance did not become ready after restart")
        restarts += 1
        if not _wait_effective(conn, targets, timeout=restart_wait_s):
            raise EvaluationError("candidate settings did not become effective after restart")
    return restarts


def _restore_scope(conn, baseline, names, restart_command, restart_wait_s, dsn):
    current = _current_settings(conn, names)
    changed_restart = []
    for name in names:
        info = baseline.get(name)
        if not info:
            continue
        was_changed = (
            name not in current
            or not _same_setting(current[name].get("setting"), info["setting"], current[name])
        )
        # Rewrite every baseline value: a previous ALTER SYSTEM may still be
        # pending in postgresql.auto.conf while pg_settings shows the old value.
        with conn.cursor() as cur:
            if info["sourcefile"].endswith("postgresql.auto.conf"):
                cur.execute("ALTER SYSTEM SET %s = %%s" % name, (info["setting"],))
            else:
                cur.execute("ALTER SYSTEM RESET %s" % name)
        if was_changed and info.get("context") == "postmaster":
            changed_restart.append(name)
    expected = {name: baseline[name]["setting"] for name in names if name in baseline}
    dynamic_expected = {name: value for name, value in expected.items()
                        if baseline.get(name, {}).get("context") != "postmaster"}
    if names:
        _reload(conn)
        if dynamic_expected and not _wait_effective(conn, dynamic_expected):
            raise EvaluationError("baseline settings did not become effective after reload")
    restarts = 0
    if changed_restart:
        if not (restart_command and _run_restart(restart_command, restart_wait_s, dsn)):
            raise EvaluationError("evaluation instance did not restart to restore baseline")
        restarts += 1
        if not _wait_effective(conn, expected, timeout=restart_wait_s):
            raise EvaluationError("baseline settings did not become effective after restart")
    return restarts


def _is_write(sql):
    head = sql.lstrip().split(None, 1)
    if not head:
        return False
    return head[0].lower() in {
        "insert", "update", "delete", "merge", "create", "drop", "alter",
        "truncate", "grant", "revoke", "copy", "vacuum", "analyze", "reindex",
    }


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExceeded()


def _run_pass(conn, statements, deadline=None):
    with conn.cursor() as cur:
        for sql in statements:
            _check_deadline(deadline)
            if _is_write(sql):
                cur.execute(sql)
                conn.rollback()
            else:
                cur.execute(sql)
                if cur.description:
                    cur.fetchall()
    conn.rollback()


def _run_pass_timed(conn, statements, deadline=None):
    times = []
    with conn.cursor() as cur:
        for sql in statements:
            _check_deadline(deadline)
            started = time.perf_counter()
            if _is_write(sql):
                cur.execute(sql)
                conn.rollback()
            else:
                cur.execute(sql)
                if cur.description:
                    cur.fetchall()
            times.append(time.perf_counter() - started)
            _check_deadline(deadline)
    conn.rollback()
    return times


def _weighted_seconds(statements, times):
    return sum(_statement_weight(sql) * elapsed for sql, elapsed in zip(statements, times))


def _parallel_latency(conns, statements, deadline=None):
    errors = []
    durations = [0.0] * len(conns)

    def worker(index):
        try:
            times = _run_pass_timed(conns[index], statements, deadline)
            durations[index] = _weighted_seconds(statements, times)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(conns))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        for exc in errors:
            if isinstance(exc, BudgetExceeded):
                raise exc
        raise EvaluationError(str(errors[0]))
    return max(durations)


def _measure_latency(conn, dsn, statements, warmup, reps, concurrency, deadline=None):
    conns = [conn]
    if concurrency > 1:
        conns = [_connect(dsn, autocommit=False) for _ in range(concurrency)]
    try:
        for idx, item in enumerate(conns):
            item.autocommit = False
        for _ in range(warmup):
            _check_deadline(deadline)
            if concurrency > 1:
                _parallel_latency(conns, statements, deadline)
            else:
                _run_pass(conns[0], statements, deadline)
        samples = []
        for _ in range(reps):
            _check_deadline(deadline)
            if concurrency > 1:
                samples.append(_parallel_latency(conns, statements, deadline))
            else:
                times = _run_pass_timed(conns[0], statements, deadline)
                samples.append(_weighted_seconds(statements, times))
        return sorted(samples)[len(samples) // 2]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.autocommit = True
        for item in conns[1:]:
            try:
                item.close()
            except Exception:
                pass


def _throughput_sample(conns, statements, duration_s, deadline=None):
    counts = [0.0] * len(conns)
    services = [0.0] * len(conns)
    errors = []
    started = time.perf_counter()
    stop_at = None if duration_s is None else started + duration_s
    has_write = any(_is_write(sql) for sql in statements)

    def worker(index):
        cur = conns[index].cursor()
        try:
            while stop_at is None or time.perf_counter() < stop_at:
                _check_deadline(deadline)
                for sql in statements:
                    _check_deadline(deadline)
                    began = time.perf_counter()
                    cur.execute(sql)
                    if cur.description:
                        cur.fetchall()
                    elapsed = time.perf_counter() - began
                    weight = _statement_weight(sql)
                    counts[index] += weight
                    services[index] += weight * elapsed
                conns[index].rollback()
                if stop_at is None or has_write:
                    break
        except Exception as exc:
            errors.append(exc)
        finally:
            try:
                cur.close()
            except Exception:
                pass

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(conns))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        for exc in errors:
            if isinstance(exc, BudgetExceeded):
                raise exc
        raise EvaluationError(str(errors[0]))
    return sum(count / max(service, 1e-9) for count, service in zip(counts, services))


def _measure_throughput(conn, dsn, statements, warmup, reps, concurrency, duration_s,
                        deadline=None):
    conns = [conn]
    if concurrency > 1:
        conns = [_connect(dsn, autocommit=True) for _ in range(concurrency)]
    try:
        for item in conns:
            item.autocommit = True
        for _ in range(warmup):
            _check_deadline(deadline)
            _throughput_sample(conns, statements, duration_s, deadline)
        samples = []
        for _ in range(reps):
            _check_deadline(deadline)
            samples.append(_throughput_sample(conns, statements, duration_s, deadline))
        return sorted(samples)[len(samples) // 2]
    finally:
        for item in conns[1:]:
            try:
                item.close()
            except Exception:
                pass


def _measure(conn, dsn, statements, objective, warmup, reps, concurrency, duration_s,
             deadline=None):
    if objective == "latency":
        value = _measure_latency(conn, dsn, statements, warmup, reps, concurrency, deadline)
        return value, value
    value = _measure_throughput(
        conn, dsn, statements, warmup, reps, concurrency, duration_s, deadline
    )
    return 1.0 / max(value, 1e-9), value


def _journal_path(state_dir):
    return os.path.join(JOURNAL_DIR, "restore_journal.json")


def _write_journal(state_dir, dsn, baseline, restart_command, restart_wait_s):
    payload = {
        "dsn": dsn,
        "completed": False,
        "touched": [],
        "baseline": baseline,
        "restart_command": restart_command,
        "restart_wait_s": restart_wait_s,
    }
    temp = _journal_path(state_dir) + ".tmp"
    with JOURNAL_LOCK:
        try:
            os.makedirs(JOURNAL_DIR, exist_ok=True)
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, _journal_path(state_dir))
        except OSError:
            pass


def _journal_touch(state_dir, names):
    path = _journal_path(state_dir)
    with JOURNAL_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
        payload["touched"] = sorted(set(payload.get("touched") or []) | set(names))
        temp = path + ".tmp"
        try:
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, path)
        except OSError:
            pass


def _journal_clear(state_dir):
    path = _journal_path(state_dir)
    with JOURNAL_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
        payload["touched"] = []
        temp = path + ".tmp"
        try:
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, path)
        except OSError:
            pass


def _complete_journal(state_dir):
    path = _journal_path(state_dir)
    with JOURNAL_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
        payload["completed"] = True
        temp = path + ".tmp"
        try:
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, path)
        except OSError:
            pass


def _replay_journal(state_dir):
    path = _journal_path(state_dir)
    with JOURNAL_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
    if not payload.get("completed"):
        touched = payload.get("touched") or []
        dsn = payload.get("dsn")
        baseline = payload.get("baseline") or {}
        if touched and dsn:
            try:
                conn = _connect(dsn)
                try:
                    _restore_scope(
                        conn, baseline, touched,
                        payload.get("restart_command"),
                        payload.get("restart_wait_s", 60),
                        dsn,
                    )
                finally:
                    conn.close()
            except Exception as exc:
                print(f"[journal] restore failed: {type(exc).__name__}", flush=True)
                return
    with JOURNAL_LOCK:
        try:
            os.remove(path)
        except OSError:
            pass


def _handle_termination(signum, frame):
    TERMINATING.set()
    SHUTDOWN_FLAG.set()
    print(f"[signal] {signum} received; restoring baseline", flush=True)
    _replay_journal(LAST_STATE["dir"])
    server = SERVER_REF.get("server")
    if server is not None:
        threading.Thread(target=server.shutdown, daemon=True).start()


LEARNING_STATE_ENTRIES = (
    "state.json", "state.json.tmp",
    "best.json", "best.json.tmp",
    "models", "smac",
    "opentuner.db", "opentuner.db-journal",
    "opentuner.db-shm", "opentuner.db-wal",
)


def _clear_learning_state(state_dir):
    """Delete persisted learning artifacts; restore_journal.json is preserved."""
    state_dir = os.path.abspath(state_dir)
    if state_dir == os.path.abspath(os.sep):
        return
    for name in LEARNING_STATE_ENTRIES:
        path = os.path.join(state_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


class _KnobInterface(MeasurementInterface):
    def __init__(self, args, manipulator, version):
        super().__init__(args=args)
        self._manipulator = manipulator
        self._version = version

    def manipulator(self):
        return self._manipulator

    def program_name(self):
        return "postgres-knobs"

    def program_version(self):
        return self._version

    def save_final_config(self, config):
        pass


def _degrade_payload(objective, eval_mode, budget, config, note, mode="online-training", extra=None):
    metadata = {
        "strategy_type": "model-free ensemble search",
        "optimization_time": 0.0,
        "objective": objective,
        "num_rounds": 0,
        "budget_used": dict(budget, time_used_s=0.0, rounds_used=0, restarts_used=0),
        "eval_mode": eval_mode,
        "excluded_restart_knobs": [],
        "mode": mode,
        "config": config,
        "degrade_note": note,
    }
    if extra:
        metadata.update(extra)
    return {"recommended_knobs": [], "metadata": metadata}


def _entries_from_config(cfg, knobs, settings_now):
    entries = []
    for name, knob in knobs.items():
        target = knob.format(cfg[name])
        row = settings_now.get(name)
        if row is not None and _same_setting(row.get("setting"), target, row):
            continue
        entry = {
            "knob": name,
            "value": target,
            "ddl": "ALTER SYSTEM SET %s = '%s';" % (name, target.replace("'", "''")),
            "restart_required": knob.restart_required,
            "current_value": row.get("setting") if row else None,
        }
        if knob.kind in ("integer", "real"):
            entry["value_range"] = [knob.low, knob.high]
        entries.append(entry)
    return entries


def _run_search(conn, eval_dsn, statements, objective, manipulator, knobs, baseline,
                config, budget, state_dir, key, start_time):
    args = opentuner.default_argparser().parse_args([])
    run_dir = tempfile.mkdtemp(prefix="opentuner-")
    args.database = "sqlite:///" + os.path.join(run_dir, "opentuner.db")
    args.label = key
    args.parallelism = 1
    args.no_dups = True
    args.quiet = True
    args.display_frequency = 1000000000
    if config["technique"]:
        args.technique = [config["technique"]]
    if budget["time_budget_s"] is not None:
        args.stop_after = budget["time_budget_s"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    args.seed = config["seed"]

    deadline = None
    if budget["time_budget_s"] is not None:
        deadline = start_time + budget["time_budget_s"]

    rounds = 0
    restarts = 0
    errors = 0
    skipped_restarts = 0
    best = None
    note = None
    max_rounds = budget["max_rounds"] if budget["max_rounds"] is not None else 20

    args.test_limit = max_rounds
    interface = _KnobInterface(args, manipulator, "pg")
    manager = TuningRunManager(interface, args)
    try:
        for _ in range(max_rounds):
            if deadline is not None and time.monotonic() > deadline:
                note = "time budget exhausted before the next round"
                break
            desired = manager.get_next_desired_result()
            if desired is None:
                note = "search produced no further candidates"
                break
            candidate = desired.configuration.data
            need_restart = any(
                knob.restart_required
                and not _same_setting(knob.format(candidate[name]), baseline[name]["setting"], knob.row)
                for name, knob in knobs.items()
            )
            if need_restart and restarts + 2 > budget["max_restarts"]:
                manager.report_result(desired, Result(time=PENALTY))
                rounds += 1
                skipped_restarts += 1
                continue
            metric = PENALTY
            value = None
            try:
                _journal_touch(state_dir, list(knobs))
                restarts += _apply_config(
                    conn, knobs, candidate, config["restart_command"],
                    config["restart_wait_s"], eval_dsn,
                )
                metric, value = _measure(
                    conn, eval_dsn, statements, objective,
                    config["warmup_passes"], config["measure_reps"],
                    config["concurrency"], config["duration_s"], deadline,
                )
            except BudgetExceeded:
                note = "time budget exhausted during a candidate"
                break
            except Exception:
                errors += 1
                metric = PENALTY
            finally:
                try:
                    restarts += _restore_scope(
                        conn, baseline, list(knobs), config["restart_command"],
                        config["restart_wait_s"], eval_dsn,
                    )
                    _journal_clear(state_dir)
                except Exception:
                    errors += 1
            manager.report_result(desired, Result(time=metric))
            rounds += 1
            if metric < PENALTY and (best is None or metric < best[0]):
                best = (metric, dict(candidate), value)
            if deadline is not None and time.monotonic() > deadline:
                note = "time budget exhausted; returning best-so-far"
                break
    finally:
        try:
            if manager.get_best_result() is not None:
                manager.finish()
            else:
                manager.session.close()
        except Exception:
            try:
                manager.session.close()
            except Exception:
                pass
        shutil.rmtree(run_dir, ignore_errors=True)
    return {
        "best": best,
        "rounds": rounds,
        "restarts": restarts,
        "errors": errors,
        "skipped_restarts": skipped_restarts,
        "note": note,
    }


def _record_state(state_dir, key, rounds, best_metric, objective, eval_mode, mode):
    with STATE_LOCK:
        state = _load_state(state_dir)
        state["requests"] = int(state.get("requests", 0)) + 1
        state["last_key"] = key
        state["last_rounds"] = rounds
        state["last_best_metric"] = best_metric
        state["last_objective"] = objective
        state["last_eval_mode"] = eval_mode
        state["last_mode"] = mode
        state["best_keys"] = []
        STATE_CACHE[state_dir] = dict(state)
        LAST_STATE["dir"] = state_dir


def _load_state(state_dir):
    with STATE_LOCK:
        if state_dir in STATE_CACHE:
            return dict(STATE_CACHE[state_dir])
        state = {
            "requests": 0,
            "last_key": None,
            "last_rounds": 0,
            "last_best_metric": None,
            "last_objective": None,
            "last_eval_mode": None,
            "last_mode": None,
            "best_keys": [],
        }
        STATE_CACHE[state_dir] = state
        return dict(state)


def _state_summary():
    with STATE_LOCK:
        state_dir = LAST_STATE["dir"]
    state = _load_state(state_dir)
    summary = {
        "idle": not BUSY.is_set(),
        "mode": state.get("last_mode") or "online-training",
        "state_dir": state_dir,
    }
    summary.update({key: state.get(key) for key in (
        "requests", "last_key", "last_rounds", "last_best_metric",
        "last_objective", "last_eval_mode", "best_keys",
    )})
    return summary


def _parse_apply_scope(value):
    if value is None or value == "":
        return "dynamic"
    if not isinstance(value, str):
        raise BadRequest("apply_scope must be a string")
    text = value.strip()
    if text in ("dynamic", "all"):
        return text
    names = []
    for part in text.split(","):
        name = part.strip()
        if not name:
            continue
        if not KNOB_RE.fullmatch(name):
            raise BadRequest(f"invalid knob name in apply_scope: {name!r}")
        names.append(name.lower())
    if not names:
        raise BadRequest("apply_scope is empty")
    return names


def _apply_entries(conn, entries, apply_scope):
    if apply_scope == "dynamic":
        selected = [entry for entry in entries if not entry["restart_required"]]
    elif apply_scope == "all":
        selected = list(entries)
    else:
        names = set(apply_scope)
        selected = [entry for entry in entries if entry["knob"].lower() in names]
    results = []
    applied = False
    for entry in selected:
        try:
            with conn.cursor() as cur:
                cur.execute(entry["ddl"])
            applied = True
            results.append({
                "knob": entry["knob"],
                "status": "applied",
                "error": None,
                "restart_pending": entry["restart_required"],
            })
        except Exception as exc:
            results.append({
                "knob": entry["knob"],
                "status": "failed",
                "error": str(exc)[:200],
                "restart_pending": entry["restart_required"],
            })
    if applied:
        try:
            _reload(conn)
        except Exception:
            pass
    return results


def _require_str(req, key):
    value = req.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"missing required parameter: {key}")
    return value.strip()


def _space_description(knobs):
    result = []
    for name, knob in knobs.items():
        item = {"knob": name, "type": knob.kind}
        if knob.kind in ("integer", "real"):
            item["min"] = knob.low
            item["max"] = knob.high
        else:
            item["options"] = knob.options
        result.append(item)
    return result


def _tune(req):
    started = time.monotonic()
    dsn = _require_str(req, "dsn")
    if "workload" not in req or req.get("workload") is None:
        raise BadRequest("missing required parameter: workload")
    objective = _require_str(req, "objective")
    if objective not in OBJECTIVES:
        raise BadRequest("objective must be latency or throughput")
    config = _parse_config(req.get("config"))
    budget = _parse_budget(req.get("budget"))
    mode = "online-training"
    deadline = None
    if budget["time_budget_s"] is not None:
        deadline = started + budget["time_budget_s"]

    eval_dsn_raw = req.get("eval_dsn")
    dedicated = (
        isinstance(eval_dsn_raw, str)
        and bool(eval_dsn_raw.strip())
        and eval_dsn_raw.strip() != dsn
    )
    eval_mode = "dedicated" if dedicated else "in-place"
    eval_dsn = eval_dsn_raw.strip() if dedicated else dsn

    apply_flag = req.get("apply", False)
    if not isinstance(apply_flag, bool):
        raise BadRequest("apply must be a boolean")
    apply_scope = _parse_apply_scope(req.get("apply_scope")) if apply_flag else None

    state_dir = _resolve_state_dir(config["state_dir"])
    _clear_learning_state(state_dir)
    apply_done = False
    statements = _workload_statements(req.get("workload"))
    if not statements:
        payload = _degrade_payload(
            objective, eval_mode, budget, config, "workload contains no SQL statements", mode
        )
        _record_state(state_dir, None, 0, None, objective, eval_mode, mode)
        return payload

    try:
        conn = _connect(dsn)
    except Exception:
        payload = _degrade_payload(
            objective, eval_mode, budget, config, "database connection failed", mode
        )
        _record_state(state_dir, None, 0, None, objective, eval_mode, mode)
        return payload

    try:
        settings = _read_settings(conn)
        manipulator, knobs, skipped, excluded_restart, scope_names = _build_space(
            settings, config, eval_mode, budget
        )
        scope_unknown = sorted(scope_names - set(settings)) if scope_names else []
        key = _payload_key(dsn, eval_dsn, objective, statements, knobs.keys())

        if not knobs:
            note = "tuning_scope contains no tunable knobs"
            if excluded_restart:
                note = "all requested knobs require a restart and were excluded"
            if scope_unknown:
                note = "tuning_scope contains no knob present on the target"
            payload = _degrade_payload(
                objective, eval_mode, budget, config, note, mode,
                {"excluded_restart_knobs": excluded_restart,
                 "skipped_knobs": skipped[:100],
                 "scope_unknown_knobs": scope_unknown},
            )
            _record_state(state_dir, key, 0, None, objective, eval_mode, mode)
            return payload

        baseline = _snapshot_scope(settings, list(knobs))
        _write_journal(
            state_dir, eval_dsn, baseline,
            config["restart_command"], config["restart_wait_s"],
        )
        try:
            baseline_cost, baseline_value = _measure(
                conn, eval_dsn, statements, objective,
                config["warmup_passes"], config["measure_reps"],
                config["concurrency"], config["duration_s"], deadline,
            )
        except Exception as exc:
            note = "workload evaluation failed"
            if isinstance(exc, BudgetExceeded):
                note = "time budget exhausted during baseline measurement"
            payload = _degrade_payload(
                objective, eval_mode, budget, config, note, mode,
                {"excluded_restart_knobs": excluded_restart,
                 "skipped_knobs": skipped[:100],
                 "scope_unknown_knobs": scope_unknown,
                 "tuning_space": _space_description(knobs)},
            )
            _record_state(state_dir, key, 0, None, objective, eval_mode, mode)
            return payload

        if budget["max_rounds"] == 0:
            payload = _degrade_payload(
                objective, eval_mode, budget, config,
                "budget allows no candidate rounds", mode,
                {"baseline_metric": baseline_value,
                 "excluded_restart_knobs": excluded_restart,
                 "tuning_space": _space_description(knobs)},
            )
            _record_state(state_dir, key, 0, None, objective, eval_mode, mode)
            return payload

        if budget["time_budget_s"] is not None and time.monotonic() - started > budget["time_budget_s"]:
            payload = _degrade_payload(
                objective, eval_mode, budget, config,
                "time budget exhausted during baseline measurement", mode,
                {"baseline_metric": baseline_value,
                 "excluded_restart_knobs": excluded_restart,
                 "tuning_space": _space_description(knobs)},
            )
            _record_state(state_dir, key, 0, None, objective, eval_mode, mode)
            return payload

        result = _run_search(
            conn, eval_dsn, statements, objective, manipulator, knobs, baseline,
            config, budget, state_dir, key, started,
        )
        elapsed = time.monotonic() - started
        evaluated = max(0, result["rounds"] - result["skipped_restarts"])
        live = _read_settings(conn)
        budget_used = dict(
            budget,
            time_used_s=round(elapsed, 4),
            rounds_used=evaluated,
            restarts_used=result["restarts"],
            skipped_restart_candidates=result["skipped_restarts"],
        )
        common = {
            "strategy_type": "model-free ensemble search",
            "optimization_time": round(elapsed, 4),
            "objective": objective,
            "num_rounds": evaluated,
            "budget_used": budget_used,
            "eval_mode": eval_mode,
            "excluded_restart_knobs": excluded_restart,
            "mode": mode,
            "baseline_metric": baseline_value,
            "technique": config["technique"],
            "config": config,
            "tuning_space": _space_description(knobs),
            "skipped_knobs": skipped[:100],
            "scope_unknown_knobs": scope_unknown,
        }
        best = result["best"]
        if best is None:
            note = result["note"] or "no candidate produced a usable result"
            payload = _degrade_payload(objective, eval_mode, budget, config, note, mode, common)
            _record_state(state_dir, key, evaluated, None, objective, eval_mode, mode)
            return payload
        best_cost, best_cfg, best_value = best
        if best_cost >= baseline_cost:
            note = result["note"] or "no candidate improved on the baseline"
            payload = _degrade_payload(objective, eval_mode, budget, config, note, mode, common)
            _record_state(state_dir, key, evaluated, best_cost, objective, eval_mode, mode)
            return payload

        entries = _entries_from_config(best_cfg, knobs, live)
        if not entries:
            note = "best configuration equals the current configuration"
            payload = _degrade_payload(objective, eval_mode, budget, config, note, mode, common)
            _record_state(state_dir, key, evaluated, best_cost, objective, eval_mode, mode)
            return payload

        if objective == "latency":
            impact = (baseline_value - best_value) / baseline_value * 100.0 if baseline_value else 0.0
        else:
            impact = (best_value - baseline_value) / baseline_value * 100.0 if baseline_value else 0.0
        metadata = dict(common)
        metadata["tuned_metric"] = best_value
        metadata["estimated_impact"] = round(impact, 2)
        if result["note"]:
            metadata["degrade_note"] = result["note"]
        if result["errors"]:
            metadata["failed_candidates"] = result["errors"]
        payload = {"recommended_knobs": entries, "metadata": metadata}
        if apply_flag:
            if dedicated:
                main_conn = _connect(dsn)
                try:
                    payload["apply_results"] = _apply_entries(main_conn, entries, apply_scope)
                finally:
                    main_conn.close()
            else:
                payload["apply_results"] = _apply_entries(conn, entries, apply_scope)
                apply_done = True
        _record_state(state_dir, key, evaluated, best_cost, objective, eval_mode, mode)
        return payload
    finally:
        restore_error = None
        try:
            if not apply_done and "baseline" in locals() and "knobs" in locals() and knobs:
                _journal_touch(state_dir, list(knobs))
                for _ in range(2):
                    try:
                        _restore_scope(
                            conn, baseline, list(knobs), config["restart_command"],
                            config["restart_wait_s"], eval_dsn,
                        )
                        restore_error = None
                        break
                    except Exception as exc:
                        restore_error = "{}: {}".format(type(exc).__name__, exc)
                        time.sleep(0.5)
        except Exception as exc:
            restore_error = "{}: {}".format(type(exc).__name__, exc)
        if restore_error is None:
            _journal_clear(state_dir)
            _complete_journal(state_dir)
        else:
            print(f"[restore] baseline restore failed: {restore_error}", flush=True)
            current_payload = locals().get("payload")
            if isinstance(current_payload, dict):
                metadata = current_payload.setdefault("metadata", {})
                metadata["restore_error"] = restore_error
                note = "baseline restore failed: " + restore_error
                metadata["degrade_note"] = (
                    metadata["degrade_note"] + "; " + note
                    if metadata.get("degrade_note") else note
                )
        try:
            conn.close()
        except Exception:
            pass


def _send_json(handler, status, payload):
    data = json.dumps(payload).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError):
        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "KnobTuning/1.0"

    def log_message(self, format, *args):
        return

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BadRequest("invalid Content-Length")
        if length < 0 or length > 10_000_000:
            raise BadRequest("request body too large")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            raise BadRequest("request body is empty")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise BadRequest("invalid JSON body")
        if not isinstance(payload, dict):
            raise BadRequest("request body must be a JSON object")
        return payload

    def do_GET(self):
        started = time.monotonic()
        path = self.path.split("?", 1)[0]
        if path == "/health":
            _send_json(self, 200, {"status": "ok"})
        elif path == "/state":
            _send_json(self, 200, _state_summary())
        else:
            _send_json(self, 404, {"error": "unknown endpoint"})
        print(f"[req] {path} status={200 if path in ('/health', '/state') else 404} "
              f"duration={time.monotonic() - started:.3f}s", flush=True)

    def do_POST(self):
        started = time.monotonic()
        path = self.path.split("?", 1)[0]
        if path == "/shutdown":
            _send_json(self, 200, {"status": "shutting down"})
            SHUTDOWN_FLAG.set()
            server = SERVER_REF.get("server")
            if server is not None:
                threading.Thread(target=server.shutdown, daemon=True).start()
            print(f"[req] /shutdown status=200 duration={time.monotonic() - started:.3f}s", flush=True)
            return
        if path != "/tune":
            _send_json(self, 404, {"error": "unknown endpoint"})
            print(f"[req] {path} status=404 duration={time.monotonic() - started:.3f}s", flush=True)
            return

        req = {}
        try:
            req = self._read_json()
            if SHUTDOWN_FLAG.is_set():
                _send_json(self, 503, {"error": "server is shutting down"})
                return
            with TUNE_LOCK:
                BUSY.set()
                try:
                    payload = _tune(req)
                finally:
                    BUSY.clear()
            _send_json(self, 200, payload)
            metadata = payload.get("metadata") or {}
            print(f"[req] /tune status=200 knobs={len(payload.get('recommended_knobs') or [])} "
                  f"rounds={metadata.get('num_rounds')} "
                  f"degrade={'yes' if metadata.get('degrade_note') else 'no'} "
                  f"duration={time.monotonic() - started:.3f}s", flush=True)
        except BadRequest as exc:
            _send_json(self, 400, {"error": str(exc)})
            print(f"[req] /tune status=400 error={exc} duration={time.monotonic() - started:.3f}s", flush=True)
        except Exception:
            BUSY.clear()
            objective = req.get("objective") if isinstance(req, dict) else ""
            payload = _degrade_payload(
                objective or "", "in-place", dict(BUDGET_DEFAULTS), {},
                "internal recommendation failure", "online-training",
            )
            _send_json(self, 200, payload)
            print(f"[req] /tune status=200 degrade=yes duration={time.monotonic() - started:.3f}s", flush=True)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    _clear_learning_state(LAST_STATE["dir"])
    _replay_journal(LAST_STATE["dir"])
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)
    port = int(os.environ.get("PORT", "8765"))
    server = Server(("0.0.0.0", port), Handler)
    SERVER_REF["server"] = server
    print(f"[start] listening on 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()