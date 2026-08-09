from __future__ import annotations
import json, os, time, random, hashlib, math
from pathlib import Path
from typing import Any
import numpy as np
import torch

SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION = 3
CODE_VERSION_COMPAT = ("git_commit", "git_dirty", "weightwatcher_version", "wwpgd_commit", "torch_version", "optimizer_implementation_version")
REQUIRED_COMPAT=("configuration_hash","data_hash","tokenizer_hash","initialization_hash","model_configuration_hash","training_configuration_hash","wwpgd_configuration_hash","validation_probe_hash","training_probe_hash","scientific_schema_version","optimizer_fingerprint", *CODE_VERSION_COMPAT)

WWPGD_PROJECTION_DOSE_DEFINITION = (
    "applied_projection_delta_frobenius_over_preprojection_weight_frobenius"
)
WWPGD_PROJECTION_EVENT_INDEX_BASE = 0
WWPGD_PROJECTION_APPLICATION_INDEX_SCHEMA_VERSION = 1


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()

def rng_state() -> dict[str, Any]:
    out={"python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(), "torch_cpu_rng_state": torch.get_rng_state()}
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    mps_state = None
    try:
        if torch.backends.mps.is_available():
            mps_state = torch.mps.get_rng_state()
    except (AttributeError, RuntimeError):
        # Older PyTorch releases may expose MPS execution without RNG-state APIs.
        mps_state = None
    out["torch_cuda_rng_states"] = cuda_states
    out["torch_mps_rng_state"] = mps_state
    out["accelerator_rng_states"] = {"cuda": cuda_states, "mps": mps_state}
    return out

def restore_rng_state(state: dict[str, Any]) -> None:
    if "python_random_state" in state: random.setstate(state["python_random_state"])
    if "numpy_random_state" in state: np.random.set_state(state["numpy_random_state"])
    if "torch_cpu_rng_state" in state: torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_states"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])
    mps_state = state.get("torch_mps_rng_state")
    if mps_state is None:
        mps_state = (state.get("accelerator_rng_states") or {}).get("mps")
    if mps_state is not None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.set_rng_state(mps_state)
        except (AttributeError, RuntimeError):
            pass

REQUIRED_CHECKPOINT_KEYS = (
    "model_state_dict","optimizer_state_dict","base_optimizer_state_dict","scheduler_state_dict","gradient_scaler_state_dict",
    "current_step","next_step","tokens_processed","training_reader_position","seed",
    "wwpgd_state","python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_states","accelerator_rng_states",
    "device_type","precision_policy","gradient_accumulation_position","artifact_commits",
    "resolved_config","optimizer_fingerprint","data_hash","tokenizer_hash",
    "scientific_schema_version","checkpoint_schema_version","created_at",
)

INVENTORY_FIELDS = (
    "checkpoint","current_step","next_step","tokens_processed","created_at","sha256",
    "size_bytes","verified","compatibility_hash","checkpoint_schema_version",
)

def _sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _projection_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0", ""}:
        return False
    raise ValueError(f"{field} must be boolean-compatible, got {value!r}")


def _projection_event_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("projection_event must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("projection_event must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("projection_event must be an integer")
    event = int(numeric)
    if event < 0:
        raise ValueError("projection_event must be non-negative")
    return event


def _projection_application_index_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.application_counts.json")


def _scan_projection_application_counts(path: Path) -> dict[str, int]:
    import csv
    if not path.exists() or not path.stat().st_size:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"layer_name", "dose_applied", "layer_application_index"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                "existing WWPGD projection CSV uses an older telemetry schema; "
                f"missing fields {missing}. Start a fresh run instead of appending."
            )
        counts: dict[str, int] = {}
        for row in reader:
            layer_name = str(row.get("layer_name") or "").strip()
            if not layer_name:
                raise ValueError("existing WWPGD projection row is missing layer_name")
            if _projection_bool(row.get("dose_applied", False), field="dose_applied"):
                counts[layer_name] = counts.get(layer_name, 0) + 1
        return counts


def _valid_projection_application_counts(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    counts: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        name = str(raw_name).strip()
        if not name or isinstance(raw_count, bool):
            return None
        try:
            count = int(raw_count)
        except (TypeError, ValueError, OverflowError):
            return None
        if count < 0 or count != raw_count:
            return None
        counts[name] = count
    return counts


def _load_projection_application_counts(path: Path) -> dict[str, int]:
    if not path.exists() or not path.stat().st_size:
        return {}
    stat = path.stat()
    index_path = _projection_application_index_path(path)
    try:
        index = json.loads(index_path.read_text())
    except (OSError, UnicodeError, TypeError, json.JSONDecodeError):
        index = None
    if isinstance(index, dict):
        counts = _valid_projection_application_counts(index.get("counts"))
        try:
            schema_version = int(index.get("schema_version", -1))
            source_size_bytes = int(index.get("source_size_bytes", -1))
            source_mtime_ns = int(index.get("source_mtime_ns", -1))
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if (
                schema_version == WWPGD_PROJECTION_APPLICATION_INDEX_SCHEMA_VERSION
                and source_size_bytes == stat.st_size
                and source_mtime_ns == stat.st_mtime_ns
                and counts is not None
            ):
                return counts
    return _scan_projection_application_counts(path)


def _store_projection_application_counts(path: Path, counts: dict[str, int]) -> None:
    stat = path.stat()
    _atomic_write_json(
        _projection_application_index_path(path),
        {
            "schema_version": WWPGD_PROJECTION_APPLICATION_INDEX_SCHEMA_VERSION,
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "counts": dict(sorted(counts.items())),
        },
    )


def _enrich_wwpgd_projection_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Add exact dose and resume-safe first-application identity to projection rows.

    ``projection_event`` is zero-based.  First actual application is tracked per
    layer from a self-validating durable-history index.  The index is rebuilt
    from the reconciled CSV only after a crash, truncation, or metadata mismatch,
    avoiding an O(n^2) history scan during interval-one experiments.
    """
    application_counts = _load_projection_application_counts(path)
    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if "is_first_projection_event" in row:
            raise ValueError(
                "ambiguous is_first_projection_event is unsupported; use "
                "is_first_scheduled_projection_event and is_first_applied_projection"
            )
        if "projection_event" not in row:
            raise ValueError("WWPGD projection row is missing projection_event")
        event = _projection_event_index(row["projection_event"])

        raw_dose = row.get("relative_frobenius_change_applied")
        if raw_dose is None or raw_dose == "" or isinstance(raw_dose, bool):
            raise ValueError(
                "WWPGD projection row is missing relative_frobenius_change_applied"
            )
        try:
            dose = float(raw_dose)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "relative_frobenius_change_applied must be numeric"
            ) from exc
        if not math.isfinite(dose) or dose < 0.0:
            raise ValueError(
                "relative_frobenius_change_applied must be finite and non-negative"
            )

        layer_name = str(row.get("layer_name") or "").strip()
        if not layer_name:
            raise ValueError("WWPGD projection row is missing layer_name")
        changed = _projection_bool(row.get("changed", False), field="changed")
        dose_applied = changed and dose > 0.0
        application_index = None
        if dose_applied:
            application_index = application_counts.get(layer_name, 0) + 1
            application_counts[layer_name] = application_index
        row.update(
            {
                "dose_definition": WWPGD_PROJECTION_DOSE_DEFINITION,
                "dose_value": dose,
                "dose_relative_frobenius": dose,
                "dose_applied": dose_applied,
                "layer_application_index": application_index,
                "is_first_applied_projection": application_index == 1,
                "projection_event_index_base": WWPGD_PROJECTION_EVENT_INDEX_BASE,
                "projection_event_number": event + 1,
                "is_first_scheduled_projection_event": event == 0,
            }
        )
        enriched.append(row)
    return enriched, application_counts


def append_csv_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Durably append records. A flush boundary is a transaction boundary."""
    import csv
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    projection_application_counts = None
    if path.name == "wwpgd_projection.csv":
        rows, projection_application_counts = _enrich_wwpgd_projection_rows(path, rows)
    # Cached-endpoint fast-relaxation rows already have the dedicated
    # wwpgd_endpoint_relaxation.csv stream. Do not duplicate them into
    # wwpgd_controller.csv, whose stable schema is the slow measurement record.
    if path.name == "wwpgd_controller.csv":
        rows = [
            row
            for row in rows
            if str(row.get("action_type", "")) != "fast_endpoint_relaxation"
        ]
        if not rows:
            return
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as f:
            fields = next(csv.reader(f))
        if path.name == "wwpgd_endpoint_relaxation.csv":
            # Terminal convergence/invalidation rows intentionally omit movement
            # fields. They may be a subset of the established relaxation schema,
            # but introducing any new field remains a hard schema error.
            field_set = set(fields)
            for row in rows:
                extra = [key for key in row if key not in field_set]
                if extra:
                    raise ValueError(
                        f"CSV schema changed while appending {path}: "
                        f"unexpected fields {extra}; existing fields {fields}"
                    )
        else:
            for row in rows:
                observed = list(row)
                if observed != fields:
                    raise ValueError(
                        f"CSV schema changed while appending {path}: "
                        f"{fields} != {observed}"
                    )
    else:
        fields = (
            list(dict.fromkeys(key for row in rows for key in row))
            if path.name == "wwpgd_endpoint_relaxation.csv"
            else list(rows[0])
        )
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    if projection_application_counts is not None:
        _store_projection_application_counts(path, projection_application_counts)

def csv_commit(path: Path) -> dict[str, Any]:
    """Describe the exact durable prefix referenced by a checkpoint."""
    import csv
    path = Path(path)
    if not path.exists():
        return {"rows": 0, "sha256": None, "size_bytes": 0}
    with path.open(newline="") as f:
        rows = sum(1 for _ in csv.DictReader(f))
    return {"rows": rows, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}

def reconcile_csv_artifacts(run_dir: Path, commits: dict[str, dict[str, Any]]) -> None:
    """Verify committed prefixes and discard only bytes appended after them."""
    run_dir = Path(run_dir)
    for name, commit in commits.items():
        path = run_dir / name
        size = int(commit.get("size_bytes", 0)); expected = commit.get("sha256")
        if not path.exists():
            if size: raise RuntimeError(f"committed artifact is missing: {name}")
            continue
        if path.stat().st_size < size:
            raise RuntimeError(f"committed artifact was shortened: {name}")
        with path.open("rb") as f:
            prefix = f.read(size)
        actual = hashlib.sha256(prefix).hexdigest()
        if actual != expected:
            raise RuntimeError(f"committed artifact hash mismatch: {name}")
        if path.stat().st_size > size:
            with path.open("r+b") as f:
                f.truncate(size); f.flush(); os.fsync(f.fileno())
        verified = csv_commit(path)
        if int(verified["rows"]) != int(commit.get("rows", 0)):
            raise RuntimeError(f"committed artifact row-count mismatch: {name}")

def validate_checkpoint_keys(obj: dict) -> None:
    missing=[k for k in REQUIRED_CHECKPOINT_KEYS if k not in obj]
    if missing:
        raise ValueError("checkpoint missing required keys: "+", ".join(missing))
    if int(obj.get("checkpoint_schema_version", -1)) != SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {obj.get('checkpoint_schema_version')}")

def _atomic_write_json(path: Path, data: dict) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('w') as f:
        f.write(json.dumps(data, indent=2, sort_keys=True, default=str)+"\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _append_inventory_atomic(path: Path, row: dict) -> None:
    import csv
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    if path.exists():
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
    rows.append({k: row.get(k, '') for k in INVENTORY_FIELDS})
    tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(INVENTORY_FIELDS)); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_torch_save(obj, path: Path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    loaded=torch.load(tmp, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "checkpoint_schema_version" in loaded:
        validate_checkpoint_keys(loaded)
    sha=_sha256_file(tmp); size=tmp.stat().st_size
    os.replace(tmp, path); return path, sha, size

def save_checkpoint(run_dir: Path, state: dict):
    ck=Path(run_dir)/"checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    step=int(state.get("current_step", state.get("step",0)))
    created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    compatibility=state.get("compatibility", {})
    full={**state,"step":step,"current_step":step,"next_step":int(state.get("next_step", step+1)),"checkpoint_schema_version":SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION,"created_at":state.get("created_at") or created,"saved_at":created}
    for k, v in compatibility.items():
        full.setdefault(k, v)
    path=ck/f"checkpoint_step_{step:06d}.pt"
    path, sha, size = atomic_torch_save(full,path)
    meta={"checkpoint":path.name,"current_step":step,"next_step":full["next_step"],"tokens_processed":full.get("tokens_processed",0),"created_at":full["created_at"],"sha256":sha,"size_bytes":size,"verified":True,"compatibility_hash":stable_hash(compatibility),"checkpoint_schema_version":SCIENTIFIC_CHECKPOINT_SCHEMA_VERSION}
    _atomic_write_json(ck/"latest.json", meta)
    _append_inventory_atomic(ck/"checkpoint_inventory.csv", meta)
    return path


def complete_test_checkpoint_state(**overrides) -> dict:
    """Explicit test helper for constructing a complete scientific checkpoint."""
    state = {
        "model_state_dict": {}, "optimizer_state_dict": {}, "base_optimizer_state_dict": {}, "scheduler_state_dict": None,
        "gradient_scaler_state_dict": None, "current_step": 0, "next_step": 1,
        "tokens_processed": 0, "training_reader_position": 0, "reader_position": 0,
        "seed": 0, "wwpgd_state": {}, **rng_state(), "device_type": "cpu",
        "precision_policy": "torch_default", "gradient_accumulation_position": 0,
        "artifact_commits": {},
        "scientific_schema_version": 0, "compatibility": {}, "resolved_config": {}, "optimizer_fingerprint": "", "data_hash": "", "tokenizer_hash": "",
    }
    state.update(overrides)
    return state

def load_latest_checkpoint(run_dir: Path):
    ck=Path(run_dir)/"checkpoints"; latest=ck/"latest.json"
    if not latest.exists(): raise FileNotFoundError(f"missing latest checkpoint pointer: {latest}")
    meta=json.loads(latest.read_text()); path=ck/meta["checkpoint"]
    if not path.exists(): raise FileNotFoundError(f"latest checkpoint missing: {path}")
    size=path.stat().st_size
    if int(meta.get("size_bytes", -1)) != size: raise RuntimeError(f"checkpoint size mismatch for {path}")
    sha=_sha256_file(path)
    if meta.get("sha256") != sha: raise RuntimeError(f"checkpoint sha256 mismatch for {path}")
    obj=torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_keys(obj)
    return obj

def compatibility_mismatches(checkpoint: dict, expected: dict):
    got=checkpoint.get("compatibility",{})
    return {k:{"checkpoint":got.get(k),"expected":expected.get(k)} for k in REQUIRED_COMPAT if expected.get(k) is not None and got.get(k)!=expected.get(k)}

def assert_checkpoint_compatible(checkpoint: dict, expected: dict, *, allow_code_version_mismatch: bool = False) -> dict:
    mm=compatibility_mismatches(checkpoint, expected)
    blocking = {k: v for k, v in mm.items() if not (allow_code_version_mismatch and k in CODE_VERSION_COMPAT)}
    if blocking:
        raise RuntimeError("checkpoint compatibility validation failed: "+json.dumps(mm, sort_keys=True, default=str))
    return mm

def inspect_checkpoint(path: Path):
    path=Path(path)
    sha=_sha256_file(path); size=path.stat().st_size
    obj=torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_keys(obj)
    keys=("checkpoint_schema_version","scientific_schema_version","run_directory","pair_id","optimizer_name","seed","level","token_multiplier","current_step","next_step","tokens_processed","training_reader_position","reader_position","gradient_accumulation_position","next_projection_event_index","completed_projection_event_indexes","compatibility","data_hash","tokenizer_hash","validation_probe_hash","training_probe_hash","weightwatcher_version","weightwatcher_configuration","wwpgd_commit","git_commit","device_type","precision_policy","created_at","saved_at")
    out={k: obj.get(k) for k in keys}
    out.update({"sha256": sha, "size_bytes": size, "sha256_verified": True, "size_verified": True})
    return out

def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def expected_compatibility_from_run(run_dir: Path) -> dict:
    run_dir=Path(run_dir)
    manifest=_load_json(run_dir/"manifest.json")
    config=_load_json(run_dir/"config.json")
    data_manifest=_load_json(run_dir/"data_manifest.json")
    tokenizer_manifest=_load_json(run_dir/"tokenizer_manifest.json")
    init_hash=(run_dir/"initialization_hash.txt").read_text().strip()
    return {
        "configuration_hash": manifest.get("configuration_hash", stable_hash(config)),
        "data_hash": manifest.get("data_hash", data_manifest.get("corpus_hash")),
        "tokenizer_hash": manifest.get("tokenizer_hash", tokenizer_manifest.get("tokenizer_hash")),
        "initialization_hash": init_hash,
        "model_configuration_hash": manifest.get("model_configuration_hash", stable_hash(config.get("model", {}))),
        "training_configuration_hash": manifest.get("training_configuration_hash", stable_hash(config.get("train", {}))),
        "wwpgd_configuration_hash": manifest.get("wwpgd_configuration_hash", stable_hash(config.get("wwpgd", {}))),
        "validation_probe_hash": manifest.get("validation_probe_hash"),
        "training_probe_hash": manifest.get("training_probe_hash"),
        "scientific_schema_version": manifest.get("scientific_schema_version"),
        "optimizer_fingerprint": manifest.get("optimizer_fingerprint"),
    }


def validate_resume(run_dir: Path, expected: dict|None=None):
    ck=load_latest_checkpoint(run_dir); exp=expected or expected_compatibility_from_run(run_dir); mm=compatibility_mismatches(ck, exp)
    if mm:
        raise RuntimeError("checkpoint compatibility validation failed: "+json.dumps(mm, sort_keys=True, default=str))
    return {"compatible":True,"mismatches":mm,"next_step":int(ck.get("next_step", int(ck.get("step",0))+1)),"token_position":ck.get("training_reader_position", ck.get("reader_position")),"checkpoint_step":ck.get("step")}
