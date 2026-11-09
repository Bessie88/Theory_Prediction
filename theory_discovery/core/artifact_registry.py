"""Atomic, locked registry for the future integrated workflow.

Coordinator-only: business modules never receive this object, its index, or paths.
Instantiation is inert. Explicit commit/read APIs perform file I/O only when called.
"""
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from typing import get_type_hints
from .workflow_contracts import Record, EditorKey, MetadataItem, decode, encode, nonempty
from .schemas import require, stable_id
from .storage import write_json
from .stage_dependencies import BY_NAME, downstream_stages


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value):
    raw = json.dumps(encode(value), sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ArtifactEntry(Record):
    artifact_id: str
    stage_name: str
    stage_version: str
    experiment_id: str
    configuration_id: str
    editor: EditorKey
    artifact_type: str
    schema_name: str
    schema_version_name: str
    storage_location: str
    content_sha256: str
    upstream_artifact_ids: tuple[str, ...]
    generation_configuration: tuple[MetadataItem, ...]
    model_versions: tuple[str, ...]
    prompt_versions: tuple[str, ...]
    commit_id: str
    access_class: str
    status: str
    created_at_utc: str

    def validate(self):
        nonempty(self.artifact_id, self.stage_name, self.configuration_id, self.storage_location)
        require(self.status in ("COMPLETE", "STALE"), "Invalid artifact state")


@dataclass(frozen=True)
class StageFailure(Record):
    stage_name: str
    editor: EditorKey
    configuration_id: str
    exception_type: str
    reason_code: str
    occurred_at_utc: str


@dataclass(frozen=True)
class CohortArtifactEntry(Record):
    artifact_id: str
    configuration_id: str
    experiment_id: str
    schema_name: str
    schema_version_name: str
    storage_location: str
    content_sha256: str
    upstream_artifact_ids: tuple[str, ...]
    status: str
    created_at_utc: str


class RegistryIntegrityError(ValueError):
    pass


class ArtifactRegistry:
    def __init__(self, root, experiment_id, configuration_id):
        self.root = Path(root).resolve()
        self.experiment_id = experiment_id
        self.configuration_id = configuration_id

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "registry.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def record_configuration(self, config):
        require(config.configuration_id == self.configuration_id and
                config.experiment_id == self.experiment_id, "Configuration identity mismatch")
        with self._locked():
            path = self.root / "configurations" / (self.configuration_id + ".json")
            if path.exists():
                require(json.loads(path.read_text(encoding="utf-8")) == config.to_dict(),
                        "Resolved configuration snapshot changed")
            else:
                write_json(path, config.to_dict())

    def _read(self):
        path = self.root / "registry.json"
        if not path.exists():
            return {"schema": "integrated-registry-1", "entries": [], "failures": []}
        raw = json.loads(path.read_text(encoding="utf-8"))
        require(raw.get("schema") == "integrated-registry-1" and
                set(raw) in ({"schema", "entries", "failures"}, {"schema", "entries", "failures", "cohorts"}), "Unknown registry schema")
        return raw

    def _entries(self, raw):
        return [ArtifactEntry.from_dict(e) for e in raw["entries"]]

    def _payload(self, entry):
        path = (self.root / entry.storage_location).resolve()
        require(path.is_relative_to(self.root), "Artifact path escapes registry")
        raw = json.loads(path.read_text(encoding="utf-8"))
        require(content_hash(raw) == entry.content_sha256, "Artifact integrity mismatch")
        if entry.schema_name == "NativeTheoryBundle":
            # The native output is registered by reference, never copied/reformatted.
            for ref in (raw["theory"], *raw["sidecars"]):
                require(file_hash(ref["location"]) == ref["sha256"], "Native source changed")
        return raw

    def _fresh(self, entry, index, active=None):
        active = set() if active is None else active
        if entry.artifact_id in active:
            raise RegistryIntegrityError("Registry dependency cycle")
        if entry.status != "COMPLETE":
            return False
        active.add(entry.artifact_id)
        try:
            self._payload(entry)
            for key in entry.upstream_artifact_ids:
                upstream = index.get(key)
                if upstream is None or not self._fresh(upstream, index, active):
                    return False
            return True
        except (OSError, ValueError, TypeError):
            return False
        finally:
            active.remove(entry.artifact_id)

    def completed(self, spec, editor, upstream):
        """Resume requires one coherent commit with exact upstream identities."""
        with self._locked():
            entries = self._entries(self._read())
            index = {e.artifact_id: e for e in entries}
            expected = tuple(sorted(e.artifact_id for e in upstream))
            candidates = [
                e for e in entries if e.stage_name == spec.name and
                e.stage_version == spec.version and e.editor == editor and
                e.schema_version_name == spec.output_type.schema_version and
                e.configuration_id == self.configuration_id and
                e.experiment_id == self.experiment_id and
                e.upstream_artifact_ids == expected and self._fresh(e, index)
            ]
            for commit in reversed(tuple(dict.fromkeys(e.commit_id for e in candidates))):
                selected = {e.artifact_type: e for e in candidates if e.commit_id == commit}
                if set(selected) == {f.name for f in fields(spec.output_type)}:
                    return selected
            return None

    def read_for_binding(self, entry, consumer, binding):
        """Only declared ports may resolve; returns typed payload, never a location."""
        require(binding in consumer.bindings, "Undeclared stage input")
        require((entry.stage_name, entry.artifact_type) ==
                (binding.producer, binding.output_field), "Wrong upstream port")
        producer = BY_NAME[binding.producer]
        require(entry.access_class == dict(producer.access)[binding.output_field],
                "Artifact access-class mismatch")
        expected_type = get_type_hints(consumer.input_type)[binding.input_field]
        require(expected_type.__name__ == entry.schema_name and
                expected_type.schema_version == entry.schema_version_name,
                "Artifact contract version mismatch")
        # Consumer input dataclasses are the access allowlist. An annotator has
        # no binding for predictions or theory mappings, including via config.
        with self._locked():
            current = self._entries(self._read())
            index = {e.artifact_id: e for e in current}
            require(index.get(entry.artifact_id) == entry and self._fresh(entry, index),
                    "Stale or unregistered input")
            return decode(expected_type, self._payload(entry))

    def commit(self, spec, editor, outputs, upstream, generation=(), models=(), prompts=()):
        require(type(outputs) is spec.output_type, "Wrong stage output type")
        require({e.stage_name for e in upstream} == set(spec.dependencies),
                "Missing/extra upstream producer")
        for producer in spec.dependencies:
            require({e.artifact_type for e in upstream if e.stage_name == producer} ==
                    {f.name for f in fields(BY_NAME[producer].output_type)},
                    "Incomplete upstream stage commit")
        require(all(e.editor == editor and e.configuration_id == self.configuration_id and
                    e.experiment_id == self.experiment_id for e in upstream),
                "Mixed editor/configuration/experiment lineage")
        def check_editor(value):
            if isinstance(value, EditorKey):
                require(value == editor, "Output belongs to another editor")
            elif isinstance(value, Record):
                for field in fields(value):
                    check_editor(getattr(value, field.name))
            elif isinstance(value, tuple):
                for item in value:
                    check_editor(item)
        check_editor(outputs)
        now = datetime.now(timezone.utc).isoformat()
        upstream_ids = tuple(sorted(e.artifact_id for e in upstream))
        body_hash = content_hash(outputs)
        commit = stable_id("stage_commit", [self.configuration_id, editor.to_dict(),
                                           spec.name, spec.version, upstream_ids, body_hash])
        with self._locked():
            raw = self._read()
            entries = self._entries(raw)
            index = {e.artifact_id: e for e in entries}
            require(all(index.get(e.artifact_id) == e and self._fresh(e, index)
                        for e in upstream), "Upstream changed before commit")
            # A newly requested commit supersedes this stage and all dependent
            # stages for this editor; original files are preserved for inspection.
            affected = downstream_stages(spec.name)
            for e in raw["entries"]:
                if (e["stage_name"] in affected and e["editor"] == editor.to_dict() and
                        e["experiment_id"] == self.experiment_id):
                    e["status"] = "STALE"
            created = {}
            for field in fields(outputs):
                payload = getattr(outputs, field.name)
                digest = content_hash(payload)
                artifact_id = stable_id("artifact", [commit, field.name, digest])
                location = "objects/" + artifact_id + ".json"
                write_json(self.root / location, encode(payload))
                entry = ArtifactEntry(
                    artifact_id, spec.name, spec.version, self.experiment_id,
                    self.configuration_id, editor, field.name, type(payload).__name__,
                    payload.schema_version, location, digest, upstream_ids, generation,
                    models, prompts, commit, dict(spec.access)[field.name], "COMPLETE", now)
                # Remove metadata duplicate only; immutable object bytes remain.
                raw["entries"] = [e for e in raw["entries"] if e["artifact_id"] != artifact_id]
                raw["entries"].append(entry.to_dict())
                created[field.name] = entry
            # One atomic registry commit after ALL output objects are written.
            write_json(self.root / "registry.json", raw)
            return created

    def invalidate(self, stage_name, editor):
        affected = downstream_stages(stage_name)
        with self._locked():
            raw = self._read()
            for entry in raw["entries"]:
                if (entry["stage_name"] in affected and entry["editor"] == editor.to_dict() and
                        entry["experiment_id"] == self.experiment_id):
                    entry["status"] = "STALE"
            write_json(self.root / "registry.json", raw)

    def record_failure(self, stage_name, editor, exception):
        # Never persist arbitrary exception text containing outcome data.
        failure = StageFailure(stage_name, editor, self.configuration_id,
                               type(exception).__name__, "STAGE_FAILED_NO_RESULT",
                               datetime.now(timezone.utc).isoformat())
        with self._locked():
            raw = self._read()
            raw["failures"].append(failure.to_dict())
            write_json(self.root / "registry.json", raw)


    def read_completed_port(self, entry, stage_name, port):
        """Coordinator-only typed read for cross-editor aggregation."""
        spec = BY_NAME[stage_name]
        require(entry.stage_name == stage_name and entry.artifact_type == port, "Wrong report port")
        expected = get_type_hints(spec.output_type)[port]
        with self._locked():
            index = {e.artifact_id: e for e in self._entries(self._read())}
            require(index.get(entry.artifact_id) == entry and self._fresh(entry, index) and
                    entry.schema_version_name == expected.schema_version, "Stale report")
            return decode(expected, self._payload(entry))

    def commit_cohort(self, payload, upstream):
        """Explicit cohort scope inside the ONE registry/index; no invented EditorKey."""
        require(upstream and all(e.stage_name == "final_report" and e.artifact_type == "research_report" and
                e.configuration_id == self.configuration_id and e.experiment_id == self.experiment_id
                for e in upstream), "Cohort requires compatible registered final reports")
        require({e.editor for e in upstream} == {r.editor for r in payload.editor_reports},
                "Cohort editor lineage differs")
        with self._locked():
            raw = self._read()
            index = {e.artifact_id: e for e in self._entries(raw)}
            require(all(index.get(e.artifact_id) == e and self._fresh(e, index) for e in upstream),
                    "Cohort input changed")
            digest = content_hash(payload)
            ids = tuple(sorted(e.artifact_id for e in upstream))
            identity = stable_id("cohort_artifact", [self.configuration_id, ids, digest])
            location = "objects/" + identity + ".json"
            write_json(self.root / location, encode(payload))
            entry = CohortArtifactEntry(identity, self.configuration_id, self.experiment_id,
                type(payload).__name__, payload.schema_version, location, digest, ids, "COMPLETE", datetime.now(timezone.utc).isoformat())
            previous = raw.setdefault("cohorts", [])
            raw["cohorts"] = [e for e in previous if e["artifact_id"] != identity] + [entry.to_dict()]
            write_json(self.root / "registry.json", raw)
            return entry

    def read_cohort(self, entry, expected_type):
        with self._locked():
            raw = self._read()
            require(entry.to_dict() in raw.get("cohorts", []) and entry.configuration_id == self.configuration_id and
                    entry.schema_version_name == expected_type.schema_version and
                    entry.schema_name == expected_type.__name__, "Wrong cohort contract")
            index = {e.artifact_id: e for e in self._entries(raw)}
            require(self._fresh(entry, index), "Cohort dependencies stale")
            return decode(expected_type, self._payload(entry))
