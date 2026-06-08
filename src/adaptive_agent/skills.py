from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .schemas import ToolDigest, ToolManifest, ToolSpec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillStore:
    """Persists generated tools as versioned skill directories and reloads them on demand."""

    def __init__(self, skills_dir: Path | str) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def _dir(self, name: str) -> Path:
        # Reject names containing path separators or traversal segments so a
        # skill name can never escape skills_dir.
        if name != Path(name).name:
            raise ValueError("잘못된 skill 이름입니다")
        return self.skills_dir / name

    def read_manifest(self, name: str) -> ToolManifest:
        """Read and parse the manifest for a persisted skill."""
        data = json.loads((self._dir(name) / "manifest.json").read_text("utf-8"))
        return ToolManifest.model_validate(data)

    def persist(self, spec: ToolSpec) -> ToolManifest:
        """Write or update a skill directory; increments version on each call."""
        d = self._dir(spec.name)
        d.mkdir(parents=True, exist_ok=True)
        version = 1
        created = _now()
        if (d / "manifest.json").exists():
            prev = self.read_manifest(spec.name)
            version = prev.version + 1
            created = prev.created_at
        manifest = ToolManifest(
            name=spec.name,
            description=spec.description,
            inputSchema=spec.input_schema,
            outputSchema=spec.output_schema,
            entrypoint=spec.entrypoint,
            runtime="python",
            createdAt=created,
            updatedAt=_now(),
            usageCount=0,
            trustedStatus="persisted",
            version=version,
            source="generated",
        )
        (d / "tool.py").write_text(spec.code, encoding="utf-8")
        (d / "manifest.json").write_text(
            json.dumps(manifest.model_dump(by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Human-readable summary for the skill directory.
        (d / "SKILL.md").write_text(
            f"# {spec.name}\n\n{spec.description}\n", encoding="utf-8"
        )
        return manifest

    def load_digests(self) -> list[ToolDigest]:
        """Return lightweight digests for all persisted skills, sorted by name."""
        digests = []
        for d in sorted(self.skills_dir.iterdir()):
            mf = d / "manifest.json"
            if d.is_dir() and mf.exists():
                m = ToolManifest.model_validate(json.loads(mf.read_text("utf-8")))
                digests.append(
                    ToolDigest(name=m.name, description=m.description, origin="generated")
                )
        return digests

    def load_spec(self, name: str) -> ToolSpec:
        """Lazily load the full ToolSpec (including code) for a named skill."""
        m = self.read_manifest(name)
        code = (self._dir(name) / "tool.py").read_text("utf-8")
        return ToolSpec(
            name=m.name,
            description=m.description,
            code=code,
            entrypoint=m.entrypoint,
            inputSchema=m.input_schema,
            outputSchema=m.output_schema,
        )
