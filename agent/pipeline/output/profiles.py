"""Non sensitive adaptive output route profiles keyed by immutable contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ...application.io import atomic_write
from .models import CompilationOutcome, OutputRoute

PROFILE_VERSION = 3
MAX_PROFILES = 256
DEMOTION_THRESHOLD = 2
RouteBaseline = Literal["basic_schema", "basic_json", "text_projection"]
_ADAPTIVE_ROUTES: tuple[OutputRoute, ...] = (
    "content_schema",
    "content_json",
    "tagged_text",
)
_BASELINES: dict[str, OutputRoute] = {
    "basic_schema": "content_schema",
    "basic_json": "content_json",
    "text_projection": "tagged_text",
}


@dataclass
class ContractProfile:
    key: str
    model_digest: str
    policy_id: str
    prompt_hash: str
    contract_version: str
    route: OutputRoute
    success_count: int = 0
    failure_count: int = 0
    quality_failure_count: int = 0
    consecutive_structural_failures: int = 0
    last_failure_run_id: str | None = None


def profile_key(model_digest: str, policy_id: str, prompt_hash: str, contract_version: str) -> str:
    raw = "\0".join((model_digest, policy_id, prompt_hash, contract_version)).encode()
    return hashlib.sha256(raw).hexdigest()


class ModelContractProfileStore:
    """Atomic route learning that never persists prompts, generated text, or fallback routes."""

    def __init__(self, path: str | Path = ".cache/model-contract-profiles.json") -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, ContractProfile]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        # Older profiles mixed transport success with recovered content success. Relearn safely.
        if not isinstance(raw, dict) or raw.get("version") != PROFILE_VERSION:
            return {}
        profiles: dict[str, ContractProfile] = {}
        for item in raw.get("profiles", []):
            try:
                profile = ContractProfile(**item)
            except (TypeError, ValueError):
                continue
            if profile.route in _ADAPTIVE_ROUTES:
                profiles[profile.key] = profile
        return profiles

    def _save(self, profiles: dict[str, ContractProfile]) -> None:
        ordered = list(profiles.values())[-MAX_PROFILES:]
        payload = {"version": PROFILE_VERSION, "profiles": [asdict(item) for item in ordered]}
        atomic_write(self.path, json.dumps(payload, indent=2, sort_keys=True))

    def route_for(
        self,
        *,
        model_digest: str,
        policy_id: str,
        prompt_hash: str,
        contract_version: str,
        baseline: RouteBaseline = "basic_schema",
    ) -> OutputRoute:
        key = profile_key(model_digest, policy_id, prompt_hash, contract_version)
        profile = self._load().get(key)
        return profile.route if profile else _BASELINES[baseline]

    def record(
        self,
        *,
        model_digest: str,
        policy_id: str,
        prompt_hash: str,
        contract_version: str,
        route: OutputRoute,
        outcome: CompilationOutcome,
        run_id: str | None = None,
        structural_failure: bool = False,
        confirmed_incompatibility: bool = False,
    ) -> ContractProfile:
        profiles = self._load()
        key = profile_key(model_digest, policy_id, prompt_hash, contract_version)
        safe_route = route if route in _ADAPTIVE_ROUTES else _BASELINES["basic_schema"]
        profile = profiles.get(key) or ContractProfile(
            key=key,
            model_digest=model_digest,
            policy_id=policy_id,
            prompt_hash=prompt_hash,
            contract_version=contract_version,
            route=safe_route,
        )
        if structural_failure:
            profile.failure_count += 1
            failure_run_id = run_id or f"unscoped:{profile.failure_count}"
            if failure_run_id != profile.last_failure_run_id:
                profile.consecutive_structural_failures += 1
                profile.last_failure_run_id = failure_run_id
            if confirmed_incompatibility or (
                profile.consecutive_structural_failures >= DEMOTION_THRESHOLD
            ):
                index = _ADAPTIVE_ROUTES.index(profile.route)
                profile.route = _ADAPTIVE_ROUTES[min(index + 1, len(_ADAPTIVE_ROUTES) - 1)]
                profile.consecutive_structural_failures = 0
                profile.last_failure_run_id = None
        elif outcome in {"exact", "normalized"}:
            profile.success_count += 1
            profile.consecutive_structural_failures = 0
            profile.last_failure_run_id = None
        else:
            profile.failure_count += 1
            profile.quality_failure_count += 1
        profiles.pop(key, None)
        profiles[key] = profile
        self._save(profiles)
        return profile

    def reset(self) -> None:
        self._save({})


__all__ = [
    "ContractProfile",
    "DEMOTION_THRESHOLD",
    "ModelContractProfileStore",
    "PROFILE_VERSION",
    "profile_key",
]
