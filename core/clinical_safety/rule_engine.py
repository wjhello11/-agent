from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SafetyFinding:
    rule_id: str
    severity: str
    category: str
    action: str
    message: str
    recommendation: str


class SafetyRuleEngine:
    def __init__(self, rules_path: str | Path):
        self.rules_path = Path(rules_path)
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self.rules = payload.get("rules", [])

    def evaluate(self, context: dict[str, Any]) -> list[SafetyFinding]:
        findings: list[SafetyFinding] = []
        normalized = _normalize_context(context)
        for rule in self.rules:
            if _matches_group(rule.get("if", {}), normalized):
                then = rule.get("then", {})
                findings.append(
                    SafetyFinding(
                        rule_id=str(rule.get("rule_id", "")),
                        severity=str(rule.get("severity", "warn")),
                        category=str(rule.get("category", "")),
                        action=str(then.get("action", "warn")),
                        message=str(then.get("message", "")),
                        recommendation=str(then.get("recommendation", "")),
                    )
                )
        return findings


def _normalize_context(context: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in context.items():
        if isinstance(value, list):
            normalized[key] = [_normalize_text(item) for item in value]
        elif isinstance(value, dict):
            normalized[key] = _normalize_context(value)
        else:
            normalized[key] = _normalize_text(value)
    return normalized


def _matches_group(group: dict[str, Any], context: dict[str, Any]) -> bool:
    if "all" in group:
        return all(_matches_clause(item, context) for item in group.get("all", []))
    if "any" in group:
        return any(_matches_clause(item, context) for item in group.get("any", []))
    if "not" in group:
        return not _matches_clause(group.get("not", {}), context)
    return False


def _matches_clause(clause: dict[str, Any], context: dict[str, Any]) -> bool:
    if any(key in clause for key in ("all", "any", "not")):
        return _matches_group(clause, context)
    return _matches_condition(clause, context)


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
    field_value = _get_field(context, str(condition.get("field", "")))
    values = [_normalize_text(item) for item in condition.get("values", [])]
    operator = condition.get("operator")

    if operator == "exists":
        return field_value is not None
    if operator == "equals":
        return _normalize_text(field_value) in values
    if operator == "contains_any":
        return any(_contains_value(field_value, item) for item in values)
    if operator == "contains_all":
        return all(_contains_value(field_value, item) for item in values)
    if operator == "greater_than":
        return _compare_number(field_value, values, lambda left, right: left > right)
    if operator == "less_than":
        return _compare_number(field_value, values, lambda left, right: left < right)
    return False


def _get_field(context: dict[str, Any], field_path: str) -> Any:
    current: Any = context
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _contains_value(field_value: Any, expected: str) -> bool:
    if field_value is None:
        return False
    if isinstance(field_value, list):
        return any(_text_contains_expected(_normalize_text(item), expected) for item in field_value)
    return _text_contains_expected(_normalize_text(field_value), expected)


def _text_contains_expected(text: str, expected: str) -> bool:
    if not expected:
        return False
    if _is_ascii_word(expected):
        normalized_text = _normalize_ascii_token_separators(text)
        normalized_expected = _normalize_ascii_token_separators(expected)
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_expected)}(?![a-z0-9])", normalized_text) is not None
    return expected in text


def _is_ascii_word(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_ -]+", value))


def _normalize_ascii_token_separators(value: str) -> str:
    return re.sub(r"[_\s-]+", " ", value.lower()).strip()


def _compare_number(field_value: Any, values: list[str], predicate) -> bool:
    if field_value is None or not values:
        return False
    try:
        left = float(field_value)
        right = float(values[0])
    except (TypeError, ValueError):
        return False
    return bool(predicate(left, right))


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()
