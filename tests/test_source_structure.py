from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


def test_source_has_no_duplicate_top_level_definitions() -> None:
    duplicates: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        names = Counter(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        duplicates.extend(
            f"{path}:{name}" for name, count in names.items() if count > 1
        )
    assert duplicates == []


def test_analysis_does_not_hardcode_level_zero_multiplier_twenty() -> None:
    source = Path("src/wwgpt/analysis.py").read_text()
    assert '"level_00" / "multiplier_20"' not in source
