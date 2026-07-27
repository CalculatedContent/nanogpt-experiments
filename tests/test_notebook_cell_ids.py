import json
from pathlib import Path


def test_all_notebook_cells_have_stable_ids():
    missing: dict[str, list[int]] = {}
    duplicate: dict[str, list[str]] = {}

    for path in sorted(Path("notebooks").glob("*.ipynb")):
        payload = json.loads(path.read_text())
        ids: list[str] = []
        missing_indexes: list[int] = []
        for index, cell in enumerate(payload.get("cells", [])):
            cell_id = cell.get("id")
            if not isinstance(cell_id, str) or not cell_id.strip():
                missing_indexes.append(index)
            else:
                ids.append(cell_id)
        if missing_indexes:
            missing[str(path)] = missing_indexes
        repeated = sorted({cell_id for cell_id in ids if ids.count(cell_id) > 1})
        if repeated:
            duplicate[str(path)] = repeated

    assert not missing, f"notebook cells missing stable ids: {missing}"
    assert not duplicate, f"notebook cells have duplicate ids: {duplicate}"
