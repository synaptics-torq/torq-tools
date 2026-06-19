from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(getattr(item, "path", item.fspath)))
        parts = set(path.parts)
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        if "graph_edit" in parts:
            item.add_marker(pytest.mark.graph_edit)
