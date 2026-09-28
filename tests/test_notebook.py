from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _generator():
    spec = importlib.util.spec_from_file_location("nb", Path("scripts/make_colab_notebook.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_notebook_matches_its_generator() -> None:
    committed = json.loads(Path("notebooks/qwen3_8b_qlora_colab.ipynb").read_text())
    assert committed == _generator().notebook()


def test_notebook_is_marked_not_run_and_has_no_outputs() -> None:
    nb = _generator().notebook()
    assert "not run" in "".join(nb["cells"][0]["source"])
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    source = "".join("".join(c["source"]) for c in code)
    assert "bnb_4bit_compute_dtype=torch.float16" in source  # the T4 has no bf16
    assert "TaskType.SEQ_CLS" in source
