import pytest  # noqa: F401

import os
import tempfile
import shutil

from mempalace.convo_miner import mine_convos
from mempalace.vector_store import RECORDS_COLLECTION, get_vector_store, reset_singletons


def test_convo_mining():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
        f.write(
            "> What is memory?\nMemory is persistence.\n\n> Why does it matter?\nIt enables continuity.\n\n> How do we build it?\nWith structured storage.\n"
        )

    palace_path = os.path.join(tmpdir, "palace")
    mine_convos(tmpdir, palace_path)

    reset_singletons()
    vs = get_vector_store(palace_path)
    assert vs.count(RECORDS_COLLECTION) >= 2

    # Verify search works
    results = vs.query(RECORDS_COLLECTION, query_texts=["memory persistence"], n_results=1)
    assert results.ids and results.ids[0], "expected at least one hit"

    reset_singletons()
    shutil.rmtree(tmpdir, ignore_errors=True)


pytestmark = pytest.mark.integration
