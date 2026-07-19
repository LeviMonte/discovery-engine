"""Data-loading tests. Skipped gracefully if the datasets were not downloaded."""
import pytest

from discovery.data import DATA_DIR, load_molecular, load_superconductor


@pytest.mark.skipif(not (DATA_DIR / "superconductor" / "train.csv").exists(),
                    reason="superconductor data not downloaded")
def test_superconductor_shape():
    raw = load_superconductor()
    assert raw.n > 20000
    assert raw.table.shape[1] == 81
    assert raw.y.min() >= 0


@pytest.mark.skipif(not (DATA_DIR / "esol.csv").exists(),
                    reason="esol not downloaded")
def test_molecular_load():
    raw = load_molecular("esol")
    assert raw.n > 1000
    assert raw.smiles is not None
    assert len(raw.smiles) == raw.n
