import numpy as np

from discovery.featurize import (
    Standardizer, bemis_murcko_scaffold, morgan_and_descriptors,
    scaffold_split, _scaffold_split_from_ids,
)

SMILES = [
    "CCO", "CCCO", "c1ccccc1", "c1ccccc1C", "c1ccccc1CC",
    "CC(=O)O", "CCN", "c1ccncc1", "CCOC(=O)C", "C1CCCCC1",
    "c1ccc2ccccc2c1", "CCCCCC", "CC(C)O", "c1ccccc1O", "CCCl",
]


def test_morgan_shapes_and_validity():
    X, valid, nb, nd = morgan_and_descriptors(SMILES, n_bits=256, use_descriptors=True)
    assert X.shape == (len(SMILES), 256 + nd)
    assert valid.all()
    assert nb == 256 and nd > 0


def test_invalid_smiles_flagged():
    X, valid, nb, nd = morgan_and_descriptors(["CCO", "not_a_smiles"], n_bits=64)
    assert valid[0] and not valid[1]


def test_scaffold_split_no_shared_scaffold():
    tr, te = scaffold_split(SMILES, test_frac=0.3, seed=0)
    assert len(set(tr) & set(te)) == 0
    scaf_tr = {bemis_murcko_scaffold(SMILES[i]) for i in tr}
    scaf_te = {bemis_murcko_scaffold(SMILES[i]) for i in te}
    # zero scaffolds shared between train and test
    assert scaf_tr.isdisjoint(scaf_te)
    assert len(te) > 0 and len(tr) > 0


def test_scaffold_split_from_ids_disjoint():
    scaffolds = np.array([bemis_murcko_scaffold(s) for s in SMILES])
    tr, te = _scaffold_split_from_ids(scaffolds, 0.3, seed=1)
    assert set(scaffolds[tr]).isdisjoint(set(scaffolds[te]))


def test_standardizer_uses_train_stats_only():
    rng = np.random.RandomState(0)
    X = rng.normal(5.0, 3.0, size=(100, 4)).astype(np.float32)
    train_idx = np.arange(60)
    std = Standardizer.fit(X, train_idx)
    # mean/std must match TRAIN slice, not the whole array
    assert np.allclose(std.mean, X[train_idx].mean(axis=0), atol=1e-4)
    Xz = std.transform(X)
    assert np.allclose(Xz[train_idx].mean(axis=0), 0.0, atol=1e-4)
    # full-array mean would differ -> confirm we did NOT use it
    assert not np.allclose(std.mean, X.mean(axis=0), atol=1e-3)
