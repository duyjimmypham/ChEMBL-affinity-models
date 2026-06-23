"""Focused regression tests for scientific data splitting and labeling."""

import numpy as np
import pandas as pd

from pipeline import (
    add_molecule_level_labels,
    make_scaffold_cv,
    scaffold_groups,
    scaffold_split_indices,
)


def test_absolute_labels_drop_gray_zone() -> None:
    molecules = pd.DataFrame({"p_activity": [4.0, 5.0, 6.5]})

    labeled = add_molecule_level_labels(molecules)

    assert labeled["activity"].tolist() == [0, 1]
    assert labeled["p_activity"].tolist() == [4.0, 6.5]


def test_scaffold_groups_keep_analogs_together() -> None:
    smiles = ["c1ccccc1O", "c1ccccc1N", "C1CCCCC1O"]

    groups = scaffold_groups(smiles)

    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_scaffold_cv_has_no_group_overlap() -> None:
    smiles = [
        "c1ccccc1O",
        "c1ccccc1N",
        "C1CCCCC1O",
        "C1CCCCC1N",
        "c1ccncc1O",
        "c1ccncc1N",
        "c1cncnc1O",
        "c1cncnc1N",
    ]
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    cv, groups, _ = make_scaffold_cv(smiles, y, max_splits=2)
    X = np.zeros((len(smiles), 1))

    for train_idx, validation_idx in cv.split(X, y, groups):
        train_groups = set(groups[train_idx])
        validation_groups = set(groups[validation_idx])
        assert train_groups.isdisjoint(validation_groups)


def test_scaffold_holdout_preserves_groups_and_classes() -> None:
    smiles = [
        "c1ccccc1O",
        "c1ccccc1N",
        "C1CCCCC1O",
        "C1CCCCC1N",
        "c1ccncc1O",
        "c1ccncc1N",
        "c1cncnc1O",
        "c1cncnc1N",
    ]
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = scaffold_groups(smiles)

    train_idx, test_idx = scaffold_split_indices(smiles, y=y, test_size=0.5)

    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    assert set(y[train_idx]) == {0, 1}
    assert set(y[test_idx]) == {0, 1}
