"""Tests for solvents module."""

from __future__ import annotations

import pandas.testing as pdt

from doptools.chem.solvents import SolventVectorizer


def test_solvent_vectorizer_selects_columns() -> None:
    """SolventVectorizer selects requested Katalan parameters."""
    vectorizer = SolventVectorizer(sp=True, sdp=False, sa=True, sb=False)

    result = vectorizer.transform(["water", None])

    assert vectorizer.get_feature_names() == ["SP Katalan", "SA Katalan"]
    expected = [[0.681, 1.062], [0.0, 0.0]]
    pdt.assert_frame_equal(
        result.reset_index(drop=True),
        vectorizer.transform(["water", None]).reset_index(drop=True),
    )
    assert result.values.tolist() == expected
