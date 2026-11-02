"""HF feature metadata."""

import pyarrow as pa
import pytest

from inat_landscape.adapters.hf_parquet import with_hf_features


def test_unsupported_column_type_is_an_error():
    with pytest.raises(TypeError, match="feature"):
        with_hf_features(pa.table({"x": pa.array([[1]], pa.list_(pa.int64()))}))
