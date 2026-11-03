"""HF feature metadata."""

import json

import pyarrow as pa
import pytest

from inat_landscape.adapters.hf_parquet import with_hf_features


def test_unsupported_column_type_is_an_error():
    with pytest.raises(TypeError, match="feature"):
        with_hf_features(pa.table({"x": pa.array([[1]], pa.list_(pa.int64()))}))


def test_large_strings_have_a_feature():
    table = with_hf_features(pa.table({"t": pa.array(["a"], pa.large_string())}))
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    assert features["t"] == {"dtype": "large_string", "_type": "Value"}
