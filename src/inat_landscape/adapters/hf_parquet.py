"""Parquet metadata that makes the Hugging Face viewer show images and column types."""

import json

import pyarrow as pa

IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
_DTYPES = {
    pa.int64(): "int64",
    pa.float64(): "float64",
    pa.string(): "string",
    pa.bool_(): "bool",
}


def image_column(images: list[bytes], paths: list[str]) -> pa.Array:
    """Return an image column in the layout of the `datasets` Image feature."""
    return pa.array(
        [{"bytes": b, "path": p} for b, p in zip(images, paths, strict=True)], IMAGE_TYPE
    )


def with_hf_features(table: pa.Table) -> pa.Table:
    """Add the `huggingface` schema metadata that declares each column's feature."""
    features = {f.name: _feature(f.type) for f in table.schema}
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    return table.replace_schema_metadata(meta)


def _feature(dtype: pa.DataType) -> dict[str, str]:
    if dtype == IMAGE_TYPE:
        return {"_type": "Image"}
    if dtype not in _DTYPES:
        raise TypeError(f"no Hugging Face feature for column type {dtype}")
    return {"dtype": _DTYPES[dtype], "_type": "Value"}
