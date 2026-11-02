"""Photo licences and attribution text, from the iNaturalist Open Data README."""

ALLOWED_LICENSES = ("CC0", "CC-BY", "CC-BY-NC")


def is_allowed(license_code: str | None) -> bool:
    """Tell if a photo licence is one that the dataset keeps."""
    # Any default that is not a licence is equivalent, so the line is not mutated.
    code = (license_code or "").upper()  # pragma: no mutate
    return code in ALLOWED_LICENSES


def attribution(license_code: str, name: str | None, login: str) -> str:
    """Return the attribution statement for one photo."""
    code = license_code.upper()
    if code not in ALLOWED_LICENSES:
        raise ValueError(f"licence not allowed: {license_code}")
    who = name or login
    if code == "CC0":
        return f"{who}, no rights reserved (CC0)"
    return f"© {who}, some rights reserved ({code})"
