"""Licences and attribution, as the iNaturalist Open Data README defines them."""

import pytest

from inat_landscape.domain.licensing import ALLOWED_LICENSES, attribution, is_allowed


def test_allowed_licenses_are_the_three_open_data_licenses():
    assert ALLOWED_LICENSES == ("CC0", "CC-BY", "CC-BY-NC")


@pytest.mark.parametrize(
    ("lic", "ok"),
    [
        ("CC0", True),
        ("cc-by", True),
        ("CC-BY-NC", True),
        ("CC-BY-SA", False),
        ("", False),
        (None, False),
    ],
)
def test_is_allowed(lic, ok):
    assert is_allowed(lic) is ok


def test_cc0_attribution_uses_no_rights_reserved():
    assert attribution("CC0", "Ana B", "anab") == "Ana B, no rights reserved (CC0)"


def test_cc_attribution_falls_back_to_login():
    assert attribution("CC-BY-NC", "", "anab") == "© anab, some rights reserved (CC-BY-NC)"
    assert attribution("cc-by", None, "anab") == "© anab, some rights reserved (CC-BY)"


def test_attribution_rejects_disallowed_license():
    with pytest.raises(ValueError, match="licen"):
        attribution("CC-BY-SA", "x", "y")
