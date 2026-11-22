import pytest
from src.safety_eval.data import tag_identities

def test_tag_identities():
    text = "The Muslim community and Jewish leaders gathered with Christian priests."
    tags = tag_identities(text)
    assert tags["muslim"] == 1
    assert tags["jewish"] == 1
    assert tags["christian"] == 1
    assert tags["lgbtq"] == 0

def test_tag_identities_empty():
    tags = tag_identities("")
    assert all(v == 0 for v in tags.values())
