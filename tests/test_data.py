import pytest

# datasets is only installed in the training environment (requirements-train.txt).
# Skip this test file when datasets is not available (e.g., serving-only CI).
pytest.importorskip("datasets", reason="datasets package not installed; install requirements-train.txt to run data tests")

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
