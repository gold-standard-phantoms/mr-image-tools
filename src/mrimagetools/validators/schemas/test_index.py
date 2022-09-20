""" JSON schema index tests"""
from mrimagetools.validators.schemas.index import load_schemas


def test_get_schemas() -> None:
    """Check that loading the schemas is successful
    and does not throw any SchemaError"""
    assert len(load_schemas()) > 0  # we have more than one schema saved
