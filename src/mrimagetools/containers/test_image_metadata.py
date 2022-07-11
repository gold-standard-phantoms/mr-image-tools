"""ImageMetadata tests"""
from mrimagetools.containers.image_metadata import ImageMetadata


def test_image_metadata() -> None:
    """Do a very basic test on the ImageMetadata class. Set a single
    argument and read it out"""
    i = ImageMetadata(
        series_description="foo",
    )
    assert i.series_description == "foo"
    # Check that unset values are removed when using "exclude_unset"
    assert i.dict(exclude_unset=True) == {"series_description": "foo"}
