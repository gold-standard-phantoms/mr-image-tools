"""ImageMetadata tests"""


from mrimagetools.containers.bids_metadata import BidsMetadata
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


def test_simple_bids_conversion() -> None:
    """Do some simple ImageMetadata to BIDS conversion"""

    image_metadata = ImageMetadata(
        echo_time=5, bolus_cut_off_flag=True, label_duration=1.0
    )

    bids_metadata = image_metadata.to_bids()
    assert bids_metadata == BidsMetadata(
        EchoTime=5.0, BolusCutOffFlag=True, LabelingDuration=1.0  # type: ignore
    )
