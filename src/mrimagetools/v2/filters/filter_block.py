""" FilterBlock class """

from abc import abstractmethod
from typing import List, Optional

from mrimagetools.v2.filters.basefilter import BaseFilter


class FilterBlock(BaseFilter):
    """A filter made from multiple, chained filters. Used for when
    the same configuration of filters is used multiple times, or needs
    to be tested as a whole"""

    @abstractmethod
    def _create_filter_block(self) -> BaseFilter:
        """
        Constructs the filter block and returns the filter which is at the
        end of the FilterBlock (the final filter with the desired outputs)
        This is where all of the filters should be constructed and linked.
        NOTE: the filters should not be run here. This is taken care of
        when the FilterBlock is run.
        THIS SHOULD BE OVERWRITTEN IN THE SUBCLASS
        """

    def _run(self) -> None:
        pass  # do nothing

    def run(
        self, validate_only: bool = False, history: Optional[list["BaseFilter"]] = None
    ) -> None:
        """
        Calls the BaseFilter's run method to make sure all of the
        inputs of this FilterBlock are up-to-date and valid. Then runs
        this FilterBlock's output filter, and populates the outputs
        to this FilterBlock.
        """
        super().run(validate_only=validate_only, history=history)
        filter_block = self._create_filter_block()
        filter_block.run(validate_only=validate_only)
        self.outputs = filter_block.outputs
