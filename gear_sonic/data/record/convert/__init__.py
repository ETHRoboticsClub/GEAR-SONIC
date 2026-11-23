"""Cut invalid rows out of a recorded episode.

``segment_invalid`` is the schema CALLS function
``convert.segment_invalid(rows_valid_mask, *, min_segment_frames)``.
"""

from gear_sonic.data.record.convert.segment import segment_invalid

__all__ = ["segment_invalid"]
