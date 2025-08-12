"""
Thumbnails Package
Contains thumbnail generation and brief creation tools
"""

from .brief import (
    generate_thumbnail_brief,
    generate_thumbnail_brief_from_script,
    save_thumbnail_brief,
)


__all__ = [
    "generate_thumbnail_brief",
    "generate_thumbnail_brief_from_script",
    "save_thumbnail_brief",
]
