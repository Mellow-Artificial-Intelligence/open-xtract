from .main import Extract
from .main import extract_content_info as extract_content_info

# Backwards compatibility: alias OpenXtract to the new Extract class
OpenXtract = Extract

__all__ = ["Extract", "OpenXtract", "extract_content_info"]

