from importlib.metadata import version

from . import find_retained_cells, templates

__all__ = ["find_retained_cells", "templates"]
__version__ = version("qcatch")
