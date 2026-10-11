"""QuantToolBox: econometrics, portfolio optimization, and risk analytics.

Python port of the MATLAB QuantToolBox library. See README.md and
docs/migration_map.md for architecture notes and the original-file mapping.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quanttoolbox")
except PackageNotFoundError:
    # Source checkouts may not have installed package metadata.
    __version__ = "0.0.0+unknown"
