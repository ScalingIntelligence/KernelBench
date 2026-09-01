try:
    from ._version import __version__ as __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"
