"""
Tensorboard Wrapper

This is a wrapper class for the Tensorflow's Tensorboard application.
"""

from ._version import get_versions

from .tensorboard import Tensorboard
from .exceptions import BoardAlreadyExistsException


__version__ = get_versions()['version']
__author__ = "Nathan Gavenski"
__credits__ = "Machine Learning Theory and Applications Lab"
del get_versions
