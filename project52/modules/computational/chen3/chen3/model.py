# File: chen3/model.py

"""ChenModel encapsulates the three‐factor parameters."""

from .datatypes import ModelParams

class ChenModel:
    def __init__(self, params: ModelParams):
        self.params = params
