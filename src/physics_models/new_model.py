#!/usr/bin/env python3

from . import base_model

class NewModel(base_model.BaseModel):
    def __init__(self):
        pass


    def construct_H(self, L):
        raise NotImplementedError
