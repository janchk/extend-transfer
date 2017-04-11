from json import JSONEncoder
# Reassigning fd to make progress out
# updating current state when style.py try to write out progress
from kombu.utils import json


class Fdout:
    def __init__(self, **kwargs):
        print(kwargs)

    def write(self, smtxt):
        # print(smtxt)
        print(smtxt)


# emulation for parser's namespace arguments
# must specify 10 arguments at least
# json serialise
class Namespace:
    # def default(self, o):
    #     if isinstance(o, Namespace):
    #         return self.__dict__
    #     return JSONEncoder.default(self, o)
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    # def __repr__(self):
    #     return self.__dict__.__repr__()