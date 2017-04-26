from json import JSONEncoder
from celery.result import AsyncResult
from celery import Celery
# app = Celery('style', backend='redis://localhost', broker='pyamqp://guest@localhost//')
# Reassigning fd to make progress out
# updating current state when style.py try to write out progress
from kombu.utils import json


class Fdout:
    def __init__(self, tsk):
        self.tsk_id = tsk
        print(tsk, 'lol')

    def write(self, smtxt):
        #  todo need to reassign as backend progress
        self.tsk_id.update_state(state=smtxt)
        # print(smtxt)
        print(smtxt) # n, AsyncResult(id=self.tsk_id, app=app).get())


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