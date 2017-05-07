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
        print("task ", tsk, "started")

    def write(self, prgrs):
        self.tsk_id.update_state(state="PROGRESS", meta=prgrs)
        print(prgrs)


# emulation for parser's namespace arguments
# must specify 10 arguments at least
# json serialise
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
