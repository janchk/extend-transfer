class Fdout:
    def __init__(self, tsk):
        self.tsk_id = tsk
        print("task ", tsk, "started")

    def write(self, prgrs):
        self.tsk_id.update_state(state="PROGRESS", meta=prgrs)
        print(prgrs)

    def flush(self) -> None:
        pass


# emulation for parser's namespace arguments
# must specify 10 arguments at least
# json serialise
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
