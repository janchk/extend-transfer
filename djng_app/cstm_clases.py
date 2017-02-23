# Reassigning fd to make progress out
class Fdout:
    # def __init__(self, **kwargs):
    def write(self, smtxt):
        print (smtxt)


# emulation for parser's namespace arguments
# must specify arguments 10 at least
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)