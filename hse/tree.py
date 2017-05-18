from collections import defaultdict
from copy import deepcopy
from functools import reduce
from operator import add
from json import dumps

class Tree(defaultdict):
    def __init__(self, parent=None, default_attrs=None):
        self.parent = parent
        self.attrs = deepcopy(default_attrs)
        defaultdict.__init__(self, lambda: Tree(self, default_attrs))

    def __str__(self, *args, **kwargs):
        return dumps(self, sort_keys=True, indent=4)

    def at(self, path):
        for key in path:
            self = self[key]
        return self

    def freeze(self):
        self.default_factory = None    
        for key in self:
            self[key].freeze()
        return self

    def deepcopy(self):
        return deepcopy(self)

    def rootdistance(self):
        return 0 if self.parent is None else 1+self.parent.rootdistance()

    def maxdepth(self):
        return 0 if not self else 1 + max((self[k].maxdepth() for k in self))

    def nsiblings(self):
        return len(self.parent)

    def nchildren(self):
        return len(self)

    def ndescendants(self, dist):
        if dist < 0:
            return 0
        elif dist == 0:
            return 1
        elif dist == 1:
            return self.nchildren()
        else:
            return sum((self[k].ndescendants(dist-1) for k in self))

    def descendantpaths(self,prefix=(),maxdist=-1,partial=False):
        if not self or maxdist==0:
            return (prefix,)
        else:
            return reduce(add,(self[k].descendantpaths(prefix+(k,),maxdist-1,partial) for k in self)) + (() if not partial else (prefix,))
