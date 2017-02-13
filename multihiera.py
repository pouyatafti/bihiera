import numpy as np
from scipy import sparse

from functools import reduce
from operator import mul

from .tree import Tree

# exposes .i2c .c2i (idx<->code) and .T (sparse propagation matrix)
class HieraCode:
    def __init__(self,codes,code2path,path2code):
        self._c2p = code2path
        self._p2c = path2code
        self._load(codes)

    def _load(self,codes):
        # create and freeze tree
        self._t = Tree()
        for c in codes:
            self._t.at(self._c2p(c))
        self._t.freeze()

        # index and create transfer matrix
        self._i2p = sorted(((),) + self._t.descendantpaths(partial=True))
        self._p2i = { p:i for i,p in enumerate(self._i2p) }

        self.i2c = [self._p2c(p) for p in self._i2p]
        self.c2i = { c:i for i,c in enumerate(self.i2c) }

        a = np.fromiter((
            (i,j,self._transweight(pi,pj))
            for i,pi in enumerate(self._i2p)
            for j,pj in enumerate(self._i2p)
            if HieraCode._connected(pi,pj)
        ), dtype=[("i",int),("j",int),("w",float)])
        self.T = sparse.csr_matrix((a["w"],(a["i"],a["j"])))

    @staticmethod
    def _connected(path1,path2):
        l1,l2 = len(path1),len(path2)
        snr = l1 < l2
        return (snr and (path1==path2[:l1])) or (not snr and (path1[:l2]==path2))
        
    def _transweight(self,path1,path2):
        l1,l2 = len(path1),len(path2)
        if l1 >= l2: # path2 is more senior than path1
            # 1 if path1 is a descendant of path2, otherwise 0
            return 1. if path1[:l2] == path2 else 0.
        else: # path1 is more senior than path2
            # prod(1/nchildren) up to the level of path2 if it is a descendant of
            # path1, otherwise 0
            return 1./reduce(
                mul,(self._t.at(path2[:i]).nchildren() for i in range(l1,l2))
            ) if path1 == path2[:l1] else 0.


# key is the name of the dimension, code is its value (e.g. "dx"->"i20.")
class MultiHiera:
    def __init__(self,hicode_dict,list_query_fun):
        self._hicode_dict = hicode_dict
        self._keys = set(hicode_dict)
        self._qfun = list_query_fun
             
    def get_tree(self,target_key,given_key_code_pairs):
        simple_counts = self._qfun(target_key,given_key_code_pairs)
        propagated_counts = self._propagate(target_key,simple_counts)
        return self._counts2tree(target_key,propagated_counts) 
         
    def _propagate(self,key,simple_counts):
        hc = self._hicode_dict[key]
        # this also aggregates repeats of the same code
        _a = np.fromiter((
            (hc.c2i[c], simple_counts[c])
            for c in simple_counts
        ), dtype=[("i","int"),("n",float)])
        a = sparse.csr_matrix((_a["n"],(0,_a["i"])))
        return a.dot(hc.T).todok()

    def _counts2tree(self,key,counts):
        hc = self._hicode_dict[key]
        N = counts[hc._p2i[()]]
        v = dict(counts)
        t = Tree(default_attrs={"code": None, "N": 0, "p": 0})
        for _,i in v:
            c = hc.i2c[i]
            t.at(hc._i2p[i]).attrs = {
                "code": c,
                "N": v[(0,i)],
                "p": v[(0,i)]/N
            }
        return t
