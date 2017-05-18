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


class BiHiera:
    def __init__(self,hicode1,hicode2,codepair_counts):
        self._hicode1 = hicode1
        self._hicode2 = hicode2
        self._load(codepair_counts)
             
    def _load(self, cpc):
        a = np.fromiter((
            (self._hicode1.c2i[c1],self._hicode2.c2i[c2],cpc[(c1,c2)])
            for c1,c2 in cpc
        ), dtype=[("i",int),("j",int),("n",float)])
        self._M = sparse.csr_matrix((a["n"],(a["i"],a["j"])))
        self._J = self._hicode1.T.transpose().dot(self._M).dot(self._hicode2.T).todok()

    def tree_given1(self,code1):
        i1 = self._hicode1.c2i[code1]
        N = self._J[i1,self._hicode2._p2i[()]]
        v = dict(self._J[i1,:])
        t = Tree(default_attrs={"code": None, "N": 0, "p": 0})
        for _,i2 in v:
            c2 = self._hicode2.i2c[i2]
            t.at(self._hicode2._i2p[i2]).attrs = {
                "code": c2,
                "N": v[(0,i2)],
                "p": v[(0,i2)]/N
            }
        return t

    def tree_given2(self,code2):
        i2 = self._hicode2.c2i[code2]
        N = self._J[self._hicode1._p2i[()],i2]
        v = dict(self._J[:,i2])
        t = Tree(default_attrs={"code": None, "N": 0, "p": 0})
        for i1,_ in v:
            c1 = self._hicode1.i2c[i1]
            t.at(self._hicode1._i2p[i1]).attrs = {
                "code": c1,
                "N": v[(i1,0)],
                "p": v[(i1,0)]/N
            }
        return t
