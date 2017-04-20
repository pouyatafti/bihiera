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
        if l1 >= l2:    # path2 is more senior than path1
            # 1 if path1 is a descendant of path2, otherwise 0
            return 1. if path1[:l2] == path2 else 0.
        else:           # path1 is more senior than path2
            # prod(1/nchildren) up to the level of path2 if it is a descendant
            # of path1, otherwise 0
            return 1./reduce(
                mul,(self._t.at(path2[:i]).nchildren() for i in range(l1,l2))
            ) if path1 == path2[:l1] else 0.


# key is the name of the dimension, code is its value (e.g. "dx"->"i20.")
class MultiHiera:
    # query_fun must follow the definition
    #
    #   query_fun(target_key, given) -> dict(target_code:count)
    #
    # where:
    #
    #   target_key is the key for the tree on which we marginalise, and
    #   given is of the form
    #       list( (
    #           given_key,
    #           given_code_weights = dict(given_code:weight)
    #       ) ),
    #
    # and where for each target_code, the returned count is the weighted
    # aggregate
    #
    # sum_{i in observations}
    #   prod_{gk,gws in given}
    #       [codes[i,gk] ∩ gws]*max(gws[c] for c in codes[i,gk] ∩ gws])
    #
    # the idea is that for each value of target_code, we count the
    # observations that are a match in every given_key for one of the
    # corresponding given_codes (weighted by the transfer weight associated
    # with each given code)
    #
    # for the special value target_key = None, query_fun must return the total
    # number of observations fulfilling the given conditions (not
    # double-counting observations that match multiple target codes)
    def __init__(self,hicode_dict,list_query_fun):
        self._hicode_dict = hicode_dict
        self._keys = set(hicode_dict)
        self._qfun = list_query_fun

    # target codes with a depth below target_minimum_depth won't be considered
    def get_tree(self,target_key,given_key_code_list,target_minimum_depth=0):
        collected_counts,N = self._collect(target_key,given_key_code_list)
        propagated_counts = self._propagate(target_key,collected_counts,target_minimum_depth)
        # XXX hack for presentation
        N0 = max(propagated_counts.values()) if propagated_counts.values() else 0
        return self._counts2tree(target_key,propagated_counts,N)

    # collect counts from the given_key_code hierarchy
    def _collect(self,target_key,given_key_code_list):
        hcd = self._hicode_dict
        # map each code c to dict of codes ci and transfer weights ci->c within
        # the hierarchy, for aggregation by self._qfun()
        given = [
            (
                k, {
                    hcd[k].i2c[i] : hcd[k].T[i,hcd[k].c2i[c]] 
                    for i in hcd[k].T[:,hcd[k].c2i[c]].tocoo().row
                }
            ) 
            for k,c in given_key_code_list
        ]
        collected_counts = self._qfun(target_key,given)
        N = self._qfun(None,given)
        return collected_counts,N

    # propagate counts through target hierarchy
    def _propagate(self,target_key,collected_counts,target_minimum_depth):
        if len(collected_counts) == 0:
            return dict()

        hc = self._hicode_dict[target_key]
        # this also aggregates repeats of the same code
        _a = np.fromiter((
            (hc.c2i[c], collected_counts[c])
            for c in collected_counts
            if len(hc._c2p(c)) >= target_minimum_depth
        ), dtype=[("i","int"),("n",float)])

        a = sparse.csr_matrix(
            (_a["n"],(np.repeat(0,len(_a)).astype("int"),_a["i"])),
            shape=(1,hc.T.shape[0])
        )

        return a.dot(hc.T).todok()

    def _counts2tree(self,key,counts,N):
        t = Tree(default_attrs={"code": None, "N": 0, "p": 0})
        if len(counts) == 0:
            return t

        hc = self._hicode_dict[key]
        #N = counts[(0,hc._p2i[()])]
        v = dict(counts)
        for _,i in v:
            c = hc.i2c[i]
            t.at(hc._i2p[i]).attrs = {
                "code": c,
                "N": v[(0,i)],
                "p": v[(0,i)]/N
            }
        return t
