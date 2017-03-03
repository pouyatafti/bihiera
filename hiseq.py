import warnings

import numpy as np
from scipy import sparse
import tensorflow as tf

from functools import reduce
from operator import mul

from tree import Tree

def subpaths(p):
    return ( p[:i] for i in range(0,len(p)+1) )

class HierarchicalCode:
    def __init__(self, symbols, sym2path, path2sym):
        self._s2p_fun = sym2path
        self._p2s_fun = path2sym
        self._s2s_fun = lambda s: path2sym(sym2path(s))
        self._p2p_fun = lambda s: sym2path(path2sym(s))

        self._load(symbols)

    def _extend_symbols(self, symbols):
        if not all(s == self._s2s_fun(s) for s in symbols):
            raise ValueError("path2sym o sym2path is not idempotent on symbols.")

        px = reduce(
            lambda x,y: set(x) | set(y),
            (subpaths(self._s2p_fun(s)) for s in symbols)
        )
        if not all(p == self._p2p_fun(p) for p in px):
            for p in px:
                if p != self._p2p_fun(p):
                    print("error: sym2path(path2sym(%s) != %s" % (p,p))
            raise ValueError("sym2path o path2sym is not idempotent.")

        sx = { self._p2s_fun(p) for p in px }
        if not all(s == self._s2s_fun(s) for s in sx):
            for s in sx:
                if s != self._s2s_fun(s):
                    print("error: path2sym(sym2path(%s) != %s" % (s,s))
            raise ValueError("sym2path o path2sym is not idempotent.")

        return sx,px

    def _load(self, symbols):
        symbols_x,paths_x = self._extend_symbols(symbols)

        self._s2p = {s:self._s2p_fun(s) for s in symbols_x}
        self._p2s = {p:s for s,p in self._s2p.items()}

        # create tree
        self._tree = Tree()
        for p in self._p2s:
            self._tree.at(p)

        self._i2p = {i:p for i,p in enumerate(sorted(paths_x))}
        self._p2i = {p:i for i,p in self._i2p.items()}

        self.i2s = {i:self._p2s[p] for i,p in self._i2p.items()}
        self.s2i = {s:i for i,s in self.i2s.items()}

        # identify leaf nodes
        self.i_leaves = [i for i in self.i2s if self._is_leaf_node(i)]

        # create various connectivity matrices
        self.D = self._descendants()
        self.T = self._transfer_weights()

    def _is_leaf_node(self,i):
        return self._tree.at(self._i2p[i]).nchildren() == 0

    def _relationship_pair(self,i1,i2):
        p1, p2 = self._i2p[i1], self._i2p[i2]
        l1, l2 = len(p1), len(p2)
        two_descends_from_one = (l1 < l2) and (p1 == p2[:l1])
        one_descends_from_two = (l1 > l2) and (p1[:l2] == p2)

        return 1 if two_descends_from_one else -1 if one_descends_from_two else 0

    def _transfer_weight_pair(self, i1, i2):
        p1, p2 = self._i2p[i1], self._i2p[i2]
        l1, l2 = len(p1), len(p2)
        if (l1 < l2) and (p1 == p2[:l1]): # two descends from one
            return 1./reduce(
                mul,(self._tree.at(p2[:i]).nchildren() for i in range(l1,l2))
            )
        elif (l1 >= l2) and (p1[:l2] == p2): # one descends from two or equal
            return 1.
        else:
            return 0.

    def _descendants(self):
        a = np.fromiter((
            (i,j,True)
            for i in self.i2s
            for j in self.i2s
            if self._relationship_pair(i,j) > 0
        ), dtype=[("i","int"),("j","int"),("v","bool")])

        return sparse.csr_matrix((a["v"],(a["i"],a["j"])), shape=(len(self.i2s),len(self.i2s)))

    def _transfer_weights(self):
        a = np.fromiter((
            (i,j,self._transfer_weight_pair(i,j))
            for i in self.i2s
            for j in self.i2s
            if i==j or self._relationship_pair(i,j) != 0
        ), dtype=[("i","int"),("j","int"),("v","float")])

        return sparse.csr_matrix((a["v"],(a["i"],a["j"])), shape=(len(self.i2s),len(self.i2s)))


class HierarchicalSequenceEncoder:
    def __init__(self, symbols, sym2path_fun, path2sym_fun, **kwargs):
        self._hc = HierarchicalCode(symbols, sym2path_fun, path2sym_fun)
        self._g = tf.Graph()
        self._create_graph()
        self._session = tf.Session(graph=self._g, **kwargs)

    def close(self):
        self._session.close()

    def _create_graph(self):
        nnodes = len(self._hc.i2s)
        nleaves = len(self._hc.i_leaves)
        # XXX this should be re-implemented with sparse matrices once support
        # is added to TensorFlow
        with self._g.as_default():
            self._g_D = tf.constant(
                self._hc.D.todense(),
                dtype=tf.int32,
                shape=(nnodes,nnodes)
            )
            self._g_T_from_leaves = tf.constant(
                self._hc.T[self._hc.i_leaves,:].todense(),
                dtype=tf.float32,
                shape=(nleaves,nnodes)
            )
            self._g_T_to_leaves = tf.constant(
                self._hc.T[:,self._hc.i_leaves].todense(),
                dtype=tf.float32,
                shape=(nnodes,nleaves)
            )

            # XXX tf.sparse_placeholder() is buggy and can't convert type of
            # shape
            self._g_x = tf.sparse_placeholder(
                dtype=tf.int32
                #shape=np.array((nnodes,1),dtype=np.int64)
            )

            self._g_x0 = tf.sparse_tensor_to_dense(self._g_x)
            self._g_x1 = tf.matmul(self._g_D, self._g_x0,
                a_is_sparse=True, b_is_sparse=True)
            self._g_x2 = tf.logical_and(
                (self._g_x0 > 0), (self._g_x1 <= 0)
            )
            self._g_x3 = tf.matmul(
                tf.to_float(self._g_x2),
                self._g_T_to_leaves,
                transpose_a=True, a_is_sparse=True, b_is_sparse=True
            )
            self._g_x4 = tf.matmul(
                self._g_x3,
                self._g_T_from_leaves,
                a_is_sparse=True, b_is_sparse=True
            )
            self._g_y = tf.clip_by_value(self._g_x4, 0., 1.)

    def encode(self, seq):
        i_seq = sorted([(self._hc.s2i[s],0) for s in seq])
        v_seq = [True]*len(i_seq)
        shape=np.array((len(self._hc.i2s),1),dtype=np.int64)

        return self._session.run(self._g_y, feed_dict={self._g_x: tf.SparseTensorValue(i_seq,v_seq,shape)}).T
