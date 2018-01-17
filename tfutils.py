import tensorflow as tf
from functools import wraps
import numpy as np

def tffunc(num_tensors, list_tensors=None):
    def tffunc_apply(func):
        @wraps(func)
        def newfunc(*args, output_collections=(), name=None, **kwargs):
            with tf.name_scope(name, func.__name__):
                if list_tensors:
                    tensors = [map_nlist(x, tf.convert_to_tensor) for x in args[:list_tensors]]
                    tensors += [tf.convert_to_tensor(x) for x in args[list_tensors:num_tensors]]
                else:
                    tensors = [tf.convert_to_tensor(x) for x in args[:num_tensors]]
                result = func(*tensors, *args[num_tensors:], **kwargs)
                tf.add_to_collection(output_collections, result)
                return result
        return newfunc
    return tffunc_apply

def tfmethod(num_tensors, list_tensors=None):
    def tffunc_apply(func):
        @wraps(func)
        def newfunc(self, *args, output_collections=(), name=None, **kwargs):
            with tf.name_scope(name, func.__name__):
                if list_tensors:
                    tensors = [map_nlist(x, tf.convert_to_tensor) for x in args[:list_tensors]]
                    tensors += [tf.convert_to_tensor(x) for x in args[list_tensors:num_tensors]]
                else:
                    tensors = [tf.convert_to_tensor(x) for x in args[:num_tensors]]
                result = func(self, *tensors, *args[num_tensors:], **kwargs)
                tf.add_to_collection(output_collections, result)
                return result
        return newfunc
    return tffunc_apply

class cdiagrank1:
    '''implements a matrix consisiting of a constant diagonal matrix and a rank 1
        gI + uv^T'''

    def __init__(self, g, u, v, dtype='float64'):
        self.g = g
        self.u = u
        self.v = v
        self.D = tf.size(self.u)

    def dot(self, A):
        return self.g*A + tf.matmul(self.u,
                                    tf.matmul(self.v, A, transpose_a=True))

    def dense(self):
        return self.g*tf.eye(self.D, dtype=self.dtype)+tf.matmul(self.u, self.v, transpose_b=True)

class Householder(cdiagrank1):
    '''Implements Householder reflection'''
    def __init__(self, v, dtype='float64'):
        self.g = 1.

        self.u = -2.*v
        self.v = v
        self.D = tf.size(v)
        self.dtype = dtype

class HouseholderChain:
    def __init__(self, V):
        self.V = V
        self.H = [Householder(v[:, None]) for v in tf.unstack(V)]

    def dot(self, A):
        for h in self.H:
            A = h.dot(A)
        return A

def householderproduct(V):
    C = tf.matmul(V, V, transpose_a=True)
    L = tf.cholesky(C)
    CiV = tf.cholesky(L, tf.transpose(V))
    return tf.eye(V.shape[0]) - 2.*tf.matmul(V, CiV)

class Givens:
    def __init__(self, v, coords, D, dtype='float64'):
        self.D = D
        self.v = v
        self.coords = coords
        self.dtype = dtype
        flip = tf.convert_to_tensor(([[0.,-1.],[1.,0.]]), dtype='float64')
        self.R = tf.concat([v,tf.matmul(flip, v)], axis=1) - tf.eye(2, dtype=self.dtype)
        self.P = tf.constant(np.eye(D, dtype=self.dtype)[coords].T)

    def dot(self, A):
        return A + tf.matmul(self.P, tf.matmul(self.R, tf.gather(A,self.coords)))

    def dense(self):
        return np.eye(self.D, dtype=self.dtype) + tf.matmul(tf.matmul(self.P, self.R), self.P, transpose_b=True)

class GivensChain:
    def __init__(self, V, coords, D, dtype='float64'):
        self.D = D
        self.V = V
        self.dtype = dtype
        if coords is None:
            self.coords = [[i,j] for j in range(D) for i in range(D)[-1::-1] if i>j]
            assert(V.shape[0] == len(self.coords))
        else:
            self.coords = coords
        self.G = [Givens(v[:, None], coord, D, dtype=self.dtype) for v, coord in zip(tf.unstack(V), self.coords)]

    def dot(self, A):
        for g in self.G:
            A = g.dot(A)
        return A





def map_nlist(nlist, fun):
    try:
        new_list=[]
        for i in range(len(nlist)):
            if isinstance(nlist[i],list):
                new_list += [map_nlist(nlist[i],fun)]
            else:
                new_list += [fun(nlist[i])]
        return new_list
    except TypeError:
        return fun(nlist)
