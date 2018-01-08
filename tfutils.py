import tensorflow as tf
from functools import wraps

def tffunc(num_tensors):
    def tffunc_apply(func):
        @wraps(func)
        def newfunc(*args, output_collections=(), name=None, **kwargs):
            with tf.name_scope(name, func.__name__):
                tensors = [map_nlist(x, tf.convert_to_tensor) for x in args[:num_tensors]]
                result = func(*tensors, *args[num_tensors:], **kwargs)
                tf.add_to_collection(output_collections, result)
                return result
        return newfunc
    return tffunc_apply

def tfmethod(num_tensors):
    def tffunc_apply(func):
        @wraps(func)
        def newfunc(self, *args, output_collections=(), name=None, **kwargs):
            with tf.name_scope(name, func.__name__):
                tensors = [map_nlist(x, tf.convert_to_tensor) for x in args[:num_tensors]]
                result = func(self, *tensors, *args[num_tensors:], **kwargs)
                tf.add_to_collection(output_collections, result)
                return result
        return newfunc
    return tffunc_apply

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
