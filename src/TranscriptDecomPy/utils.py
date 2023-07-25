import torch
import numpy as np
from dask import core
from dask.utils import ndimlist
from dask.array import core as da_core

###############################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################

def _concatenate3(arrays):

    NDARRAY_ARRAY_FUNCTION = getattr(np.ndarray, "__array_function__", None)

    arrays = da_core.concrete(arrays)
    if not arrays:
        return np.empty(0)

    advanced = max(
        core.flatten(arrays, container=(list, tuple)),
        key=lambda x: getattr(x, "__array_priority__", 0),
    )

    if not all(
        NDARRAY_ARRAY_FUNCTION
        is getattr(type(arr), "__array_function__", NDARRAY_ARRAY_FUNCTION)
        for arr in core.flatten(arrays, container=(list, tuple))
    ):
        try:
            x = da_core.unpack_singleton(arrays)
            return da_core._concatenate2(arrays, axes=tuple(range(x.ndim)))
        except TypeError:
            pass

    if da_core.concatenate_lookup.dispatch(type(advanced)) is not np.concatenate:
        x = da_core.unpack_singleton(arrays)
        return da_core._concatenate2(arrays, axes=list(range(x.ndim)))

    ndim = ndimlist(arrays)
    if not ndim:
        return arrays
    chunks = da_core.chunks_from_arrays(arrays)
    shape = tuple(map(sum, chunks))

    def dtype(x):
        try:
            return x.dtype
        except AttributeError:
            return type(x)

    arr_type = dtype(da_core.deepfirst(arrays))
    if arr_type == np.float64:
        ten_type = torch.float64
    elif arr_type == np.int64:
        ten_type = torch.int64
    else:
        raise NotImplemented
    
    result = torch.empty(shape, dtype=ten_type)

    for idx, arr in zip(
        da_core.slices_from_chunks(chunks), core.flatten(arrays, 
                                                    container=(list, tuple))
    ):
        if hasattr(arr, "ndim"):
            while arr.ndim < ndim:
                arr = arr[None, ...]
        result[idx] = arr.a

    return result

da_core.concatenate3 = _concatenate3

###############################################################################

import dask.array as da

###############################################################################

# utils

def general_determinant(A: torch.tensor, zero_dec: int) -> float:
    """Calculates a general determinant that applies to singular matrices, i.e.,
    matrices that are not of full-rank. The determinant here corresponds to the
    product of all the nonzero eigenvalues of the matrix.

        Here, we make use of the assumptions of A being a diagonal matrix, so
    that we can use a banded solver as these are much faster than general sparse
    solvers.

    Args:
        A (sparse.dia_matrix): real symmetric matrix.
        zero (float): let e be an eigenvalue, if 0 <= e < zero, then e is
            considered to be zero, otherwise e has its own value.


    Returns:
        log_det (float): logarithm of the general determinant of A.
    
    Requires:
        A should only have nonzero values in the main diagonal and the first
            offset diagonals of the matrix.
        A.data should be ordered such that the main diagonal is the middle 
            element of the array, and that the the top offset diagonal should be
            the first element of the array.
        zero >= 0
        zero should be a very small number.
    """

    evs = torch.linalg.eigvalsh(A)
    non_zero_evs = evs[torch.round(evs, decimals=zero_dec) != 0]
    log_det = torch.sum(torch.log(non_zero_evs))

    return log_det


def gen_tensor(value: float or list, shape: int or tuple=None) -> torch.tensor:
    """Creates a float64 tensor on the GPU.

    Args:
        value (float or list): value which we want to convert one instance or 
            multiple instance of into a tensor.
        shape (int, optional): in case the value is a single float that we would
            like to repeat to form a 1-D vector, additionally to the value, the
            length of the tensor should also be provided. Defaults to None.

    Returns:
        torch.tensor: tensor created.
    """

    if shape:
        ten_shape = shape if isinstance(shape, tuple) else (shape,)
        return torch.full(ten_shape, value, dtype=torch.float64, 
                                device=device)
    return torch.tensor(value, dtype=torch.float64, device=device)


###############################################################################

HANDLED_FUNCTIONS = {}

def implements(np_function):
    "Register an __array_function__ implementation for DaskTensor objects."
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.transpose)
def transpose(arr, **kwargs):
    if isinstance(arr, DaskTensor):
        if 'axes' in kwargs:
            return DaskTensor(arr.a.transpose(*kwargs['axes']))
        else:
            return DaskTensor(arr.a.t())
    else:
        raise NotImplementedError


@implements(np.ones_like)
def ones(arr, **kwargs):
    if isinstance(arr, DaskTensor):
        return DaskTensor(gen_tensor(1, kwargs['shape']))
    else:
        raise NotImplementedError


@implements(np.zeros_like)
def zeros(arr, **kwargs):
    if isinstance(arr, DaskTensor):
        return DaskTensor(gen_tensor(0, kwargs['shape']))
    else:
        raise NotImplementedError


@implements(np.empty)
def empty(**kwargs):
    return NotImplementedError


@implements(np.linalg.cholesky)
def cholesky(arr):
    if isinstance(arr, DaskTensor):
        return DaskTensor(torch.linalg.cholesky(arr.a))
    
    raise NotImplementedError


@implements(np.sum)
def cholesky(arr, **kwargs):
    if isinstance(arr, DaskTensor):
        if "dtype" in kwargs:
            if kwargs['dtype'] == np.float64:
                dtype = torch.float64
            elif kwargs['dtype'] == np.int64:
                dtype = torch.int64
            else:
                raise NotImplementedError
            return DaskTensor(torch.sum(arr.a, dtype=dtype))
        return DaskTensor(torch.sum(arr.a))
    
    raise NotImplementedError


@implements(np.concatenate)
def concatenate(arrs, **kwargs):
    if len(arrs) > 1:
        if "axis" in kwargs:
            new_arrs = []
            for curr_arr in arrs:
                while curr_arr.ndim < kwargs["axis"] + 1:
                    curr_arr = DaskTensor(curr_arr.a[None])
                new_arrs.append(curr_arr.a)
            return DaskTensor(torch.cat(new_arrs, kwargs['axis']))
        
        raise NotImplementedError
    else:
        arr1 = arrs[0]
        if "axis" in kwargs:
            while arr1.ndim < kwargs["axis"] + 1:
                arr1 = DaskTensor(arr1.a[None])
            return arr1
        raise NotImplementedError


@implements(np.diag)
def diag(arr, **kwargs):
    if isinstance(arr, DaskTensor):
        if 'k' in kwargs:
            return DaskTensor(torch.diag(arr.a, kwargs['k']))
        return DaskTensor(torch.diag(arr.a))
    elif isinstance(arr, list) and isinstance(arr[1], DaskTensor):
        if 'k' in kwargs:
            return DaskTensor(torch.diagarr[1].a, kwargs['k'])
        return DaskTensor(torch.diag(arr[1].a))
    raise NotImplementedError


class DaskTensor:
    """Wrapper for pytorch tensors. These allow pytorch tensors to be used as
    the backend for the chunks in dask.
    """
    def __init__(self, a):
        self.a = a

    #####################
    #  math operations  #
    #####################

    def __add__(self, b):
        if isinstance(b, DaskTensor):
            return DaskTensor(torch.add(self.a, b.a))
        return DaskTensor(torch.add(self.a, b))
    
    def __radd__(self, b):
        return DaskTensor(torch.add(b, self.a))

    def __sub__(self, b):
        if isinstance(b, DaskTensor):
            return DaskTensor(torch.sub(self.a, b.a))
        return DaskTensor(torch.sub(self.a, b))
    
    def __rsub__(self, b):
        return DaskTensor(torch.sub(b, self.a))

    def __mul__(self, b):
        if isinstance(b, DaskTensor):
            return DaskTensor(torch.mul(self.a, b.a))
        return  DaskTensor(torch.mul(self.a, b))

    def __rmul__(self, b):
        return DaskTensor(torch.mul(b, self.a))
    
    def __truediv__(self, b):
        if isinstance(b, DaskTensor):
            return DaskTensor(torch.div(self.a, b.a))
        return DaskTensor(torch.div(self.a, b))

    def __rtruediv__(self, b):
        return DaskTensor(torch.div(b, self.a))
    

    ######################
    #  array operations  #
    ######################

    def __repr__(self):
        return repr(self.a)
    
    def __array__(self, some):
        return DaskTensor(self.a)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            downcast_inputs = []
            for input in inputs:
                if isinstance(input, self.__class__):
                    downcast_inputs.append(input.a)
                elif isinstance(input, np.ndarray):
                    input = torch.tensor(input)
                    downcast_inputs.append(input)
                else:
                    return NotImplemented
            return self.__class__(ufunc(*downcast_inputs, **kwargs))
        else:
            return NotImplemented
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
    @property
    def T(self):
        """Calculates the transpose of the tensor."""
        return DaskTensor(self.a.T)

    @property
    def shape(self) -> tuple:
        return tuple(self.a.shape)
    
    @property
    def ndim(self) -> int:
        return self.a.ndim
    
    @property
    def dtype(self):
        if self.a.dtype == torch.int64:
            return np.array(0).dtype
        elif self.a.dtype == torch.float64:
            return np.array(0.23).dtype
        return self.a.dtype
    
    def reshape(self, shape):
        if shape == (0,0) or shape == (0,):
            return DaskTensor(self.a)
        raise NotImplementedError
    
    def __getitem__(self, key):
        return type(self)(self.a[key])
    
    def __setitem__(self, key, value):
        self.a[key] = value