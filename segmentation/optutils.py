import ctypes
import numpy as np
import platform

if platform.system() == 'Windows':
    optutils_lib = ctypes.CDLL('segmentation/optutils.dll')
else:
    optutils_lib = ctypes.CDLL('segmentation/optutils.so')

# samples ha dimensioni nsamples x nchannels
def compute_running_divergence(samples: np.array, window_size: int):

    nsamples, nchannels = samples.shape

    divergences = np.zeros(nsamples, dtype=np.float64)
    samples = np.array(samples, dtype=np.float64)

    types_in = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    types_out = None

    c_divergences = divergences.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_samples = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    optutils_lib.compute_running_divergence(c_samples, nsamples, nchannels, window_size, c_divergences)

    return divergences






