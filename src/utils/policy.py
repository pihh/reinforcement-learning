import numpy as np 
import tensorflow.keras.backend as K

# Helpers
def gaussian_likelihood(log_std, lib="keras"): # for keras custom loss
    _exp = K.exp
    _log = K.log
    _sum = K.sum
    if lib == "numpy":
        _exp = np.exp
        _log = np.log
        _sum = np.sum

    def fn(actions,pred):
        pre_sum = -0.5 * (((actions-pred)/(_exp(log_std)+1e-8))**2 + 2*log_std + _log(2*np.pi))
        return _sum(pre_sum, axis=1)

    return fn