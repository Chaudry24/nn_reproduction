import numpy as np
from scipy.special import gamma, kv
import autograd
import tensorflow as tf
import torch

# AUTOGRAD

v = 1.0
x = 10.0
h = 1e-3

gradient_autograd = autograd.grad(kv)

gradient_approx = (kv(v+h, x) - kv(v, x)) * 1/h

# gradient_autograd(v, h).astype(float)  # error


# TORCH

# v = torch.tensor(1.0, requires_grad=True)
# x = torch.tensor(10.0)
# h = torch.tensor(1e-3)
#
# obj_function = kv(v, x)
# obj_function.backward()
# torch.autograd.Function.


# TENSORFLOW

v = tf.Variable(1.0)
x = tf.Variable(10.0)
h = tf.Variable(1e-3)

with tf.GradientTape() as tape:
    obj_fnc = kv(v, x)

dobj_dv = tape.gradient(obj_fnc, v)
asd