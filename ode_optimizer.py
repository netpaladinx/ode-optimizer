from __future__ import absolute_import, division, print_function

import tensorflow as tf

from odeint import odeint


def _compute_nested(nested_li, leaf_fn, **kwargs):
    nested0 = nested_li[0]
    if isinstance(nested0, dict):
        ret = dict()
        for k, v in nested0.iteritems():
            n_li = [v]
            for i, nested in enumerate(nested_li[1:]):
                assert isinstance(nested, dict), '`nested_li[%d]` should be a dict' % (i+1)
                assert k in nested, '"%s" should be a key of `nested_li[%d]`' % (k, (i+1))
                n_li.append(nested[k])
            ret[k] = _compute_nested(n_li, leaf_fn, **kwargs)
        return ret
    elif isinstance(nested0, (tuple, list)):
        ret = list()
        for i, v in enumerate(nested0):
            n_li = [v]
            for j, nested in enumerate(nested_li[1:]):
                assert isinstance(nested, (tuple, list)), '`nested_li[%d]` should be a tuple or list' % (j+1)
                assert i < len(nested), 'index %d should be less than the length of `nested_li[%d]` %d' % (i, (j+1), len(nested))
                n_li.append(nested[i])
            ret.append(_compute_nested(n_li, leaf_fn, **kwargs))
        return ret
    else:
        return leaf_fn(*nested_li, **kwargs)


def _get_solution(solution, i):
    return _compute_nested([solution], lambda x, i=i: x[i], i=i)


def _zeros_like(nested):
    return _compute_nested([nested], lambda x: tf.zeros_like(nested))


def _add_nested(nested1, nested2):
    return _compute_nested([nested1, nested2], lambda n1, n2: n1 + n2)


class ODEOptimizer(object):
    def __init__(self, learning_rate=0.001, optimizer='adam', atol=1e-6, rtol=1e-3):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.atol = atol
        self.rtol = rtol

    def compute_gradients(self, model, ts, batches, debug=False):
        mparams_0 = model.mparams
        dparams = model.dparams
        forward_fn = model.forward_fn
        gradient_fn = model.gradient_fn
        backward_fn = model.backward_fn

        # forward computation
        forward_solution = self._forward(forward_fn, mparams_0, ts, debug)
        mparams_N = _get_solution(forward_solution, -1)

        # backward computatoin
        mparams = mparams_N
        adjoint_mparams = _zeros_like(mparams)
        adjoint_t = tf.constant(0.)
        adjoint_dparams = _zeros_like(dparams)
        for i in range(ts.shape[0]-1, 0, -1):
            cur_t = ts[i]
            pre_t = ts[i-1]
            images, labels = batches[i]

            # compute loss gradients wrt `mparams` and `t` at t
            grads_mparams, grads_t = gradient_fn(mparams, images, labels)
            adjoint_mparams = _add_nested(adjoint_mparams, grads_mparams)
            adjoint_t = adjoint_t + grads_t

            # update augmented backward states
            augmented = (mparams, adjoint_mparams, adjoint_t, adjoint_dparams)
            backward_solution = self._backward(backward_fn, augmented, [cur_t, pre_t], debug)
            mparams, adjoint_mparams, adjoint_t, adjoint_dparams = _get_solution(backward_solution, -1)

        # compute loss gradients wrt `mparams` and `t` at t0
        images, labels = batches[0]
        grads_mparams, grads_t = gradient_fn(mparams, images, labels)
        adjoint_mparams = _add_nested(adjoint_mparams, grads_mparams)
        adjoint_t = adjoint_t + grads_t

        return adjoint_mparams, adjoint_t, adjoint_dparams

    def _forward(self, func, mparams_0, ts, debug):
        if debug:
            mparams_solution, info_dict = odeint(func, mparams_0, ts,
                                                 atol=self.atol, rtol=self.rtol, full_output=True)
        else:
            mparams_solution = odeint(func, mparams_0, ts,
                                      atol=self.atol, rtol=self.rtol, )
        return mparams_solution

    def _backward(self, func, augmented_0, ts, debug):
        if debug:
            augmented_solution, info_dict = odeint(func, augmented_0, ts,
                                                   atol=self.atol, rtol=self.rtol, full_output=True)
        else:
            augmented_solution = odeint(func, augmented_0, ts,
                                        atol=self.atol, rtol=self.rtol, )
        return augmented_solution
