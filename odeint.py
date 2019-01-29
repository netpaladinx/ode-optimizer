# Based on tf.contrib.integrate.odeint
"""ODE solvers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import linalg_ops

_ButcherTableau = collections.namedtuple('_ButcherTableau',
                                         'alpha beta c_sol c_mid c_error')

# Parameters from Shampine (1986), section 4.
_DORMAND_PRINCE_TABLEAU = _ButcherTableau(
    alpha=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.],
    beta=[
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    c_sol=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    c_mid=[
        6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
    ],
    c_error=[
        1951 / 21600 - 35 / 384,
        0,
        22642 / 50085 - 500 / 1113,
        451 / 720 - 125 / 192,
        -12231 / 42400 - -2187 / 6784,
        649 / 6300 - 11 / 84,
        1 / 60,
    ],)


def _get_nested_dtype(nested, dtype_fn=lambda v: v.dtype):
    if isinstance(nested, dict):
        for v in nested.itervalues():
            return _get_nested_dtype(v, dtype_fn=dtype_fn)
        else:
            return None
    elif isinstance(nested, (list, tuple)):
        for v in nested:
            return _get_nested_dtype(v, dtype_fn=dtype_fn)
        else:
            return None
    else:
        return dtype_fn(nested)


def _check_input_types(ys0, ts, dt=None):
    ys0_dtype = _get_nested_dtype(ys0)
    if not ys0_dtype.is_floating or ys0_dtype.is_complex:
        raise TypeError('`ys0` must have a floating point or complex floating point dtype')
    if not ts.dtype.is_floating:
        raise TypeError('`ts` must have a floating point dtype')
    if dt is not None and not dt.dtype.is_floating:
        raise TypeError('`dt` must have a floating point dtype')


def _assert_monotonicity(ts):
  assert_monotonicity = control_flow_ops.Assert(
      math_ops.logical_or(math_ops.reduce_all(ts[1:] >= ts[:-1]), math_ops.reduce_all(ts[1:] <= ts[:-1])),
      ['`t` must be monotonic increasing or decreasing'])
  return ops.control_dependencies([assert_monotonicity])


def _compute_nested(nested_li, leaf_fn, name=None, **kwargs):
    nested0 = nested_li[0]
    if isinstance(nested0, dict):
        ret = dict()
        for k, v in nested0.iteritems():
            n_li = [v]
            for i, nested in enumerate(nested_li[1:]):
                assert isinstance(nested, dict), '`nested_li[%d]` should be a dict' % (i+1)
                assert k in nested, '"%s" should be a key of `nested_li[%d]`' % (k, (i+1))
                n_li.append(nested[k])
            ret[k] = _compute_nested(n_li, leaf_fn,
                                     name='%s_%s' % (name, k) if name is not None else None, **kwargs)
        return ret
    elif isinstance(nested0, (tuple, list)):
        ret = list()
        for i, v in enumerate(nested0):
            n_li = [v]
            for j, nested in enumerate(nested_li[1:]):
                assert isinstance(nested, (tuple, list)), '`nested_li[%d]` should be a tuple or list' % (j+1)
                assert i < len(nested), 'index %d should be less than the length of `nested_li[%d]` %d' % (i, (j+1), len(nested))
                n_li.append(nested[i])
            ret.append(_compute_nested(n_li, leaf_fn,
                                       name='%s_%d' % (name, i) if name is not None else None, **kwargs))
        return ret
    else:
        args = nested_li + [name]
        return leaf_fn(*args, **kwargs)


def _compute_nested_and_flatten(nested_li, leaf_fn, name=None, **kwargs):
    ret = []
    nested0 = nested_li[0]
    if isinstance(nested0, dict):
        for k, v in nested0.iteritems():
            n_li = [v]
            for i, nested in enumerate(nested_li[1:]):
                assert isinstance(nested, dict), '`nested_li[%d]` should be a dict' % (i+1)
                assert k in nested, '"%s" should be a key of `nested_li[%d]`' % (k, (i+1))
                n_li.append(nested[k])
            ret += _compute_nested_and_flatten(n_li, leaf_fn,
                                               name='%s_%s' % (name, k) if name is not None else None, **kwargs)
    elif isinstance(nested0, (tuple, list)):
        for i, v in enumerate(nested0):
            n_li = [v]
            for j, nested in enumerate(nested_li[1:]):
                assert isinstance(nested, (tuple, list)), '`nested_li[%d]` should be a tuple or list' % (j+1)
                assert i < len(nested), 'index %d should be less than the length of `nested_li[%d]` %d' % (i, (j+1), len(nested))
                n_li.append(nested[i])
            ret += _compute_nested_and_flatten(n_li, leaf_fn,
                                               name='%s_%d' % (name, i) if name is not None else None, **kwargs)
    else:
        args = nested_li + [name]
        ret.append(leaf_fn(*args, **kwargs))
    return ret


def _traverse_and_do(nested, do_fn, name=None):
    _compute_nested([nested], do_fn, name=name)


def _traverse_and_return_nested(nested, return_fn, name=None):
    return _compute_nested([nested], return_fn, name=name)


def _traverse_and_return_flattened(nested, return_fn, name=None):
    return _compute_nested_and_flatten([nested], return_fn, name=name)


def _multi_traverse_and_return_nested(nested_li, return_fn, name=None):
    return _compute_nested(nested_li, return_fn, name=name)


def _multi_traverse_and_do(nested_li, do_fn, name=None):
    _compute_nested(nested_li, do_fn, name=name)


def _possibly_nonzero(x):
    return isinstance(x, ops.Tensor) or x != 0


def _scaled_dot_product(scale, xs, ys, name=None):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    with ops.name_scope(name, 'scaled_dot_product', [scale, xs, ys]) as scope:
        # Some of the parameters in our Butcher tableau include zeros. Using
        # _possibly_nonzero lets us avoid wasted computation.
        return math_ops.add_n(
            [(scale * x) * y for x, y in zip(xs, ys) if _possibly_nonzero(x) and _possibly_nonzero(y)], name=scope)


def _abs_square(x):
    if x.dtype.is_complex:
        return math_ops.square(math_ops.real(x)) + math_ops.square(math_ops.imag(x))
    else:
        return math_ops.square(x)


def _dot_product(xs, ys, name=None):
    """Calculate the vector inner product between two lists of Tensors."""
    with ops.name_scope(name, 'dot_product', [xs, ys]) as scope:
        try:
            return math_ops.add_n([x * y for x, y in zip(xs, ys)], name=scope)
        except ValueError, e:
            print(e)


def _ta_append(tensor_array, value):
    """Append a value to the end of a tf.TensorArray."""
    return tensor_array.write(tensor_array.size(), value)


def _norm(x):
    """Compute RMS norm."""
    return linalg_ops.norm(x) / math_ops.sqrt(math_ops.cast(array_ops.size(x), dtype=dtypes.float32))


def _select_initial_step(ts):
    return min(abs(ts[1:] - ts[:-1])) * 0.1


class _History(
    collections.namedtuple('_History', 'integrate_points, error_ratio')):
    """Saved integration history for use in `info_dict`."""


class _RungeKuttaState(
    collections.namedtuple('_RungeKuttaState', 'ys1, fs1, t0, t1, us1, dt, interp_coeff')):
    """Saved state of the Runge Kutta solver."""


def _runge_kutta_step(func,
                      ys0,
                      fs0,
                      t0,
                      us0,
                      dt,
                      tableau=_DORMAND_PRINCE_TABLEAU,
                      name=None):
    """Take an arbitrary Runge-Kutta step and estimate error."""
    with ops.name_scope(name, 'runge_kutta_step', [ys0, fs0, t0, us0, dt]) as scope:
        ys0 = _traverse_and_return_nested(ys0, lambda y, name_: ops.convert_to_tensor(y, name=name_), name='ys0')
        fs0 = _traverse_and_return_nested(fs0, lambda f, name_: ops.convert_to_tensor(f, name=name_), name='fs0')
        t0 = ops.convert_to_tensor(t0, name='t0')
        if us0 is not None:
            us0 = _traverse_and_return_nested(us0, lambda u, name_: ops.convert_to_tensor(u, name=name_), name='us0')
        dt = ops.convert_to_tensor(dt, name='dt')
        dt_cast = math_ops.cast(dt, _get_nested_dtype(ys0))

        ks = [fs0]
        for alpha_i, beta_i in zip(tableau.alpha, tableau.beta):
            ti = t0 + alpha_i * dt
            ysi = _multi_traverse_and_return_nested(
                [ys0] + ks, lambda y, *k: y + _scaled_dot_product(dt_cast, beta_i, k[:-1]))
            ks.append(func(ysi, ti, us0))

        if not (tableau.c_sol[-1] == 0 and tableau.c_sol[:-1] == tableau.beta[-1]):
            # This property (true for Dormand-Prince) lets us save a few FLOPs.
            ysi = _multi_traverse_and_return_nested(
                [ys0] + ks, lambda y, *k: y + _scaled_dot_product(dt_cast, tableau.c_sol, k[:-1]))

        ys1 = _traverse_and_return_nested(
            ysi, lambda y, name_: array_ops.identity(y, name=name_), name='%s/ys1' % scope)
        fs1 = _traverse_and_return_nested(
            ks[-1], lambda f, name_: array_ops.identity(f, name=name_), name='%s/fs1' % scope)
        ys1_error = _multi_traverse_and_return_nested(
            ks, lambda *k: _scaled_dot_product(dt_cast, tableau.c_error, k[:-1], name='%s/ys1_error' % scope))
        return ys1, fs1, ys1_error, ks


def _interp_fit(ys0, ys1, ys_mid, fs0, fs1, dt):
    """Fit coefficients for 4th order polynomial interpolation.
    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.
    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    # a, b, c, d, e = sympy.symbols('a b c d e')
    # x, dt, y0, y1, y_mid, f0, f1 = sympy.symbols('x dt y0 y1 y_mid f0 f1')
    # p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    # sympy.solve([p.subs(x, 0) - y0,
    #              p.subs(x, 1 / 2) - y_mid,
    #              p.subs(x, 1) - y1,
    #              (p.diff(x) / dt).subs(x, 0) - f0,
    #              (p.diff(x) / dt).subs(x, 1) - f1],
    #             [a, b, c, d, e])
    # {a: -2.0*dt*f0 + 2.0*dt*f1 - 8.0*y0 - 8.0*y1 + 16.0*y_mid,
    #  b: 5.0*dt*f0 - 3.0*dt*f1 + 18.0*y0 + 14.0*y1 - 32.0*y_mid,
    #  c: -4.0*dt*f0 + dt*f1 - 11.0*y0 - 5.0*y1 + 16.0*y_mid,
    #  d: dt*f0,
    #  e: y0}
    a = _multi_traverse_and_return_nested(
        [ys0, ys1, ys_mid, fs0, fs1],
        lambda y0, y1, y_mid, f0, f1, _: _dot_product([-2 * dt, 2 * dt, -8, -8, 16], [f0, f1, y0, y1, y_mid]))
    b = _multi_traverse_and_return_nested(
        [ys0, ys1, ys_mid, fs0, fs1],
        lambda y0, y1, y_mid, f0, f1, _: _dot_product([5 * dt, -3 * dt, 18, 14, -32], [f0, f1, y0, y1, y_mid]))
    c = _multi_traverse_and_return_nested(
        [ys0, ys1, ys_mid, fs0, fs1],
        lambda y0, y1, y_mid, f0, f1, _: _dot_product([-4 * dt, dt, -11, -5, 16], [f0, f1, y0, y1, y_mid]))
    d = _traverse_and_return_nested(fs0, lambda f, _: dt * f)
    e = ys0
    return [a, b, c, d, e]


def _interp_fit_rk(ys0, ys1, ks, dt, tableau=_DORMAND_PRINCE_TABLEAU):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    with ops.name_scope('interp_fit_rk'):
        dt = math_ops.cast(dt, _get_nested_dtype(ys0))
        ys_mid = _multi_traverse_and_return_nested(
            [ys0] + ks, lambda y, *k: y + _scaled_dot_product(dt, tableau.c_mid, k[:-1]))
        fs0 = ks[0]
        fs1 = ks[-1]
    return _interp_fit(ys0, ys1, ys_mid, fs0, fs1, dt)


def _optimal_step_size(last_step,
                       error_ratio,
                       safety=0.9,
                       ifactor=10.0,
                       dfactor=0.2,
                       order=5,
                       name=None):
    """Calculate the optimal size for the next Runge-Kutta step."""
    with ops.name_scope(name, 'optimal_step_size', [last_step, error_ratio]) as scope:
        error_ratio = math_ops.cast(error_ratio, last_step.dtype)
        exponent = math_ops.cast(1 / order, last_step.dtype)
        # this looks more complex than necessary, but importantly it keeps
        # error_ratio in the numerator so we can't divide by zero:
        factor = math_ops.maximum(1 / ifactor, math_ops.minimum(error_ratio**exponent / safety, 1 / dfactor))
        return math_ops.div(last_step, factor, name=scope)


def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.
    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar Tensor giving the start of the interval.
        t1: scalar Tensor giving the end of the interval.
        t: scalar Tensor giving the desired interpolation point.
    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """
    with ops.name_scope('interp_evaluate'):
        t0 = ops.convert_to_tensor(t0)
        t1 = ops.convert_to_tensor(t1)
        t = ops.convert_to_tensor(t)

    dtype = _get_nested_dtype(coefficients[0])

    assert_op = control_flow_ops.Assert((t0 <= t) & (t <= t1),
                                        ['invalid interpolation, fails `t0 <= t <= t1`:', t0, t, t1])
    with ops.control_dependencies([assert_op]):
        x = math_ops.cast((t - t0) / (t1 - t0), dtype)

    xs = [constant_op.constant(1, dtype), x]
    for _ in range(2, len(coefficients)):
        xs.append(xs[-1] * x)

    return _multi_traverse_and_return_nested(coefficients, lambda *coeff: _dot_product(coeff[:-1], reversed(xs)))


def _dopri5(func,
            ys0,
            ts,
            us,
            rtol,
            atol,
            full_output=False,
            first_step=None,
            safety=0.9,
            ifactor=10.0,
            dfactor=0.2,
            max_num_steps=1000,
            name=None):
    """Solve an ODE for `odeint` using method='dopri5'."""
    us0 = us if us is None else us[0]
    if first_step is None:
        first_step = _select_initial_step(ts)

    with ops.name_scope(name, 'dopri5', [
        ys0, ts, us, rtol, atol, safety, ifactor, dfactor, max_num_steps
    ]) as scope:
        first_step = ops.convert_to_tensor(first_step, dtype=ts.dtype, name='first_step')
        safety = ops.convert_to_tensor(safety, dtype=ts.dtype, name='safety')
        ifactor = ops.convert_to_tensor(ifactor, dtype=ts.dtype, name='ifactor')
        dfactor = ops.convert_to_tensor(dfactor, dtype=ts.dtype, name='dfactor')
        max_num_steps = ops.convert_to_tensor(max_num_steps, dtype=dtypes.int32, name='max_num_steps')

        with _assert_monotonicity(ts):
            num_times = array_ops.size(ts)
            first_step = control_flow_ops.cond(math_ops.reduce_all(ts[1:]>=ts[:-1]), lambda: first_step, lambda: -first_step)

        def adaptive_runge_kutta_step(rk_state, history, n_steps):
            """Take an adaptive Runge-Kutta step to integrate the ODE."""
            ys0, fs0, _, t0, us0, dt, interp_coeff = rk_state
            with ops.name_scope('assertions'):
                check_underflow = control_flow_ops.Assert(
                    (t0 + dt > t0 and first_step > 0) or (t0 + dt < t0 and first_step < 0), ['underflow in dt', dt])
                check_max_num_steps = control_flow_ops.Assert(n_steps < max_num_steps, ['max_num_steps exceeded'])
                check_numerics = _traverse_and_return_flattened(ys0, lambda y, _: control_flow_ops.Assert(
                    math_ops.reduce_all(math_ops.is_finite(abs(y))), ['non-finite values in state `y`', y]))
            with ops.control_dependencies([check_underflow, check_max_num_steps] + check_numerics):
                ys1, fs1, ys1_error, ks = _runge_kutta_step(func, ys0, fs0, t0, us0, dt)

            with ops.name_scope('error_ratio'):
                # We use the same approach as the dopri5 fortran code.
                error_tol = _multi_traverse_and_return_nested(
                    [ys0, ys1], lambda y0, y1, _: atol + rtol * math_ops.maximum(abs(y0), abs(y1)))
                tensor_error_ratio = _multi_traverse_and_return_nested(
                    [ys1_error, error_tol], lambda err, tol, _: _abs_square(err) / _abs_square(tol))
                # Could also use reduce_maximum here.
                error_ratio = math_ops.sqrt(math_ops.reduce_mean(_traverse_and_return_flattened(
                    tensor_error_ratio, lambda err, _: math_ops.reduce_mean(err))))
                accept_step = error_ratio <= 1

            with ops.name_scope('update/rk_state'):
                # If we don't accept the step, the _RungeKuttaState will be useless
                # (covering a time-interval of size 0), but that's OK, because in such
                # cases we always immediately take another Runge-Kutta step.
                ys_next = control_flow_ops.cond(accept_step, lambda: ys1, lambda: ys0)
                fs_next = control_flow_ops.cond(accept_step, lambda: fs1, lambda: fs0)
                ts_next = control_flow_ops.cond(accept_step, lambda: t0 + dt, lambda: t0)
                us_next = us0
                interp_coeff = control_flow_ops.cond(
                    accept_step, lambda: _interp_fit_rk(ys0, ys1, ks, dt), lambda: interp_coeff)
                dt_next = _optimal_step_size(dt, error_ratio, safety, ifactor, dfactor)
                rk_state = _RungeKuttaState(ys_next, fs_next, t0, ts_next, us_next, dt_next, interp_coeff)

            with ops.name_scope('update/history'):
                history = _History(
                    _ta_append(history.integrate_points, t0 + dt),
                    _ta_append(history.error_ratio, error_ratio))
            return rk_state, history, n_steps + 1

        def interpolate(solution, history, rk_state, i):
            """Interpolate through the next time point, integrating as necessary."""
            with ops.name_scope('interpolate'):
                us1 = None if us is None else (us[0] if len(us) == 1 else us[i])
                ys1, fs1, t0, t1, _, dt, interp_coeff = rk_state
                rk_state = _RungeKuttaState(ys1, fs1, t0, t1, us1, dt, interp_coeff)

                rk_state, history, _ = control_flow_ops.while_loop(
                    lambda rk_s, *_: (ts[i] > rk_s.t1 and first_step > 0) or (ts[i] < rk_s.t1 and first_step < 0),
                    adaptive_runge_kutta_step, (rk_state, history, 0),
                    name='integrate_loop')

                ys = _interp_evaluate(rk_state.interp_coeff, rk_state.t0, rk_state.t1, ts[i])
                solution = _multi_traverse_and_return_nested(
                    [solution, ys], lambda sol, y, _: sol.write(i, y))

                return solution, history, rk_state, i + 1

        solution = _traverse_and_return_nested(
            ys0, lambda y, _: tensor_array_ops.TensorArray(y.dtype, size=num_times).write(0, y))
        history = _History(
            integrate_points=tensor_array_ops.TensorArray(ts.dtype, size=0, dynamic_size=True),
            error_ratio=tensor_array_ops.TensorArray(rtol.dtype, size=0, dynamic_size=True))
        rk_state = _RungeKuttaState(ys0, func(ys0, ts[0], us0), ts[0], ts[0], us0, first_step, [ys0] * 5)

        solution, history, _, _ = control_flow_ops.while_loop(
            lambda _, __, ___, i: i < num_times, interpolate, (solution, history, rk_state, 1),
            name='interpolate_loop')

        ys = _traverse_and_return_nested(solution, lambda s, _: s.stack(name=scope))
        _multi_traverse_and_do([ys, ys0], lambda y, y0, _: y.set_shape(ts.get_shape().concatenate(y0.get_shape())))
        if not full_output:
            return ys
        else:
            integrate_points = history.integrate_points.stack()
            info_dict = {
                'num_func_evals': 6 * array_ops.size(integrate_points) + 1,
                'integrate_points': integrate_points,
                'error_ratio': history.error_ratio.stack()
            }
            return (ys, info_dict)


def odeint(func,  # func(ys0, ts[i], us[i] or us[0])
           ys0,
           ts,
           us=None,  # control variables, size(us)==size(ts) or size(us)=1
           rtol=1e-6,
           atol=1e-12,
           method=None,
           options=None,
           full_output=False,
           name=None):
    """Integrate a system of ordinary differential equations.
    Solves the initial value problem for a non-stiff system of first order ODEs:
    ```
    dy/dt = func(y, t), y(t[0]) = y0
    ```
    where y can be a Tensor, a list of Tensors, a dict or a nested data structure.

    Currently, implements 5th order Runge-Kutta with adaptive step size control
    and dense output, using the Dormand-Prince method. Similar to the 'dopri5'
    method of `scipy.integrate.ode` and MATLAB's `ode45`.

    Based on: Shampine, Lawrence F. (1986), "Some Practical Runge-Kutta Formulas",
    Mathematics of Computation, American Mathematical Society, 46 (173): 135-150,
    doi:10.2307/2008219.
    """

    if method is not None and method != 'dopri5':
        raise ValueError('invalid method: %r' % method)

    if options is None:
        options = {}
    elif method is None:
        raise ValueError('cannot supply `options` without specifying `method`')

    with ops.name_scope(name, 'odeint', [ys0, ts, us, rtol, atol]) as scope:
        ys0 = _traverse_and_return_nested(ys0, lambda y, name_: ops.convert_to_tensor(y, name=name_), name='ys0')
        ts = ops.convert_to_tensor(ts, preferred_dtype=dtypes.float32, name='ts')
        if us is not None:
            us = _traverse_and_return_nested(us, lambda u, name_: ops.convert_to_tensor(u, name=name_), name='us')
        _check_input_types(ys0, ts)

        error_dtype = _get_nested_dtype(ys0, dtype_fn=lambda v: abs(v).dtype)
        rtol = ops.convert_to_tensor(rtol, dtype=error_dtype, name='rtol')
        atol = ops.convert_to_tensor(atol, dtype=error_dtype, name='atol')

        return _dopri5(
            func,
            ys0,
            ts,
            us,
            rtol,
            atol,
            full_output=full_output,
            name=scope,
            **options)