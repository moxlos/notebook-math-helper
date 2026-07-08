"""
SymPy Visual Helper - Interactive Command Palette for Jupyter Notebooks

Usage:
    from sympy_helper import SympyHelper
    helper = SympyHelper()  # displays the tabbed UI

Install: pip install sympy ipywidgets matplotlib numpy
"""

import html
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ipywidgets as widgets
from IPython.display import display, Math, HTML, Image, clear_output

import sympy
from sympy import (
    symbols, Symbol, sympify, latex, pi, oo, I, E, S,
    sin, cos, tan, exp, log, ln, sqrt, Abs, floor, ceiling,
    asin, acos, atan, sinh, cosh, tanh,
    factorial, binomial,
    diff, integrate, limit, series, summation,
    solve, simplify, expand, factor, apart, cancel, trigsimp,
    Matrix, Function, Eq, dsolve, lambdify,
    factorint, isprime, gcd, lcm, totient, divisors,
    laplace_transform, inverse_laplace_transform,
    fourier_transform, inverse_fourier_transform,
    Heaviside, DiracDelta,
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor,
)

# Parsing transformations: allow ^ for power, implicit multiplication
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Pre-declare common symbols
x, y, z, t, n, k, a, b, c, s, r, u, v, w = symbols("x y z t n k a b c s r u v w")
_COMMON_LOCALS = {
    "x": x, "y": y, "z": z, "t": t, "n": n, "k": k,
    "a": a, "b": b, "c": c, "s": s, "r": r, "u": u, "v": v, "w": w,
    "pi": pi, "E": E, "e": E, "I": I, "oo": oo,
    "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log, "ln": ln,
    "sqrt": sqrt, "abs": Abs, "floor": floor, "ceil": ceiling,
    "asin": asin, "acos": acos, "atan": atan,
    "arcsin": asin, "arccos": acos, "arctan": atan,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "factorial": factorial, "binomial": binomial,
    "Heaviside": Heaviside, "DiracDelta": DiracDelta,
}

# Symbols declared by the user in the "Symbols" tab, with assumptions
# (positive, integer, ...). They take precedence over the plain built-in
# symbols in every expression input.
_USER_SYMBOLS = {}        # name -> Symbol with assumptions
_USER_SYMBOL_SPECS = {}   # name -> dict of assumption kwargs (for display)

# Session-wide computation log, shown in the History tab
_HISTORY = []             # dicts: n, time, kind, code, result
_HISTORY_LISTENERS = []   # callbacks to refresh UI when the log changes


def _log(kind, code, result=None):
    """Record a computation in the session history."""
    import datetime
    _HISTORY.append({
        "n": len(_HISTORY) + 1,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "kind": kind,
        # The UI accepts ^ for powers, but logged code must be valid Python
        "code": code.replace("^", "**"),
        "result": "" if result is None else str(result)[:300],
    })
    for cb in _HISTORY_LISTENERS:
        try:
            cb()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Layout helpers — always return NEW Layout instances to avoid shared state
# ---------------------------------------------------------------------------
def _wide():
    return widgets.Layout(width="95%")

def _med():
    return widgets.Layout(width="50%")

def _short():
    return widgets.Layout(width="220px")

def _out_layout():
    return widgets.Layout(border="1px solid #ccc", min_height="60px",
                          padding="8px", width="98%")

def _btn_layout():
    return widgets.Layout(width="auto", min_width="110px")

_STY = {"description_width": "initial"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(text, extra=None):
    """Parse user expression string into a SymPy expression."""
    loc = dict(_COMMON_LOCALS)
    # Pull in user-exported SymPy objects from the notebook namespace
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            for name, obj in ip.user_ns.items():
                if name not in loc and not name.startswith("_") and isinstance(obj, sympy.Basic):
                    loc[name] = obj
    except Exception:
        pass
    loc.update(_USER_SYMBOLS)
    if extra:
        loc.update(extra)
    text = text.strip()
    try:
        return parse_expr(text, local_dict=loc, transformations=_TRANSFORMS)
    except Exception:
        # Fallback parser has no convert_xor transform, so map ^ to ** here too
        return sympify(text.replace("^", "**"), locals=loc)


def _esc(s):
    """Escape user-supplied text before embedding it in HTML output."""
    return html.escape(str(s), quote=False)


def _code_details(code_str, label="Show SymPy code"):
    """Collapsible <details> block with escaped code."""
    return HTML(
        f'<details><summary style="cursor:pointer;color:#666">{label}</summary>'
        f'<pre style="background:#f4f4f4;padding:8px;margin-top:4px">'
        f'{_esc(code_str)}</pre></details>'
    )


def _plot_var(expr, default):
    """Return the single free symbol of *expr*, or *default* if constant."""
    free = expr.free_symbols
    if len(free) > 1:
        names = ", ".join(sorted(str(s) for s in free))
        raise ValueError(f"Expression has several variables ({names}); expected one.")
    return free.pop() if free else default


def _plot_vals(f_np, *grids):
    """Evaluate a lambdified function on grid(s); broadcast constants,
    mask complex values so matplotlib gets a real array."""
    vals = np.asarray(f_np(*grids))
    if np.iscomplexobj(vals):
        vals = np.where(np.abs(vals.imag) < 1e-12, vals.real, np.nan)
    return np.broadcast_to(vals.astype(float), grids[0].shape)


def _result_html(result, code_str, output_area, input_latex=None):
    """Show LaTeX-rendered result and collapsible SymPy code.

    If *input_latex* is given it is displayed first as
    ``input_latex = result_latex`` so the user can verify parsing.
    """
    _log("calc", code_str, result)
    with output_area:
        clear_output(wait=True)
        ltx = None
        try:
            ltx = latex(result)
            if input_latex:
                display(Math(input_latex + " = " + ltx))
            else:
                display(Math(ltx))
        except Exception:
            display(HTML(f"<pre>{_esc(result)}</pre>"))
        if ltx is not None:
            display(_code_details(ltx, label="Show LaTeX"))
        display(_code_details(code_str))


def _error(msg, output_area):
    with output_area:
        clear_output(wait=True)
        display(HTML(f'<div style="color:#c00;padding:4px"><b>Error:</b> {_esc(msg)}</div>'))


def _save_fig_and_display(fig):
    """Save a matplotlib figure to temp PNG, display it, clean up."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=tmp.name))
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


def _make_output():
    return widgets.Output(layout=_out_layout())


def _label(text):
    return widgets.HTML(f"<b style='font-size:13px'>{text}</b>")


# ===================================================================
# Computation core — plain functions, unit-testable without widgets
# ===================================================================

def integrate_trapezoid(f_np, a, b, n):
    """Composite trapezoidal rule. Returns (result, xs, ys)."""
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = _plot_vals(f_np, xs)
    result = h * (ys[0] / 2 + np.sum(ys[1:-1]) + ys[-1] / 2)
    return result, xs, ys


def integrate_simpson(f_np, a, b, n):
    """Composite Simpson rule (n rounded up to even).
    Returns (result, xs, ys, n)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = _plot_vals(f_np, xs)
    result = (h / 3) * (ys[0] + ys[-1]
                        + 4 * np.sum(ys[1:-1:2])
                        + 2 * np.sum(ys[2:-1:2]))
    return result, xs, ys, n


def _check_finite(val, i, t):
    if not np.isfinite(val):
        raise ValueError(
            f"Solution diverged at step {i} (t={t:.4g}). "
            f"Try smaller step size or different equation.")


def solve_ivp_euler(f_np, t0, y0, tf, steps):
    """Explicit Euler for dy/dt = f(t, y). Returns (ts, ys).
    Raises ValueError if the solution diverges."""
    h = (tf - t0) / steps
    ts = np.zeros(steps + 1); ys = np.zeros(steps + 1)
    ts[0], ys[0] = t0, y0
    for i in range(steps):
        _check_finite(ys[i], i, ts[i])
        ys[i + 1] = ys[i] + h * f_np(ts[i], ys[i])
        ts[i + 1] = ts[i] + h
    _check_finite(ys[-1], steps, ts[-1])
    return ts, ys


def solve_ivp_rk4(f_np, t0, y0, tf, steps):
    """Classic Runge-Kutta 4 for dy/dt = f(t, y). Returns (ts, ys).
    Raises ValueError if the solution diverges."""
    h = (tf - t0) / steps
    ts = np.zeros(steps + 1); ys = np.zeros(steps + 1)
    ts[0], ys[0] = t0, y0
    for i in range(steps):
        _check_finite(ys[i], i, ts[i])
        k1 = h * f_np(ts[i], ys[i])
        k2 = h * f_np(ts[i] + h / 2, ys[i] + k1 / 2)
        k3 = h * f_np(ts[i] + h / 2, ys[i] + k2 / 2)
        k4 = h * f_np(ts[i] + h, ys[i] + k3)
        ys[i + 1] = ys[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ts[i + 1] = ts[i] + h
    _check_finite(ys[-1], steps, ts[-1])
    return ts, ys


def solve_ivp_rk4_system(f_np, g_np, t0, x0, y0, tf, steps):
    """RK4 for the 2D system dx/dt = f(t,x,y), dy/dt = g(t,x,y).
    Returns (ts, xs, ys). Raises ValueError if the solution diverges."""
    h = (tf - t0) / steps
    ts = np.empty(steps + 1); xs = np.empty(steps + 1); ys = np.empty(steps + 1)
    ts[0], xs[0], ys[0] = t0, x0, y0
    for i in range(steps):
        if not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            raise ValueError(
                f"Solution diverged at step {i} (t={ts[i]:.4g}). "
                f"Try more steps or a shorter interval.")
        ti, xi, yi = ts[i], xs[i], ys[i]
        k1x = h * f_np(ti, xi, yi);          k1y = h * g_np(ti, xi, yi)
        k2x = h * f_np(ti + h/2, xi + k1x/2, yi + k1y/2)
        k2y = h * g_np(ti + h/2, xi + k1x/2, yi + k1y/2)
        k3x = h * f_np(ti + h/2, xi + k2x/2, yi + k2y/2)
        k3y = h * g_np(ti + h/2, xi + k2x/2, yi + k2y/2)
        k4x = h * f_np(ti + h, xi + k3x, yi + k3y)
        k4y = h * g_np(ti + h, xi + k3x, yi + k3y)
        xs[i+1] = xi + (k1x + 2*k2x + 2*k3x + k4x) / 6
        ys[i+1] = yi + (k1y + 2*k2y + 2*k3y + k4y) / 6
        ts[i+1] = ti + h
    return ts, xs, ys


def root_bisection(f_np, a, b, tol, maxiter):
    """Bisection root finding. Returns (root, iterations, converged).
    Raises ValueError if f(a) and f(b) have the same sign."""
    fa, fb = f_np(a), f_np(b)
    if fa * fb > 0:
        raise ValueError(
            f"f(a)={fa:.4g} and f(b)={fb:.4g} have the same sign. "
            f"Bisection requires opposite signs at endpoints.")
    iterations = []
    converged = False
    lo, hi = a, b
    for i in range(maxiter):
        mid = (lo + hi) / 2.0
        fm = f_np(mid)
        iterations.append({"i": i + 1, "a": lo, "b": hi,
                           "mid": mid, "f(mid)": fm})
        if abs(fm) < tol or (hi - lo) / 2 < tol:
            converged = True
            break
        if fa * fm < 0:
            hi = mid
        else:
            lo = mid
            fa = fm
    return (lo + hi) / 2.0, iterations, converged


def root_newton(f_np, df_np, x0, tol, maxiter):
    """Newton's method. Returns (root, iterations, converged).
    Raises ValueError on near-zero derivative or divergence."""
    iterations = []
    converged = False
    xn = x0
    for i in range(maxiter):
        fxn = f_np(xn)
        dfxn = df_np(xn)
        if abs(dfxn) < 1e-15:
            raise ValueError(
                f"Derivative is near zero at x={xn:.8g}. "
                f"Newton's method cannot continue.")
        xn1 = xn - fxn / dfxn
        iterations.append({"i": i + 1, "x_n": xn, "f(x_n)": fxn,
                           "f'(x_n)": dfxn, "x_{n+1}": xn1})
        if abs(xn1 - xn) < tol:
            xn = xn1
            converged = True
            break
        if not np.isfinite(xn1):
            raise ValueError(f"Newton's method diverged at iteration {i + 1}.")
        xn = xn1
    return xn, iterations, converged


def finite_difference(f_np, x0, h, method="Central Difference"):
    """Forward or central finite-difference derivative estimate."""
    if method == "Forward Difference":
        return (f_np(x0 + h) - f_np(x0)) / h
    return (f_np(x0 + h) - f_np(x0 - h)) / (2 * h)


def gradient(f, vs):
    """Gradient of scalar field f w.r.t. symbols vs, as a column Matrix."""
    return Matrix([diff(f, v_) for v_ in vs])


def divergence(F, vs):
    """Divergence of vector field F (list of exprs) w.r.t. symbols vs."""
    if len(F) != len(vs):
        raise ValueError(
            f"Field has {len(F)} components but {len(vs)} variables.")
    return sum(diff(Fi, v_) for Fi, v_ in zip(F, vs))


def curl(F, vs):
    """Curl: scalar for a 2-component field, Matrix for a 3-component one."""
    if len(F) == 2 and len(vs) >= 2:
        return diff(F[1], vs[0]) - diff(F[0], vs[1])
    if len(F) == 3 and len(vs) == 3:
        x1, x2, x3 = vs
        F1, F2, F3 = F
        return Matrix([
            diff(F3, x2) - diff(F2, x3),
            diff(F1, x3) - diff(F3, x1),
            diff(F2, x1) - diff(F1, x2),
        ])
    raise ValueError(
        "Curl needs a 2-component field (planar, any two variables) "
        "or a 3-component field with 3 variables.")


def laplacian(f, vs):
    """Laplacian of scalar field f."""
    return sum(diff(f, v_, 2) for v_ in vs)


def directional_derivative(f, vs, u_comps):
    """Derivative of f along (normalized) direction u_comps."""
    if len(u_comps) != len(vs):
        raise ValueError(
            f"Direction has {len(u_comps)} components but "
            f"{len(vs)} variables.")
    u_vec = Matrix(u_comps)
    norm = sqrt(sum(c ** 2 for c in u_comps))
    return simplify(gradient(f, vs).dot(u_vec) / norm)


def analyze_function(f, pv):
    """Full curve discussion of a single-variable function.

    Returns a dict with keys: fp, fpp, domain, y_intercept, roots, minima,
    maxima, saddle, inflection, vertical_asymptotes, asym_pos, asym_neg,
    oblique, notes. Every part is best-effort; failures land in notes.
    """
    fp = diff(f, pv)
    fpp = diff(fp, pv)
    res = {"fp": fp, "fpp": fpp, "domain": None, "y_intercept": None,
           "roots": [], "minima": [], "maxima": [], "saddle": [],
           "inflection": [], "vertical_asymptotes": [],
           "asym_pos": None, "asym_neg": None, "oblique": None, "notes": []}

    try:
        from sympy.calculus.util import continuous_domain
        res["domain"] = continuous_domain(f, pv, S.Reals)
    except Exception:
        pass

    try:
        y0 = f.subs(pv, 0)
        if y0.is_finite:
            res["y_intercept"] = y0
    except Exception:
        pass

    try:
        res["roots"] = solve(Eq(f, 0), pv)
    except Exception:
        res["notes"].append("Could not solve f = 0 symbolically.")

    try:
        crit = solve(Eq(fp, 0), pv)
        for p in crit:
            try:
                curv = fpp.subs(pv, p)
                if curv.is_positive:
                    res["minima"].append(p)
                elif curv.is_negative:
                    res["maxima"].append(p)
                else:
                    res["saddle"].append(p)
            except Exception:
                res["saddle"].append(p)
        if not crit:
            res["notes"].append("No critical points (f' has no zeros).")
    except Exception:
        res["notes"].append("Could not solve f' = 0 symbolically.")

    try:
        res["inflection"] = solve(Eq(fpp, 0), pv)
    except Exception:
        pass

    try:
        from sympy.calculus.util import singularities
        res["vertical_asymptotes"] = list(singularities(f, pv, S.Reals))
    except Exception:
        pass

    try:
        lim_p = limit(f, pv, oo)
        lim_m = limit(f, pv, -oo)
        if lim_p.is_finite:
            res["asym_pos"] = lim_p
        else:
            m = limit(f / pv, pv, oo)
            if m.is_finite and not m.is_zero:
                b0 = limit(f - m * pv, pv, oo)
                if b0.is_finite:
                    res["oblique"] = m * pv + b0
        if lim_m.is_finite and lim_m != lim_p:
            res["asym_neg"] = lim_m
    except Exception:
        pass

    return res


# ===================================================================
# Main class
# ===================================================================

class SympyHelper:
    """Interactive SymPy command palette with ipywidgets tabs."""

    def __init__(self):
        tab = widgets.Tab(layout=widgets.Layout(width="100%"))
        children = [
            self._calculus_tab(),
            self._vectorcalc_tab(),
            self._algebra_tab(),
            self._plotting_tab(),
            self._analysis_tab(),
            self._interactive_tab(),
            self._linalg_tab(),
            self._ode_tab(),
            self._odesystem_tab(),
            self._transforms_tab(),
            self._numerical_tab(),
            self._numtheory_tab(),
            self._freeinput_tab(),
            self._history_tab(),
            self._symbols_tab(),
            self._reference_tab(),
        ]
        titles = [
            "Calculus", "Vector Calculus", "Algebra", "Plotting",
            "Function Analysis", "Interactive Plot", "Linear Algebra",
            "Diff Equations", "ODE Systems", "Transforms", "Numerical Methods",
            "Number Theory", "Free Input", "History", "Symbols",
            "Quick Reference",
        ]
        tab.children = children
        for i, title in enumerate(titles):
            tab.set_title(i, title)
        display(tab)

    # ---------------------------------------------------------------
    # 1. Calculus
    # ---------------------------------------------------------------
    def _calculus_tab(self):
        op = widgets.ToggleButtons(
            options=["Derivative", "Integral", "Limit", "Taylor Series", "Summation"],
            button_style="info",
        )
        expr_in = widgets.Text(
            value="sin(x)*exp(x)",
            placeholder="e.g. sin(x)*exp(x), x^3 + 2*x",
            description="Expression:", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        var_in = widgets.Text(value="x", description="Variable:", layout=_med())
        params_in = widgets.Text(
            value="",
            placeholder="e.g. k, A, omega",
            description="Parameters:", style=_STY,
            layout=_med(), continuous_update=False,
        )
        params_hint = widgets.HTML(
            "<div style='color:#666;font-size:12px;padding-left:4px'>"
            "Extra symbolic constants beyond the built-in "
            "<code>a b c k n r s t u v w</code>."
            "</div>"
        )

        # -- Derivative --
        order_in = widgets.BoundedIntText(value=1, min=1, max=20,
                                          description="Order:", layout=_short())
        order_box = widgets.HBox([order_in])

        # -- Integral --
        def_chk = widgets.Checkbox(value=False, description="Definite integral")
        lo_in = widgets.Text(value="0", description="Lower:", layout=_short())
        hi_in = widgets.Text(value="1", description="Upper:", layout=_short())
        bounds_box = widgets.HBox([lo_in, hi_in])

        # -- Limit --
        lim_pt = widgets.Text(value="0", description="Point:", layout=_short())
        lim_dir = widgets.Dropdown(
            options=[("both sides", "+-"), ("from left", "-"), ("from right", "+")],
            description="Direction:", layout=_short(),
        )
        limit_box = widgets.HBox([lim_pt, lim_dir])

        # -- Taylor --
        tay_ctr = widgets.Text(value="0", description="Center:", layout=_short())
        tay_ord = widgets.BoundedIntText(value=5, min=1, max=30,
                                         description="Order:", layout=_short())
        taylor_box = widgets.HBox([tay_ctr, tay_ord])

        # -- Summation --
        sum_lo = widgets.Text(value="1", description="From k=", layout=_short())
        sum_hi = widgets.Text(value="oo", description="To k=", layout=_short())
        sum_box = widgets.HBox([sum_lo, sum_hi])

        def _toggle(change):
            val = change["new"]
            order_box.layout.display  = "flex" if val == "Derivative" else "none"
            def_chk.layout.display    = "flex" if val == "Integral" else "none"
            bounds_box.layout.display = "flex" if val == "Integral" and def_chk.value else "none"
            limit_box.layout.display  = "flex" if val == "Limit" else "none"
            taylor_box.layout.display = "flex" if val == "Taylor Series" else "none"
            sum_box.layout.display    = "flex" if val == "Summation" else "none"

        op.observe(_toggle, "value")
        def_chk.observe(lambda c: setattr(bounds_box.layout, "display",
                                          "flex" if c["new"] else "none"), "value")

        out = _make_output()
        btn = widgets.Button(description="Calculate", button_style="success", layout=_btn_layout())

        def _calc(b):
            try:
                sv = Symbol(var_in.value.strip())
                extra = {var_in.value.strip(): sv}
                for p in params_in.value.split(","):
                    p = p.strip()
                    if p:
                        extra[p] = Symbol(p)
                f = _parse(expr_in.value, extra)
                choice = op.value

                f_ltx = latex(f)
                v_ltx = latex(sv)
                in_ltx = None

                if choice == "Derivative":
                    nd = order_in.value
                    result = diff(f, sv, nd)
                    code = f"diff({expr_in.value}, {var_in.value}, {nd})"
                    if nd == 1:
                        in_ltx = r"\frac{d}{d" + v_ltx + r"}\left(" + f_ltx + r"\right)"
                    else:
                        in_ltx = (r"\frac{d^{" + str(nd) + r"}}{d" + v_ltx
                                  + r"^{" + str(nd) + r"}}\left(" + f_ltx + r"\right)")

                elif choice == "Integral":
                    if def_chk.value:
                        lo = _parse(lo_in.value)
                        hi = _parse(hi_in.value)
                        result = integrate(f, (sv, lo, hi))
                        code = f"integrate({expr_in.value}, ({var_in.value}, {lo_in.value}, {hi_in.value}))"
                        in_ltx = (r"\int_{" + latex(lo) + r"}^{" + latex(hi)
                                  + r"}" + f_ltx + r"\, d" + v_ltx)
                    else:
                        result = integrate(f, sv)
                        code = f"integrate({expr_in.value}, {var_in.value})"
                        in_ltx = r"\int " + f_ltx + r"\, d" + v_ltx

                elif choice == "Limit":
                    pt = _parse(lim_pt.value)
                    d = lim_dir.value
                    result = limit(f, sv, pt, d)
                    code = f"limit({expr_in.value}, {var_in.value}, {lim_pt.value}, '{d}')"
                    dir_str = ""
                    if d == "+":
                        dir_str = "^{+}"
                    elif d == "-":
                        dir_str = "^{-}"
                    in_ltx = (r"\lim_{" + v_ltx + r" \to " + latex(pt)
                              + dir_str + r"}" + f_ltx)

                elif choice == "Taylor Series":
                    ctr = _parse(tay_ctr.value)
                    nd = tay_ord.value
                    result = series(f, sv, ctr, nd + 1)
                    code = f"series({expr_in.value}, {var_in.value}, {tay_ctr.value}, {nd + 1})"
                    in_ltx = (f_ltx + r"\;\text{expanded around}\;" + v_ltx
                              + "=" + latex(ctr) + r"\;\text{to order}\;" + str(nd))

                elif choice == "Summation":
                    lo = _parse(sum_lo.value)
                    hi = _parse(sum_hi.value)
                    result = summation(f, (sv, lo, hi))
                    code = f"summation({expr_in.value}, ({var_in.value}, {sum_lo.value}, {sum_hi.value}))"
                    in_ltx = (r"\sum_{" + v_ltx + "=" + latex(lo)
                              + r"}^{" + latex(hi) + r"}" + f_ltx)

                _result_html(result, code, out, input_latex=in_ltx)
            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)
        _toggle({"new": op.value})

        return widgets.VBox([
            op, expr_in, var_in,
            params_in, params_hint,
            order_box, def_chk, bounds_box,
            limit_box, taylor_box, sum_box,
            btn, out,
        ])

    # ---------------------------------------------------------------
    # 1b. Vector Calculus
    # ---------------------------------------------------------------
    def _vectorcalc_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>Vector Calculus</b> — scalar fields are plain expressions "
            "(<code>x^2 + y^2</code>); vector fields are component lists "
            "(<code>[-y, x]</code> or <code>-y, x</code>)."
            "</div>"
        )
        op = widgets.ToggleButtons(
            options=["Gradient", "Divergence", "Curl", "Laplacian",
                     "Jacobian", "Hessian", "Directional Derivative"],
            button_style="info",
        )
        field_in = widgets.Text(
            value="x^2*y + y^3",
            description="f =", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        vars_in = widgets.Text(value="x, y", description="Variables:",
                               style=_STY, layout=_med(),
                               continuous_update=False)
        dir_in = widgets.Text(value="1, 1", description="Direction u:",
                              style=_STY, layout=_med(),
                              continuous_update=False)
        dir_in.layout.display = "none"

        plot_chk = widgets.Checkbox(
            value=True, description="Quiver plot (2D fields)",
        )
        rmin_w = widgets.FloatText(value=-5, description="Range min:",
                                   style=_STY, layout=_short())
        rmax_w = widgets.FloatText(value=5, description="Range max:",
                                   style=_STY, layout=_short())
        plot_box = widgets.HBox([plot_chk, rmin_w, rmax_w])

        _defaults = {
            "Gradient":               ("x^2*y + y^3",   "x, y", "f ="),
            "Divergence":             ("[-y, x]",       "x, y", "F ="),
            "Curl":                   ("[-y, x, 0]",    "x, y, z", "F ="),
            "Laplacian":              ("x^2 + y^2",     "x, y", "f ="),
            "Jacobian":               ("[x*y, x + y]",  "x, y", "F ="),
            "Hessian":                ("x^3 + x*y^2",   "x, y", "f ="),
            "Directional Derivative": ("x^2 + y^2",     "x, y", "f ="),
        }

        def _toggle(change):
            val = change["new"]
            expr, vs, desc = _defaults[val]
            field_in.value = expr
            vars_in.value = vs
            field_in.description = desc
            dir_in.layout.display = "flex" if val == "Directional Derivative" else "none"

        op.observe(_toggle, "value")

        out = _make_output()
        btn = widgets.Button(description="Calculate", button_style="success",
                             layout=_btn_layout())

        def _parse_syms(text):
            names = [p.strip() for p in text.split(",") if p.strip()]
            if not names:
                raise ValueError("Enter at least one variable.")
            return [Symbol(nm) for nm in names]

        def _parse_field(text, extra):
            """Parse a vector field: '[-y, x]' or '-y, x' -> list of exprs."""
            text = text.strip()
            if not text.startswith("["):
                text = "[" + text + "]"
            val = _parse(text, extra)
            if not isinstance(val, (list, tuple)):
                val = [val]
            return list(val)

        def _quiver(Fx, Fy, v1, v2, title):
            a, bb = float(rmin_w.value), float(rmax_w.value)
            g = np.linspace(a, bb, 20)
            X, Y = np.meshgrid(g, g)
            U = _plot_vals(lambdify((v1, v2), Fx, modules=["numpy"]), X, Y)
            V = _plot_vals(lambdify((v1, v2), Fy, modules=["numpy"]), X, Y)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.quiver(X, Y, U, V, color="tab:blue")
            ax.set_xlabel(str(v1))
            ax.set_ylabel(str(v2))
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            _save_fig_and_display(fig)

        def _calc(b):
            try:
                vs = _parse_syms(vars_in.value)
                extra = {str(v_): v_ for v_ in vs}
                choice = op.value
                fstr = field_in.value.strip()
                vstr = ", ".join(str(v_) for v_ in vs)
                quiver_field = None   # (Fx, Fy) to draw if 2D

                if choice == "Gradient":
                    f = _parse(fstr, extra)
                    result = gradient(f, vs)
                    code = f"Matrix([diff({fstr}, v) for v in ({vstr})])"
                    in_ltx = r"\nabla\left(" + latex(f) + r"\right)"
                    if len(vs) == 2:
                        quiver_field = (result[0], result[1])

                elif choice == "Divergence":
                    F = _parse_field(fstr, extra)
                    result = divergence(F, vs)
                    code = (f"sum(diff(Fi, v) for Fi, v in "
                            f"zip({fstr}, ({vstr})))")
                    in_ltx = (r"\nabla\cdot" + latex(Matrix(F)))
                    if len(F) == 2:
                        quiver_field = (F[0], F[1])

                elif choice == "Curl":
                    F = _parse_field(fstr, extra)
                    result = curl(F, vs)
                    if len(F) == 2:
                        code = (f"diff(F[1], {vs[0]}) - diff(F[0], {vs[1]})"
                                f"  # F = {fstr}")
                        quiver_field = (F[0], F[1])
                    else:
                        x1, x2, x3 = vs
                        code = (f"# curl of F = {fstr} over ({vstr})\n"
                                f"Matrix([diff(F3,{x2})-diff(F2,{x3}), "
                                f"diff(F1,{x3})-diff(F3,{x1}), "
                                f"diff(F2,{x1})-diff(F1,{x2})])")
                    in_ltx = (r"\nabla\times" + latex(Matrix(F)))

                elif choice == "Laplacian":
                    f = _parse(fstr, extra)
                    result = laplacian(f, vs)
                    code = f"sum(diff({fstr}, v, 2) for v in ({vstr}))"
                    in_ltx = r"\nabla^{2}\left(" + latex(f) + r"\right)"

                elif choice == "Jacobian":
                    F = _parse_field(fstr, extra)
                    result = Matrix(F).jacobian(Matrix(vs))
                    code = f"Matrix({fstr}).jacobian(Matrix([{vstr}]))"
                    in_ltx = (r"\mathbf{J}_{" + latex(Matrix(F)) + r"}")
                    if len(F) == 2 and len(vs) == 2:
                        quiver_field = (F[0], F[1])

                elif choice == "Hessian":
                    f = _parse(fstr, extra)
                    result = sympy.hessian(f, vs)
                    code = f"hessian({fstr}, ({vstr}))"
                    in_ltx = (r"\mathbf{H}\left(" + latex(f) + r"\right)")

                elif choice == "Directional Derivative":
                    f = _parse(fstr, extra)
                    u_comps = _parse_field(dir_in.value, extra)
                    result = directional_derivative(f, vs, u_comps)
                    code = (f"grad = Matrix([diff({fstr}, v) for v in ({vstr})])\n"
                            f"u = Matrix([{dir_in.value}])\n"
                            f"grad.dot(u) / u.norm()")
                    in_ltx = (r"D_{" + latex(Matrix(u_comps).T) + r"}\left("
                              + latex(f) + r"\right)")

                _result_html(result, code, out, input_latex=in_ltx)

                if plot_chk.value and quiver_field is not None and len(vs) >= 2:
                    with out:
                        _quiver(quiver_field[0], quiver_field[1],
                                vs[0], vs[1],
                                f"{choice}: field over ({vs[0]}, {vs[1]})")

            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([
            info, op, field_in, vars_in, dir_in,
            plot_box, btn, out,
        ])

    # ---------------------------------------------------------------
    # 2. Algebra
    # ---------------------------------------------------------------
    def _algebra_tab(self):
        op = widgets.ToggleButtons(
            options=["Solve", "Simplify", "Expand", "Factor",
                     "Partial Fractions", "Substitute"],
            button_style="info",
        )
        expr_in = widgets.Text(
            value="x**2 - 5*x + 6",
            placeholder="e.g. x**2 - 5*x + 6  or  x^2 - 5x + 6",
            description="Expression:", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        var_in = widgets.Text(value="x", description="Variable(s):", layout=_med())

        rhs_in = widgets.Text(value="0", description="= RHS:", layout=_short())
        solve_hint = widgets.HTML(
            "<small style='color:#888'>Systems: separate equations with <code>;</code> "
            "and list variables comma-separated. Use <code>**</code> or <code>^</code> for powers.</small>"
        )
        solve_box = widgets.VBox([rhs_in, solve_hint])

        sub_from = widgets.Text(value="x", description="Replace:", layout=_short())
        sub_to = widgets.Text(value="2", description="With:", layout=_short())
        sub_box = widgets.HBox([sub_from, sub_to])

        def _toggle(change):
            val = change["new"]
            solve_box.layout.display = "flex" if val == "Solve" else "none"
            sub_box.layout.display   = "flex" if val == "Substitute" else "none"

        op.observe(_toggle, "value")
        _toggle({"new": op.value})

        out = _make_output()
        btn = widgets.Button(description="Calculate", button_style="success", layout=_btn_layout())

        def _calc(b):
            try:
                f = _parse(expr_in.value)
                vname = var_in.value.strip()
                choice = op.value

                if choice == "Solve":
                    rhs = _parse(rhs_in.value)
                    vnames = [s.strip() for s in vname.split(",")]
                    vs = [Symbol(vn) for vn in vnames]

                    if len(vs) == 1:
                        eq = Eq(f, rhs)
                        result = solve(eq, vs[0])
                        code = f"solve(Eq({expr_in.value}, {rhs_in.value}), {vname})"
                    else:
                        eqn_strs = [s.strip() for s in expr_in.value.split(";")]
                        eqs = []
                        for es in eqn_strs:
                            if "==" in es:
                                lhs_s, rhs_s = es.split("==", 1)
                                eqs.append(Eq(_parse(lhs_s), _parse(rhs_s)))
                            else:
                                eqs.append(Eq(_parse(es), rhs))
                        result = solve(eqs, vs)
                        code = f"solve([{'; '.join(eqn_strs)}], [{vname}])"

                elif choice == "Simplify":
                    result = simplify(f)
                    code = f"simplify({expr_in.value})"

                elif choice == "Expand":
                    result = expand(f)
                    code = f"expand({expr_in.value})"

                elif choice == "Factor":
                    result = factor(f)
                    code = f"factor({expr_in.value})"

                elif choice == "Partial Fractions":
                    sv = Symbol(vname)
                    result = apart(f, sv)
                    code = f"apart({expr_in.value}, {vname})"

                elif choice == "Substitute":
                    sf = Symbol(sub_from.value.strip())
                    st = _parse(sub_to.value)
                    result = f.subs(sf, st)
                    code = f"({expr_in.value}).subs({sub_from.value}, {sub_to.value})"

                _result_html(result, code, out)
            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([op, expr_in, var_in, solve_box, sub_box, btn, out])

    # ---------------------------------------------------------------
    # 3. Plotting  (uses matplotlib + lambdify directly)
    # ---------------------------------------------------------------
    def _plotting_tab(self):
        plot_type = widgets.ToggleButtons(
            options=["2D Plot", "Parametric", "3D Surface", "Implicit"],
            button_style="info",
        )
        expr_in = widgets.Text(
            value="sin(x)",
            placeholder="e.g. sin(x), x^2 - 1 — separate several curves with ;",
            description="f(x) =", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        expr2_in = widgets.Text(
            value="cos(t)",
            placeholder="e.g. cos(t)",
            description="y(t) =", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        expr2_in.layout.display = "none"

        xmin_w = widgets.FloatText(value=-5, description="x min:", layout=_short())
        xmax_w = widgets.FloatText(value=5, description="x max:", layout=_short())
        ymin_w = widgets.FloatText(value=-5, description="y min:", layout=_short())
        ymax_w = widgets.FloatText(value=5, description="y max:", layout=_short())
        range1 = widgets.HBox([xmin_w, xmax_w])
        range2 = widgets.HBox([ymin_w, ymax_w])
        range2.layout.display = "none"

        color_in = widgets.Dropdown(
            options=["blue", "red", "green", "purple", "orange", "black"],
            description="Color:", layout=_short(),
        )
        grid_chk = widgets.Checkbox(value=True, description="Grid")
        title_in = widgets.Text(value="", description="Title:", layout=_short())
        appearance = widgets.HBox([color_in, grid_chk, title_in])

        def _toggle(change):
            val = change["new"]
            expr2_in.layout.display = "flex" if val == "Parametric" else "none"
            range2.layout.display   = "flex" if val in ("3D Surface", "Implicit") else "none"
            labels = {"2D Plot": "f(x) =", "Parametric": "x(t) =",
                      "3D Surface": "f(x,y) =", "Implicit": "F(x,y) ="}
            expr_in.description = labels.get(val, "f(x) =")

        plot_type.observe(_toggle, "value")

        out = _make_output()
        btn = widgets.Button(description="Plot", button_style="success", layout=_btn_layout())

        def _calc(b):
            try:
                choice = plot_type.value
                ttl = title_in.value.strip() or None
                clr = color_in.value
                grid = grid_chk.value

                with out:
                    clear_output(wait=True)

                    if choice == "2D Plot":
                        # Several curves may be given, separated by ;
                        expr_strs = [p.strip() for p in expr_in.value.split(";")
                                     if p.strip()]
                        if not expr_strs:
                            raise ValueError("Enter at least one expression.")
                        fs = [_parse(es) for es in expr_strs]
                        allfree = set().union(*(f.free_symbols for f in fs))
                        if len(allfree) > 1:
                            names = ", ".join(sorted(str(s_) for s_ in allfree))
                            raise ValueError(
                                f"Expressions have several variables ({names}); "
                                f"expected one shared variable.")
                        pv = allfree.pop() if allfree else x
                        xs = np.linspace(float(xmin_w.value), float(xmax_w.value), 500)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        for i, (es, f) in enumerate(zip(expr_strs, fs)):
                            f_np = lambdify(pv, f, modules=["numpy"])
                            ys = _plot_vals(f_np, xs)
                            # single curve keeps the chosen color; several
                            # curves use matplotlib's color cycle
                            kw = {"color": clr} if len(fs) == 1 else {"label": es}
                            ax.plot(xs, ys, **kw)
                        if len(fs) > 1:
                            ax.legend()
                        if grid: ax.grid(True, alpha=0.3)
                        if ttl: ax.set_title(ttl)
                        ax.set_xlabel(str(pv))
                        ax.set_ylabel("y")
                        code = (f"plot({', '.join(expr_strs)}, "
                                f"({pv}, {xmin_w.value}, {xmax_w.value}))")

                    elif choice == "Parametric":
                        fx = _parse(expr_in.value, {"t": t})
                        fy = _parse(expr2_in.value, {"t": t})
                        pv = _plot_var(fx + fy, t)
                        fx_np = lambdify(pv, fx, modules=["numpy"])
                        fy_np = lambdify(pv, fy, modules=["numpy"])
                        ts = np.linspace(float(xmin_w.value), float(xmax_w.value), 500)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(_plot_vals(fx_np, ts), _plot_vals(fy_np, ts), color=clr)
                        if grid: ax.grid(True, alpha=0.3)
                        if ttl: ax.set_title(ttl)
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_aspect("equal")
                        code = (f"plot_parametric({expr_in.value}, {expr2_in.value}, "
                                f"(t, {xmin_w.value}, {xmax_w.value}))")

                    elif choice == "3D Surface":
                        f = _parse(expr_in.value)
                        if not f.free_symbols <= {x, y}:
                            raise ValueError("Surface expression must use only x and y.")
                        f_np = lambdify((x, y), f, modules=["numpy"])
                        xs = np.linspace(float(xmin_w.value), float(xmax_w.value), 80)
                        ys = np.linspace(float(ymin_w.value), float(ymax_w.value), 80)
                        X, Y = np.meshgrid(xs, ys)
                        Z = _plot_vals(f_np, X, Y)

                        fig = plt.figure(figsize=(8, 6))
                        ax = fig.add_subplot(111, projection="3d")
                        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
                        if ttl: ax.set_title(ttl)
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_zlabel("z")
                        code = (f"plot3d({expr_in.value}, (x, {xmin_w.value}, {xmax_w.value}), "
                                f"(y, {ymin_w.value}, {ymax_w.value}))")

                    elif choice == "Implicit":
                        f = _parse(expr_in.value)
                        if not f.free_symbols <= {x, y}:
                            raise ValueError("Implicit expression must use only x and y.")
                        f_np = lambdify((x, y), f, modules=["numpy"])
                        xs = np.linspace(float(xmin_w.value), float(xmax_w.value), 300)
                        ys = np.linspace(float(ymin_w.value), float(ymax_w.value), 300)
                        X, Y = np.meshgrid(xs, ys)
                        Z = _plot_vals(f_np, X, Y)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.contour(X, Y, Z, levels=[0], colors=[clr])
                        if grid: ax.grid(True, alpha=0.3)
                        if ttl: ax.set_title(ttl)
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_aspect("equal")
                        code = (f"plot_implicit({expr_in.value}, "
                                f"(x, {xmin_w.value}, {xmax_w.value}), "
                                f"(y, {ymin_w.value}, {ymax_w.value}))")

                    _save_fig_and_display(fig)
                    _log("plot", code)
                    display(_code_details(code))

            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([
            plot_type, expr_in, expr2_in,
            range1, range2, appearance,
            btn, out,
        ])

    # ---------------------------------------------------------------
    # 3b. Function Analysis — full curve discussion of f(x)
    # ---------------------------------------------------------------
    def _analysis_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>Function Analysis</b> — enter a single-variable function; get "
            "roots, extrema, inflection points, asymptotes and an annotated plot."
            "</div>"
        )
        expr_in = widgets.Text(
            value="x**3 - 3*x + 1",
            placeholder="e.g. x^3 - 3x + 1, (x^2-1)/(x-2), exp(-x)*sin(x)",
            description="f(x) =", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        xmin_w = widgets.FloatText(value=-5, description="Plot x min:",
                                   style=_STY, layout=_short())
        xmax_w = widgets.FloatText(value=5, description="Plot x max:",
                                   style=_STY, layout=_short())
        range_box = widgets.HBox([xmin_w, xmax_w])
        plot_chk = widgets.Checkbox(value=True, description="Annotated plot")

        out = _make_output()
        btn = widgets.Button(description="Analyze", button_style="success",
                             layout=_btn_layout())

        def _real_pts(sols):
            """Keep solutions that are real numbers, as (sympy, float) pairs."""
            pts = []
            for p in sols:
                try:
                    if p.is_real:
                        pts.append((p, float(p.evalf())))
                except Exception:
                    continue
            return pts

        def _calc(b):
            try:
                f = _parse(expr_in.value)
                pv = _plot_var(f, x)
                res = analyze_function(f, pv)
                fp, fpp = res["fp"], res["fpp"]
                roots = res["roots"]
                minima, maxima, saddle = res["minima"], res["maxima"], res["saddle"]
                infl = res["inflection"]
                sing = res["vertical_asymptotes"]
                notes = res["notes"]

                lines = []       # (label, sympy object) for Math display
                if res["domain"] is not None:
                    lines.append((r"\text{Domain: }", res["domain"]))
                if res["y_intercept"] is not None:
                    lines.append((r"\text{y-intercept: }", res["y_intercept"]))
                if roots:
                    lines.append((r"\text{Roots: }", roots))
                if minima:
                    lines.append((r"\text{Local minima at: }", minima))
                if maxima:
                    lines.append((r"\text{Local maxima at: }", maxima))
                if saddle:
                    lines.append((r"\text{Other critical points: }", saddle))
                if infl:
                    lines.append((r"\text{Inflection candidates (f''=0): }", infl))
                if sing:
                    lines.append((r"\text{Vertical asymptotes at: }", sing))
                if res["asym_pos"] is not None:
                    lines.append((r"\text{Horizontal asymptote } x\to+\infty:\;",
                                  res["asym_pos"]))
                if res["oblique"] is not None:
                    lines.append((r"\text{Oblique asymptote } x\to+\infty:\;",
                                  res["oblique"]))
                if res["asym_neg"] is not None:
                    lines.append((r"\text{Horizontal asymptote } x\to-\infty:\;",
                                  res["asym_neg"]))

                code = (
                    f"from sympy import *\n"
                    f"{pv} = symbols('{pv}')\n"
                    f"f = {expr_in.value}\n"
                    f"solve(Eq(f, 0), {pv})            # roots\n"
                    f"solve(Eq(diff(f, {pv}), 0), {pv})     # critical points\n"
                    f"solve(Eq(diff(f, {pv}, 2), 0), {pv})  # inflection candidates\n"
                    f"limit(f, {pv}, oo), limit(f, {pv}, -oo)  # asymptotes"
                )

                with out:
                    clear_output(wait=True)
                    display(Math(r"f(" + latex(pv) + r") = " + latex(f)))
                    display(Math(r"f' = " + latex(fp) + r",\qquad f'' = " + latex(fpp)))
                    for lbl, val in lines:
                        display(Math(lbl + latex(val)))
                    for msg in notes:
                        display(HTML(f"<div style='color:#666'>{_esc(msg)}</div>"))

                    if plot_chk.value:
                        a, bb = float(xmin_w.value), float(xmax_w.value)
                        f_np = lambdify(pv, f, modules=["numpy"])
                        xs = np.linspace(a, bb, 800)
                        with np.errstate(all="ignore"):
                            ys = _plot_vals(f_np, xs).copy()
                        # Blank out jumps across vertical asymptotes
                        for p, pf in _real_pts(sing):
                            ys[np.abs(xs - pf) < (bb - a) / 200] = np.nan

                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(xs, ys, "b-", linewidth=2, label="f")
                        ax.axhline(0, color="k", linewidth=0.5)

                        def _mark(pts, style, lbl):
                            shown = False
                            for p, pf in _real_pts(pts):
                                if a <= pf <= bb:
                                    try:
                                        yv = float(f.subs(pv, p).evalf())
                                    except Exception:
                                        continue
                                    ax.plot(pf, yv, style, markersize=9,
                                            label=None if shown else lbl)
                                    shown = True

                        _mark(roots, "ko", "root")
                        _mark(minima, "g^", "min")
                        _mark(maxima, "rv", "max")
                        _mark(infl, "ms", "inflection")
                        for p, pf in _real_pts(sing):
                            if a <= pf <= bb:
                                ax.axvline(pf, color="orange", linestyle="--",
                                           linewidth=1)

                        ax.set_xlabel(str(pv))
                        ax.set_ylabel(f"f({pv})")
                        ax.grid(True, alpha=0.3)
                        if ax.get_legend_handles_labels()[0]:
                            ax.legend()
                        _save_fig_and_display(fig)

                    _log("analysis", code)
                    display(_code_details(code))

            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([info, expr_in, range_box, plot_chk, btn, out])

    # ---------------------------------------------------------------
    # 3c. Interactive Plot — auto-generated parameter sliders
    # ---------------------------------------------------------------
    def _interactive_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>Interactive Plot</b> — enter an expression with parameters "
            "(e.g. <code>a*sin(k*x) + b</code>). Each parameter gets a slider; "
            "the plot updates when you release a slider."
            "</div>"
        )
        expr_in = widgets.Text(
            value="a*sin(k*x)",
            placeholder="e.g. a*sin(k*x) + b, exp(-a*x)*cos(w*x)",
            description="f(x) =", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        xmin_w = widgets.FloatText(value=-5, description="x min:", layout=_short())
        xmax_w = widgets.FloatText(value=5, description="x max:", layout=_short())
        range_box = widgets.HBox([xmin_w, xmax_w])

        setup_btn = widgets.Button(description="Create sliders",
                                   button_style="success", layout=_btn_layout())
        sliders_box = widgets.VBox([])
        out = _make_output()

        state = {}  # f_np, plot var, param symbols, slider widgets

        def _redraw(change=None):
            if "f_np" not in state:
                return
            try:
                pvals = [sl.value for sl in state["sliders"]]
                a, bb = float(xmin_w.value), float(xmax_w.value)
                xs = np.linspace(a, bb, 500)
                with np.errstate(all="ignore"):
                    ys = _plot_vals(lambda g: state["f_np"](g, *pvals), xs)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(xs, ys, color="blue", linewidth=2)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(str(state["pv"]))
                ax.set_ylabel("f")
                if state["params"]:
                    vals = ", ".join(f"{p} = {v:g}" for p, v in
                                     zip(state["params"], pvals))
                    ax.set_title(vals)
                with out:
                    clear_output(wait=True)
                    _save_fig_and_display(fig)
            except Exception as e:
                _error(str(e), out)

        def _setup(b):
            try:
                f = _parse(expr_in.value)
                free = sorted(f.free_symbols, key=lambda s_: s_.name)
                if not free:
                    raise ValueError("Expression is constant — nothing to vary.")
                # Plot over x if present, otherwise the first symbol
                pv = next((s_ for s_ in free if s_.name == "x"), free[0])
                params = [s_ for s_ in free if s_ != pv]

                sliders = [
                    widgets.FloatSlider(
                        value=1.0, min=-5.0, max=5.0, step=0.1,
                        description=str(p), continuous_update=False,
                        layout=_med(),
                    )
                    for p in params
                ]
                for sl in sliders:
                    sl.observe(_redraw, "value")

                state.update({
                    "f_np": lambdify([pv] + params, f, modules=["numpy"]),
                    "pv": pv, "params": params, "sliders": sliders,
                })
                sliders_box.children = sliders if sliders else [widgets.HTML(
                    "<div style='color:#666'>No parameters found — plain plot.</div>"
                )]
                _redraw()
            except Exception as e:
                state.pop("f_np", None)
                sliders_box.children = []
                _error(str(e), out)

        setup_btn.on_click(_setup)
        xmin_w.observe(_redraw, "value")
        xmax_w.observe(_redraw, "value")

        return widgets.VBox([info, expr_in, range_box, setup_btn,
                             sliders_box, out])

    # ---------------------------------------------------------------
    # 4. Linear Algebra
    # ---------------------------------------------------------------
    def _linalg_tab(self):
        op = widgets.ToggleButtons(
            options=["Determinant", "Inverse", "Transpose", "Eigenvalues",
                     "Eigenvectors", "RREF", "Rank", "Nullspace", "Multiply"],
            button_style="info",
        )
        mat_in = widgets.Textarea(
            value="[[1,2],[3,4]]",
            placeholder="e.g. [[1,2],[3,4]]",
            description="Matrix A:", style=_STY,
            layout=widgets.Layout(width="95%", height="70px"),
        )
        mat2_in = widgets.Textarea(
            value="[[5,6],[7,8]]",
            placeholder="e.g. [[5,6],[7,8]]",
            description="Matrix B:", style=_STY,
            layout=widgets.Layout(width="95%", height="70px"),
        )
        mat2_in.layout.display = "none"

        def _toggle(change):
            mat2_in.layout.display = "flex" if change["new"] == "Multiply" else "none"
        op.observe(_toggle, "value")

        out = _make_output()
        btn = widgets.Button(description="Calculate", button_style="success", layout=_btn_layout())

        def _parse_matrix(text):
            import ast
            data = ast.literal_eval(text.strip())
            return Matrix(data)

        def _calc(b):
            try:
                A = _parse_matrix(mat_in.value)
                choice = op.value
                mv = mat_in.value.strip()

                if choice == "Determinant":
                    result = A.det()
                    code = f"Matrix({mv}).det()"
                elif choice == "Inverse":
                    result = A.inv()
                    code = f"Matrix({mv}).inv()"
                elif choice == "Transpose":
                    result = A.T
                    code = f"Matrix({mv}).T"
                elif choice == "Eigenvalues":
                    result = A.eigenvals()
                    code = f"Matrix({mv}).eigenvals()"
                elif choice == "Eigenvectors":
                    result = A.eigenvects()
                    code = f"Matrix({mv}).eigenvects()"
                elif choice == "RREF":
                    rref_mat, pivots = A.rref()
                    result = rref_mat
                    code = f"Matrix({mv}).rref()"
                elif choice == "Rank":
                    result = A.rank()
                    code = f"Matrix({mv}).rank()"
                elif choice == "Nullspace":
                    result = A.nullspace()
                    code = f"Matrix({mv}).nullspace()"
                elif choice == "Multiply":
                    B = _parse_matrix(mat2_in.value)
                    result = A * B
                    code = f"Matrix({mv}) * Matrix({mat2_in.value.strip()})"

                _result_html(result, code, out)
            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([op, mat_in, mat2_in, btn, out])

    # ---------------------------------------------------------------
    # 5. Differential Equations
    # ---------------------------------------------------------------
    def _ode_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>ODE Solver</b><br>"
            "Enter the ODE using <code>f(x)</code> notation. The equation is set = 0.<br>"
            "Examples:<br>"
            "<code>f(x).diff(x) + f(x) - sin(x)</code><br>"
            "<code>f(x).diff(x, 2) + f(x)</code><br>"
            "<code>f(x).diff(x) + k*f(x) - A*cos(omega*x)</code> "
            "(with parameters <code>k, A, omega</code>)"
            "</div>"
        )
        expr_in = widgets.Text(
            value="f(x).diff(x, 2) + f(x)",
            placeholder="e.g. f(x).diff(x, 2) + f(x)",
            description="ODE (= 0):", style=_STY,
            layout=_wide(), continuous_update=False,
        )
        func_in = widgets.Text(value="f", description="Function:", layout=_short())
        var_in_ode = widgets.Text(value="x", description="Variable:", layout=_short())
        params_in = widgets.Text(
            value="",
            placeholder="e.g. k, A, omega",
            description="Parameters:", style=_STY,
            layout=_med(), continuous_update=False,
        )
        funcvar_box = widgets.HBox([func_in, var_in_ode])
        params_hint = widgets.HTML(
            "<div style='color:#666;font-size:12px;padding-left:4px'>"
            "Extra symbolic constants beyond the built-in "
            "<code>a b c k n r s t u v w</code>."
            "</div>"
        )

        ic_chk = widgets.Checkbox(value=False, description="Initial / boundary conditions")
        ic_in = widgets.Text(
            value="f(0): 1, f'(0): 0",
            placeholder="f(0): 1, f'(0): 0  or  f(0): 1, f(1): 0",
            description="Conditions:", style=_STY, layout=_wide(),
        )
        ic_in.layout.display = "none"
        ic_hint = widgets.HTML(
            "<div style='color:#666;font-size:12px;padding-left:4px'>"
            "IVP: <code>f(0): 1, f'(0): 0</code> &nbsp;|&nbsp; "
            "BVP: <code>f(0): 1, f(1): 0</code>"
            "</div>"
        )
        ic_hint.layout.display = "none"

        export_chk = widgets.Checkbox(value=False, description="Export solution")
        export_name = widgets.Text(
            value="sol",
            placeholder="variable name",
            description="Name:", layout=_short(),
        )
        export_box = widgets.HBox([export_chk, export_name])

        plot_chk = widgets.Checkbox(value=False, description="Plot solution")
        plot_lo = widgets.FloatText(value=-5, description="x min:", layout=_short())
        plot_hi = widgets.FloatText(value=5, description="x max:", layout=_short())
        plot_range = widgets.HBox([plot_lo, plot_hi])
        plot_range.layout.display = "none"

        def _toggle_ic(change):
            vis = "flex" if change["new"] else "none"
            ic_in.layout.display = vis
            ic_hint.layout.display = "block" if change["new"] else "none"
        ic_chk.observe(_toggle_ic, "value")

        def _toggle_plot(change):
            plot_range.layout.display = "flex" if change["new"] else "none"
        plot_chk.observe(_toggle_plot, "value")

        out = _make_output()
        btn = widgets.Button(description="Solve ODE", button_style="success", layout=_btn_layout())

        def _calc(b):
            try:
                xname = var_in_ode.value.strip()
                fname = func_in.value.strip()
                xv = Symbol(xname)
                fv = Function(fname)

                loc = dict(_COMMON_LOCALS)
                loc.update(_USER_SYMBOLS)
                loc[fname] = fv
                loc[xname] = xv

                # Declare extra parameters
                param_names = []
                param_syms = []
                if params_in.value.strip():
                    for p in params_in.value.split(","):
                        p = p.strip()
                        if p and p not in loc:
                            ps = Symbol(p)
                            loc[p] = ps
                            param_names.append(p)
                            param_syms.append(ps)

                raw = expr_in.value.strip()
                ode_expr = sympify(raw, locals=loc)
                eq = Eq(ode_expr, 0)

                code = f"from sympy import *\n"
                code += f"{xname} = symbols('{xname}')\n"
                if param_names:
                    code += f"{', '.join(param_names)} = symbols('{' '.join(param_names)}')\n"
                code += f"{fname} = Function('{fname}')\n"

                # Parse initial conditions
                ics = None
                if ic_chk.value and ic_in.value.strip():
                    import re
                    ics = {}
                    for part in ic_in.value.split(","):
                        part = part.strip()
                        if ":" not in part:
                            continue
                        lhs, rhs = part.split(":", 1)
                        lhs = lhs.strip()
                        rhs_val = sympify(rhs.strip())
                        if "'" in lhs:
                            order = lhs.count("'")
                            m = re.search(r'\((.+?)\)', lhs)
                            pt = sympify(m.group(1)) if m else 0
                            ics[fv(xv).diff(xv, order).subs(xv, pt)] = rhs_val
                        else:
                            m = re.search(r'\((.+?)\)', lhs)
                            pt = sympify(m.group(1)) if m else 0
                            ics[fv(pt)] = rhs_val

                if ics:
                    result = dsolve(eq, fv(xv), ics=ics)
                    code += f"dsolve(Eq({raw}, 0), {fname}({xname}), ics=...)"
                else:
                    result = dsolve(eq, fv(xv))
                    code += f"dsolve(Eq({raw}, 0), {fname}({xname}))"

                # dsolve may return a list of solution branches
                first_sol = result[0] if isinstance(result, list) else result

                with out:
                    clear_output(wait=True)
                    display(Math(latex(result)))
                    display(_code_details(latex(result), label="Show LaTeX"))
                    if isinstance(result, list):
                        display(HTML(
                            "<div style='color:#666'>Multiple solutions found; "
                            "export/plot use the first one.</div>"
                        ))

                    if export_chk.value:
                        varname = export_name.value.strip() or "sol"
                        sol_expr = first_sol.rhs
                        from IPython import get_ipython
                        ip = get_ipython()
                        if ip is not None:
                            ip.user_ns[varname] = sol_expr
                            # also export the independent variable and params
                            ip.user_ns[xname] = xv
                            for pn, ps in zip(param_names, param_syms):
                                ip.user_ns[pn] = ps
                        display(HTML(
                            f"<div style='padding:4px;background:#e8f5e9;border-radius:4px'>"
                            f"Exported <code>{_esc(varname)}</code> = "
                            f"<code>{_esc(sol_expr)}</code> to notebook namespace."
                            f"</div>"
                        ))

                    if plot_chk.value:
                        try:
                            sol_expr = first_sol.rhs
                            # A general solution still contains integration
                            # constants (C1, ...) and parameters; set them to 1
                            # so the curve can be drawn.
                            consts = sorted(sol_expr.free_symbols - {xv},
                                            key=lambda s_: s_.name)
                            if consts:
                                sol_expr = sol_expr.subs({cs: 1 for cs in consts})
                                display(HTML(
                                    f"<div style='color:#666'>Plotting with "
                                    f"{_esc(', '.join(map(str, consts)))} = 1.</div>"
                                ))
                            f_np = lambdify(xv, sol_expr, modules=["numpy"])
                            xs = np.linspace(float(plot_lo.value), float(plot_hi.value), 500)
                            ys = _plot_vals(f_np, xs)

                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.plot(xs, ys, color="blue")
                            ax.grid(True, alpha=0.3)
                            ax.set_xlabel(xname)
                            ax.set_ylabel(fname)
                            _save_fig_and_display(fig)
                        except Exception as pe:
                            display(HTML(
                                f"<div style='color:orange'>Could not plot: {_esc(pe)}</div>"
                            ))

                    _log("ode", code, result)
                    display(_code_details(code))

            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([
            info, expr_in, funcvar_box,
            params_in, params_hint,
            ic_chk, ic_in, ic_hint,
            export_box, plot_chk, plot_range,
            btn, out,
        ])

    # ---------------------------------------------------------------
    # 5a. ODE Systems — coupled equations, symbolic and numerical
    # ---------------------------------------------------------------
    def _odesystem_tab(self):
        mode = widgets.ToggleButtons(
            options=["Numerical / Phase Plane", "Symbolic (dsolve)"],
            button_style="info",
        )

        # ---- Numerical mode widgets ----
        num_info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>2D system</b> &nbsp;dx/dt = f(t, x, y), &nbsp;"
            "dy/dt = g(t, x, y) — integrated with Runge-Kutta 4.<br>"
            "Autonomous systems (no <code>t</code>) also get a phase-plane "
            "streamplot with equilibria."
            "</div>"
        )
        fx_in = widgets.Text(
            value="y", description="dx/dt =", style=_STY, layout=_wide(),
            continuous_update=False,
            placeholder="e.g. y   (pendulum: y)",
        )
        gy_in = widgets.Text(
            value="-sin(x) - 0.2*y", description="dy/dt =", style=_STY,
            layout=_wide(), continuous_update=False,
            placeholder="e.g. -sin(x) - 0.2*y",
        )
        x0_in = widgets.FloatText(value=2.5, description="x(t0):", layout=_short())
        y0_in = widgets.FloatText(value=0.0, description="y(t0):", layout=_short())
        ic_box = widgets.HBox([x0_in, y0_in])
        t0_in = widgets.FloatText(value=0, description="t0:", layout=_short())
        tf_in = widgets.FloatText(value=30, description="t_final:", layout=_short())
        steps_in = widgets.BoundedIntText(value=1000, min=10, max=200000,
                                          description="Steps:", layout=_short())
        trange_box = widgets.HBox([t0_in, tf_in, steps_in])
        ts_chk = widgets.Checkbox(value=True, description="Time series plot")
        pp_chk = widgets.Checkbox(value=True, description="Phase plane plot")
        exp_chk = widgets.Checkbox(value=False, description="Export arrays")
        numopts_box = widgets.HBox([ts_chk, pp_chk, exp_chk])
        num_box = widgets.VBox([num_info, fx_in, gy_in, ic_box, trange_box,
                                numopts_box])

        # ---- Symbolic mode widgets ----
        sym_info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>Symbolic system</b> — one equation per line (or separated by "
            "<code>;</code>), each set = 0, using <code>x(t)</code>, "
            "<code>y(t)</code> notation:<br>"
            "<code>x(t).diff(t) - x(t) - y(t) ; y(t).diff(t) - 4*x(t) - y(t)</code>"
            "</div>"
        )
        sym_eqs_in = widgets.Textarea(
            value="x(t).diff(t) - x(t) - y(t); y(t).diff(t) - 4*x(t) - y(t)",
            description="Equations:", style=_STY,
            layout=widgets.Layout(width="95%", height="60px"),
        )
        sym_funcs_in = widgets.Text(value="x, y", description="Functions:",
                                    style=_STY, layout=_short())
        sym_var_in = widgets.Text(value="t", description="Variable:",
                                  layout=_short())
        sym_fv_box = widgets.HBox([sym_funcs_in, sym_var_in])
        sym_ic_chk = widgets.Checkbox(value=False, description="Initial conditions")
        sym_ic_in = widgets.Text(
            value="x(0): 1, y(0): 0",
            description="Conditions:", style=_STY, layout=_wide(),
        )
        sym_ic_in.layout.display = "none"
        sym_ic_chk.observe(
            lambda c: setattr(sym_ic_in.layout, "display",
                              "flex" if c["new"] else "none"), "value")
        sym_box = widgets.VBox([sym_info, sym_eqs_in, sym_fv_box,
                                sym_ic_chk, sym_ic_in])
        sym_box.layout.display = "none"

        def _toggle_mode(change):
            num = change["new"] == "Numerical / Phase Plane"
            num_box.layout.display = "flex" if num else "none"
            sym_box.layout.display = "none" if num else "flex"
        mode.observe(_toggle_mode, "value")

        out = _make_output()
        btn = widgets.Button(description="Solve", button_style="success",
                             layout=_btn_layout())

        def _calc_numeric():
            tv, xv, yv = Symbol("t"), Symbol("x"), Symbol("y")
            loc = {"t": tv, "x": xv, "y": yv}
            f_sym = _parse(fx_in.value, loc)
            g_sym = _parse(gy_in.value, loc)
            f_np = lambdify((tv, xv, yv), f_sym, modules=["numpy"])
            g_np = lambdify((tv, xv, yv), g_sym, modules=["numpy"])
            autonomous = tv not in (f_sym.free_symbols | g_sym.free_symbols)

            t0, tf = float(t0_in.value), float(tf_in.value)
            steps = int(steps_in.value)
            h = (tf - t0) / steps
            ts, xs, ys = solve_ivp_rk4_system(
                f_np, g_np, t0, float(x0_in.value), float(y0_in.value),
                tf, steps)

            # Equilibria (autonomous only, best effort)
            equil = []
            if autonomous:
                try:
                    sols = solve([Eq(f_sym, 0), Eq(g_sym, 0)], [xv, yv], dict=True)
                    for s_ in sols:
                        ex, ey = s_.get(xv), s_.get(yv)
                        if ex is not None and ey is not None \
                                and ex.is_real and ey.is_real:
                            equil.append((float(ex), float(ey)))
                except Exception:
                    pass

            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"t, x, y = symbols('t x y')\n"
                f"f = lambdify((t, x, y), {fx_in.value}, 'numpy')\n"
                f"g = lambdify((t, x, y), {gy_in.value}, 'numpy')\n"
                f"t0, tf, steps = {t0}, {tf}, {steps}\n"
                f"x0, y0 = {float(x0_in.value)}, {float(y0_in.value)}\n"
                f"h = (tf - t0) / steps\n"
                f"# RK4 on the state vector (x0, y0) — see helper source for the loop"
            )

            with out:
                clear_output(wait=True)
                display(HTML(
                    f"<h4>Runge-Kutta 4 — 2D system</h4>"
                    f"<table style='border-collapse:collapse'>"
                    f"<tr><td style='padding:2px 12px'><b>x({tf:g})</b></td>"
                    f"<td>{xs[-1]:.10g}</td></tr>"
                    f"<tr><td style='padding:2px 12px'><b>y({tf:g})</b></td>"
                    f"<td>{ys[-1]:.10g}</td></tr>"
                    f"<tr><td style='padding:2px 12px'>Steps</td><td>{steps}"
                    f"</td></tr></table>"
                ))
                if equil:
                    eq_str = ", ".join(f"({ex:g}, {ey:g})" for ex, ey in equil)
                    display(HTML(f"<div>Equilibria: <code>{_esc(eq_str)}</code></div>"))

                if ts_chk.value:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(ts, xs, "b-", label="x(t)")
                    ax.plot(ts, ys, "r-", label="y(t)")
                    ax.set_xlabel("t")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_title("Time series")
                    _save_fig_and_display(fig)

                if pp_chk.value:
                    fig, ax = plt.subplots(figsize=(7, 7))
                    if autonomous:
                        pad_x = 0.15 * (xs.max() - xs.min() or 1.0)
                        pad_y = 0.15 * (ys.max() - ys.min() or 1.0)
                        gx = np.linspace(xs.min() - pad_x, xs.max() + pad_x, 25)
                        gy_ = np.linspace(ys.min() - pad_y, ys.max() + pad_y, 25)
                        X, Y = np.meshgrid(gx, gy_)
                        with np.errstate(all="ignore"):
                            U = _plot_vals(lambda a_, b_: f_np(0.0, a_, b_), X, Y)
                            V = _plot_vals(lambda a_, b_: g_np(0.0, a_, b_), X, Y)
                        ax.streamplot(X, Y, U, V, color="0.75", density=1.1,
                                      linewidth=0.8)
                    ax.plot(xs, ys, "b-", linewidth=1.8, label="trajectory")
                    ax.plot(xs[0], ys[0], "go", markersize=9, label="start")
                    ax.plot(xs[-1], ys[-1], "rs", markersize=8, label="end")
                    for ex, ey in equil:
                        ax.plot(ex, ey, "k*", markersize=13)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_title("Phase plane" +
                                 ("" if autonomous else " (non-autonomous: no field)"))
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    _save_fig_and_display(fig)

                if exp_chk.value:
                    self._export_arrays({"ts": ts, "xs": xs, "ys": ys})

                _log("ode-system", code, (xs[-1], ys[-1]))
                display(_code_details(code, label="Show NumPy code"))

        def _calc_symbolic():
            import re
            tname = sym_var_in.value.strip() or "t"
            tv = Symbol(tname)
            fnames = [p.strip() for p in sym_funcs_in.value.split(",") if p.strip()]
            if len(fnames) < 2:
                raise ValueError("List at least two function names, e.g. x, y")
            funcs = {nm: Function(nm) for nm in fnames}

            loc = dict(_COMMON_LOCALS)
            loc.update(_USER_SYMBOLS)
            loc[tname] = tv
            loc.update(funcs)

            raw = sym_eqs_in.value.replace("\n", ";")
            eq_strs = [p.strip() for p in raw.split(";") if p.strip()]
            if len(eq_strs) != len(fnames):
                raise ValueError(
                    f"{len(eq_strs)} equations for {len(fnames)} functions — "
                    f"they must match.")
            eqs = [Eq(sympify(es, locals=loc), 0) for es in eq_strs]
            unknowns = [funcs[nm](tv) for nm in fnames]

            ics = None
            if sym_ic_chk.value and sym_ic_in.value.strip():
                ics = {}
                for part in sym_ic_in.value.split(","):
                    part = part.strip()
                    if ":" not in part:
                        continue
                    lhs, rhs = part.split(":", 1)
                    lhs = lhs.strip()
                    m = re.match(r"(\w+)\((.+?)\)", lhs)
                    if not m or m.group(1) not in funcs:
                        raise ValueError(f"Cannot parse condition '{part}'. "
                                         f"Use e.g. x(0): 1")
                    pt = sympify(m.group(2))
                    ics[funcs[m.group(1)](pt)] = sympify(rhs.strip())

            if ics:
                result = dsolve(eqs, unknowns, ics=ics)
            else:
                result = dsolve(eqs, unknowns)

            code = (
                f"from sympy import *\n"
                f"{tname} = symbols('{tname}')\n"
                f"{', '.join(fnames)} = symbols('{' '.join(fnames)}', cls=Function)\n"
                f"eqs = [Eq({', 0), Eq('.join(eq_strs)}, 0)]\n"
                f"dsolve(eqs, [{', '.join(nm + '(' + tname + ')' for nm in fnames)}]"
                + (", ics=...)" if ics else ")")
            )

            with out:
                clear_output(wait=True)
                sols = result if isinstance(result, list) else [result]
                for s_ in sols:
                    display(Math(latex(s_)))
                display(_code_details(latex(sols if len(sols) > 1 else sols[0]),
                                      label="Show LaTeX"))
                _log("ode-system", code, result)
                display(_code_details(code))

        def _calc(b):
            try:
                if mode.value == "Numerical / Phase Plane":
                    _calc_numeric()
                else:
                    _calc_symbolic()
            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([mode, num_box, sym_box, btn, out])

    # ---------------------------------------------------------------
    # 5b. Transforms — Laplace and Fourier
    # ---------------------------------------------------------------
    def _transforms_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>Integral Transforms</b> — Laplace and Fourier transforms and "
            "their inverses. <code>Heaviside(t)</code> and "
            "<code>DiracDelta(t)</code> are available in expressions."
            "</div>"
        )
        op = widgets.ToggleButtons(
            options=["Laplace", "Inverse Laplace", "Fourier", "Inverse Fourier"],
            button_style="info",
        )
        expr_in = widgets.Text(
            value="exp(-2*t)*sin(3*t)",
            placeholder="e.g. exp(-2*t)*sin(3*t), Heaviside(t-1)",
            description="f =", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        from_in = widgets.Text(value="t", description="From var:",
                               style=_STY, layout=_short())
        to_in = widgets.Text(value="s", description="To var:",
                             style=_STY, layout=_short())
        vars_box = widgets.HBox([from_in, to_in])
        params_in = widgets.Text(
            value="",
            placeholder="e.g. a, omega",
            description="Parameters:", style=_STY,
            layout=_med(), continuous_update=False,
        )
        params_hint = widgets.HTML(
            "<div style='color:#666;font-size:12px;padding-left:4px'>"
            "Extra symbolic constants; they are assumed <i>positive</i> here "
            "so the transforms can evaluate (e.g. <code>a</code> in "
            "<code>exp(-a*t)</code>)."
            "</div>"
        )

        _defaults = {
            "Laplace":         ("exp(-2*t)*sin(3*t)", "t", "s"),
            "Inverse Laplace": ("1/(s**2 + 1)",       "s", "t"),
            "Fourier":         ("exp(-x**2)",         "x", "k"),
            "Inverse Fourier": ("exp(-pi*k**2)",      "k", "x"),
        }

        def _toggle(change):
            expr, fv, tv = _defaults[change["new"]]
            expr_in.value = expr
            from_in.value = fv
            to_in.value = tv

        op.observe(_toggle, "value")

        out = _make_output()
        btn = widgets.Button(description="Transform", button_style="success",
                             layout=_btn_layout())

        _OPS = {
            "Laplace":         (r"\mathcal{L}", "laplace_transform",
                                lambda f, a, b: laplace_transform(f, a, b, noconds=True)),
            "Inverse Laplace": (r"\mathcal{L}^{-1}", "inverse_laplace_transform",
                                inverse_laplace_transform),
            "Fourier":         (r"\mathcal{F}", "fourier_transform",
                                fourier_transform),
            "Inverse Fourier": (r"\mathcal{F}^{-1}", "inverse_fourier_transform",
                                inverse_fourier_transform),
        }

        def _calc(b):
            try:
                fname = from_in.value.strip()
                tname = to_in.value.strip()
                if not fname or not tname:
                    raise ValueError("Both variable fields are required.")
                fv = Symbol(fname, positive=True) if op.value == "Laplace" \
                    else Symbol(fname)
                tv = Symbol(tname)

                extra = {fname: fv, tname: tv}
                for p in params_in.value.split(","):
                    p = p.strip()
                    if p:
                        extra[p] = Symbol(p, positive=True)

                f = _parse(expr_in.value, extra)
                sym_ltx, fn_name, fn = _OPS[op.value]

                result = fn(f, fv, tv)
                extra_arg = ", noconds=True" if op.value == "Laplace" else ""
                code = (
                    f"from sympy import *\n"
                    f"{fname}, {tname} = symbols('{fname} {tname}')\n"
                    f"{fn_name}({expr_in.value}, {fname}, {tname}{extra_arg})"
                )
                in_ltx = (sym_ltx + r"\left\{" + latex(f) + r"\right\}\!("
                          + latex(tv) + r")")
                _result_html(result, code, out, input_latex=in_ltx)
            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([
            info, op, expr_in, vars_box,
            params_in, params_hint,
            btn, out,
        ])

    # ---------------------------------------------------------------
    # 6. Numerical Methods
    # ---------------------------------------------------------------
    def _numerical_tab(self):
        # -- Category / method selectors --
        category = widgets.ToggleButtons(
            options=["Integration", "ODE Solver", "Root Finding", "Differentiation"],
            button_style="info",
        )
        method_dd = widgets.Dropdown(
            options=["Trapezoidal Rule", "Simpson's Rule"],
            description="Method:", style=_STY, layout=_med(),
        )

        _methods_map = {
            "Integration":      ["Trapezoidal Rule", "Simpson's Rule"],
            "ODE Solver":       ["Euler's Method", "Runge-Kutta 4"],
            "Root Finding":     ["Bisection", "Newton's Method"],
            "Differentiation":  ["Forward Difference", "Central Difference"],
        }

        # -- Shared inputs --
        expr_in = widgets.Text(
            value="sin(x)", placeholder="e.g. sin(x), x^2 - 1",
            description="f(x) =", style=_STY, layout=_wide(),
            continuous_update=False,
        )
        var_in = widgets.Text(value="x", description="Variable:", layout=_short())

        # -- Integration parameters --
        int_a = widgets.FloatText(value=0, description="a:", layout=_short())
        int_b = widgets.FloatText(value=3.14159, description="b:", layout=_short())
        int_n = widgets.BoundedIntText(value=100, min=2, max=100000,
                                       description="n (panels):", style=_STY,
                                       layout=_short())
        int_box = widgets.HBox([int_a, int_b, int_n])

        # -- ODE parameters --
        ode_hint = widgets.HTML(
            "<small style='color:#888'>Enter dy/dt = f(t, y). Use <code>t</code> and "
            "<code>y</code> as variables.</small>"
        )
        ode_t0 = widgets.FloatText(value=0, description="t0:", layout=_short())
        ode_y0 = widgets.FloatText(value=1, description="y0:", layout=_short())
        ode_tf = widgets.FloatText(value=5, description="t_final:", layout=_short())
        ode_steps = widgets.BoundedIntText(value=50, min=2, max=100000,
                                           description="Steps:", layout=_short())
        ode_box1 = widgets.HBox([ode_t0, ode_y0])
        ode_box2 = widgets.HBox([ode_tf, ode_steps])

        # -- Root-finding parameters --
        rf_a = widgets.FloatText(value=0, description="a:", layout=_short())
        rf_b = widgets.FloatText(value=2, description="b:", layout=_short())
        rf_x0 = widgets.FloatText(value=1.5, description="x0 (Newton):",
                                  style=_STY, layout=_short())
        rf_tol = widgets.FloatText(value=1e-8, description="Tolerance:",
                                   style=_STY, layout=_short())
        rf_maxiter = widgets.BoundedIntText(value=50, min=1, max=1000,
                                            description="Max iter:", style=_STY,
                                            layout=_short())
        rf_box1 = widgets.HBox([rf_a, rf_b, rf_x0])
        rf_box2 = widgets.HBox([rf_tol, rf_maxiter])

        # -- Differentiation parameters --
        diff_x0 = widgets.FloatText(value=1.0, description="At x =", style=_STY,
                                    layout=_short())
        diff_h = widgets.FloatText(value=0.01, description="h:", layout=_short())
        diff_conv_chk = widgets.Checkbox(
            value=True, description="Show convergence table",
        )
        diff_box = widgets.HBox([diff_x0, diff_h, diff_conv_chk])

        # -- Options --
        plot_chk = widgets.Checkbox(value=True, description="Plot result")
        export_chk = widgets.Checkbox(value=False, description="Export arrays")
        exact_chk = widgets.Checkbox(
            value=False, description="Compare with exact (may be slow)",
            style=_STY,
        )
        options_box = widgets.HBox([plot_chk, export_chk, exact_chk])

        out = _make_output()
        btn = widgets.Button(description="Calculate", button_style="success",
                             layout=_btn_layout())

        # -- Visibility toggling --
        def _toggle_category(change):
            cat = change["new"]
            method_dd.options = _methods_map[cat]

            int_box.layout.display   = "flex" if cat == "Integration" else "none"
            ode_hint.layout.display  = "block" if cat == "ODE Solver" else "none"
            ode_box1.layout.display  = "flex" if cat == "ODE Solver" else "none"
            ode_box2.layout.display  = "flex" if cat == "ODE Solver" else "none"
            rf_box1.layout.display   = "flex" if cat == "Root Finding" else "none"
            rf_box2.layout.display   = "flex" if cat == "Root Finding" else "none"
            diff_box.layout.display  = "flex" if cat == "Differentiation" else "none"

            if cat == "ODE Solver":
                expr_in.description = "dy/dt ="
                expr_in.value = "-2*y + sin(t)"
                var_in.layout.display = "none"
            elif cat == "Root Finding":
                expr_in.description = "f(x) ="
                expr_in.value = "x**3 - x - 2"
                var_in.layout.display = "flex"
            elif cat == "Differentiation":
                expr_in.description = "f(x) ="
                expr_in.value = "sin(x)"
                var_in.layout.display = "flex"
            else:
                expr_in.description = "f(x) ="
                expr_in.value = "sin(x)"
                var_in.layout.display = "flex"

        def _toggle_method(change):
            meth = change["new"]
            if meth == "Newton's Method":
                rf_x0.layout.display = "flex"
                rf_a.layout.display = "none"
                rf_b.layout.display = "none"
            elif meth == "Bisection":
                rf_x0.layout.display = "none"
                rf_a.layout.display = "flex"
                rf_b.layout.display = "flex"

        category.observe(_toggle_category, "value")
        method_dd.observe(_toggle_method, "value")

        # Fire initial toggle
        _toggle_category({"new": category.value})

        # -- Calculation callback --
        def _calc(b):
            cat = category.value
            meth = method_dd.value

            try:
                vname = var_in.value.strip()
                sv = Symbol(vname)

                if cat == "Integration":
                    self._numerical_integration(
                        meth, expr_in.value, sv, vname,
                        float(int_a.value), float(int_b.value), int(int_n.value),
                        plot_chk.value, export_chk.value, exact_chk.value, out,
                    )
                elif cat == "ODE Solver":
                    self._numerical_ode(
                        meth, expr_in.value,
                        float(ode_t0.value), float(ode_y0.value),
                        float(ode_tf.value), int(ode_steps.value),
                        plot_chk.value, export_chk.value, out,
                    )
                elif cat == "Root Finding":
                    self._numerical_rootfind(
                        meth, expr_in.value, sv, vname,
                        float(rf_a.value), float(rf_b.value), float(rf_x0.value),
                        float(rf_tol.value), int(rf_maxiter.value),
                        plot_chk.value, export_chk.value, out,
                    )
                elif cat == "Differentiation":
                    self._numerical_diff(
                        meth, expr_in.value, sv, vname,
                        float(diff_x0.value), float(diff_h.value),
                        diff_conv_chk.value,
                        plot_chk.value, export_chk.value, out,
                    )

            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([
            category, method_dd,
            expr_in, var_in,
            int_box,
            ode_hint, ode_box1, ode_box2,
            rf_box1, rf_box2,
            diff_box,
            options_box, btn, out,
        ])

    # -- Numerical Integration helpers ---------------------------------
    def _numerical_integration(self, method, expr_str, sv, vname,
                               a, b, n, do_plot, do_export, do_exact, out):
        f_sym = _parse(expr_str, {vname: sv})
        f_np = lambdify(sv, f_sym, modules=["numpy"])

        if method == "Trapezoidal Rule":
            result, xs, ys = integrate_trapezoid(f_np, a, b, n)
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"{vname} = symbols('{vname}')\n"
                f"f = lambdify({vname}, {expr_str}, 'numpy')\n"
                f"a, b, n = {a}, {b}, {n}\n"
                f"h = (b - a) / n\n"
                f"xs = np.linspace(a, b, n + 1)\n"
                f"ys = f(xs)\n"
                f"result = h * (ys[0]/2 + np.sum(ys[1:-1]) + ys[-1]/2)"
            )
        else:  # Simpson's Rule
            result, xs, ys, n = integrate_simpson(f_np, a, b, n)
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"{vname} = symbols('{vname}')\n"
                f"f = lambdify({vname}, {expr_str}, 'numpy')\n"
                f"a, b, n = {a}, {b}, {n}\n"
                f"h = (b - a) / n\n"
                f"xs = np.linspace(a, b, n + 1)\n"
                f"ys = f(xs)\n"
                f"result = (h/3) * (ys[0] + ys[-1] + 4*np.sum(ys[1:-1:2]) + 2*np.sum(ys[2:-1:2]))"
            )

        h = (b - a) / n

        # SymPy exact integral for comparison — opt-in, since symbolic
        # integration can take arbitrarily long and blocks the kernel
        exact_str = ""
        if do_exact:
            try:
                exact = integrate(f_sym, (sv, a, b))
                exact_f = float(exact.evalf())
                err = abs(result - exact_f)
                exact_str = (
                    f"<tr><td>SymPy exact</td><td>{exact_f:.12g}</td></tr>"
                    f"<tr><td>Absolute error</td><td>{err:.2e}</td></tr>"
                )
            except Exception:
                pass

        with out:
            clear_output(wait=True)
            display(HTML(
                f"<h4>{method}</h4>"
                f"<table style='border-collapse:collapse'>"
                f"<tr><td style='padding:2px 12px'><b>Numerical result</b></td>"
                f"<td>{result:.12g}</td></tr>"
                f"<tr><td style='padding:2px 12px'>Panels (n)</td><td>{n}</td></tr>"
                f"<tr><td style='padding:2px 12px'>Step size (h)</td><td>{h:.6g}</td></tr>"
                f"{exact_str}</table>"
            ))

            if do_plot:
                fig, ax = plt.subplots(figsize=(8, 5))
                xs_fine = np.linspace(a, b, 500)
                ys_fine = _plot_vals(f_np, xs_fine)
                ax.plot(xs_fine, ys_fine, "b-", linewidth=2, label="f(x)")

                if method == "Trapezoidal Rule":
                    for i in range(n):
                        trap_x = [xs[i], xs[i], xs[i + 1], xs[i + 1]]
                        trap_y = [0, ys[i], ys[i + 1], 0]
                        ax.fill(trap_x, trap_y, alpha=0.15, color="green",
                                edgecolor="green", linewidth=0.5)
                else:  # Simpson — shade parabolic panels in pairs
                    for i in range(0, n, 2):
                        xp = np.linspace(xs[i], xs[i + 2], 50)
                        # Lagrange interpolation through 3 points
                        x0, x1, x2 = xs[i], xs[i + 1], xs[i + 2]
                        y0, y1, y2 = ys[i], ys[i + 1], ys[i + 2]
                        L0 = ((xp - x1) * (xp - x2)) / ((x0 - x1) * (x0 - x2))
                        L1 = ((xp - x0) * (xp - x2)) / ((x1 - x0) * (x1 - x2))
                        L2 = ((xp - x0) * (xp - x1)) / ((x2 - x0) * (x2 - x1))
                        yp = y0 * L0 + y1 * L1 + y2 * L2
                        ax.fill_between(xp, 0, yp, alpha=0.15, color="green")

                ax.set_xlabel(vname)
                ax.set_ylabel(f"f({vname})")
                ax.set_title(f"{method}  (n={n}, result={result:.8g})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                _save_fig_and_display(fig)

            if do_export:
                self._export_arrays({"xs": xs, "ys": ys, "result": result})

            _log("numerical", code, result)
            display(_code_details(code, label="Show NumPy code"))

    # -- Numerical ODE helpers -----------------------------------------
    def _numerical_ode(self, method, expr_str,
                       t0, y0, tf, steps, do_plot, do_export, out):
        tv = Symbol("t")
        yv = Symbol("y")
        f_sym = _parse(expr_str, {"t": tv, "y": yv})
        f_np = lambdify((tv, yv), f_sym, modules=["numpy"])

        h = (tf - t0) / steps

        if method == "Euler's Method":
            ts, ys = solve_ivp_euler(f_np, t0, y0, tf, steps)
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"t, y = symbols('t y')\n"
                f"f = lambdify((t, y), {expr_str}, 'numpy')\n"
                f"t0, y0, tf, steps = {t0}, {y0}, {tf}, {steps}\n"
                f"h = (tf - t0) / steps\n"
                f"ts = np.zeros(steps + 1); ys = np.zeros(steps + 1)\n"
                f"ts[0] = t0; ys[0] = y0\n"
                f"for i in range(steps):\n"
                f"    ys[i+1] = ys[i] + h * f(ts[i], ys[i])\n"
                f"    ts[i+1] = ts[i] + h"
            )
        else:  # Runge-Kutta 4
            ts, ys = solve_ivp_rk4(f_np, t0, y0, tf, steps)
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"t, y = symbols('t y')\n"
                f"f = lambdify((t, y), {expr_str}, 'numpy')\n"
                f"t0, y0, tf, steps = {t0}, {y0}, {tf}, {steps}\n"
                f"h = (tf - t0) / steps\n"
                f"ts = np.zeros(steps + 1); ys = np.zeros(steps + 1)\n"
                f"ts[0] = t0; ys[0] = y0\n"
                f"for i in range(steps):\n"
                f"    k1 = h * f(ts[i], ys[i])\n"
                f"    k2 = h * f(ts[i] + h/2, ys[i] + k1/2)\n"
                f"    k3 = h * f(ts[i] + h/2, ys[i] + k2/2)\n"
                f"    k4 = h * f(ts[i] + h, ys[i] + k3)\n"
                f"    ys[i+1] = ys[i] + (k1 + 2*k2 + 2*k3 + k4) / 6\n"
                f"    ts[i+1] = ts[i] + h"
            )

        with out:
            clear_output(wait=True)
            display(HTML(
                f"<h4>{method}</h4>"
                f"<table style='border-collapse:collapse'>"
                f"<tr><td style='padding:2px 12px'><b>y({tf})</b></td>"
                f"<td>{ys[-1]:.12g}</td></tr>"
                f"<tr><td style='padding:2px 12px'>Steps</td><td>{steps}</td></tr>"
                f"<tr><td style='padding:2px 12px'>Step size (h)</td><td>{h:.6g}</td></tr>"
                f"</table>"
            ))

            # Show first/last few steps
            n_show = min(6, steps + 1)
            rows = ""
            for i in range(n_show):
                rows += f"<tr><td>{i}</td><td>{ts[i]:.6g}</td><td>{ys[i]:.8g}</td></tr>"
            if steps + 1 > n_show:
                rows += "<tr><td>...</td><td>...</td><td>...</td></tr>"
                rows += (f"<tr><td>{steps}</td><td>{ts[-1]:.6g}</td>"
                         f"<td>{ys[-1]:.8g}</td></tr>")
            display(HTML(
                f"<details><summary style='cursor:pointer;color:#666'>"
                f"Show step table</summary>"
                f"<table style='border-collapse:collapse;margin-top:4px'>"
                f"<tr><th style='padding:2px 8px'>Step</th>"
                f"<th style='padding:2px 8px'>t</th>"
                f"<th style='padding:2px 8px'>y</th></tr>"
                f"{rows}</table></details>"
            ))

            if do_plot:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(ts, ys, "b-o", markersize=2, linewidth=1.5, label=method)
                ax.set_xlabel("t")
                ax.set_ylabel("y")
                ax.set_title(f"{method}  (steps={steps}, h={h:.4g})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                _save_fig_and_display(fig)

            if do_export:
                self._export_arrays({"ts": ts, "ys": ys})

            _log("numerical", code, ys[-1])
            display(_code_details(code, label="Show NumPy code"))

    # -- Root-finding helpers ------------------------------------------
    def _numerical_rootfind(self, method, expr_str, sv, vname,
                            a, b, x0, tol, maxiter,
                            do_plot, do_export, out):
        f_sym = _parse(expr_str, {vname: sv})
        f_np = lambdify(sv, f_sym, modules=["numpy"])

        if method == "Bisection":
            root, iterations, converged = root_bisection(f_np, a, b, tol, maxiter)
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"{vname} = symbols('{vname}')\n"
                f"f = lambdify({vname}, {expr_str}, 'numpy')\n"
                f"a, b, tol = {a}, {b}, {tol}\n"
                f"for i in range({maxiter}):\n"
                f"    mid = (a + b) / 2\n"
                f"    if abs(f(mid)) < tol or (b - a)/2 < tol:\n"
                f"        break\n"
                f"    if f(a) * f(mid) < 0:\n"
                f"        b = mid\n"
                f"    else:\n"
                f"        a = mid\n"
                f"root = (a + b) / 2"
            )

        else:  # Newton's Method
            df_sym = diff(f_sym, sv)
            df_np = lambdify(sv, df_sym, modules=["numpy"])
            root, iterations, converged = root_newton(f_np, df_np, x0, tol, maxiter)
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"{vname} = symbols('{vname}')\n"
                f"f_expr = {expr_str}\n"
                f"df_expr = diff(f_expr, {vname})\n"
                f"f = lambdify({vname}, f_expr, 'numpy')\n"
                f"df = lambdify({vname}, df_expr, 'numpy')\n"
                f"xn = {x0}\n"
                f"for i in range({maxiter}):\n"
                f"    fxn = f(xn); dfxn = df(xn)\n"
                f"    xn1 = xn - fxn / dfxn\n"
                f"    if abs(xn1 - xn) < {tol}: break\n"
                f"    xn = xn1\n"
                f"root = xn"
            )

        n_iter = len(iterations)

        with out:
            clear_output(wait=True)
            warn_str = ""
            if not converged:
                warn_str = (
                    f"<tr><td style='padding:2px 12px;color:#c60'><b>Warning</b></td>"
                    f"<td style='color:#c60'>tolerance not reached within "
                    f"{maxiter} iterations</td></tr>"
                )
            display(HTML(
                f"<h4>{method}</h4>"
                f"<table style='border-collapse:collapse'>"
                f"<tr><td style='padding:2px 12px'><b>Root</b></td>"
                f"<td>{root:.12g}</td></tr>"
                f"<tr><td style='padding:2px 12px'>f(root)</td>"
                f"<td>{f_np(root):.2e}</td></tr>"
                f"<tr><td style='padding:2px 12px'>Iterations</td>"
                f"<td>{n_iter}</td></tr>"
                f"{warn_str}</table>"
            ))

            # Convergence table
            if iterations:
                if method == "Bisection":
                    hdr = ("<th style='padding:2px 8px'>i</th>"
                           "<th style='padding:2px 8px'>a</th>"
                           "<th style='padding:2px 8px'>b</th>"
                           "<th style='padding:2px 8px'>mid</th>"
                           "<th style='padding:2px 8px'>f(mid)</th>")
                    rows = ""
                    for it in iterations:
                        rows += (f"<tr><td>{it['i']}</td>"
                                 f"<td>{it['a']:.8g}</td>"
                                 f"<td>{it['b']:.8g}</td>"
                                 f"<td>{it['mid']:.8g}</td>"
                                 f"<td>{it['f(mid)']:.4e}</td></tr>")
                else:
                    hdr = ("<th style='padding:2px 8px'>i</th>"
                           "<th style='padding:2px 8px'>x_n</th>"
                           "<th style='padding:2px 8px'>f(x_n)</th>"
                           "<th style='padding:2px 8px'>f'(x_n)</th>"
                           "<th style='padding:2px 8px'>x_{n+1}</th>")
                    rows = ""
                    for it in iterations:
                        dfxn = it["f'(x_n)"]
                        xn1 = it["x_{n+1}"]
                        rows += (f"<tr><td>{it['i']}</td>"
                                 f"<td>{it['x_n']:.8g}</td>"
                                 f"<td>{it['f(x_n)']:.4e}</td>"
                                 f"<td>{dfxn:.4e}</td>"
                                 f"<td>{xn1:.8g}</td></tr>")
                display(HTML(
                    f"<details><summary style='cursor:pointer;color:#666'>"
                    f"Show convergence table ({n_iter} iterations)</summary>"
                    f"<table style='border-collapse:collapse;margin-top:4px'>"
                    f"<tr>{hdr}</tr>{rows}</table></details>"
                ))

            if do_plot:
                fig, ax = plt.subplots(figsize=(8, 5))
                plot_a = a if method == "Bisection" else root - 2
                plot_b = b if method == "Bisection" else root + 2
                xs_fine = np.linspace(plot_a, plot_b, 500)
                ys_fine = _plot_vals(f_np, xs_fine)
                ax.plot(xs_fine, ys_fine, "b-", linewidth=2, label="f(x)")
                ax.axhline(y=0, color="k", linewidth=0.5)
                ax.plot(root, 0, "ro", markersize=10, label=f"root={root:.6g}")

                if method == "Newton's Method" and iterations:
                    for it in iterations[:5]:  # show first 5 tangent lines
                        xn_val = it["x_n"]
                        fxn_val = it["f(x_n)"]
                        dfxn_val = it["f'(x_n)"]
                        xn1_val = it["x_{n+1}"]
                        # tangent line from (xn, f(xn)) to (xn1, 0)
                        tx = np.array([xn_val, xn1_val])
                        ty = np.array([fxn_val, 0.0])
                        ax.plot(tx, ty, "g--", alpha=0.5, linewidth=1)
                        ax.plot(xn_val, fxn_val, "gs", markersize=4)

                ax.set_xlabel(vname)
                ax.set_ylabel(f"f({vname})")
                ax.set_title(f"{method}  (root={root:.8g})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                _save_fig_and_display(fig)

            if do_export:
                self._export_arrays({"root": root, "iterations": iterations})

            _log("numerical", code, root)
            display(_code_details(code, label="Show NumPy code"))

    # -- Numerical Differentiation helpers -----------------------------
    def _numerical_diff(self, method, expr_str, sv, vname,
                        x0, h, show_conv, do_plot, do_export, out):
        f_sym = _parse(expr_str, {vname: sv})
        f_np = lambdify(sv, f_sym, modules=["numpy"])

        approx = finite_difference(f_np, x0, h, method)
        if method == "Forward Difference":
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"{vname} = symbols('{vname}')\n"
                f"f = lambdify({vname}, {expr_str}, 'numpy')\n"
                f"x0, h = {x0}, {h}\n"
                f"deriv = (f(x0 + h) - f(x0)) / h"
            )
        else:  # Central Difference
            code = (
                f"import numpy as np\n"
                f"from sympy import *\n"
                f"{vname} = symbols('{vname}')\n"
                f"f = lambdify({vname}, {expr_str}, 'numpy')\n"
                f"x0, h = {x0}, {h}\n"
                f"deriv = (f(x0 + h) - f(x0 - h)) / (2 * h)"
            )

        # SymPy exact derivative for comparison
        exact_str = ""
        exact_val = None
        try:
            df_sym = diff(f_sym, sv)
            exact_val = float(df_sym.subs(sv, x0).evalf())
            err = abs(approx - exact_val)
            exact_str = (
                f"<tr><td style='padding:2px 12px'>Exact (SymPy)</td>"
                f"<td>{exact_val:.12g}</td></tr>"
                f"<tr><td style='padding:2px 12px'>Absolute error</td>"
                f"<td>{err:.2e}</td></tr>"
            )
        except Exception:
            pass

        with out:
            clear_output(wait=True)
            display(HTML(
                f"<h4>{method}</h4>"
                f"<table style='border-collapse:collapse'>"
                f"<tr><td style='padding:2px 12px'><b>f'({x0})</b></td>"
                f"<td>{approx:.12g}</td></tr>"
                f"<tr><td style='padding:2px 12px'>h</td><td>{h:.2e}</td></tr>"
                f"{exact_str}</table>"
            ))

            # Convergence table across multiple h values
            if show_conv:
                hs = [h * (10 ** -i) for i in range(6)]
                rows = ""
                for hv in hs:
                    val = finite_difference(f_np, x0, hv, method)
                    err_v = abs(val - exact_val) if exact_val is not None else float("nan")
                    rows += (f"<tr><td>{hv:.2e}</td>"
                             f"<td>{val:.12g}</td>"
                             f"<td>{err_v:.2e}</td></tr>")
                display(HTML(
                    f"<details open><summary style='cursor:pointer;color:#666'>"
                    f"Convergence table</summary>"
                    f"<table style='border-collapse:collapse;margin-top:4px'>"
                    f"<tr><th style='padding:2px 8px'>h</th>"
                    f"<th style='padding:2px 8px'>Approximation</th>"
                    f"<th style='padding:2px 8px'>|Error|</th></tr>"
                    f"{rows}</table></details>"
                ))

            if do_plot:
                fig, ax = plt.subplots(figsize=(8, 5))
                dx = max(abs(h) * 20, 1.0)
                xs_fine = np.linspace(x0 - dx, x0 + dx, 500)
                ys_fine = _plot_vals(f_np, xs_fine)
                ax.plot(xs_fine, ys_fine, "b-", linewidth=2, label="f(x)")

                # Mark the point and tangent line
                fx0 = f_np(x0)
                ax.plot(x0, fx0, "ro", markersize=8, label=f"x = {x0}")

                # Tangent line using numerical derivative
                tang_xs = np.linspace(x0 - dx / 2, x0 + dx / 2, 100)
                tang_ys = fx0 + approx * (tang_xs - x0)
                ax.plot(tang_xs, tang_ys, "r--", linewidth=1.5,
                        label=f"tangent (slope={approx:.4g})")

                # Show the difference stencil points
                if method == "Forward Difference":
                    ax.plot([x0, x0 + h], [f_np(x0), f_np(x0 + h)],
                            "gs-", markersize=6, label="stencil points")
                else:
                    ax.plot([x0 - h, x0 + h], [f_np(x0 - h), f_np(x0 + h)],
                            "gs-", markersize=6, label="stencil points")

                ax.set_xlabel(vname)
                ax.set_ylabel(f"f({vname})")
                ax.set_title(f"{method}  (f'({x0}) ≈ {approx:.8g})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                _save_fig_and_display(fig)

            if do_export:
                self._export_arrays({"derivative": approx, "x0": x0, "h": h})

            _log("numerical", code, approx)
            display(_code_details(code, label="Show NumPy code"))

    # -- Export helper --------------------------------------------------
    @staticmethod
    def _export_arrays(data):
        """Export arrays/values to notebook namespace."""
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                names = []
                for name, val in data.items():
                    ip.user_ns[name] = val
                    names.append(name)
                display(HTML(
                    f"<div style='padding:4px;background:#e8f5e9;border-radius:4px'>"
                    f"Exported to notebook namespace: "
                    f"<code>{', '.join(names)}</code></div>"
                ))
        except Exception:
            pass

    # ---------------------------------------------------------------
    # 7. Number Theory
    # ---------------------------------------------------------------
    def _numtheory_tab(self):
        op = widgets.ToggleButtons(
            options=["Factor Integer", "Primality Test", "GCD / LCM",
                     "Mod / power_mod", "Euler Phi", "Divisors"],
            button_style="info",
        )
        n_in = widgets.Text(value="360", description="n:", layout=_med(),
                            continuous_update=False)
        m_in = widgets.Text(value="24", description="m:", layout=_short())
        exp_in = widgets.Text(value="", description="exp:", layout=_short())
        extra_box = widgets.HBox([m_in, exp_in])

        def _toggle(change):
            val = change["new"]
            if val == "GCD / LCM":
                extra_box.layout.display = "flex"
                m_in.layout.display = "flex"
                m_in.description = "m:"
                exp_in.layout.display = "none"
            elif val == "Mod / power_mod":
                extra_box.layout.display = "flex"
                m_in.layout.display = "flex"
                m_in.description = "mod:"
                exp_in.layout.display = "flex"
                exp_in.description = "exp (opt):"
            else:
                extra_box.layout.display = "none"

        op.observe(_toggle, "value")
        _toggle({"new": op.value})

        out = _make_output()
        btn = widgets.Button(description="Calculate", button_style="success", layout=_btn_layout())

        def _calc(b):
            try:
                choice = op.value
                nv = int(n_in.value.strip())

                if choice == "Factor Integer":
                    result = factorint(nv)
                    factors_str = " \u00d7 ".join(
                        f"{p}^{e}" if e > 1 else str(p)
                        for p, e in sorted(result.items())
                    )
                    with out:
                        clear_output(wait=True)
                        display(HTML(f"<b>{nv}</b> = {factors_str}"))
                        display(HTML(f"<pre>Dict: {result}</pre>"))
                        _log("calc", f"factorint({nv})", result)
                        display(_code_details(f"factorint({nv})"))
                    return

                elif choice == "Primality Test":
                    result = isprime(nv)
                    code = f"isprime({nv})"

                elif choice == "GCD / LCM":
                    mv = int(m_in.value.strip())
                    g, l_val = gcd(nv, mv), lcm(nv, mv)
                    with out:
                        clear_output(wait=True)
                        display(HTML(f"<b>GCD({nv}, {mv})</b> = {g}"))
                        display(HTML(f"<b>LCM({nv}, {mv})</b> = {l_val}"))
                        _log("calc", f"gcd({nv}, {mv})\nlcm({nv}, {mv})", (g, l_val))
                        display(_code_details(f"gcd({nv}, {mv})\nlcm({nv}, {mv})"))
                    return

                elif choice == "Mod / power_mod":
                    mv = int(m_in.value.strip())
                    if exp_in.value.strip():
                        ev = int(exp_in.value.strip())
                        result = pow(nv, ev, mv)
                        code = f"pow({nv}, {ev}, {mv})"
                    else:
                        result = nv % mv
                        code = f"{nv} % {mv}"

                elif choice == "Euler Phi":
                    result = totient(nv)
                    code = f"totient({nv})"

                elif choice == "Divisors":
                    result = divisors(nv)
                    code = f"divisors({nv})"

                _result_html(result, code, out)
            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([op, n_in, extra_box, btn, out])

    # ---------------------------------------------------------------
    # 8. Free Input — run arbitrary SymPy expressions
    # ---------------------------------------------------------------
    def _freeinput_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>Free Input</b> — type any SymPy expression and evaluate it.<br>"
            "You can combine operations: "
            "<code>simplify(diff(sin(x)*exp(x), x) - integrate(cos(x), x))</code><br>"
            "All standard SymPy functions are available. Use <code>;</code> to "
            "separate multiple expressions (last one is displayed).<br>"
            "Declare extra symbols in the Parameters field if needed."
            "</div>"
        )
        expr_in = widgets.Textarea(
            value="simplify(diff(sin(x)**2, x))",
            placeholder="e.g. simplify(diff(sin(x)*exp(x), x) - integrate(cos(x), x))",
            description="Expression:", style=_STY,
            layout=widgets.Layout(width="95%", height="80px"),
        )
        params_in = widgets.Text(
            value="",
            placeholder="e.g. alpha, beta, omega",
            description="Parameters:", style=_STY,
            layout=_med(), continuous_update=False,
        )
        params_hint = widgets.HTML(
            "<div style='color:#666;font-size:12px;padding-left:4px'>"
            "Extra symbolic constants beyond the built-in "
            "<code>a b c k n r s t u v w</code>."
            "</div>"
        )
        export_chk = widgets.Checkbox(value=False, description="Export result")
        export_name = widgets.Text(
            value="result",
            placeholder="variable name",
            description="Name:", layout=_short(),
        )
        export_box = widgets.HBox([export_chk, export_name])

        out = _make_output()
        btn = widgets.Button(description="Evaluate", button_style="success",
                             layout=_btn_layout())

        def _calc(b):
            try:
                extra = {}
                for p in params_in.value.split(","):
                    p = p.strip()
                    if p:
                        extra[p] = Symbol(p)

                # Build a rich local namespace with all SymPy functions
                loc = dict(_COMMON_LOCALS)
                loc.update(_USER_SYMBOLS)
                loc.update(extra)
                # Add common SymPy operations to the namespace
                loc.update({
                    "diff": diff, "integrate": integrate,
                    "limit": limit, "series": series, "summation": summation,
                    "solve": solve, "simplify": simplify,
                    "expand": expand, "factor": factor,
                    "apart": apart, "cancel": cancel, "trigsimp": trigsimp,
                    "Matrix": Matrix, "Eq": Eq, "dsolve": dsolve,
                    "Function": Function, "lambdify": lambdify,
                    "laplace_transform": laplace_transform,
                    "inverse_laplace_transform": inverse_laplace_transform,
                    "fourier_transform": fourier_transform,
                    "inverse_fourier_transform": inverse_fourier_transform,
                    "factorint": factorint, "isprime": isprime,
                    "gcd": gcd, "lcm": lcm, "totient": totient,
                    "divisors": divisors, "latex": latex,
                    "Abs": Abs, "floor": floor, "ceiling": ceiling,
                    "factorial": factorial, "binomial": binomial,
                    "symbols": symbols, "Symbol": Symbol,
                })

                raw = expr_in.value.strip()
                # Convert ^ to ** so users can write powers naturally
                raw = raw.replace("^", "**")

                # Support multiple expressions separated by ;
                parts = [p.strip() for p in raw.split(";") if p.strip()]
                result = None
                for part in parts:
                    result = eval(part, {"__builtins__": {}}, loc)  # noqa: S307
                    # Make result available for subsequent expressions
                    if result is not None:
                        loc["_"] = result

                code = raw

                with out:
                    clear_output(wait=True)
                    try:
                        if hasattr(result, '__iter__') and not isinstance(result, (str, sympy.Basic)):
                            # Lists, tuples, dicts — show as HTML
                            display(HTML(f"<pre>{_esc(result)}</pre>"))
                        else:
                            display(Math(latex(result)))
                            display(_code_details(latex(result), label="Show LaTeX"))
                    except Exception:
                        display(HTML(f"<pre>{_esc(result)}</pre>"))

                    if export_chk.value:
                        varname = export_name.value.strip() or "result"
                        try:
                            from IPython import get_ipython
                            ip = get_ipython()
                            if ip is not None:
                                ip.user_ns[varname] = result
                                display(HTML(
                                    f"<div style='padding:4px;background:#e8f5e9;"
                                    f"border-radius:4px'>"
                                    f"Exported <code>{_esc(varname)}</code> to notebook "
                                    f"namespace.</div>"
                                ))
                        except Exception:
                            pass

                    _log("free-input", code, result)
                    display(_code_details(code))

            except Exception as e:
                _error(str(e), out)

        btn.on_click(_calc)

        return widgets.VBox([
            info, expr_in,
            params_in, params_hint,
            export_box, btn, out,
        ])

    # ---------------------------------------------------------------
    # 8a. History — session log of every computation
    # ---------------------------------------------------------------
    def _history_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>History</b> — every computation in this session is logged "
            "here with its equivalent code. Select an entry to view its full "
            "code, or collect the whole session as a runnable script."
            "</div>"
        )
        table_html = widgets.HTML("")
        entry_dd = widgets.Dropdown(options=[], description="Entry:",
                                    style=_STY, layout=_wide())
        show_btn = widgets.Button(description="Show code", layout=_btn_layout())
        script_btn = widgets.Button(description="Session as script",
                                    layout=_btn_layout())
        export_btn = widgets.Button(description="Export to notebook",
                                    layout=_btn_layout())
        clear_btn = widgets.Button(description="Clear history",
                                   button_style="warning", layout=_btn_layout())
        btn_box = widgets.HBox([show_btn, script_btn, export_btn, clear_btn])
        code_area = widgets.Textarea(
            value="", description="Code:", style=_STY,
            placeholder="Selected entry / session script appears here — "
                        "copy it into a notebook cell.",
            layout=widgets.Layout(width="95%", height="140px"),
        )
        out = _make_output()

        def _refresh():
            if not _HISTORY:
                table_html.value = ("<div style='color:#666;padding:4px'>"
                                    "No computations yet.</div>")
                entry_dd.options = []
                return
            rows = ""
            for e in reversed(_HISTORY[-30:]):
                code_short = e["code"].replace("\n", "; ")
                if len(code_short) > 70:
                    code_short = code_short[:67] + "..."
                res_short = e["result"].replace("\n", " ")
                if len(res_short) > 40:
                    res_short = res_short[:37] + "..."
                rows += (
                    f"<tr><td style='padding:2px 8px'>{e['n']}</td>"
                    f"<td style='padding:2px 8px'>{e['time']}</td>"
                    f"<td style='padding:2px 8px'>{_esc(e['kind'])}</td>"
                    f"<td style='padding:2px 8px'><code>{_esc(code_short)}"
                    f"</code></td>"
                    f"<td style='padding:2px 8px;color:#555'>{_esc(res_short)}"
                    f"</td></tr>"
                )
            extra = ""
            if len(_HISTORY) > 30:
                extra = (f"<div style='color:#888'>... showing last 30 of "
                         f"{len(_HISTORY)} entries</div>")
            table_html.value = (
                "<table style='border-collapse:collapse'>"
                "<tr><th style='padding:2px 8px'>#</th>"
                "<th style='padding:2px 8px'>Time</th>"
                "<th style='padding:2px 8px'>Kind</th>"
                "<th style='padding:2px 8px;text-align:left'>Code</th>"
                "<th style='padding:2px 8px;text-align:left'>Result</th></tr>"
                f"{rows}</table>{extra}"
            )
            entry_dd.options = [
                (f"#{e['n']} [{e['kind']}] {e['code'].splitlines()[0][:60]}",
                 e["n"])
                for e in reversed(_HISTORY)
            ]

        def _show(b):
            if entry_dd.value is None:
                return
            e = _HISTORY[entry_dd.value - 1]
            code_area.value = e["code"]

        def _script(b):
            if not _HISTORY:
                code_area.value = ""
                return
            header = (
                "from sympy import *\n"
                "import numpy as np\n"
                "x, y, z, t, n, k, a, b, c, s, r, u, v, w = "
                "symbols('x y z t n k a b c s r u v w')\n"
            )
            code_area.value = header + "\n" + "\n\n".join(
                f"# --- {e['n']} [{e['kind']}] at {e['time']}\n{e['code']}"
                for e in _HISTORY
            )

        def _export(b):
            try:
                from IPython import get_ipython
                ip = get_ipython()
                if ip is not None:
                    ip.user_ns["helper_history"] = list(_HISTORY)
                with out:
                    clear_output(wait=True)
                    display(HTML(
                        "<div style='padding:4px;background:#e8f5e9;"
                        "border-radius:4px'>Exported <code>helper_history"
                        "</code> (list of dicts) to notebook namespace.</div>"
                    ))
            except Exception as e_:
                _error(str(e_), out)

        def _clear(b):
            _HISTORY.clear()
            code_area.value = ""
            _refresh()
            with out:
                clear_output(wait=True)
                display(HTML("<div>History cleared.</div>"))

        show_btn.on_click(_show)
        script_btn.on_click(_script)
        export_btn.on_click(_export)
        clear_btn.on_click(_clear)

        _HISTORY_LISTENERS.append(_refresh)
        _refresh()

        return widgets.VBox([
            info, table_html, entry_dd, btn_box, code_area, out,
        ])

    # ---------------------------------------------------------------
    # 8b. Symbols — declare symbols with assumptions, used by every tab
    # ---------------------------------------------------------------
    def _symbols_tab(self):
        info = widgets.HTML(
            "<div style='padding:6px;background:#f0f7ff;border-radius:4px'>"
            "<b>Symbol assumptions</b> — declare symbols with assumptions "
            "(positive, integer, ...). Every expression input in every tab "
            "then uses them, which changes results: with plain <code>a</code>, "
            "<code>simplify(sqrt(a^2))</code> stays put and "
            "<code>integrate(exp(-a*x), (x, 0, oo))</code> is conditional; "
            "with <i>positive</i> <code>a</code> they become <code>a</code> "
            "and <code>1/a</code>."
            "</div>"
        )
        name_in = widgets.Text(
            value="", placeholder="e.g. a, omega, n",
            description="Symbol(s):", style=_STY, layout=_med(),
            continuous_update=False,
        )
        chk_positive = widgets.Checkbox(value=True, description="positive")
        chk_negative = widgets.Checkbox(value=False, description="negative")
        chk_real     = widgets.Checkbox(value=False, description="real")
        chk_integer  = widgets.Checkbox(value=False, description="integer")
        chk_nonzero  = widgets.Checkbox(value=False, description="nonzero")
        chk_even     = widgets.Checkbox(value=False, description="even")
        chk_odd      = widgets.Checkbox(value=False, description="odd")
        checks = [chk_positive, chk_negative, chk_real, chk_integer,
                  chk_nonzero, chk_even, chk_odd]
        chk_box = widgets.HBox(checks, layout=widgets.Layout(flex_flow="row wrap"))

        declare_btn = widgets.Button(description="Declare",
                                     button_style="success", layout=_btn_layout())
        remove_dd = widgets.Dropdown(options=[], description="Declared:",
                                     style=_STY, layout=_med())
        remove_btn = widgets.Button(description="Remove", layout=_btn_layout())
        clear_btn = widgets.Button(description="Clear all", button_style="warning",
                                   layout=_btn_layout())
        manage_box = widgets.HBox([remove_dd, remove_btn, clear_btn])

        table_html = widgets.HTML("")
        out = _make_output()

        def _refresh():
            remove_dd.options = sorted(_USER_SYMBOLS)
            if not _USER_SYMBOLS:
                table_html.value = ("<div style='color:#666;padding:4px'>"
                                    "No symbols declared.</div>")
                return
            rows = ""
            for nm in sorted(_USER_SYMBOLS):
                kws = ", ".join(sorted(_USER_SYMBOL_SPECS.get(nm, {}))) or "—"
                rows += (f"<tr><td style='padding:2px 12px'><code>{_esc(nm)}"
                         f"</code></td><td>{_esc(kws)}</td></tr>")
            table_html.value = (
                "<table style='border-collapse:collapse'>"
                "<tr><th style='padding:2px 12px;text-align:left'>Symbol</th>"
                "<th style='text-align:left'>Assumptions</th></tr>"
                f"{rows}</table>"
            )

        def _declare(b):
            try:
                names = [p.strip() for p in name_in.value.split(",") if p.strip()]
                if not names:
                    raise ValueError("Enter at least one symbol name.")
                kwargs = {c.description: True for c in checks if c.value}
                declared = []
                for nm in names:
                    if not nm.isidentifier():
                        raise ValueError(f"'{nm}' is not a valid symbol name.")
                    sym = Symbol(nm, **kwargs)   # raises on contradictions
                    _USER_SYMBOLS[nm] = sym
                    _USER_SYMBOL_SPECS[nm] = kwargs
                    declared.append(nm)
                _refresh()
                with out:
                    clear_output(wait=True)
                    kws = ", ".join(sorted(kwargs)) or "no assumptions"
                    display(HTML(
                        f"<div style='padding:4px;background:#e8f5e9;"
                        f"border-radius:4px'>Declared "
                        f"<code>{_esc(', '.join(declared))}</code> ({_esc(kws)}). "
                        f"Used by all tabs from now on.</div>"
                    ))
                    display(_code_details(
                        f"{', '.join(declared)} = symbols("
                        f"'{' '.join(declared)}'"
                        + "".join(f", {k}=True" for k in sorted(kwargs)) + ")"
                    ))
            except Exception as e:
                _error(str(e), out)

        def _remove(b):
            nm = remove_dd.value
            if nm:
                _USER_SYMBOLS.pop(nm, None)
                _USER_SYMBOL_SPECS.pop(nm, None)
                _refresh()
                with out:
                    clear_output(wait=True)
                    display(HTML(f"<div>Removed <code>{_esc(nm)}</code>.</div>"))

        def _clear(b):
            _USER_SYMBOLS.clear()
            _USER_SYMBOL_SPECS.clear()
            _refresh()
            with out:
                clear_output(wait=True)
                display(HTML("<div>All symbol declarations removed.</div>"))

        declare_btn.on_click(_declare)
        remove_btn.on_click(_remove)
        clear_btn.on_click(_clear)
        _refresh()

        return widgets.VBox([
            info, name_in, chk_box, declare_btn,
            _label("Declared symbols"), table_html,
            manage_box, out,
        ])

    # ---------------------------------------------------------------
    # 9. Quick Reference / Symbol Palette
    # ---------------------------------------------------------------
    def _reference_tab(self):
        clipboard = widgets.Textarea(
            value="",
            placeholder="Click a button below -- its SymPy syntax appears here. Copy into your notebook.",
            description="Clipboard:", style=_STY,
            layout=widgets.Layout(width="95%", height="50px"),
        )

        def _insert(text):
            def handler(b):
                clipboard.value = text
            return handler

        groups = [
            ("Setup & Variables", [
                ("import",     "from sympy import *"),
                ("symbols",    "x, y, z = symbols('x y z')"),
                ("positive",   "x = symbols('x', positive=True)"),
                ("integer",    "n = symbols('n', integer=True)"),
                ("function",   "f = Function('f')"),
                ("init print", "init_printing()"),
            ]),
            ("Calculus", [
                ("derivative",     "diff(expr, x)"),
                ("nth deriv",      "diff(expr, x, n)"),
                ("integral",       "integrate(expr, x)"),
                ("def. integral",  "integrate(expr, (x, a, b))"),
                ("limit",          "limit(expr, x, a)"),
                ("limit +",        "limit(expr, x, a, '+')"),
                ("limit -",        "limit(expr, x, a, '-')"),
                ("series",         "series(expr, x, 0, 6)"),
                ("summation",      "summation(expr, (k, 1, n))"),
            ]),
            ("Algebra", [
                ("solve",       "solve(Eq(expr, 0), x)"),
                ("simplify",    "simplify(expr)"),
                ("expand",      "expand(expr)"),
                ("factor",      "factor(expr)"),
                ("apart",       "apart(expr, x)"),
                ("cancel",      "cancel(expr)"),
                ("trig simp",   "trigsimp(expr)"),
                ("subs",        "expr.subs(x, value)"),
                ("numerical",   "expr.evalf()"),
            ]),
            ("Common Functions", [
                ("sin",    "sin(x)"),
                ("cos",    "cos(x)"),
                ("tan",    "tan(x)"),
                ("exp",    "exp(x)"),
                ("log",    "log(x)"),
                ("sqrt",   "sqrt(x)"),
                ("abs",    "Abs(x)"),
                ("arcsin", "asin(x)"),
                ("arccos", "acos(x)"),
                ("arctan", "atan(x)"),
                ("oo",     "oo"),
                ("pi",     "pi"),
                ("e",      "E"),
                ("i",      "I"),
            ]),
            ("Linear Algebra", [
                ("matrix",      "Matrix([[1,2],[3,4]])"),
                ("det",         "M.det()"),
                ("inverse",     "M.inv()"),
                ("transpose",   "M.T"),
                ("eigenvals",   "M.eigenvals()"),
                ("eigenvects",  "M.eigenvects()"),
                ("rref",        "M.rref()"),
                ("nullspace",   "M.nullspace()"),
            ]),
            ("Vector Calculus", [
                ("gradient",    "Matrix([diff(f, v) for v in (x, y)])"),
                ("divergence",  "sum(diff(Fi, v) for Fi, v in zip(F, (x, y)))"),
                ("curl 2D",     "diff(F2, x) - diff(F1, y)"),
                ("laplacian",   "sum(diff(f, v, 2) for v in (x, y))"),
                ("jacobian",    "Matrix(F).jacobian(Matrix([x, y]))"),
                ("hessian",     "hessian(f, (x, y))"),
                ("directional", "grad.dot(u) / u.norm()"),
            ]),
            ("Transforms", [
                ("Laplace",      "laplace_transform(f, t, s, noconds=True)"),
                ("inv Laplace",  "inverse_laplace_transform(F, s, t)"),
                ("Fourier",      "fourier_transform(f, x, k)"),
                ("inv Fourier",  "inverse_fourier_transform(F, k, x)"),
                ("Heaviside",    "Heaviside(t)"),
                ("Dirac delta",  "DiracDelta(t)"),
            ]),
            ("Plotting", [
                ("2D plot",      "plot(sin(x), (x, -pi, pi))"),
                ("parametric",   "plot_parametric(cos(t), sin(t), (t, 0, 2*pi))"),
                ("3D surface",   "plot3d(sin(x)*cos(y), (x,-3,3), (y,-3,3))"),
                ("implicit",     "plot_implicit(x**2 + y**2 - 1)"),
            ]),
            ("Numerical Methods", [
                ("trapezoidal",  "h*(y[0]/2 + sum(y[1:-1]) + y[-1]/2)"),
                ("simpson",      "(h/3)*(y[0] + y[-1] + 4*sum(y[1::2]) + 2*sum(y[2::2]))"),
                ("euler",        "y[i+1] = y[i] + h*f(t[i], y[i])"),
                ("RK4",          "k1=h*f(t,y); k2=h*f(t+h/2,y+k1/2); ..."),
                ("bisection",    "mid=(a+b)/2; if f(a)*f(mid)<0: b=mid; else: a=mid"),
                ("newton",       "x = x - f(x)/f'(x)"),
                ("fwd diff",     "(f(x+h) - f(x)) / h"),
                ("central diff", "(f(x+h) - f(x-h)) / (2*h)"),
            ]),
            ("Free Input examples", [
                ("simplify diff", "simplify(diff(sin(x)**2, x))"),
                ("diff - int",    "simplify(diff(sin(x)*exp(x), x) - integrate(cos(x), x))"),
                ("solve + subs",  "solve(x**2 - a, x)[0].subs(a, 4)"),
                ("compose",       "limit(integrate(exp(-t**2), (t, 0, x)), x, oo)"),
                ("matrix eig",    "Matrix([[1,2],[3,4]]).eigenvals()"),
            ]),
            ("Display", [
                ("show LaTeX",  "display(Math(latex(expr)))"),
                ("pretty",      "pprint(expr)"),
                ("get LaTeX",   "print(latex(expr))"),
            ]),
        ]

        sections = []
        for title, items in groups:
            btns = []
            for btn_label, code in items:
                b = widgets.Button(
                    description=btn_label,
                    layout=widgets.Layout(width="auto", min_width="70px", margin="2px"),
                    button_style="",
                )
                b.on_click(_insert(code))
                btns.append(b)
            sections.append(widgets.VBox([
                _label(title),
                widgets.HBox(btns, layout=widgets.Layout(flex_flow="row wrap")),
            ]))

        tip = widgets.HTML(
            "<div style='color:#555;margin-top:8px;padding:4px;background:#fffbe6;"
            "border-radius:4px'>"
            "<b>Tip:</b> Click any button to copy its SymPy syntax to the clipboard "
            "area above, then paste into your notebook cell.</div>"
        )

        return widgets.VBox([clipboard, tip] + sections)
