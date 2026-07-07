# notebook-math-helper

An interactive, tabbed helper for symbolic mathematics in Jupyter notebooks, built on
[SymPy](https://www.sympy.org/). Instead of memorizing SymPy's API, you type an expression,
pick an operation from a widget, and get typeset LaTeX output — with the equivalent SymPy
code shown right below it so you can learn or copy it.

It's designed to live *inside* your notebook: launch the UI in one cell, then keep using the
rest of the notebook for your own calculations. Every symbol you declare and every result you
compute is tracked and reusable.

```python
from sympy_helper import SympyHelper

helper = SympyHelper()   # renders the tabbed UI
```

## Features

A single `ipywidgets.Tab` UI with 16 tabs:

| Tab                   | What it does                                                 |
| --------------------- | ------------------------------------------------------------ |
| **Calculus**          | Derivatives, integrals, limits, Taylor series, summations    |
| **Vector Calculus**   | Gradient, divergence, curl, Laplacian, directional derivative |
| **Algebra**           | Solve, simplify, expand, factor, substitute, partial fractions |
| **Plotting**          | 2D/3D plots, multiple curves per axes with copy-able LaTeX   |
| **Function Analysis** | Domain, roots, extrema, inflection points, asymptotes        |
| **Interactive Plot**  | Live parameter sliders that re-plot on the fly               |
| **Linear Algebra**    | Matrix arithmetic, determinant, inverse, eigenvalues, RREF   |
| **Diff Equations**    | Symbolic ODE solving via `dsolve`                            |
| **ODE Systems**       | Numerical phase planes + symbolic system solving             |
| **Transforms**        | Laplace / inverse Laplace, Fourier / inverse Fourier         |
| **Numerical Methods** | Quadrature, root finding, IVP integration, finite differences |
| **Number Theory**     | Factorization, primality, gcd/lcm, totient, divisors         |
| **Free Input**        | Evaluate arbitrary SymPy expressions directly                |
| **History**           | Log of every computation; export the session as a script     |
| **Symbols**           | Declare symbols with assumptions (real, positive, integer, …) |
| **Quick Reference**   | Cheat sheet of syntax and supported functions                |

Nice touches throughout:

- `^` works as a power operator and implicit multiplication is allowed (`2x` → `2*x`).
- Every result shows its equivalent, runnable SymPy/NumPy code in a collapsible panel.
- Symbols declared in the **Symbols** tab (with assumptions) are picked up by every other tab,
  so `sqrt(q**2)` simplifies to `q` once `q` is declared positive.
- The **History** tab can export your whole session as a standalone Python script.

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install sympy ipywidgets matplotlib numpy
```

## Usage

Open the template notebook and run the first cell:

```bash
jupyter notebook template_sympy.ipynb
```

Or in any notebook of your own:

```python
from sympy_helper import SympyHelper
helper = SympyHelper()
```

The UI renders inline. Use the tabs for common operations, and use the rest of the notebook
for free-form work — symbols and results carry over.


## Project structure

```
sympy_helper.py         # the SympyHelper class + the module-level computation core
template_sympy.ipynb    # ready-to-run template notebook
```

The math itself lives in plain module-level functions (`integrate_simpson`, `solve_ivp_rk4`,
`gradient`, `analyze_function`, …) that the widget callbacks delegate to, so the numerics are
unit-testable independently of the UI.
