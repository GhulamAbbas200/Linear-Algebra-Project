# LU Decomposition Solver — Data Science Application

An interactive Streamlit application implementing **LU Decomposition (Doolittle's
Method)** from scratch, applied to a real Data Science problem: fitting a multiple
linear regression model via the **normal equations**.

Built for a Linear Algebra course project — Application of Systems of Linear
Equations in Computer Science.

**Live demo:** *(add your deployed Streamlit Community Cloud link here)*

---

## Problem Statement

**Field:** Data Science

**Task:** Predict a house's price from three features — Area, number of Bedrooms,
and Age — using the linear model:

```
price = b0 + b1·Area + b2·Bedrooms + b3·Age
```

Given a dataset of *n* houses, the best-fit coefficients (b0, b1, b2, b3) that
minimize the sum of squared errors are the solution to the **normal equations** of
least squares:

```
(XᵀX) b = Xᵀy
```

where **X** is the (n × 4) design matrix (a column of 1's for the intercept plus
the 3 feature columns) and **y** is the vector of observed prices.

This produces a **square, symmetric 4×4 system of linear equations** — solved here
using LU decomposition rather than direct matrix inversion, because in real
Data Science pipelines `XᵀX` is often reused across many right-hand sides
(cross-validation folds, ridge regression trials, etc.). Decomposing once into
`L` and `U` means every subsequent solve is a cheap forward/backward substitution
instead of repeating full Gaussian elimination — this is the same principle behind
MATLAB's `\` operator and SciPy's `linalg.lu_factor`.

---

## Method

**LU Decomposition — Doolittle's Method (no pivoting)**

For a square matrix `A`, we decompose `A = LU` where `L` is unit-lower-triangular
and `U` is upper-triangular. Multiplying `L·U` symbolically and matching each
entry to the corresponding entry of `A` gives one equation per unknown, solved in
order — the same method as classic hand-worked textbook examples.

The system `Ax = b` is then solved in two cheap steps instead of one expensive one:

1. **Forward substitution:** solve `Ly = b` for `y`
2. **Backward substitution:** solve `Ux = y` for `x`

**All arithmetic uses Python's `fractions.Fraction`** — every intermediate value
and the final answer is an *exact* fraction, never a rounded decimal.

The engine is fully generic and works for any **2×2, 3×3, or 4×4** system.

---

## Features

- 🧮 **Manual System Solver** — enter any A and b (2×2 / 3×3 / 4×4), see the full
  derivation: symbolic `A = L·U` matrix setup, every unknown solved one equation
  at a time (using the standard `a, b, c, ...` letter labeling), forward and
  backward substitution shown as complete equations, and a final exact-fraction
  answer with verification.
- 🏠 **Data Science Application** — editable house-price dataset, builds the
  design matrix and normal equations live, solves the resulting 4×4 system with
  the same engine, displays the fitted regression formula, and lets you predict
  a price for a new house.
- ✅ **Exact arithmetic throughout** — no floating-point rounding anywhere in the
  computation or display.
- 🔎 **Built-in verification** — every solve is checked against `A · x = b`
  exactly (zero tolerance needed, since there's no rounding error to tolerate).

---

## Project Structure

```
.
├── app.py              # Streamlit UI (two tabs: manual solver, regression demo)
├── lu_engine.py         # Core LU decomposition engine (exact-fraction, generic n x n)
├── latex_helpers.py     # Renders the engine's output as LaTeX matrices/equations
├── requirements.txt     # Python dependencies
├── DEMO_QUESTIONS.md    # Pre-verified example systems for demoing the app
└── README.md
```

---

## Running Locally

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Deployment

Deployed via **Streamlit Community Cloud** (`share.streamlit.io`), connected
directly to this GitHub repository, pointing at `app.py`.

> Note: Streamlit apps require a persistent server process with an open
> WebSocket connection, which is incompatible with serverless platforms like
> Vercel. Streamlit Community Cloud, Render, or Railway are the correct hosts
> for this kind of app.

---

## Example

For the system:

```
2x + 3y - 4z = 3
 x + 2y - 6z = 1
4x +  y +  z = 7
```

the app derives:

```
L = [ 1     0    0 ]     U = [ 2   3   -4 ]
    [ 1/2   1    0 ]         [ 0  1/2  -4 ]
    [ 2   -10    1 ]         [ 0   0  -31 ]
```

and solves to the exact answer `x = 53/31, y = 1/31, z = 4/31`, verified exactly
against `A·x = b`. See `DEMO_QUESTIONS.md` for more worked examples, including the
2×2 and 4×4 cases and the regression walkthrough.

---

## Tech Stack

- **Python 3** — core LU decomposition logic, pure `fractions.Fraction` arithmetic
- **Streamlit** — interactive UI
- **Pandas** — dataset/matrix input grids

---

## Author

*(your name, course, and submission date here)*
