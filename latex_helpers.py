"""
LaTeX rendering helpers -- turns the exact-fraction computation from
lu_engine.py into the visual matrix / equation layout used in the notebook
(symbolic L*U matrices, the "multiplied out" product matrix with cells like
ag, bg+d, cg+e, and boxed final answers for each unknown).
"""

from fractions import Fraction


def latex_num(fr):
    """Fraction -> LaTeX. Never a decimal. e.g. Fraction(1,2) -> '\\frac{1}{2}'."""
    fr = Fraction(fr)
    if fr.denominator == 1:
        return str(fr.numerator)
    sign = "-" if fr.numerator < 0 else ""
    return f"{sign}\\frac{{{abs(fr.numerator)}}}{{{fr.denominator}}}"


def bmatrix(rows):
    body = " \\\\ ".join(" & ".join(row) for row in rows)
    return "\\begin{bmatrix}" + body + "\\end{bmatrix}"


# ------------------------------------------------------------------------------
# Symbolic (unresolved) L, U and their product -- the "Step II: A = LU" layout
# ------------------------------------------------------------------------------
def _sym_L_entry(i, k):
    if k == i:
        return "1"
    if k > i:
        return "0"
    return f"l_{{{i+1}{k+1}}}"


def _sym_U_entry(k, j):
    if k > j:
        return "0"
    return f"u_{{{k+1}{j+1}}}"


def symbolic_L_latex(n):
    return bmatrix([[_sym_L_entry(i, k) for k in range(n)] for i in range(n)])


def symbolic_U_latex(n):
    return bmatrix([[_sym_U_entry(k, j) for j in range(n)] for k in range(n)])


def symbolic_A_latex(A):
    n = len(A)
    return bmatrix([[latex_num(A[i][j]) for j in range(n)] for i in range(n)])


def symbolic_product_latex(n):
    """The 'multiplied out' matrix: each cell (i,j) = sum of L*U products,
    e.g. cell (1,0) = l_{21}*u_{11}, matching the notebook's 'ag' style cell."""
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            terms = []
            for k in range(min(i, j) + 1):
                Lt, Ut = _sym_L_entry(i, k), _sym_U_entry(k, j)
                if Lt == "0" or Ut == "0":
                    continue
                terms.append(Ut if Lt == "1" else f"{Lt}{Ut}")
            row.append(" + ".join(terms) if terms else "0")
        rows.append(row)
    return bmatrix(rows)


def numeric_LU_latex(L, U):
    n = len(L)
    L_str = bmatrix([[latex_num(L[i][j]) for j in range(n)] for i in range(n)])
    U_str = bmatrix([[latex_num(U[i][j]) for j in range(n)] for i in range(n)])
    return L_str, U_str


# ------------------------------------------------------------------------------
# One unknown's equation, LaTeX with a boxed final answer
# ------------------------------------------------------------------------------
def step_to_latex(s):
    """Renders one structured LU step (from lu_engine) as a LaTeX equation."""
    if s["type"] == "U":
        if not s["terms"]:
            return f"{s['var']} = {s['a_label']} = \\boxed{{{latex_num(s['value'])}}}"
        sym_terms = " + ".join(f"{cl}\\!\\cdot\\!{ol}" for cl, cv, ol, ov in s["terms"])
        num_terms = " + ".join(f"({latex_num(cv)})({latex_num(ov)})" for cl, cv, ol, ov in s["terms"])
        return (
            f"{s['var']} = {s['a_label']} - \\left[{sym_terms}\\right] "
            f"= {latex_num(s['a_value'])} - \\left[{num_terms}\\right] "
            f"= {latex_num(s['a_value'])} - \\left({latex_num(s['total'])}\\right) "
            f"= \\boxed{{{latex_num(s['value'])}}}"
        )
    else:
        if not s["terms"]:
            return (f"{s['var']} = \\dfrac{{{s['a_label']}}}{{{s['divisor_label']}}} "
                    f"= \\dfrac{{{latex_num(s['a_value'])}}}{{{latex_num(s['divisor_value'])}}} "
                    f"= \\boxed{{{latex_num(s['value'])}}}")
        sym_terms = " + ".join(f"{cl}\\!\\cdot\\!{ol}" for cl, cv, ol, ov in s["terms"])
        num_terms = " + ".join(f"({latex_num(cv)})({latex_num(ov)})" for cl, cv, ol, ov in s["terms"])
        numerator = s["a_value"] - s["total"]
        return (
            f"{s['var']} = \\dfrac{{{s['a_label']} - \\left[{sym_terms}\\right]}}{{{s['divisor_label']}}} "
            f"= \\dfrac{{{latex_num(s['a_value'])} - \\left[{num_terms}\\right]}}{{{latex_num(s['divisor_value'])}}} "
            f"= \\dfrac{{{latex_num(numerator)}}}{{{latex_num(s['divisor_value'])}}} "
            f"= \\boxed{{{latex_num(s['value'])}}}"
        )


def _join_signed_terms(terms):
    """terms: list of (coefficient_Fraction_or_None, latex_body) tuples.
    Joins with correct +/- signs instead of showing '+ -10y2'."""
    parts = []
    for idx, (coef, body) in enumerate(terms):
        negative = coef is not None and coef < 0
        if idx == 0:
            parts.append(f"-{body}" if negative else body)
        else:
            parts.append(f" - {body}" if negative else f" + {body}")
    return "".join(parts)


def fwd_step_to_latex(s, n):
    """Forward substitution row as a full symbolic equation, then solved."""
    term_list = []
    for k in range(s["i"]):
        coef = s["L_row"][k]
        if coef != 0:
            mag = abs(coef)
            body = f"y_{{{k+1}}}" if mag == 1 else f"{latex_num(mag)}y_{{{k+1}}}"
            term_list.append((coef, body))
    term_list.append((None, f"y_{{{s['i']+1}}}"))
    full_eq = _join_signed_terms(term_list) + f" = {latex_num(s['b_value'])}"

    if not s["terms"]:
        solved = f"{s['var']} = {s['b_label']} = \\boxed{{{latex_num(s['value'])}}}"
    else:
        num_terms = " + ".join(f"({latex_num(cv)})({latex_num(ov)})" for cl, cv, ol, ov in s["terms"])
        solved = (
            f"{s['var']} = {s['b_label']} - \\left[{num_terms}\\right] "
            f"= {latex_num(s['b_value'])} - \\left({latex_num(s['total'])}\\right) "
            f"= \\boxed{{{latex_num(s['value'])}}}"
        )
    return full_eq, solved


def back_step_to_latex(s, n):
    """Backward substitution row as a full symbolic equation, then solved."""
    term_list = [(None, f"{latex_num(s['divisor_value'])}x_{{{s['i']+1}}}")]
    for k in range(s["i"] + 1, n):
        coef = s["U_row"][k]
        if coef != 0:
            mag = abs(coef)
            body = f"x_{{{k+1}}}" if mag == 1 else f"{latex_num(mag)}x_{{{k+1}}}"
            term_list.append((coef, body))
    full_eq = _join_signed_terms(term_list) + f" = {latex_num(s['y_value'])}"

    if not s["terms"]:
        solved = (f"{s['var']} = \\dfrac{{{s['y_label']}}}{{{s['divisor_label']}}} "
                  f"= \\dfrac{{{latex_num(s['y_value'])}}}{{{latex_num(s['divisor_value'])}}} "
                  f"= \\boxed{{{latex_num(s['value'])}}}")
    else:
        num_terms = " + ".join(f"({latex_num(cv)})({latex_num(ov)})" for cl, cv, ol, ov in s["terms"])
        numerator = s["y_value"] - s["total"]
        solved = (
            f"{s['var']} = \\dfrac{{{s['y_label']} - \\left[{num_terms}\\right]}}{{{s['divisor_label']}}} "
            f"= \\dfrac{{{latex_num(numerator)}}}{{{latex_num(s['divisor_value'])}}} "
            f"= \\boxed{{{latex_num(s['value'])}}}"
        )
    return full_eq, solved