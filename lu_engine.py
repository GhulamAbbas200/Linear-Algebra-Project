"""
Generic LU Decomposition engine with FULL SYMBOLIC STEP DERIVATION.
Every arithmetic value is a Python Fraction -- NEVER a float/decimal.

This reproduces, for ANY n x n matrix (n = 2, 3, or 4), the exact hand-worked
method from the notebook:
  1. Write A = L*U with L unit-lower-triangular and U upper-triangular.
  2. Multiply L*U symbolically and match each entry of A to an equation in
     one unknown (e.g. "ag = 1", "bg + d = 2", "ch + dh = 1" in the notebook's
     3x3 letters -- here written generically as u_ij / l_ij subscripts since
     the alphabet runs out at 4x4).
  3. Solve each equation IN ORDER, substituting already-known values, and
     record every step as a printable equation string.
  4. Forward-substitute Ly = b (again showing each equation and solve).
  5. Back-substitute Ux = y (again showing each equation and solve).

Why u_ij / l_ij instead of a, b, c ... like the notebook?
  The notebook's 3x3 example needed 9 unknowns (a..i) which just fits the
  alphabet. A 4x4 system needs 16 unknowns (10 for U, 6 for L) -- more
  letters than are convenient to keep track of. u_ij / l_ij is the standard
  textbook subscript notation for the SAME method, and it scales to any
  size without running out of letters. The arithmetic and the order of
  steps are identical to the notebook's approach.
"""

from fractions import Fraction


def F(x):
    """Convert int/str/Fraction safely to an exact Fraction (never a float)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        return Fraction(x.strip())
    if isinstance(x, int):
        return Fraction(x)
    # Only hit this for floats passed in by mistake -- convert via string
    # of a reasonably-limited denominator to avoid binary float noise.
    return Fraction(x).limit_denominator(10_000)


def to_fraction_matrix(M):
    return [[F(v) for v in row] for row in M]


def to_fraction_vector(v):
    return [F(x) for x in v]


def fmt(fr):
    """Pretty string for a Fraction: '2', '-4', '1/2', '-31' ... never decimal."""
    return str(fr)


# ==============================================================================
# STEP A: SYMBOLIC LU DECOMPOSITION  (Doolittle, no pivoting -- matches the
#          notebook exactly, since the notebook never swapped rows either)
# ==============================================================================
def lu_decompose_with_steps(A):
    """
    A: list of lists (Fractions or ints/strings convertible to Fraction), n x n

    Returns:
        L, U     : n x n matrices of Fractions
        steps    : list of STRUCTURED step dicts (not pre-formatted text),
                   one per unknown solved, in the exact order solved.
                   Each dict has keys:
                     type      : "U" or "L"
                     i, j      : 0-indexed position of the unknown
                     var       : "u_ij" or "l_ij" style label string
                     a_label   : label of the corresponding entry of A
                     a_value   : Fraction, the A[i][j] value
                     terms     : list of (coef_label, coef_value, other_label,
                                 other_value) tuples -- the already-known
                                 products being subtracted
                     total     : Fraction, sum of the terms above
                     divisor_label / divisor_value : only present for type "L"
                     value     : Fraction, the final solved value
    """
    A = to_fraction_matrix(A)
    n = len(A)
    L = [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]
    U = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    steps = []

    def uL(i, j):
        return f"u{i+1}{j+1}"

    def lL(i, j):
        return f"l{i+1}{j+1}"

    for i in range(n):
        # ---- Row i of U, columns j = i .. n-1 ----
        for j in range(i, n):
            terms = []
            total = Fraction(0)
            for k in range(i):
                total += L[i][k] * U[k][j]
                terms.append((lL(i, k), L[i][k], uL(k, j), U[k][j]))
            U[i][j] = A[i][j] - total
            steps.append({
                "type": "U", "i": i, "j": j, "var": uL(i, j),
                "a_label": f"a{i+1}{j+1}", "a_value": A[i][j],
                "terms": terms, "total": total, "value": U[i][j],
            })

        # ---- Column i of L, rows j = i+1 .. n-1 ----
        for j in range(i + 1, n):
            terms = []
            total = Fraction(0)
            for k in range(i):
                total += L[j][k] * U[k][i]
                terms.append((lL(j, k), L[j][k], uL(k, i), U[k][i]))
            numerator = A[j][i] - total
            if U[i][i] == 0:
                raise ValueError(
                    f"Zero pivot at u{i+1}{i+1} -- this plain (no-pivoting) "
                    f"Doolittle method cannot continue. Rows would need to "
                    f"be swapped, which changes the method from the "
                    f"notebook's approach."
                )
            L[j][i] = numerator / U[i][i]
            steps.append({
                "type": "L", "i": j, "j": i, "var": lL(j, i),
                "a_label": f"a{j+1}{i+1}", "a_value": A[j][i],
                "terms": terms, "total": total,
                "divisor_label": uL(i, i), "divisor_value": U[i][i],
                "value": L[j][i],
            })

    return L, U, steps


def format_step_text(s):
    """Formats one structured step dict as a plain-text equation (fractions, no decimals)."""
    if s["type"] == "U":
        if s["terms"]:
            terms_sym = [f"{cl}*{ol}" for cl, cv, ol, ov in s["terms"]]
            terms_num = [f"({fmt(cv)})({fmt(ov)})" for cl, cv, ol, ov in s["terms"]]
            return (f"{s['var']} = {s['a_label']} - [{' + '.join(terms_sym)}]"
                    f"  =  {fmt(s['a_value'])} - [{' + '.join(terms_num)}]"
                    f"  =  {fmt(s['a_value'])} - ({fmt(s['total'])})"
                    f"  =  {fmt(s['value'])}")
        return f"{s['var']} = {s['a_label']}  =  {fmt(s['value'])}"
    else:
        if s["terms"]:
            terms_sym = [f"{cl}*{ol}" for cl, cv, ol, ov in s["terms"]]
            terms_num = [f"({fmt(cv)})({fmt(ov)})" for cl, cv, ol, ov in s["terms"]]
            numerator = s["a_value"] - s["total"]
            return (f"{s['var']} = ({s['a_label']} - [{' + '.join(terms_sym)}]) / {s['divisor_label']}"
                    f"  =  ({fmt(s['a_value'])} - [{' + '.join(terms_num)}]) / {fmt(s['divisor_value'])}"
                    f"  =  ({fmt(s['a_value'])} - ({fmt(s['total'])})) / {fmt(s['divisor_value'])}"
                    f"  =  {fmt(numerator)} / {fmt(s['divisor_value'])}"
                    f"  =  {fmt(s['value'])}")
        return (f"{s['var']} = {s['a_label']} / {s['divisor_label']}"
                f"  =  {fmt(s['a_value'])} / {fmt(s['divisor_value'])}"
                f"  =  {fmt(s['value'])}")


# ==============================================================================
# STEP B: FORWARD SUBSTITUTION with steps    Ly = b
# ==============================================================================
def forward_substitution_with_steps(L, b):
    n = len(b)
    b = to_fraction_vector(b)
    y = [Fraction(0)] * n
    steps = []
    for i in range(n):
        terms = []
        total = Fraction(0)
        for k in range(i):
            total += L[i][k] * y[k]
            terms.append((f"l{i+1}{k+1}", L[i][k], f"y{k+1}", y[k]))
        y[i] = b[i] - total
        steps.append({
            "i": i, "var": f"y{i+1}", "b_label": f"b{i+1}", "b_value": b[i],
            "L_row": L[i], "terms": terms, "total": total, "value": y[i],
        })
    return y, steps


def format_fwd_step_text(s):
    if s["terms"]:
        terms_sym = [f"{cl}*{ol}" for cl, cv, ol, ov in s["terms"]]
        terms_num = [f"({fmt(cv)})({fmt(ov)})" for cl, cv, ol, ov in s["terms"]]
        return (f"{s['var']} = {s['b_label']} - [{' + '.join(terms_sym)}]"
                f"  =  {fmt(s['b_value'])} - [{' + '.join(terms_num)}]"
                f"  =  {fmt(s['b_value'])} - ({fmt(s['total'])})"
                f"  =  {fmt(s['value'])}")
    return f"{s['var']} = {s['b_label']}  =  {fmt(s['value'])}"


# ==============================================================================
# STEP C: BACKWARD SUBSTITUTION with steps    Ux = y
# ==============================================================================
def backward_substitution_with_steps(U, y):
    n = len(y)
    x = [Fraction(0)] * n
    steps = []
    for i in range(n - 1, -1, -1):
        terms = []
        total = Fraction(0)
        for k in range(i + 1, n):
            total += U[i][k] * x[k]
            terms.append((f"u{i+1}{k+1}", U[i][k], f"x{k+1}", x[k]))
        x[i] = (y[i] - total) / U[i][i]
        steps.append({
            "i": i, "var": f"x{i+1}", "y_label": f"y{i+1}", "y_value": y[i],
            "U_row": U[i], "terms": terms, "total": total,
            "divisor_label": f"u{i+1}{i+1}", "divisor_value": U[i][i],
            "value": x[i],
        })
    steps.reverse()  # display in x1, x2, x3... order (solved n..1)
    return x, steps


def format_back_step_text(s):
    if s["terms"]:
        terms_sym = [f"{cl}*{ol}" for cl, cv, ol, ov in s["terms"]]
        terms_num = [f"({fmt(cv)})({fmt(ov)})" for cl, cv, ol, ov in s["terms"]]
        numerator = s["y_value"] - s["total"]
        return (f"{s['var']} = ({s['y_label']} - [{' + '.join(terms_sym)}]) / {s['divisor_label']}"
                f"  =  ({fmt(s['y_value'])} - [{' + '.join(terms_num)}]) / {fmt(s['divisor_value'])}"
                f"  =  ({fmt(s['y_value'])} - ({fmt(s['total'])})) / {fmt(s['divisor_value'])}"
                f"  =  {fmt(numerator)} / {fmt(s['divisor_value'])}"
                f"  =  {fmt(s['value'])}")
    return (f"{s['var']} = {s['y_label']} / {s['divisor_label']}"
            f"  =  {fmt(s['y_value'])} / {fmt(s['divisor_value'])}"
            f"  =  {fmt(s['value'])}")


# ==============================================================================
# FULL SOLVER
# ==============================================================================
def solve_lu_exact(A, b):
    """
    Full pipeline. Returns a dict with L, U, x, y and every step string,
    all in exact Fractions.
    """
    A_f = to_fraction_matrix(A)
    b_f = to_fraction_vector(b)

    L, U, lu_steps = lu_decompose_with_steps(A_f)
    y, fwd_steps = forward_substitution_with_steps(L, b_f)
    x, back_steps = backward_substitution_with_steps(U, y)

    # exact verification (fraction arithmetic, zero rounding error possible)
    n = len(A_f)
    Ax = [sum(A_f[i][j] * x[j] for j in range(n)) for i in range(n)]
    exact_match = all(Ax[i] == b_f[i] for i in range(n))

    return {
        "A": A_f, "b": b_f, "L": L, "U": U, "y": y, "x": x,
        "lu_steps": lu_steps, "fwd_steps": fwd_steps, "back_steps": back_steps,
        "Ax": Ax, "exact_match": exact_match,
    }


if __name__ == "__main__":
    # ---- Reproduce the notebook's 3x3 example exactly ----
    A3 = [[2, 3, -4], [1, 2, -6], [4, 1, 1]]
    b3 = [3, 1, 7]
    result = solve_lu_exact(A3, b3)

    print("STEP II: A = LU  (symbolic derivation)")
    for s in result["lu_steps"]:
        print(" ", format_step_text(s))

    print("\nL =")
    for row in result["L"]:
        print("  ", [fmt(v) for v in row])
    print("U =")
    for row in result["U"]:
        print("  ", [fmt(v) for v in row])

    print("\nSTEP III: Ly = b  (forward substitution)")
    for s in result["fwd_steps"]:
        print(" ", format_fwd_step_text(s))

    print("\nSTEP IV: Ux = y  (backward substitution)")
    for s in result["back_steps"]:
        print(" ", format_back_step_text(s))

    print("\nFinal solution x =", [fmt(v) for v in result["x"]])
    print("Verification A@x =", [fmt(v) for v in result["Ax"]], "  b =", [fmt(v) for v in result["b"]])
    print("Exact match:", result["exact_match"])