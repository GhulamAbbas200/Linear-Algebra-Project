"""
==============================================================================
STREAMLIT APP:  LU Decomposition Solver -- Matrix-Style Visual Layout
Application: Data Science -> Multiple Linear Regression via Normal Equations
==============================================================================
Run locally with:
    pip install streamlit numpy pandas altair
    streamlit run app.py
==============================================================================
"""

from fractions import Fraction

import pandas as pd
import streamlit as st

from lu_engine import solve_lu_exact, fmt
from latex_helpers import (
    symbolic_L_latex, symbolic_U_latex, symbolic_product_latex, symbolic_A_latex,
    numeric_LU_latex, step_to_latex, fwd_step_to_latex, back_step_to_latex,
)

st.set_page_config(page_title="LU Decomposition Solver (Exact Fractions)", layout="wide")


# ==============================================================================
# HELPERS
# ==============================================================================
def matrix_to_str_df(M):
    n = len(M)
    m = len(M[0])
    return pd.DataFrame(
        [[fmt(v) for v in row] for row in M],
        index=[f"R{i+1}" for i in range(n)],
        columns=[f"C{j+1}" for j in range(m)],
    )


def render_step_grid(latex_strings, cols_per_row=3):
    """Renders a list of LaTeX equation strings as boxed cards in a grid,
    matching the notebook's side-by-side column layout."""
    for start in range(0, len(latex_strings), cols_per_row):
        chunk = latex_strings[start:start + cols_per_row]
        cols = st.columns(len(chunk))
        for c, eq in zip(cols, chunk):
            with c:
                st.latex(eq)


def render_result(result, n, x_labels=None):
    if x_labels is None:
        x_labels = [f"x_{i+1}" for i in range(n)]

    # ---- STEP 1: Set up A = LU symbolically ----
    st.subheader("Step 1 — Set up: A = L · U")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**A (given)**")
        st.latex(symbolic_A_latex(result["A"]))
    with c2:
        st.markdown("**L (unknowns)**")
        st.latex(symbolic_L_latex(n))
    with c3:
        st.markdown("**U (unknowns)**")
        st.latex(symbolic_U_latex(n))

    st.markdown("**Multiplying L · U symbolically, each cell must equal the matching cell of A:**")
    st.latex(symbolic_A_latex(result["A"]) + " = " + symbolic_product_latex(n))

    # ---- STEP 2: Solve every unknown, one equation at a time, in a grid ----
    st.subheader("Step 2 — Solve each equation for the unknowns of L and U")
    latex_eqs = [step_to_latex(s) for s in result["lu_steps"]]
    render_step_grid(latex_eqs, cols_per_row=3)

    st.markdown("**Resulting L and U:**")
    L_str, U_str = numeric_LU_latex(result["L"], result["U"])
    c1, c2 = st.columns(2)
    with c1:
        st.latex("L = " + L_str)
    with c2:
        st.latex("U = " + U_str)

    # ---- STEP 3: Forward substitution ----
    st.subheader("Step 3 — Forward substitution: L y = b")
    for s in result["fwd_steps"]:
        full_eq, solved = fwd_step_to_latex(s, n)
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.latex(full_eq)
        with cc2:
            st.latex(solved)

    y_str = "\\begin{bmatrix}" + " \\\\ ".join(fmt(v) for v in result["y"]) + "\\end{bmatrix}"
    st.latex("y = " + y_str)

    # ---- STEP 4: Backward substitution ----
    st.subheader("Step 4 — Backward substitution: U x = y")
    for s in result["back_steps"]:
        full_eq, solved = back_step_to_latex(s, n)
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.latex(full_eq)
        with cc2:
            st.latex(solved)

    # ---- Final answer ----
    st.subheader("Final Answer")
    x_str = "\\begin{bmatrix}" + " \\\\ ".join(fmt(v) for v in result["x"]) + "\\end{bmatrix}"
    st.latex("x = " + x_str)
    ans_df = pd.DataFrame({"Variable": x_labels, "Value (exact fraction)": [fmt(v) for v in result["x"]]})
    st.dataframe(ans_df, hide_index=True)

    st.subheader("Verification (exact — no rounding)")
    v1, v2 = st.columns(2)
    with v1:
        st.write("A · x =", [fmt(v) for v in result["Ax"]])
    with v2:
        st.write("b =", [fmt(v) for v in result["b"]])
    st.success("Exact match ✔") if result["exact_match"] else st.error("Mismatch — check inputs.")


# ==============================================================================
# APP LAYOUT
# ==============================================================================
st.title("🔢 LU Decomposition Solver — Exact Fractions, Full Matrix Steps")
st.caption(
    "Reproduces the hand-worked method: A = LU set up as matrices, each "
    "unknown solved one equation at a time, forward/backward substitution "
    "shown as full equations. Every value is an exact Fraction — never a "
    "decimal. Works for any 2×2, 3×3, or 4×4 system."
)

tab1, tab2 = st.tabs(
    ["🧮 Manual System Solver (2×2 / 3×3 / 4×4)",
     "🏠 Data Science App: House-Price Regression"]
)

# ------------------------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(
        "Enter A and b as **whole numbers or fractions** (e.g. `3`, `-4`, "
        "`1/2`). Values are never converted to decimals anywhere in this app."
    )

    size = st.radio("Matrix size", [2, 3, 4], horizontal=True, index=1)

    default_A = {
        2: [["4", "3"], ["6", "3"]],
        3: [["2", "3", "-4"], ["1", "2", "-6"], ["4", "1", "1"]],
        4: [["2", "1", "1", "0"], ["4", "3", "3", "1"], ["8", "7", "9", "5"], ["6", "7", "9", "8"]],
    }[size]
    default_b = {2: ["10", "12"], 3: ["3", "1", "7"], 4: ["4", "11", "31", "40"]}[size]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("**Matrix A** (type fractions like `1/2` if needed)")
        A_df = pd.DataFrame(
            default_A,
            columns=[f"C{i+1}" for i in range(size)],
            index=[f"R{i+1}" for i in range(size)],
        )
        A_edited = st.data_editor(A_df, key=f"A_editor_{size}")

    with col_b:
        st.markdown("**Vector b**")
        b_df = pd.DataFrame({"b": default_b}, index=[f"R{i+1}" for i in range(size)])
        b_edited = st.data_editor(b_df, key=f"b_editor_{size}")

    if st.button("🚀 Solve using LU Decomposition (exact fractions)", type="primary"):
        try:
            A_input = [[Fraction(str(v)) for v in row] for row in A_edited.to_numpy()]
            b_input = [Fraction(str(v)) for v in b_edited["b"].to_numpy()]
            result = solve_lu_exact(A_input, b_input)
            render_result(result, size)
        except ValueError as e:
            st.error(str(e))
        except ZeroDivisionError:
            st.error("Division by zero — check for a zero pivot in your matrix.")

# ------------------------------------------------------------------------------
# TAB 2: Data Science application
# ------------------------------------------------------------------------------
with tab2:
    st.markdown(
        """
### Problem
Predict **house price** (lakh PKR) from three features:
**Area** (hundred sq. ft.), **Bedrooms**, and **Age** (years), using the model:

`price = b0 + b1·Area + b2·Bedrooms + b3·Age`

There are **4 unknown coefficients** (`b0, b1, b2, b3`), so exactly **4 houses**
with known prices are enough to pin them down uniquely — 4 equations, 4
unknowns, one exact solution (no fitting/approximation needed). Each house
gives one row of the system:

`b0(1) + b1·Area + b2·Bedrooms + b3·Age = Price`

Stacking all 4 rows gives a square matrix equation **X b = y** (X is 4×4),
solved below with the exact-fraction LU engine and the same matrix-style
step layout as Tab 1. Edit the 4 houses (whole numbers only, so the system
stays exact) to use your own data.
        """
    )

    default_data = pd.DataFrame({
        "Area": [10, 15, 12, 20],
        "Bedrooms": [2, 3, 3, 4],
        "Age": [5, 8, 12, 3],
        "Price": [55, 78, 62, 105],
    })

    data_edited = st.data_editor(default_data, num_rows="fixed", key="house_data_editor")

    if st.button("📈 Solve for Model Coefficients via LU Decomposition (exact fractions)", type="primary"):
        df = data_edited.dropna()
        if len(df) != 4:
            st.error("Need exactly 4 houses (rows) to solve a 4×4 system uniquely.")
        else:
            X = [[1] + [int(df.iloc[i][c]) for c in ["Area", "Bedrooms", "Age"]] for i in range(4)]
            y = [int(v) for v in df["Price"].tolist()]

            Xf = [[Fraction(v) for v in row] for row in X]
            yf = [Fraction(v) for v in y]

            st.markdown("**Design matrix X** (1's column for intercept + the 3 features, one row per house):")
            st.dataframe(pd.DataFrame(X, columns=["Intercept", "Area", "Bedrooms", "Age"]))

            st.markdown("**System to solve:** X · b = y &nbsp;→&nbsp; a plain 4×4 linear system, solved directly (no normal equations needed).")
            result = solve_lu_exact(Xf, yf)
            render_result(result, 4, x_labels=["b0 (Intercept)", "b1 (Area)", "b2 (Bedrooms)", "b3 (Age)"])

            b0, b1, b2, b3 = result["x"]
            st.markdown("### Final Model")
            st.latex(
                f"price = {fmt(b0)} + \\left({fmt(b1)}\\right)\\cdot Area "
                f"+ \\left({fmt(b2)}\\right)\\cdot Bedrooms "
                f"+ \\left({fmt(b3)}\\right)\\cdot Age"
            )

            st.markdown("### Try a Prediction")
            st.caption(
                "Enter exact values for a precise fraction-based prediction, or drag the "
                "sliders for a live view. Submitting exact values also moves the sliders "
                "and chart to match."
            )

            # Sliders keep their own session-state keys so the exact-value form can push
            # values into them (must be set before the slider widgets are created below).
            if "area_slider" not in st.session_state:
                st.session_state.area_slider = 14
            if "bed_slider" not in st.session_state:
                st.session_state.bed_slider = 3
            if "age_slider" not in st.session_state:
                st.session_state.age_slider = 10

            with st.form("exact_prediction_form"):
                t1, t2, t3 = st.columns(3)
                with t1:
                    area_text = st.text_input("Area (exact, e.g. 14 or 1/2)", value=str(st.session_state.area_slider))
                with t2:
                    bed_text = st.text_input("Bedrooms (exact)", value=str(st.session_state.bed_slider))
                with t3:
                    age_text = st.text_input("Age (exact)", value=str(st.session_state.age_slider))
                submitted = st.form_submit_button("Calculate exact prediction")

            if submitted:
                try:
                    area_f = Fraction(area_text)
                    bed_f = Fraction(bed_text)
                    age_f = Fraction(age_text)
                    exact_predicted = b0 + b1 * area_f + b2 * bed_f + b3 * age_f
                    st.success(
                        f"Exact predicted price: **{fmt(exact_predicted)}** lakh PKR "
                        f"(≈ {float(exact_predicted):.2f})"
                    )
                    # Sync the sliders (and therefore the chart below) to match, clamped
                    # to each slider's range since sliders only take whole numbers.
                    st.session_state.area_slider = max(5, min(25, round(float(area_f))))
                    st.session_state.bed_slider = max(1, min(6, round(float(bed_f))))
                    st.session_state.age_slider = max(0, min(25, round(float(age_f))))
                except (ValueError, ZeroDivisionError):
                    st.error("Enter valid numbers/fractions for the prediction inputs.")

            p1, p2, p3 = st.columns(3)
            with p1:
                area_in = st.slider("Area (hundred sq ft)", min_value=5, max_value=25, key="area_slider")
            with p2:
                bed_in = st.slider("Bedrooms", min_value=1, max_value=6, key="bed_slider")
            with p3:
                age_in = st.slider("Age (years)", min_value=0, max_value=25, key="age_slider")

            b0f, b1f, b2f, b3f = float(b0), float(b1), float(b2), float(b3)
            predicted = b0f + b1f * area_in + b2f * bed_in + b3f * age_in
            st.info(f"Slider-based predicted price: **{predicted:.2f}** lakh PKR")

            # ---- Live chart: predicted price vs Area, bedrooms/age held at slider values,
            #      plus the 4 training houses as reference dots ----
            areas = list(range(5, 26))
            price_line = [b0f + b1f * a + b2f * bed_in + b3f * age_in for a in areas]
            line_df = pd.DataFrame({"Area": areas, "Price": price_line})

            train_pts_df = df[["Area", "Price"]].copy()

            current_pt_df = pd.DataFrame({"Area": [area_in], "Price": [predicted]})

            import altair as alt

            line_chart = alt.Chart(line_df).mark_line(color="#1D9E75").encode(
                x=alt.X("Area", title="Area (hundred sq ft)"),
                y=alt.Y("Price", title="Price (lakh PKR)"),
            )
            train_points = alt.Chart(train_pts_df).mark_circle(size=90, color="gray").encode(
                x="Area", y="Price", tooltip=["Area", "Price"]
            )
            current_point = alt.Chart(current_pt_df).mark_circle(size=140, color="#EF9F27").encode(
                x="Area", y="Price"
            )
            st.altair_chart(line_chart + train_points + current_point, use_container_width=True)

            st.caption(
                f"Chart shows predicted price as Area varies, with Bedrooms={bed_in} and "
                f"Age={age_in} held fixed at the slider values. Your current Area ({area_in}) "
                f"sits on this line at ≈{predicted:.2f} lakh PKR."
            )

st.divider()
st.caption(
    "Method: plain Doolittle LU decomposition (A = LU, no row-pivoting — "
    "matches the hand-worked notebook exactly), every unknown solved "
    "symbolically one equation at a time, followed by forward substitution "
    "(Ly = b) and backward substitution (Ux = y). All arithmetic uses "
    "Python's Fraction class — values are exact, never rounded to decimals."
)