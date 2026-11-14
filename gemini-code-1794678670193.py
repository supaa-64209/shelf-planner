import time
import pandas as pd
import plotly.graph_objects as go
import pulp
import streamlit as st

st.set_page_config(page_title="Assortment & Shelf Space Planner", layout="wide")
st.title("Integrated Assortment Planning & Store-Wide Shelf Allocation")
st.caption(
    "Production MIP Solver (PuLP/CBC). Solves for global profit optimality subject to "
    "segment capacities, min/max space, physical contiguity, and separation rules."
)

# ---------------- Default Sample Data ----------------
DEFAULT_CATS = pd.DataFrame(
    [
        # Category, Type, Margin, Demand, Impulse rate, Min space, Max space
        ["Milk", "Fast-mover", 0.30, 60, 1.00, 3.0, 6.0],
        ["Bread", "Fast-mover", 0.40, 50, 1.00, 3.0, 6.0],
        ["Eggs", "Fast-mover", 0.35, 40, 1.00, 2.0, 4.0],
        ["Detergent", "Fast-mover", 0.60, 30, 1.00, 3.0, 6.0],
        ["Chocolate", "Impulse", 2.00, 20, 0.45, 3.0, 6.0],
        ["Candy", "Impulse", 1.50, 15, 0.40, 2.0, 6.0],
        ["Chips", "Impulse", 1.20, 25, 0.30, 2.0, 5.0],
        ["Cookies", "Impulse", 1.60, 18, 0.35, 2.0, 5.0],
        ["Cake Mix", "Impulse", 1.80, 10, 0.35, 2.0, 4.0],
        ["Spices", "Impulse", 1.00, 12, 0.10, 2.0, 4.0],
    ],
    columns=["Category", "Type", "Margin", "Demand", "Impulse rate", "Min space", "Max space"],
)

DEFAULT_SHELVES = pd.DataFrame(
    {
        "Shelf": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "Segment": [1, 2, 3] * 3,
        "Traffic density": [0.95, 0.60, 0.90, 0.85, 0.50, 0.80, 0.40, 0.25, 0.35],
        "Capacity (ft)": [6.0] * 9,
    }
)

DEFAULT_RULES = pd.DataFrame(
    [
        ["Bread", "Detergent", True],
        ["Milk", "Detergent", False],
    ],
    columns=["Category A", "Category B", "Enforce Separate Shelves"],
)

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Scenario Parameters")
impulse_boost = st.sidebar.slider("Impulse potential multiplier", 0.5, 2.0, 1.0, 0.1)
enforce_contiguity = st.sidebar.checkbox(
    "Enforce contiguous placement (no split slots)", value=True
)
integer_facings = st.sidebar.checkbox(
    "Whole-foot increments only (integer space)", value=False
)
time_limit_sec = st.sidebar.slider("Max solver time (seconds)", 5, 60, 15, 5)

st.sidebar.divider()
st.sidebar.header("Data Import (Optional)")
cat_file = st.sidebar.file_uploader("Upload Categories CSV", type=["csv"])
shelf_file = st.sidebar.file_uploader("Upload Shelves CSV", type=["csv"])

base_cats = pd.read_csv(cat_file) if cat_file is not None else DEFAULT_CATS
base_shelves = pd.read_csv(shelf_file) if shelf_file is not None else DEFAULT_SHELVES

# ---------------- Editable Data Tables ----------------
c1, c2 = st.columns([1.3, 1.0])
with c1:
    st.subheader("1. Category Data")
    cats = st.data_editor(base_cats, use_container_width=True, hide_index=True, num_rows="dynamic")
with c2:
    st.subheader("2. Shelf Segments")
    shelves = st.data_editor(base_shelves, use_container_width=True, hide_index=True, num_rows="dynamic")

with st.expander("3. Separation / Incompatibility Rules (Optional)", expanded=True):
    rules = st.data_editor(
        DEFAULT_RULES,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Category A": st.column_config.SelectboxColumn(options=cats["Category"].dropna().tolist()),
            "Category B": st.column_config.SelectboxColumn(options=cats["Category"].dropna().tolist()),
        },
    )

# Clean empty rows if user added blank rows in data_editor
cats = cats.dropna(subset=["Category"]).copy()
shelves = shelves.dropna(subset=["Shelf", "Segment"]).copy()

cats["Max profit"] = (
    cats["Margin"]
    * cats["Demand"]
    * cats["Impulse rate"]
    * cats["Type"].map(lambda t: impulse_boost if t == "Impulse" else 1.0)
)


# ---------------- Exact MIP Solver ----------------
def solve_mip(cats_df, shelves_df, rules_df, contiguity=True, int_space=False, time_limit=15):
    start_time = time.perf_counter()

    categories = cats_df["Category"].tolist()
    shelf_names = sorted(shelves_df["Shelf"].unique().tolist())

    # Lookup dictionaries
    profit_max = dict(zip(cats_df["Category"], cats_df["Max profit"]))
    min_space = dict(zip(cats_df["Category"], cats_df["Min space"]))
    max_space = dict(zip(cats_df["Category"], cats_df["Max space"]))

    slots = [(r["Shelf"], int(r["Segment"])) for _, r in shelves_df.iterrows()]
    traffic = {(r["Shelf"], int(r["Segment"])): float(r["Traffic density"]) for _, r in shelves_df.iterrows()}
    capacity = {(r["Shelf"], int(r["Segment"])): float(r["Capacity (ft)"]) for _, r in shelves_df.iterrows()}

    # Group ordered segments per shelf
    shelf_segments = {
        s: sorted([seg for (sh, seg) in slots if sh == s]) for s in shelf_names
    }

    # Initialize MILP Problem
    prob = pulp.LpProblem("APSA_Shelf_Optimization", pulp.LpMaximize)

    # Decision Variables (using LpVariable.dict for compatibility with PuLP 2.x & 3.x)
    # y[i, s] = 1 if category i is assigned to shelf s
    y = pulp.LpVariable.dict(
        "y", [(i, s) for i in categories for s in shelf_names], cat="Binary"
    )

    # x[i, s, k] = linear feet of category i placed in segment k of shelf s
    x_cat = "Integer" if int_space else "Continuous"
    x = pulp.LpVariable.dict(
        "x", [(i, s, k) for i in categories for (s, k) in slots], lowBound=0, cat=x_cat
    )

    # z[i, s, k] = 1 if category i occupies any space in segment k of shelf s (for contiguity)
    z = pulp.LpVariable.dict(
        "z", [(i, s, k) for i in categories for (s, k) in slots], cat="Binary"
    )

    # Objective Function: Maximize traffic-weighted profit normalized by segment capacity
    prob += pulp.lpSum(
        (profit_max[i] * traffic[(s, k)] / capacity[(s, k)]) * x[(i, s, k)]
        for i in categories
        for (s, k) in slots
    )

    # Constraint 1: Each category assigned to at most 1 shelf
    for i in categories:
        prob += pulp.lpSum(y[(i, s)] for s in shelf_names) <= 1, f"SingleShelf_{i}"

    # Constraint 2: Min and Max space bounds if category i is stocked on shelf s
    for i in categories:
        for s in shelf_names:
            segs = shelf_segments[s]
            total_shelf_space = pulp.lpSum(x[(i, s, k)] for k in segs)
            prob += total_shelf_space >= min_space[i] * y[(i, s)], f"MinSpace_{i}_{s}"
            prob += total_shelf_space <= max_space[i] * y[(i, s)], f"MaxSpace_{i}_{s}"

    # Constraint 3: Segment physical capacity limits
    for (s, k) in slots:
        prob += (
            pulp.lpSum(x[(i, s, k)] for i in categories) <= capacity[(s, k)],
            f"Cap_{s}_{k}",
        )

    # Constraint 4: Link x and z variables
    for i in categories:
        for (s, k) in slots:
            prob += x[(i, s, k)] <= capacity[(s, k)] * z[(i, s, k)], f"LinkXZ_{i}_{s}_{k}"
            prob += z[(i, s, k)] <= y[(i, s)], f"LinkZY_{i}_{s}_{k}"

    # Constraint 5: Contiguity across shelf segments (prevents split allocations like Seg 1 + Seg 3 without Seg 2)
    if contiguity:
        for i in categories:
            for s in shelf_names:
                segs = shelf_segments[s]
                for idx_a in range(len(segs)):
                    for idx_c in range(idx_a + 2, len(segs)):
                        k_a, k_c = segs[idx_a], segs[idx_c]
                        for idx_b in range(idx_a + 1, idx_c):
                            k_b = segs[idx_b]
                            prob += (
                                z[(i, s, k_a)] + z[(i, s, k_c)] - z[(i, s, k_b)] <= 1,
                                f"Contig_{i}_{s}_{k_a}_{k_b}_{k_c}",
                            )

    # Constraint 6: Dynamic Incompatibility / Separation Rules
    if rules_df is not None and not rules_df.empty:
        for r_idx, rule in rules_df.iterrows():
            if rule.get("Enforce Separate Shelves"):
                cat_a, cat_b = rule.get("Category A"), rule.get("Category B")
                if cat_a in categories and cat_b in categories and cat_a != cat_b:
                    for s in shelf_names:
                        prob += (
                            y[(cat_a, s)] + y[(cat_b, s)] <= 1,
                            f"SepRule_{r_idx}_{s}",
                        )

    # Solve with CBC (with time limit so large datasets return cleanly)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    status = pulp.LpStatus[prob.status]

    rows = []
    # Extract solution if Optimal or if a feasible incumbent was found within time limit
    if pulp.value(prob.objective) is not None and status in ("Optimal", "Not Solved"):
        for i in categories:
            for (s, k) in slots:
                val = pulp.value(x[(i, s, k)])
                if val is not None and val > 1e-4:
                    rows.append(
                        dict(
                            Shelf=s,
                            Segment=k,
                            Category=i,
                            Space=round(val, 2),
                            Traffic=traffic[(s, k)],
                            Capacity=capacity[(s, k)],
                            Profit=profit_max[i] * traffic[(s, k)] * (val / capacity[(s, k)]),
                        )
                    )
        if status == "Not Solved" and rows:
            status = "Feasible (Time Limit)"

    return pd.DataFrame(rows), status, elapsed_ms


# ---------------- Run Optimization ----------------
st.subheader("4. Run Optimization")
if st.button("Optimize Assortment & Shelf Allocation", type="primary"):
    with st.spinner("Solving MIP model..."):
        res_df, status, elapsed_ms = solve_mip(
            cats,
            shelves,
            rules,
            contiguity=enforce_contiguity,
            int_space=integer_facings,
            time_limit=time_limit_sec,
        )
        st.session_state["res"] = res_df
        st.session_state["status"] = status
        st.session_state["elapsed_ms"] = elapsed_ms

res = st.session_state.get("res")
status = st.session_state.get("status")
elapsed_ms = st.session_state.get("elapsed_ms")

if status is not None:
    if res is None or len(res) == 0:
        st.error(
            f"Solver Status: **{status}** ({elapsed_ms:.1f} ms). "
            "Check if minimum space requirements or separation rules conflict with available shelf capacity."
        )
    else:
        st.success(f"Solver Status: **{status}** — Solved in **{elapsed_ms:.1f} ms**")

        st.subheader("5. Optimal Planogram & Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Weighted Profit", f"{res['Profit'].sum():,.2f}")
        m2.metric("Categories Stocked", f"{res['Category'].nunique()} / {len(cats)}")
        dropped = sorted(set(cats["Category"]) - set(res["Category"]))
        m3.metric(
            "Not Stocked",
            f"{len(dropped)} categories" if len(dropped) > 5 else (", ".join(dropped) if dropped else "None"),
            help=", ".join(dropped) if dropped else None,
        )

        # Shelf Map Visualization
        fig = go.Figure()
        res["Slot"] = res["Shelf"].astype(str) + "-Seg " + res["Segment"].astype(str)
        all_slots = [
            f"{r['Shelf']}-Seg {int(r['Segment'])} (f={r['Traffic density']}, cap={r['Capacity (ft)']}ft)"
            for _, r in shelves.iterrows()
        ]
        res["Y_Label"] = (
            res["Slot"]
            + " (f="
            + res["Traffic"].astype(str)
            + ", cap="
            + res["Capacity"].astype(str)
            + "ft)"
        )

        for cat, d in res.groupby("Category"):
            fig.add_bar(
                y=d["Y_Label"],
                x=d["Space"],
                name=cat,
                orientation="h",
                text=d.apply(lambda r: f"{r['Category']} ({r['Space']}ft)", axis=1),
                textposition="inside",
            )

        chart_height = max(480, len(all_slots) * 22)
        fig.update_layout(
            barmode="stack",
            height=chart_height,
            xaxis_title="Linear Space Used (ft)",
            yaxis=dict(
                categoryorder="array",
                categoryarray=all_slots,
                autorange="reversed",
            ),
            title="Optimal Store Shelf Map (Stacked by Segment)",
            legend_title="Category",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Category Breakdown Table
        by_cat = (
            res.groupby("Category")
            .agg(
                Shelf=("Shelf", "first"),
                Segments=("Segment", lambda s: ", ".join(map(str, sorted(s)))),
                Total_Space_ft=("Space", "sum"),
                Profit=("Profit", "sum"),
            )
            .reset_index()
        )
        st.dataframe(
            by_cat.sort_values("Profit", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download Optimal Allocation (CSV)",
            res.to_csv(index=False),
            "optimal_allocation.csv",
            mime="text/csv",
        )