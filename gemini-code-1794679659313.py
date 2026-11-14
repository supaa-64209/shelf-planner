import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
import streamlit as st

st.set_page_config(page_title="Assortment & Shelf Space Planner", layout="wide")
st.title("Integrated Assortment Planning & Store-Wide Shelf Allocation")
st.caption(
    "Production MIP Solver (SciPy / HiGHS). Solves for global profit optimality subject to "
    "segment capacities, min/max space, physical contiguity, and separation rules."
)

# ---------------- Default Sample Data ----------------
DEFAULT_CATS = pd.DataFrame(
    [
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
        ["Bread", "Laundry Detergents", True],
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


def load_csv_safe(uploaded_file, fallback_df, required_cols):
    if uploaded_file is None:
        return fallback_df.copy()
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.sidebar.error(f"Missing columns in uploaded CSV: {', '.join(missing)}")
            return fallback_df.copy()
        return df
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")
        return fallback_df.copy()


base_cats = load_csv_safe(
    cat_file,
    DEFAULT_CATS,
    ["Category", "Type", "Margin", "Demand", "Impulse rate", "Min space", "Max space"],
)
base_shelves = load_csv_safe(
    shelf_file,
    DEFAULT_SHELVES,
    ["Shelf", "Segment", "Traffic density", "Capacity (ft)"],
)

# ---------------- Editable Data Tables ----------------
c1, c2 = st.columns([1.3, 1.0])
with c1:
    st.subheader("1. Category Data")
    cats = st.data_editor(
        base_cats, use_container_width=True, hide_index=True, num_rows="dynamic"
    )
with c2:
    st.subheader("2. Shelf Segments")
    shelves = st.data_editor(
        base_shelves, use_container_width=True, hide_index=True, num_rows="dynamic"
    )

cat_options = cats["Category"].dropna().astype(str).str.strip().unique().tolist()
with st.expander("3. Separation / Incompatibility Rules (Optional)", expanded=True):
    rules = st.data_editor(
        DEFAULT_RULES,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Category A": st.column_config.SelectboxColumn(options=cat_options),
            "Category B": st.column_config.SelectboxColumn(options=cat_options),
        },
    )

# Clean & sanitize tables before solving
cats = cats.dropna(subset=["Category"]).copy()
cats["Category"] = cats["Category"].astype(str).str.strip()
cats = cats[cats["Category"] != ""].drop_duplicates(subset=["Category"], keep="first").reset_index(drop=True)

for col, default in [
    ("Margin", 1.0),
    ("Demand", 10.0),
    ("Impulse rate", 1.0),
    ("Min space", 1.0),
    ("Max space", 6.0),
]:
    cats[col] = pd.to_numeric(cats[col], errors="coerce").fillna(default)

cats["Type"] = cats["Type"].fillna("Impulse").astype(str).str.strip()
cats["Max profit"] = (
    cats["Margin"]
    * cats["Demand"]
    * cats["Impulse rate"]
    * cats["Type"].map(lambda t: impulse_boost if t == "Impulse" else 1.0)
)

shelves = shelves.dropna(subset=["Shelf", "Segment"]).copy()
shelves["Shelf"] = shelves["Shelf"].astype(str).str.strip()
shelves = shelves[shelves["Shelf"] != ""].copy()
shelves["Segment"] = pd.to_numeric(shelves["Segment"], errors="coerce").fillna(1).astype(int)
shelves["Traffic density"] = pd.to_numeric(shelves["Traffic density"], errors="coerce").fillna(0.5)
shelves["Capacity (ft)"] = pd.to_numeric(shelves["Capacity (ft)"], errors="coerce").fillna(6.0)
shelves = shelves.drop_duplicates(subset=["Shelf", "Segment"], keep="first").reset_index(drop=True)


# ---------------- Exact MIP Solver (SciPy / HiGHS) ----------------
def solve_mip(cats_df, shelves_df, rules_df, contiguity=True, int_space=False, time_limit=15):
    start_time = time.perf_counter()

    if cats_df.empty or shelves_df.empty:
        return pd.DataFrame(), "Empty Input Data", 0.0

    categories = cats_df["Category"].tolist()
    shelf_names = sorted(shelves_df["Shelf"].unique().tolist())
    slots = [(r["Shelf"], int(r["Segment"])) for _, r in shelves_df.iterrows()]

    n_cats = len(categories)
    n_shelves = len(shelf_names)
    n_slots = len(slots)

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    shelf_to_idx = {s: i for i, s in enumerate(shelf_names)}

    profit_max = cats_df["Max profit"].to_numpy(dtype=float)
    min_space = cats_df["Min space"].to_numpy(dtype=float)
    max_space = np.maximum(min_space, cats_df["Max space"].to_numpy(dtype=float))

    traffic = shelves_df["Traffic density"].to_numpy(dtype=float)
    capacity = shelves_df["Capacity (ft)"].to_numpy(dtype=float)

    shelf_slots = {s_idx: [] for s_idx in range(n_shelves)}
    for k_idx, (sh, seg) in enumerate(slots):
        shelf_slots[shelf_to_idx[sh]].append((seg, k_idx))
    for s_idx in range(n_shelves):
        shelf_slots[s_idx].sort(key=lambda t: t[0])

    # Variable layout:
    # y_{i, s}: 0 .. n_cats*n_shelves - 1 (binary)
    # x_{i, k}: n_cats*n_shelves .. n_cats*n_shelves + n_cats*n_slots - 1 (continuous/integer)
    # z_{i, k}: n_cats*n_shelves + n_cats*n_slots .. end (binary)
    offset_y = 0
    offset_x = n_cats * n_shelves
    offset_z = n_cats * n_shelves + n_cats * n_slots
    num_vars = n_cats * n_shelves + 2 * n_cats * n_slots

    def idx_y(i, s):
        return offset_y + i * n_shelves + s

    def idx_x(i, k):
        return offset_x + i * n_slots + k

    def idx_z(i, k):
        return offset_z + i * n_slots + k

    # Objective: minimize negative profit
    c = np.zeros(num_vars, dtype=float)
    for i in range(n_cats):
        for k in range(n_slots):
            if capacity[k] > 0:
                c[idx_x(i, k)] = -(profit_max[i] * traffic[k] / capacity[k])

    lb = np.zeros(num_vars, dtype=float)
    ub = np.ones(num_vars, dtype=float)
    integrality = np.ones(num_vars, dtype=int)

    for i in range(n_cats):
        for k in range(n_slots):
            ub[idx_x(i, k)] = min(capacity[k], max_space[i])
            integrality[idx_x(i, k)] = 1 if int_space else 0

    rows_a = []
    lhs_list = []
    rhs_list = []

    def add_row(entries, lhs, rhs):
        rows_a.append(entries)
        lhs_list.append(lhs)
        rhs_list.append(rhs)

    # 1. Single-shelf assignment: sum_s y_{i,s} <= 1
    for i in range(n_cats):
        add_row([(idx_y(i, s), 1.0) for s in range(n_shelves)], -np.inf, 1.0)

    # 2. Min & Max space bounds per shelf
    for i in range(n_cats):
        for s in range(n_shelves):
            k_indices = [k_idx for _, k_idx in shelf_slots[s]]
            if not k_indices:
                add_row([(idx_y(i, s), 1.0)], -np.inf, 0.0)
                continue
            add_row(
                [(idx_x(i, k), 1.0) for k in k_indices] + [(idx_y(i, s), -min_space[i])],
                0.0,
                np.inf,
            )
            add_row(
                [(idx_x(i, k), 1.0) for k in k_indices] + [(idx_y(i, s), -max_space[i])],
                -np.inf,
                0.0,
            )

    # 3. Segment physical capacity limits
    for k in range(n_slots):
        add_row([(idx_x(i, k), 1.0) for i in range(n_cats)], -np.inf, capacity[k])

    # 4. Link x & z, and z & y
    for k in range(n_slots):
        s = shelf_to_idx[slots[k][0]]
        for i in range(n_cats):
            add_row([(idx_x(i, k), 1.0), (idx_z(i, k), -capacity[k])], -np.inf, 0.0)
            add_row([(idx_z(i, k), 1.0), (idx_y(i, s), -1.0)], -np.inf, 0.0)

    # 5. Contiguity across shelf segments
    if contiguity:
        for i in range(n_cats):
            for s in range(n_shelves):
                k_list = [k_idx for _, k_idx in shelf_slots[s]]
                m = len(k_list)
                for a in range(m):
                    for c_pos in range(a + 2, m):
                        for b in range(a + 1, c_pos):
                            add_row(
                                [
                                    (idx_z(i, k_list[a]), 1.0),
                                    (idx_z(i, k_list[c_pos]), 1.0),
                                    (idx_z(i, k_list[b]), -1.0),
                                ],
                                -np.inf,
                                1.0,
                            )

    # 6. Separation rules
    if rules_df is not None and not rules_df.empty:
        for _, rule in rules_df.iterrows():
            enforce = rule.get("Enforce Separate Shelves", False)
            if pd.notna(enforce) and bool(enforce):
                ca = str(rule.get("Category A", "")).strip()
                cb = str(rule.get("Category B", "")).strip()
                if ca in cat_to_idx and cb in cat_to_idx and ca != cb:
                    ia, ib = cat_to_idx[ca], cat_to_idx[cb]
                    for s in range(n_shelves):
                        add_row([(idx_y(ia, s), 1.0), (idx_y(ib, s), 1.0)], -np.inf, 1.0)

    a_mat = lil_matrix((len(rows_a), num_vars), dtype=float)
    for r_idx, entries in enumerate(rows_a):
        for col_idx, val in entries:
            a_mat[r_idx, col_idx] = val
    a_mat = a_mat.tocsc()

    constraints = LinearConstraint(a_mat, lb=np.array(lhs_list), ub=np.array(rhs_list))
    bounds = Bounds(lb=lb, ub=ub)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.005},
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    status_map = {
        0: "Optimal",
        1: "Time Limit",
        2: "Infeasible",
        3: "Unbounded",
        4: "Solver Error",
    }
    status = status_map.get(res.status, f"Status {res.status}")

    out_rows = []
    if res.x is not None:
        for i in range(n_cats):
            for k in range(n_slots):
                val = float(res.x[idx_x(i, k)])
                if val > 1e-4:
                    sh, seg = slots[k]
                    out_rows.append(
                        dict(
                            Shelf=sh,
                            Segment=seg,
                            Category=categories[i],
                            Space=round(val, 2),
                            Traffic=traffic[k],
                            Capacity=capacity[k],
                            Profit=profit_max[i] * traffic[k] * (val / capacity[k]),
                        )
                    )
        if res.status == 1 and out_rows:
            status = "Feasible (Time Limit)"

    return pd.DataFrame(out_rows), status, elapsed_ms


# ---------------- Run Optimization ----------------
st.subheader("4. Run Optimization")
if st.button("Optimize Assortment & Shelf Allocation", type="primary"):
    with st.spinner("Solving MIP model with HiGHS..."):
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
        plot_df = res.copy()
        plot_df["Slot"] = plot_df["Shelf"].astype(str) + "-Seg " + plot_df["Segment"].astype(str)
        all_slots = [
            f"{r['Shelf']}-Seg {int(r['Segment'])} (f={r['Traffic density']}, cap={r['Capacity (ft)']}ft)"
            for _, r in shelves.iterrows()
        ]
        plot_df["Y_Label"] = (
            plot_df["Slot"]
            + " (f="
            + plot_df["Traffic"].astype(str)
            + ", cap="
            + plot_df["Capacity"].astype(str)
            + "ft)"
        )

        for cat, d in plot_df.groupby("Category"):
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