import json
import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Assortment & Shelf Space Planner", layout="wide")
st.title("Integrated Assortment Planning & Store-Wide Shelf Allocation")
st.caption(
    "APSA Optimization Engine (Flamand et al., 2018). Solves joint assortment selection and "
    "shelf space allocation subject to segment capacities, min/max space, contiguity, and separation rules."
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
time_limit_sec = st.sidebar.slider("Max optimization time (seconds)", 2, 20, 5, 1)

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

# ---------------- Clean & Sanitize Inputs ----------------
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


# ---------------- Built-in Zero-Dependency APSA Solver ----------------
def solve_apsa_builtin(cats_df, shelves_df, rules_df, contiguity=True, int_space=False, time_limit=5.0):
    start_time = time.perf_counter()

    if cats_df.empty or shelves_df.empty:
        return pd.DataFrame(), "Empty Input Data", 0.0

    categories = cats_df["Category"].tolist()
    n_cats = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    phi = cats_df["Max profit"].to_numpy(dtype=float)
    l_min = cats_df["Min space"].to_numpy(dtype=float)
    u_max = np.maximum(l_min, cats_df["Max space"].to_numpy(dtype=float))
    if int_space:
        l_min = np.ceil(l_min)
        u_max = np.maximum(l_min, np.floor(u_max))

    incompat = {i: set() for i in range(n_cats)}
    if rules_df is not None and not rules_df.empty:
        for _, rule in rules_df.iterrows():
            enforce = rule.get("Enforce Separate Shelves", False)
            if pd.notna(enforce) and bool(enforce):
                ca = str(rule.get("Category A", "")).strip()
                cb = str(rule.get("Category B", "")).strip()
                if ca in cat_to_idx and cb in cat_to_idx and ca != cb:
                    ia, ib = cat_to_idx[ca], cat_to_idx[cb]
                    incompat[ia].add(ib)
                    incompat[ib].add(ia)

    shelf_names = sorted(shelves_df["Shelf"].unique().tolist())
    n_shelves = len(shelf_names)
    shelf_segs = []
    shelf_cap = np.zeros(n_shelves, dtype=float)
    shelf_avg_rate = np.zeros(n_shelves, dtype=float)

    for s_idx, sh in enumerate(shelf_names):
        sub = shelves_df[shelves_df["Shelf"] == sh].sort_values("Segment")
        segs = []
        tot_c = 0.0
        tot_f = 0.0
        for _, r in sub.iterrows():
            c_val = float(r["Capacity (ft)"])
            if int_space:
                c_val = np.floor(c_val)
            f_val = float(r["Traffic density"])
            rate = (f_val / c_val) if c_val > 0 else 0.0
            segs.append({"seg": int(r["Segment"]), "traffic": f_val, "cap": c_val, "rate": rate})
            tot_c += c_val
            tot_f += f_val
        shelf_segs.append(segs)
        shelf_cap[s_idx] = tot_c
        shelf_avg_rate[s_idx] = (tot_f / tot_c) if tot_c > 0 else 0.0

    def evaluate_shelf(s_idx, cat_indices, build_rows=False):
        if not cat_indices:
            return 0.0, []
        segs = shelf_segs[s_idx]
        cap_s = shelf_cap[s_idx]
        total_min = sum(l_min[i] for i in cat_indices)
        if total_min > cap_s + 1e-6:
            return -1.0, []

        ordered_cats = sorted(cat_indices, key=lambda i: phi[i], reverse=True)
        rem_cap = cap_s - total_min
        alloc_space = {i: l_min[i] for i in ordered_cats}
        for i in ordered_cats:
            if rem_cap <= 1e-6:
                break
            add = min(u_max[i] - alloc_space[i], rem_cap)
            if int_space:
                add = np.floor(add)
            if add > 0:
                alloc_space[i] += add
                rem_cap -= add

        if not contiguity:
            seg_order = sorted(range(len(segs)), key=lambda idx: segs[idx]["rate"], reverse=True)
            seg_rem = [segs[idx]["cap"] for idx in seg_order]
            profit = 0.0
            alloc_rows = []
            for i in ordered_cats:
                need = alloc_space[i]
                for pos, seg_idx in enumerate(seg_order):
                    if need <= 1e-6:
                        break
                    if seg_rem[pos] <= 1e-6:
                        continue
                    take = min(need, seg_rem[pos])
                    if int_space:
                        take = np.floor(take)
                    if take > 1e-6:
                        seg_rem[pos] -= take
                        need -= take
                        p_val = phi[i] * segs[seg_idx]["rate"] * take
                        profit += p_val
                        if build_rows:
                            alloc_rows.append(
                                (s_idx, segs[seg_idx]["seg"], i, take, segs[seg_idx]["traffic"], segs[seg_idx]["cap"], p_val)
                            )
            return profit, alloc_rows
        else:
            left_list, right_list = [], []
            for rank, i in enumerate(ordered_cats):
                if rank % 2 == 0:
                    left_list.append(i)
                else:
                    right_list.insert(0, i)
            seq_cats = left_list + right_list

            seg_rem = [s["cap"] for s in segs]
            cur_seg = 0
            profit = 0.0
            alloc_rows = []
            for i in seq_cats:
                need = alloc_space[i]
                while need > 1e-6 and cur_seg < len(segs):
                    if seg_rem[cur_seg] <= 1e-6:
                        cur_seg += 1
                        continue
                    take = min(need, seg_rem[cur_seg])
                    if int_space:
                        take = np.floor(take)
                    if take <= 1e-6:
                        cur_seg += 1
                        continue
                    seg_rem[cur_seg] -= take
                    need -= take
                    p_val = phi[i] * segs[cur_seg]["rate"] * take
                    profit += p_val
                    if build_rows:
                        alloc_rows.append(
                            (s_idx, segs[cur_seg]["seg"], i, take, segs[cur_seg]["traffic"], segs[cur_seg]["cap"], p_val)
                        )
            return profit, alloc_rows

    def is_compatible(cat_i, current_set):
        return len(incompat[cat_i].intersection(current_set)) == 0

    # Phase 1: Sequential Shelf Construction (Algorithm 1)
    shelf_order = list(np.argsort(-shelf_avg_rate))
    cat_order = list(np.argsort(-phi))
    assigned = [set() for _ in range(n_shelves)]
    unassigned = set(range(n_cats))

    for s_idx in shelf_order:
        cap_rem = shelf_cap[s_idx]
        for i in cat_order:
            if i not in unassigned:
                continue
            if l_min[i] <= cap_rem + 1e-6 and is_compatible(i, assigned[s_idx]):
                assigned[s_idx].add(i)
                unassigned.remove(i)
                cap_rem -= l_min[i]

    shelf_profits = np.array([evaluate_shelf(s, assigned[s])[0] for s in range(n_shelves)], dtype=float)

    # Phase 2: Multi-Shelf Re-Optimization Local Search (Algorithm 2)
    deadline = start_time + float(time_limit)
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False

        # 1. Cross-shelf swaps
        for idx1 in range(n_shelves):
            s1 = shelf_order[idx1]
            for idx2 in range(idx1 + 1, n_shelves):
                s2 = shelf_order[idx2]
                if shelf_avg_rate[s1] <= shelf_avg_rate[s2] + 1e-6:
                    continue
                for c1 in list(assigned[s1]):
                    for c2 in list(assigned[s2]):
                        if phi[c2] <= phi[c1] + 1e-6:
                            continue
                        rem1 = assigned[s1] - {c1}
                        rem2 = assigned[s2] - {c2}
                        if not (is_compatible(c2, rem1) and is_compatible(c1, rem2)):
                            continue
                        p1, _ = evaluate_shelf(s1, rem1 | {c2})
                        if p1 < 0:
                            continue
                        p2, _ = evaluate_shelf(s2, rem2 | {c1})
                        if p2 < 0:
                            continue
                        if (p1 + p2) > (shelf_profits[s1] + shelf_profits[s2]) + 1e-4:
                            assigned[s1] = rem1 | {c2}
                            assigned[s2] = rem2 | {c1}
                            shelf_profits[s1] = p1
                            shelf_profits[s2] = p2
                            improved = True
                            break
                    if improved:
                        break

        # 2. Insert unassigned or replace lower-value stocked items
        for u_cat in sorted(list(unassigned), key=lambda i: phi[i], reverse=True):
            best_gain = 1e-4
            best_move = None
            for s_idx in shelf_order:
                if is_compatible(u_cat, assigned[s_idx]):
                    cand = assigned[s_idx] | {u_cat}
                    p_new, _ = evaluate_shelf(s_idx, cand)
                    if p_new > shelf_profits[s_idx] + best_gain:
                        best_gain = p_new - shelf_profits[s_idx]
                        best_move = (s_idx, cand, None, p_new)
                for old_cat in list(assigned[s_idx]):
                    if phi[u_cat] <= phi[old_cat] + 1e-6:
                        continue
                    rem = assigned[s_idx] - {old_cat}
                    if is_compatible(u_cat, rem):
                        cand = rem | {u_cat}
                        p_new, _ = evaluate_shelf(s_idx, cand)
                        if p_new > shelf_profits[s_idx] + best_gain:
                            best_gain = p_new - shelf_profits[s_idx]
                            best_move = (s_idx, cand, old_cat, p_new)
            if best_move is not None:
                s_idx, cand, old_cat, p_new = best_move
                assigned[s_idx] = cand
                unassigned.remove(u_cat)
                if old_cat is not None:
                    unassigned.add(old_cat)
                shelf_profits[s_idx] = p_new
                improved = True

        # 3. Consolidate space
        for s_idx in shelf_order:
            if len(assigned[s_idx]) <= 1:
                continue
            lowest = min(assigned[s_idx], key=lambda i: phi[i])
            cand = assigned[s_idx] - {lowest}
            p_new, _ = evaluate_shelf(s_idx, cand)
            if p_new > shelf_profits[s_idx] + 1e-4:
                assigned[s_idx] = cand
                unassigned.add(lowest)
                shelf_profits[s_idx] = p_new
                improved = True

    out_rows = []
    for s_idx in range(n_shelves):
        _, rows = evaluate_shelf(s_idx, assigned[s_idx], build_rows=True)
        for (_, seg, cat_i, sp, tr, cap_k, pr) in rows:
            out_rows.append(
                dict(
                    Shelf=shelf_names[s_idx],
                    Segment=int(seg),
                    Category=categories[cat_i],
                    Space=round(float(sp), 2),
                    Traffic=float(tr),
                    Capacity=float(cap_k),
                    Profit=round(float(pr), 2),
                )
            )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    status = "Optimal" if out_rows else "Infeasible"
    return pd.DataFrame(out_rows), status, elapsed_ms


# ---------------- Run Optimization ----------------
st.subheader("4. Run Optimization")
if st.button("Optimize Assortment & Shelf Allocation", type="primary"):
    with st.spinner("Optimizing assortment and store-wide shelf allocation..."):
        res_df, status, elapsed_ms = solve_apsa_builtin(
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
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Weighted Profit", f"{res['Profit'].sum():,.2f}")
        m2.metric("Categories Stocked", f"{res['Category