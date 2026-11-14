import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="APSA Retail Simulator", page_icon="🛒", layout="wide")

# ------------------------------------------------------------
# APSA-inspired interactive prototype
# Research basis: Flamand et al. (2018), Omega 81, 134-149.
# This is a transparent greedy simulation, NOT the paper's exact MIP.
# ------------------------------------------------------------

PAPER_DATA = [
    ("Bread", 9530, 0.12, 0.05),
    ("Cigarettes", 9315, 0.10, 0.03),
    ("Soda", 7657, 0.16, 0.12),
    ("Unpacked Meat", 6348, 0.22, 0.08),
    ("Coffee", 6212, 0.24, 0.10),
    ("Milk", 6144, 0.11, 0.04),
    ("Juice", 5969, 0.20, 0.15),
    ("Cheese", 6009, 0.23, 0.12),
    ("Canned Vegetables", 4620, 0.18, 0.08),
    ("Vegetables", 4641, 0.19, 0.06),
    ("Packed Cheese", 4500, 0.24, 0.12),
    ("Specialty Cheese", 3907, 0.29, 0.25),
    ("Ice", 2322, 0.20, 0.12),
    ("Dinners", 4529, 0.25, 0.18),
    ("Pizza", 2384, 0.24, 0.18),
    ("Household Cleaner", 3413, 0.21, 0.10),
    ("Salad Dressing", 2241, 0.28, 0.20),
    ("Pasta Sauce", 3203, 0.26, 0.15),
    ("Canned Fruit", 3556, 0.22, 0.12),
    ("Chocolate", 890, 0.35, 0.45),
    ("Cookies", 786, 0.32, 0.40),
    ("Chips", 822, 0.30, 0.35),
    ("Ice Cream", 2100, 0.34, 0.45),
    ("Candy", 146, 0.38, 0.50),
    ("Cake Mixes", 1378, 0.31, 0.38),
    ("Yogurt", 1324, 0.23, 0.18),
    ("Eggs", 572, 0.14, 0.05),
    ("Tea", 192, 0.30, 0.30),
    ("Nuts", 706, 0.32, 0.32),
    ("Baby Food", 438, 0.25, 0.08),
]

def default_products():
    df = pd.DataFrame(PAPER_DATA, columns=["Category", "Demand", "Profit_Margin", "Impulse_Potential"])
    # Space bounds are illustrative prototype inputs, not values reported in the paper.
    df["Min_Space"] = 0.5
    df["Max_Space"] = 2.0
    df["Target_Space"] = 1.0
    return df

def make_segments(n_shelves, n_segments, capacity, traffic_seed=7):
    # A repeatable mix of high-, medium- and low-traffic positions.
    patterns = [0.95, 0.85, 0.65, 0.55, 0.35, 0.25]
    rows = []
    for shelf in range(1, n_shelves + 1):
        for seg in range(1, n_segments + 1):
            base = patterns[(seg - 1 + (shelf - 1) * 2) % len(patterns)]
            rows.append({
                "Shelf": f"Shelf {shelf}",
                "Segment": f"Seg {seg}",
                "Capacity": float(capacity),
                "Traffic": float(base)
            })
    return pd.DataFrame(rows)

def simulate(products, n_shelves, n_segments, segment_capacity,
             demand_multiplier, margin_multiplier, impulse_multiplier,
             fast_pct, space_factor):
    df = products.copy()
    df["Demand_Adjusted"] = pd.to_numeric(df["Demand"], errors="coerce").fillna(0) * demand_multiplier
    df["Profit_Margin"] = pd.to_numeric(df["Profit_Margin"], errors="coerce").fillna(0).clip(0, 1) * margin_multiplier
    df["Impulse_Potential"] = pd.to_numeric(df["Impulse_Potential"], errors="coerce").fillna(0).clip(0, 1) * impulse_multiplier
    df["Min_Space"] = pd.to_numeric(df["Min_Space"], errors="coerce").fillna(0.5).clip(lower=0.1)
    df["Max_Space"] = pd.to_numeric(df["Max_Space"], errors="coerce").fillna(2).clip(lower=0.1)
    df["Target_Space"] = pd.to_numeric(df["Target_Space"], errors="coerce").fillna(1).clip(lower=0.1)
    df["Max_Space"] = np.maximum(df["Max_Space"], df["Min_Space"])
    df["Target_Space"] = np.minimum(np.maximum(df["Target_Space"], df["Min_Space"]), df["Max_Space"])

    # Paper identifies fast movers using a demand-based 80/20 logic.
    # Here the user controls the share classified as fast movers.
    count_fast = max(1, int(np.ceil(len(df) * fast_pct / 100))) if len(df) else 0
    fast_indices = set(df.nlargest(count_fast, "Demand_Adjusted").index)
    df["Type"] = ["Fast mover" if i in fast_indices else "Impulse / slow mover" for i in df.index]
    df["Gamma_Used"] = np.where(df["Type"] == "Fast mover", 1.0, df["Impulse_Potential"].clip(0, 1))

    # Maximum potential reward follows the paper's Phi_j = rho_j * v_j * gamma_j.
    df["Potential_Profit"] = df["Profit_Margin"] * df["Demand_Adjusted"] * df["Gamma_Used"]
    df["Space_Needed"] = np.minimum(
        df["Max_Space"],
        np.maximum(df["Min_Space"], df["Target_Space"] * space_factor)
    )
    df["Priority"] = df["Potential_Profit"] / df["Min_Space"].clip(lower=0.1)
    df = df.sort_values("Priority", ascending=False).reset_index(drop=True)

    segments = make_segments(n_shelves, n_segments, segment_capacity)
    segments["Remaining"] = segments["Capacity"]
    segments["Allocated"] = 0.0
    segments["Reward"] = 0.0
    allocations = []

    total_capacity = n_shelves * n_segments * segment_capacity
    used_capacity = 0.0

    for _, product in df.iterrows():
        required = float(product["Space_Needed"])
        minimum = float(product["Min_Space"])
        if used_capacity + minimum > total_capacity + 1e-9:
            continue

        # Choose one shelf for the entire category; allocate across that shelf's
        # most attractive available segments, preserving the paper's store-wide logic.
        shelf_options = []
        for shelf_name in segments["Shelf"].unique():
            shelf_seg = segments[segments["Shelf"] == shelf_name].copy()
            available = float(shelf_seg["Remaining"].sum())
            if available + 1e-9 >= minimum:
                weighted_attractiveness = (
                    (shelf_seg["Traffic"] * shelf_seg["Remaining"]).sum()
                    / max(shelf_seg["Remaining"].sum(), 1e-9)
                )
                shelf_options.append((weighted_attractiveness, available, shelf_name))
        if not shelf_options:
            continue
        shelf_options.sort(reverse=True)
        chosen_shelf = shelf_options[0][2]
        shelf_idx = segments.index[segments["Shelf"] == chosen_shelf].tolist()
        shelf_idx = sorted(shelf_idx, key=lambda ix: segments.loc[ix, "Traffic"], reverse=True)

        remaining_to_place = min(required, sum(segments.loc[ix, "Remaining"] for ix in shelf_idx))
        if remaining_to_place + 1e-9 < minimum:
            continue

        # Allocate the category to attractive segments on its chosen shelf.
        placed = 0.0
        for ix in shelf_idx:
            if remaining_to_place <= 1e-9:
                break
            available = float(segments.loc[ix, "Remaining"])
            chunk = min(available, remaining_to_place)
            if chunk <= 1e-9:
                continue
            segments.loc[ix, "Remaining"] -= chunk
            segments.loc[ix, "Allocated"] += chunk
            segments.loc[ix, "Reward"] += (
                float(product["Potential_Profit"]) *
                float(segments.loc[ix, "Traffic"]) *
                chunk / max(float(product["Space_Needed"]), 1e-9)
            )
            allocations.append({
                "Category": product["Category"],
                "Type": product["Type"],
                "Shelf": chosen_shelf,
                "Segment": segments.loc[ix, "Segment"],
                "Space": chunk,
                "Traffic": segments.loc[ix, "Traffic"],
                "Demand": product["Demand_Adjusted"],
                "Profit_Margin": product["Profit_Margin"],
                "Impulse_Potential": product["Impulse_Potential"],
                "Estimated_Contribution": (
                    float(product["Potential_Profit"]) *
                    float(segments.loc[ix, "Traffic"]) *
                    chunk / max(float(product["Space_Needed"]), 1e-9)
                )
            })
            placed += chunk
            remaining_to_place -= chunk

        if placed + 1e-9 >= minimum:
            used_capacity += placed

    alloc = pd.DataFrame(allocations)
    if alloc.empty:
        selected = pd.DataFrame(columns=["Category", "Type", "Demand", "Profit_Margin", "Impulse_Potential", "Space"])
        total_reward = 0.0
    else:
        selected = alloc.groupby(["Category", "Type", "Demand", "Profit_Margin", "Impulse_Potential"], as_index=False).agg(
            Space=("Space", "sum"),
            Estimated_Contribution=("Estimated_Contribution", "sum")
        )
        total_reward = float(alloc["Estimated_Contribution"].sum())

    segments["Utilization_%"] = 100 * segments["Allocated"] / segments["Capacity"]
    return df, alloc, selected, segments, total_reward, total_capacity

st.title("🛒 Integrated Assortment & Shelf Space Allocation")
st.caption("Advanced Operations Research | APSA-inspired supermarket decision-support simulation")
st.info(
    "Research basis: Flamand et al. (2018), Omega 81, 134–149. "
    "This prototype uses a transparent greedy allocation heuristic to demonstrate the paper's "
    "trade-offs. It is not the original mixed-integer programming solver."
)

with st.sidebar:
    st.header("1. Store configuration")
    n_shelves = st.slider("Number of shelves", 2, 12, 5)
    n_segments = st.slider("Segments per shelf", 2, 8, 4)
    segment_capacity = st.slider("Capacity per segment (feet)", 1.0, 10.0, 4.0, 0.5)
    st.divider()
    st.header("2. Scenario assumptions")
    demand_multiplier = st.slider("Demand multiplier", 0.5, 2.0, 1.0, 0.05)
    margin_multiplier = st.slider("Profit margin multiplier", 0.5, 1.5, 1.0, 0.05)
    impulse_multiplier = st.slider("Impulse potential multiplier", 0.5, 1.5, 1.0, 0.05)
    fast_pct = st.slider("Categories classified as fast movers (%)", 10, 50, 20, 5)
    space_factor = st.slider("Space requirement multiplier", 0.5, 1.5, 1.0, 0.05)
    run = st.button("▶ Run simulation", type="primary", use_container_width=True)

st.subheader("3. Product category input data")
st.write(
    "Edit demand, margin, impulse potential and shelf-space bounds below. "
    "The demand values are taken from Table 2 of the research paper for the listed categories. "
    "Margins and shelf-space bounds are illustrative inputs because the paper's full category-level "
    "margin and space data are not available in its displayed demand table."
)

if "products" not in st.session_state:
    st.session_state.products = default_products()

edited = st.data_editor(
    st.session_state.products,
    num_rows="dynamic",
    use_container_width=True,
    key="product_editor",
    column_config={
        "Category": st.column_config.TextColumn("Product category", required=True),
        "Demand": st.column_config.NumberColumn("Expected demand (units)", min_value=0, step=10),
        "Profit_Margin": st.column_config.NumberColumn("Profit margin (0–1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
        "Impulse_Potential": st.column_config.NumberColumn("Impulse potential (0–1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
        "Min_Space": st.column_config.NumberColumn("Minimum space (ft)", min_value=0.1, step=0.1),
        "Max_Space": st.column_config.NumberColumn("Maximum space (ft)", min_value=0.1, step=0.1),
        "Target_Space": st.column_config.NumberColumn("Target space (ft)", min_value=0.1, step=0.1),
    }
)
st.session_state.products = edited

if run or "simulation_result" not in st.session_state:
    try:
        st.session_state.simulation_result = simulate(
            edited.dropna(subset=["Category"]),
            n_shelves, n_segments, segment_capacity,
            demand_multiplier, margin_multiplier, impulse_multiplier,
            fast_pct, space_factor
        )
    except Exception as exc:
        st.error(f"Could not run simulation: {exc}")
        st.stop()

ranked, allocations, selected, segments, total_reward, total_capacity = st.session_state.simulation_result

st.divider()
st.subheader("4. Simulation results")

used_space = float(segments["Allocated"].sum())
utilization = 100 * used_space / total_capacity if total_capacity else 0
selected_count = int(selected["Category"].nunique()) if not selected.empty else 0
fast_count = int((selected["Type"] == "Fast mover").sum()) if not selected.empty else 0
impulse_count = int((selected["Type"] == "Impulse / slow mover").sum()) if not selected.empty else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected categories", f"{selected_count}")
c2.metric("Shelf utilization", f"{utilization:.1f}%")
c3.metric("Fast movers selected", f"{fast_count}")
c4.metric("Impulse categories selected", f"{impulse_count}")

st.metric("Expected profit attractiveness score", f"{total_reward:,.2f}",
          help="Relative objective score based on margin × demand × impulse factor × segment traffic × allocated space share. It is not currency.")

left, right = st.columns(2)
with left:
    st.markdown("**Assortment composition**")
    if selected_count:
        type_counts = selected["Type"].value_counts().rename_axis("Category type").reset_index(name="Number selected")
        fig = px.pie(type_counts, names="Category type", values="Number selected", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No product categories could be allocated. Increase shelf capacity or reduce minimum space.")

with right:
    st.markdown("**Shelf utilization by shelf**")
    shelf_usage = segments.groupby("Shelf", as_index=False).agg(Allocated=("Allocated", "sum"), Capacity=("Capacity", "sum"))
    shelf_usage["Utilization (%)"] = 100 * shelf_usage["Allocated"] / shelf_usage["Capacity"]
    fig = px.bar(shelf_usage, x="Shelf", y="Utilization (%)", range_y=[0, 100], text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Shelf segment map**")
heat = segments.pivot(index="Shelf", columns="Segment", values="Utilization_%")
fig = px.imshow(heat, text_auto=".1f", aspect="auto", color_continuous_scale="YlGnBu",
                labels=dict(x="Shelf segment", y="Shelf", color="Utilization (%)"),
                title="Percentage of each shelf segment occupied")
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Product category allocation**")
if not selected.empty:
    show = selected.sort_values("Estimated_Contribution", ascending=False).copy()
    show["Demand"] = show["Demand"].round(0).astype(int)
    show["Profit_Margin"] = show["Profit_Margin"].map(lambda x: f"{x:.1%}")
    show["Impulse_Potential"] = show["Impulse_Potential"].map(lambda x: f"{x:.1%}")
    show["Space"] = show["Space"].map(lambda x: f"{x:.2f} ft")
    show["Estimated_Contribution"] = show["Estimated_Contribution"].map(lambda x: f"{x:,.2f}")
    st.dataframe(show, use_container_width=True, hide_index=True)
else:
    st.write("No categories selected.")

if not allocations.empty:
    st.markdown("**Detailed shelf assignment**")
    st.dataframe(allocations[["Category", "Type", "Shelf", "Segment", "Space", "Traffic", "Estimated_Contribution"]],
                 use_container_width=True, hide_index=True)

st.divider()
st.subheader("5. What-if analysis")
st.write("Compare the current scenario with a lower- or higher-demand scenario while keeping other inputs constant.")
whatif = st.slider("Alternative demand multiplier", 0.5, 2.0, min(2.0, demand_multiplier + 0.25), 0.05, key="whatif")
if st.button("Compare scenarios"):
    alt = simulate(
        edited.dropna(subset=["Category"]),
        n_shelves, n_segments, segment_capacity,
        whatif, margin_multiplier, impulse_multiplier,
        fast_pct, space_factor
    )
    alt_selected, alt_segments, alt_reward = alt[2], alt[3], alt[4]
    comparison = pd.DataFrame([
        {"Scenario": "Current", "Demand multiplier": demand_multiplier,
         "Selected categories": selected_count, "Utilization (%)": utilization, "Profit score": total_reward},
        {"Scenario": "Alternative", "Demand multiplier": whatif,
         "Selected categories": int(alt_selected["Category"].nunique()) if not alt_selected.empty else 0,
         "Utilization (%)": 100 * alt_segments["Allocated"].sum() / total_capacity if total_capacity else 0,
         "Profit score": alt_reward}
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    metric_fig = px.bar(comparison, x="Scenario", y="Profit score", color="Scenario", text_auto=".2f",
                        title="Impact of demand change on the objective score")
    st.plotly_chart(metric_fig, use_container_width=True)

with st.expander("6. Model logic and assumptions"):
    st.markdown("""
    **Research-paper logic represented**
    - Fast movers have high expected demand and are treated with an impulse factor of 1.
    - Other categories receive an impulse-purchase potential factor.
    - Shelf segments have different traffic / attractiveness values.
    - A category is assigned to at most one shelf.
    - Shelf-segment capacity is respected.
    - The objective rewards profitable categories placed in attractive locations.

    **Simplifications in this teaching prototype**
    - The original paper formulates a mixed-integer programming model (APSA) with binary assignment variables, continuous space variables and assortment/allocation affinity constraints.
    - This app uses a greedy heuristic: it ranks categories by potential profit per minimum space, selects a feasible shelf, then allocates space to its more attractive available segments.
    - Product-level margins and space bounds are editable illustrative assumptions; demand values for the named categories are transcribed from the paper's Table 2.
    - The app does not implement the paper's full affinity / disaffinity constraints, adjacency logic, or its multi-shelf re-optimization heuristic.
    - The displayed score is a relative objective score, not a forecast of actual monetary profit.
    """)

st.caption("Academic reference: Flamand, T., Ghoniem, A., Haouari, M., & Maddah, B. (2018). Integrated assortment planning and store-wide shelf space allocation: An optimization-based approach. Omega, 81, 134–149. https://doi.org/10.1016/j.omega.2017.10.006")
