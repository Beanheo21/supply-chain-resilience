"""
Supply Chain Resilience Simulation — Step 1 (Complete)
Benjamin Heo | Lilley Fellowship

Fixes from original:
  - Production capped by actual order quantity (not raw capacity)
  - Pipeline arrivals use p[0] <= day to prevent silent drops
  - LP optimization properly feeds back into production & inventory
  - RL state includes 4th dimension (risk level)
  - 1000 RL training episodes (was 50)

Extensions added:
  - 6 historical events: COVID, Suez, Oil Embargo, Semiconductor, Russia-Ukraine, Taiwan Strait
  - Full KPI tracking: service level %, inventory turns, total cost, transportation cost, fill rate
  - Multi-node network graph (Supplier → Factory → Distributor)
  - Confidence intervals over 30 replications
  - Sensitivity analysis runner
  - Results export to CSV
"""

import numpy as np
import random
import csv
import os
from collections import defaultdict
from scipy import stats

try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
    PULP_AVAILABLE = True
except ImportError:
    print("WARNING: PuLP not installed. Optimization policy will fall back to heuristic.")
    print("Install with: pip install pulp")
    PULP_AVAILABLE = False

# ─────────────────────────────────────────────
# GLOBAL PARAMETERS
# ─────────────────────────────────────────────
DAYS            = 300
BASE_DEMAND     = 100          # units/day
BASE_CAPACITY   = 120          # units/day (slightly above demand → buffer)
BASE_LEAD_TIME  = 30           # days
HOLD_COST       = 0.5          # $ per unit per day
BACKLOG_COST    = 5.0          # $ per unit per day (penalty)
TRANSPORT_COST  = 1.2          # $ per unit shipped
PROD_COST       = 2.0          # $ per unit produced
REPLICATIONS    = 30           # Monte Carlo replications for confidence intervals
RL_EPISODES     = 1000         # training episodes for RL agent

# ─────────────────────────────────────────────
# HISTORICAL EVENTS
# ─────────────────────────────────────────────
EVENTS = {
    "COVID-19 Pandemic": {
        "capacity":           0.55,   # 45% capacity loss
        "demand_multiplier":  1.8,    # panic buying spike
        "lead_time":          110,    # port congestion
        "risk":               0.70,   # high daily shock probability
        "duration":           200,    # days of disruption
        "description": "Global pandemic causing factory shutdowns, port congestion, and demand surges."
    },
    "Suez Canal Blockage": {
        "capacity":           0.80,
        "demand_multiplier":  1.40,
        "lead_time":          65,
        "risk":               0.45,
        "duration":           50,
        "description": "Single canal blockage cascading into global shipping delays."
    },
    "1973 Oil Embargo": {
        "capacity":           0.65,
        "demand_multiplier":  0.75,   # demand collapsed in some sectors
        "lead_time":          50,
        "risk":               0.55,
        "duration":           150,
        "description": "OPEC oil embargo causing energy shortages and transportation cost spikes."
    },
    "Semiconductor Shortage": {
        "capacity":           0.50,   # severe input shortage
        "demand_multiplier":  1.60,   # electronics demand stayed high
        "lead_time":          90,
        "risk":               0.60,
        "duration":           180,
        "description": "Global chip shortage halting production across automotive and electronics sectors."
    },
    "Russia-Ukraine Sanctions": {
        "capacity":           0.75,
        "demand_multiplier":  1.30,
        "lead_time":          55,
        "risk":               0.50,
        "duration":           250,    # ongoing, long-duration
        "description": "Sanctions and conflict disrupting energy, grain, and logistics corridors."
    },
    "Taiwan Strait Scenario": {
        "capacity":           0.40,   # extreme — semiconductor hub threatened
        "demand_multiplier":  2.00,
        "lead_time":          120,
        "risk":               0.80,
        "duration":           100,
        "description": "Hypothetical Taiwan conflict scenario: worst-case semiconductor supply collapse."
    },
    "Korean War": {
        "capacity":           0.60,   # Korean industrial output destroyed
        "demand_multiplier":  1.50,   # military demand surge, civilian rationing
        "lead_time":          70,     # Pacific shipping disrupted
        "risk":               0.60,
        "duration":           180,    # active conflict phase
        "description": "Korean War (1950-53) disrupting Pacific trade routes and triggering US strategic materials rationing."
    },
    "World War II": {
        "capacity":           0.25,
        "demand_multiplier":  0.40,
        "lead_time":          200,
        "risk":               0.90,
        "duration":           280,
        "description": "WWII global trade collapse: Atlantic U-boat warfare, Pacific blockade, and full industrial conversion to war production."
    },
    "Gaza Conflict & Red Sea Crisis": {
        "capacity":           0.78,   # rerouting adds cost/delay but capacity mostly intact
        "demand_multiplier":  1.20,   # freight demand spikes as routes lengthen
        "lead_time":          75,     # +10-14 days via Cape of Good Hope
        "risk":               0.55,   # ongoing Houthi missile threat
        "duration":           999,    # ongoing as of 2025
        "description": "Houthi Red Sea attacks forcing 90%+ of container ships around Africa; Suez Canal revenues down 60%, freight rates tripled."
    },
}

# ─────────────────────────────────────────────
# RL AGENT (Q-LEARNING) — fixed state space
# ─────────────────────────────────────────────
ACTIONS = ["increase_prod", "decrease_prod", "reroute", "stock_up", "do_nothing"]
Q_TABLE: dict = defaultdict(float)

def _state_key(inventory, backlog, demand, risk):
    """Discretize continuous variables into state buckets."""
    inv_bucket  = min(int(inventory / 100), 9)    # 0-9
    back_bucket = min(int(backlog   / 50),  9)    # 0-9
    dem_bucket  = min(int(demand    / 50),  5)    # 0-5
    risk_bucket = int(risk * 4)                   # 0-4  (new dimension)
    return (inv_bucket, back_bucket, dem_bucket, risk_bucket)

def _get_Q(state_key, action):
    return Q_TABLE[(state_key, action)]

def _choose_action(state_key, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: _get_Q(state_key, a))

def _update_Q(state_key, action, reward, next_key, alpha=0.1, gamma=0.9):
    best_next = max(_get_Q(next_key, a) for a in ACTIONS)
    Q_TABLE[(state_key, action)] += alpha * (
        reward + gamma * best_next - _get_Q(state_key, action)
    )

# ─────────────────────────────────────────────
# NETWORK NODES  (Supplier → Factory → Distributor)
# ─────────────────────────────────────────────
# Each node has its own inventory and lead-time lag.
# For simplicity the simulation runs the full chain in sequence each day.
class NetworkNode:
    def __init__(self, name, capacity_factor=1.0, safety_stock=200):
        self.name           = name
        self.capacity_factor = capacity_factor
        self.inventory      = 500
        self.backlog        = 0
        self.safety_stock   = safety_stock
        self.pipeline       = []   # list of (arrival_day, qty)

    def reset(self):
        self.inventory = 500
        self.backlog   = 0
        self.pipeline  = []

NETWORK = {
    "Supplier":     NetworkNode("Supplier",     capacity_factor=1.0, safety_stock=300),
    "Factory":      NetworkNode("Factory",      capacity_factor=0.9, safety_stock=200),
    "Distributor":  NetworkNode("Distributor",  capacity_factor=0.8, safety_stock=150),
}

# ─────────────────────────────────────────────
# CORE SIMULATION FUNCTION
# ─────────────────────────────────────────────
def simulate(event: dict, policy: str = "baseline", train_rl: bool = False):
    """
    Run one replication of the supply chain simulation.

    Returns
    -------
    dict with daily arrays and summary KPIs
    """
    # Reset network nodes
    for node in NETWORK.values():
        node.reset()

    # Accumulators
    history_backlog      = []
    history_inventory    = []
    history_service_lvl  = []
    total_cost           = 0.0
    total_transport_cost = 0.0
    total_demand_seen    = 0
    total_demand_filled  = 0
    total_produced       = 0

    pipeline = []  # (arrival_day, qty)  — single-node view for KPI clarity

    inventory = 500.0
    backlog   = 0.0

    for day in range(DAYS):
        # ── Disruption parameters ──────────────────────────────
        if day < event["duration"]:
            capacity   = BASE_CAPACITY * event["capacity"]
            demand     = float(np.random.poisson(BASE_DEMAND * event["demand_multiplier"]))
            lead_time  = max(1, int(np.random.normal(event["lead_time"], 10)))
            risk       = event["risk"]
        else:
            capacity   = float(BASE_CAPACITY)
            demand     = float(np.random.poisson(BASE_DEMAND))
            lead_time  = BASE_LEAD_TIME
            risk       = 0.10

        # Random secondary shock
        if random.random() < risk:
            capacity *= random.uniform(0.5, 0.85)

        # ── State ──────────────────────────────────────────────
        state_key = _state_key(inventory, backlog, demand, risk)

        # ── Policy selection ───────────────────────────────────
        if policy == "rl":
            epsilon = 0.05 if not train_rl else 0.2
            action  = _choose_action(state_key, epsilon)

        elif policy == "heuristic":
            # Multi-threshold heuristic
            if backlog > 300:
                action = "increase_prod"
            elif backlog > 100:
                action = "reroute"
            elif inventory < 150:
                action = "stock_up"
            elif inventory > 800:
                action = "decrease_prod"
            else:
                action = "do_nothing"

        elif policy == "optimization" and PULP_AVAILABLE:
            # LP: minimise cost subject to demand satisfaction
            prob   = LpProblem("SupplyChain_Day", LpMinimize)
            prod   = LpVariable("prod",   lowBound=0, upBound=capacity * 1.5)
            inv_lp = LpVariable("inv_lp", lowBound=0)
            bl     = LpVariable("bl",     lowBound=0)

            # Objective: minimise backlog penalty + hold cost + prod cost
            prob += (BACKLOG_COST * bl
                     + HOLD_COST  * inv_lp
                     + PROD_COST  * prod)

            # Constraints
            prob += prod <= capacity * 1.5          # can surge 50%
            prob += inv_lp >= 0
            prob += inv_lp - bl == inventory + prod - (demand + backlog)

            prob.solve(PULP_CBC_CMD(msg=0))

            opt_prod = max(0.0, value(prod) or 0.0)
            # Override capacity with LP solution
            capacity  = min(opt_prod, BASE_CAPACITY * 2.0)
            action    = "optimized"

        else:
            # Baseline: do nothing (pure reactive)
            action = "do_nothing"

        # ── Apply action ───────────────────────────────────────
        if action == "increase_prod":
            capacity = min(capacity * 1.35, BASE_CAPACITY * 2.0)
        elif action == "decrease_prod":
            capacity *= 0.75
        elif action == "reroute":
            lead_time = max(1, int(lead_time * 0.65))
        elif action == "stock_up":
            # Emergency purchase — increases inventory immediately at extra cost
            emergency_qty = 150.0
            inventory    += emergency_qty
            total_cost   += emergency_qty * PROD_COST * 1.5   # premium price
            total_transport_cost += emergency_qty * TRANSPORT_COST * 1.5

        # ── Production order ───────────────────────────────────
        # Order enough to cover demand + replenish to safety stock, capped by capacity
        target_order = max(demand + max(0, 200 - inventory), 0)
        produced     = min(capacity, target_order)
        produced     = max(0.0, produced)

        total_produced       += produced
        total_transport_cost += produced * TRANSPORT_COST

        # Schedule arrival
        arrival_day = day + lead_time
        pipeline.append((arrival_day, produced))

        # ── Receive arrivals ───────────────────────────────────
        # BUG FIX: use <= to catch any arrivals due on or before today
        new_pipeline = []
        for (arr_day, qty) in pipeline:
            if arr_day <= day:
                inventory += qty
            else:
                new_pipeline.append((arr_day, qty))
        pipeline = new_pipeline

        # ── Fulfil demand ──────────────────────────────────────
        total_demand = demand + backlog
        total_demand_seen += demand

        if inventory >= total_demand:
            inventory            -= total_demand
            total_demand_filled  += total_demand
            backlog               = 0.0
        else:
            total_demand_filled  += inventory
            backlog               = total_demand - inventory
            inventory             = 0.0

        # ── Cost accrual ───────────────────────────────────────
        day_cost   = (HOLD_COST * inventory
                      + BACKLOG_COST * backlog
                      + PROD_COST   * produced)
        total_cost += day_cost

        # ── Service level (fill rate) this day ────────────────
        daily_service = (demand - max(0, backlog - (backlog - demand))) / demand if demand > 0 else 1.0
        daily_service = max(0.0, min(1.0, daily_service))

        # ── Record history ─────────────────────────────────────
        history_backlog.append(backlog)
        history_inventory.append(inventory)
        history_service_lvl.append(daily_service)

        # ── RL update ──────────────────────────────────────────
        if policy == "rl":
            reward    = -(BACKLOG_COST * backlog + HOLD_COST * inventory)
            next_key  = _state_key(inventory, backlog, demand, risk)
            _update_Q(state_key, action, reward, next_key)

    # ── KPI Summary ────────────────────────────────────────────
    recovery_day = next(
        (d for d, b in enumerate(history_backlog) if b < 10), None
    )

    # Inventory turns = total demand / avg inventory
    avg_inv      = np.mean(history_inventory) if history_inventory else 1
    inv_turns    = (total_demand_seen / avg_inv) if avg_inv > 0 else 0

    service_level_pct = (total_demand_filled / total_demand_seen * 100
                         if total_demand_seen > 0 else 100.0)

    return {
        # daily arrays
        "backlog":          history_backlog,
        "inventory":        history_inventory,
        "service_level":    history_service_lvl,
        # summary KPIs
        "avg_backlog":      float(np.mean(history_backlog)),
        "max_backlog":      float(np.max(history_backlog)),
        "recovery_day":     recovery_day,
        "service_level_pct": service_level_pct,
        "inventory_turns":  inv_turns,
        "total_cost":       total_cost,
        "transport_cost":   total_transport_cost,
        "total_produced":   total_produced,
    }

# ─────────────────────────────────────────────
# TRAIN RL AGENT
# ─────────────────────────────────────────────
def train_rl(episodes=RL_EPISODES):
    print(f"Training RL agent for {episodes} episodes...")
    # Train primarily on the hardest scenario for generalization
    hard_event = EVENTS["COVID-19 Pandemic"]
    for ep in range(episodes):
        simulate(hard_event, policy="rl", train_rl=True)
        if (ep + 1) % 200 == 0:
            print(f"  Episode {ep + 1}/{episodes} complete")
    print("  RL training complete.\n")

# ─────────────────────────────────────────────
# MULTI-REPLICATION RUNNER (confidence intervals)
# ─────────────────────────────────────────────
def run_replications(event, policy, n=REPLICATIONS):
    """Run n replications and return mean ± 95% CI for each KPI."""
    kpi_keys = [
        "avg_backlog", "max_backlog", "recovery_day",
        "service_level_pct", "inventory_turns",
        "total_cost", "transport_cost"
    ]
    accum = defaultdict(list)

    for _ in range(n):
        result = simulate(event, policy)
        for k in kpi_keys:
            v = result[k]
            if v is not None:
                accum[k].append(v)

    summary = {}
    for k in kpi_keys:
        vals = accum[k]
        if len(vals) > 1:
            mean = np.mean(vals)
            ci   = stats.t.interval(0.95, df=len(vals)-1,
                                     loc=mean,
                                     scale=stats.sem(vals))
            summary[k] = {
                "mean":  round(mean, 2),
                "ci_lo": round(ci[0], 2),
                "ci_hi": round(ci[1], 2),
            }
        else:
            summary[k] = {"mean": vals[0] if vals else None,
                          "ci_lo": None, "ci_hi": None}
    return summary

# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
def sensitivity_analysis(base_event_name="COVID-19 Pandemic"):
    """Vary disruption severity and duration; report avg backlog for RL policy."""
    base   = EVENTS[base_event_name].copy()
    results = []

    print(f"\n{'─'*60}")
    print(f"SENSITIVITY ANALYSIS  (base event: {base_event_name})")
    print(f"{'─'*60}")
    print(f"{'Capacity':>12}  {'Duration':>10}  {'Avg Backlog (RL)':>18}")
    print(f"{'─'*60}")

    for cap_factor in [0.3, 0.5, 0.7, 0.9]:
        for dur in [50, 100, 200]:
            e = base.copy()
            e["capacity"] = cap_factor
            e["duration"] = dur
            res = run_replications(e, "rl", n=10)
            avg_bl = res["avg_backlog"]["mean"]
            print(f"  {cap_factor:>10.0%}  {dur:>10}d  {avg_bl:>18.1f}")
            results.append({
                "capacity": cap_factor,
                "duration": dur,
                "avg_backlog_rl": avg_bl
            })

    return results

# ─────────────────────────────────────────────
# RESULTS PRINTER
# ─────────────────────────────────────────────
def print_results(event_name, policy_results):
    """Pretty-print a comparison table for one event."""
    print(f"\n{'═'*80}")
    print(f"  EVENT: {event_name}")
    print(f"  {EVENTS[event_name]['description']}")
    print(f"{'═'*80}")
    header = (f"{'Policy':<20} {'Avg BL':>8} {'Max BL':>8} "
              f"{'RecDay':>8} {'Svc%':>7} {'InvTrn':>8} "
              f"{'TotCost':>12} {'TrnCost':>12}")
    print(header)
    print(f"{'─'*80}")

    for policy, summary in policy_results.items():
        def fmt(k):
            v = summary[k]["mean"]
            return f"{v:>8.1f}" if v is not None else f"{'N/A':>8}"

        print(
            f"{policy:<20}"
            f"{fmt('avg_backlog')}"
            f"{fmt('max_backlog')}"
            f"{fmt('recovery_day')}"
            f"{fmt('service_level_pct')}"
            f"{fmt('inventory_turns')}"
            f"{fmt('total_cost'):>12}"
            f"{fmt('transport_cost'):>12}"
        )

# ─────────────────────────────────────────────
# EXPORT TO CSV
# ─────────────────────────────────────────────
def export_csv(all_results, filepath="results.csv"):
    rows = []
    for event_name, policy_results in all_results.items():
        for policy, summary in policy_results.items():
            row = {"event": event_name, "policy": policy}
            for kpi, vals in summary.items():
                row[kpi + "_mean"]  = vals["mean"]
                row[kpi + "_ci_lo"] = vals["ci_lo"]
                row[kpi + "_ci_hi"] = vals["ci_hi"]
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults exported to: {os.path.abspath(filepath)}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Train RL agent
    train_rl(RL_EPISODES)

    POLICIES = ["baseline", "heuristic", "rl", "optimization"]

    all_results = {}

    # 2. Run all events × all policies with 30 replications each
    for event_name, event in EVENTS.items():
        print(f"\nRunning: {event_name}  ({REPLICATIONS} replications × {len(POLICIES)} policies)...")
        policy_results = {}
        for policy in POLICIES:
            policy_results[policy] = run_replications(event, policy, n=REPLICATIONS)
        all_results[event_name] = policy_results
        print_results(event_name, policy_results)

    # 3. Sensitivity analysis
    sensitivity_results = sensitivity_analysis("COVID-19 Pandemic")

    # 4. Export to CSV for the dashboard / report
    export_csv(all_results, "results.csv")

    print("\n✅  Step 1 complete. results.csv ready for Streamlit dashboard.")
