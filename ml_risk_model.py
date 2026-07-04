"""
ML Risk Prediction Model — Supply Chain Resilience
Benjamin Heo | Lilley Fellowship

Model: Gradient Boosting classifier (XGBoost-style via sklearn)
Input features: geopolitical indicators → Output: disruption risk score (0-1)

Features used:
  - conflict_intensity    : 0-1 scale of active military/political conflict
  - trade_concentration   : % of supply from single country (0-1)
  - energy_dependency     : reliance on imported energy (0-1)
  - chokepoint_exposure   : proximity to major maritime chokepoints (0-1)
  - sanctions_active      : binary 0/1
  - gdp_shock             : % GDP deviation from trend (normalized 0-1)
  - lead_time_variability : historical std dev of lead times (normalized)
  - political_stability   : inverted — higher = more unstable (0-1)

Outputs:
  - risk_score    : float 0-1 (disruption probability)
  - risk_label    : LOW / MEDIUM / HIGH / CRITICAL
  - top_features  : which indicators drove the score most
  - recommendations: what a supply chain manager should do
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────
FEATURES = [
    "conflict_intensity",
    "trade_concentration",
    "energy_dependency",
    "chokepoint_exposure",
    "sanctions_active",
    "gdp_shock",
    "lead_time_variability",
    "political_instability",
]
print("LOADING:", __file__)
FEATURE_DESCRIPTIONS = {
    "conflict_intensity":    "Active military or political conflict (0=none, 1=war)",
    "trade_concentration":   "% of supply from single country (0=diversified, 1=single source)",
    "energy_dependency":     "Reliance on imported energy (0=self-sufficient, 1=fully dependent)",
    "chokepoint_exposure":   "Proximity to maritime chokepoints like Suez, Hormuz (0-1)",
    "sanctions_active":      "Active international sanctions in trade corridor (0=no, 1=yes)",
    "gdp_shock":             "Economic contraction severity (0=stable, 1=severe recession)",
    "lead_time_variability": "Historical shipping time volatility (0=stable, 1=highly variable)",
    "political_instability": "Government instability index (0=stable, 1=failed state)",
}

# ─────────────────────────────────────────────
# HISTORICAL TRAINING DATA
# Each row = one historical event-period observation
# Label: 0=Low, 1=Medium, 2=High, 3=Critical
# ─────────────────────────────────────────────
HISTORICAL_DATA = [
    # conflict  trade_conc  energy  chokepoint  sanctions  gdp_shock  lead_var  pol_instab  label
    # ── Normal peacetime baselines ──
    [0.05, 0.20, 0.30, 0.20, 0.00, 0.05, 0.15, 0.10,  0],  # 1990s stable trade
    [0.05, 0.25, 0.25, 0.25, 0.00, 0.03, 0.12, 0.08,  0],  # 2000s pre-crisis
    [0.08, 0.30, 0.35, 0.20, 0.05, 0.08, 0.18, 0.12,  0],  # 2010 mild disruption
    [0.05, 0.22, 0.28, 0.18, 0.00, 0.04, 0.10, 0.09,  0],  # 2015 stable
    [0.06, 0.18, 0.22, 0.22, 0.02, 0.06, 0.14, 0.11,  0],  # 2016 stable
    [0.07, 0.24, 0.30, 0.19, 0.03, 0.05, 0.13, 0.10,  0],  # 2017 stable
    [0.06, 0.26, 0.27, 0.21, 0.02, 0.04, 0.11, 0.09,  0],  # 2018 stable

    # ── Medium disruptions ──
    [0.20, 0.40, 0.45, 0.35, 0.20, 0.20, 0.35, 0.25,  1],  # 2008 financial crisis
    [0.15, 0.45, 0.40, 0.30, 0.15, 0.25, 0.30, 0.20,  1],  # 2011 Arab Spring
    [0.18, 0.38, 0.42, 0.38, 0.10, 0.18, 0.32, 0.28,  1],  # 2014 Crimea annexation
    [0.25, 0.42, 0.38, 0.40, 0.25, 0.15, 0.38, 0.30,  1],  # 2018 US-China trade war
    [0.22, 0.50, 0.35, 0.35, 0.30, 0.20, 0.40, 0.25,  1],  # 2019 tariff escalation
    [0.15, 0.35, 0.44, 0.42, 0.12, 0.22, 0.28, 0.22,  1],  # 2010 European debt crisis

    # ── High disruptions ──
    [0.35, 0.60, 0.55, 0.50, 0.40, 0.35, 0.55, 0.45,  2],  # 2021 Suez blockage
    [0.30, 0.65, 0.60, 0.45, 0.35, 0.40, 0.60, 0.40,  2],  # 2021 chip shortage
    [0.40, 0.55, 0.65, 0.55, 0.45, 0.30, 0.58, 0.50,  2],  # 2022 Russia sanctions
    [0.38, 0.58, 0.62, 0.52, 0.42, 0.32, 0.55, 0.48,  2],  # 1973 oil embargo
    [0.35, 0.52, 0.70, 0.60, 0.38, 0.35, 0.52, 0.44,  2],  # 2022 energy crisis EU
    [0.42, 0.60, 0.58, 0.48, 0.40, 0.38, 0.60, 0.52,  2],  # Korean War 1950

    # ── Critical disruptions ──
    [0.75, 0.80, 0.85, 0.80, 0.70, 0.80, 0.90, 0.85,  3],  # COVID-19 2020
    [0.80, 0.85, 0.90, 0.85, 0.75, 0.85, 0.92, 0.88,  3],  # WWII peak 1942-45
    [0.70, 0.78, 0.82, 0.78, 0.68, 0.75, 0.88, 0.82,  3],  # Taiwan strait (hypothetical)
    [0.65, 0.75, 0.80, 0.75, 0.65, 0.70, 0.85, 0.78,  3],  # COVID variant wave
    [0.72, 0.82, 0.78, 0.80, 0.72, 0.78, 0.87, 0.80,  3],  # hypothetical Gulf war
    [0.68, 0.76, 0.84, 0.82, 0.66, 0.72, 0.83, 0.76,  3],  # hypothetical Hormuz closure
]

LABELS = ["Low", "Medium", "High", "Critical"]
LABEL_COLORS = {
    "Low":      "#22c55e",
    "Medium":   "#f59e0b",
    "High":     "#ef4444",
    "Critical": "#dc2626",
}

# ─────────────────────────────────────────────
# EVENT FEATURE PROFILES
# Pre-calibrated feature vectors for each historical event
# ─────────────────────────────────────────────
EVENT_PROFILES = {
    "COVID-19 Pandemic": {
        "conflict_intensity":    0.10,
        "trade_concentration":   0.65,
        "energy_dependency":     0.55,
        "chokepoint_exposure":   0.70,
        "sanctions_active":      0.00,
        "gdp_shock":             0.85,
        "lead_time_variability": 0.90,
        "political_instability": 0.40,
        "narrative": "COVID spread globally with no active sanctions or conflict, but its economic shock was catastrophic. Trade concentration in China meant factory shutdowns in one region cascaded everywhere. Lead time variability skyrocketed as ports backed up worldwide.",
    },
    "Suez Canal Blockage": {
        "conflict_intensity":    0.05,
        "trade_concentration":   0.45,
        "energy_dependency":     0.40,
        "chokepoint_exposure":   1.00,
        "sanctions_active":      0.00,
        "gdp_shock":             0.20,
        "lead_time_variability": 0.55,
        "political_instability": 0.10,
        "narrative": "The Suez blockage was a pure chokepoint event — no conflict, no sanctions, just one ship. But because 12% of global trade passes through Suez, the ripple effects were massive. Chokepoint exposure is the dominant risk driver here.",
    },
    "1973 Oil Embargo": {
        "conflict_intensity":    0.45,
        "trade_concentration":   0.62,
        "energy_dependency":     0.80,
        "chokepoint_exposure":   0.65,
        "sanctions_active":      0.85,
        "gdp_shock":             0.55,
        "lead_time_variability": 0.50,
        "political_instability": 0.55,
        "narrative": "The oil embargo combined high energy dependency with deliberate sanctions and regional conflict (Yom Kippur War). Western nations with high oil import dependency suffered most. Energy dependency and active sanctions are the top drivers.",
    },
    "Semiconductor Shortage": {
        "conflict_intensity":    0.15,
        "trade_concentration":   0.82,
        "energy_dependency":     0.45,
        "chokepoint_exposure":   0.50,
        "sanctions_active":      0.25,
        "gdp_shock":             0.40,
        "lead_time_variability": 0.78,
        "political_instability": 0.20,
        "narrative": "The chip shortage was driven almost entirely by extreme trade concentration — 92% of advanced chips from Taiwan/Korea. No conflict, no sanctions, just catastrophic single-source dependency exposed by COVID demand shocks.",
    },
    "Russia-Ukraine Sanctions": {
        "conflict_intensity":    0.85,
        "trade_concentration":   0.55,
        "energy_dependency":     0.75,
        "chokepoint_exposure":   0.40,
        "sanctions_active":      0.95,
        "gdp_shock":             0.50,
        "lead_time_variability": 0.55,
        "political_instability": 0.80,
        "narrative": "Active war combined with the broadest sanctions regime since WWII. Europe's energy dependency on Russia (40% of gas) made this uniquely severe. Conflict intensity, sanctions, and energy dependency are all maxed out simultaneously.",
    },
    "Taiwan Strait Scenario": {
        "conflict_intensity":    0.95,
        "trade_concentration":   0.90,
        "energy_dependency":     0.70,
        "chokepoint_exposure":   0.95,
        "sanctions_active":      0.80,
        "gdp_shock":             0.90,
        "lead_time_variability": 0.95,
        "political_instability": 0.85,
        "narrative": "The ultimate tail risk — all indicators simultaneously critical. Taiwan produces 92% of advanced chips, sits astride the world's busiest shipping lanes, and any conflict would trigger global sanctions. Every feature is near-maximum.",
    },
    "Korean War": {
        "conflict_intensity":    0.80,
        "trade_concentration":   0.50,
        "energy_dependency":     0.40,
        "chokepoint_exposure":   0.55,
        "sanctions_active":      0.45,
        "gdp_shock":             0.45,
        "lead_time_variability": 0.60,
        "political_instability": 0.70,
        "narrative": "The Korean War triggered US strategic materials rationing, Pacific shipping disruption, and a Cold War realignment of global trade. Conflict intensity and political instability dominate. The US-led UN coalition's supply chain was one of the most complex ever assembled at that scale.",
    },
    "World War II": {
        "conflict_intensity":    1.00,
        "trade_concentration":   0.75,
        "energy_dependency":     0.65,
        "chokepoint_exposure":   0.90,
        "sanctions_active":      0.90,
        "gdp_shock":             0.80,
        "lead_time_variability": 0.95,
        "political_instability": 0.95,
        "narrative": "The most severe global supply chain disruption in history. U-boat warfare closed the Atlantic, the Pacific was a war zone, and all major industrial economies redirected production to war. Every civilian supply chain effectively ceased. The Allied logistics miracle — supplying D-Day and the Pacific simultaneously — remains the greatest supply chain achievement ever.",
    },
    "Gaza Conflict & Red Sea Crisis": {
        "conflict_intensity":    0.70,
        "trade_concentration":   0.35,
        "energy_dependency":     0.50,
        "chokepoint_exposure":   0.95,
        "sanctions_active":      0.30,
        "gdp_shock":             0.25,
        "lead_time_variability": 0.65,
        "political_instability": 0.75,
        "narrative": "A non-state actor (Houthis) using anti-ship missiles has effectively closed the Red Sea to most commercial shipping. Chokepoint exposure is near-maximum — the Bab-el-Mandeb and Suez Canal together handle 15% of world trade. The rerouting around Africa adds 10-14 days and has tripled Asia-Europe freight rates.",
    },
}

# ─────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────
class GeopoliticalRiskModel:
    """
    Ensemble ML model predicting supply chain disruption risk
    from geopolitical indicator features.
    """

    def __init__(self):
        self.models = {
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.08, subsample=0.85,
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=5,
                min_samples_leaf=2, random_state=42
            ),
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
    C=1.0,
    max_iter=500,
    solver="lbfgs",
    random_state=42
)),
            ]),
        }
        self.primary = "Gradient Boosting"
        self.trained = False
        self._X = None
        self._y = None

    def train(self):
        data = np.array(HISTORICAL_DATA)
        self._X = data[:, :-1]
        self._y = data[:, -1].astype(int)

        for name, model in self.models.items():
            model.fit(self._X, self._y)

        self.trained = True

    def cross_validate(self):
        """Return CV accuracy for each model."""
        if not self.trained:
            self.train()
        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, self._X, self._y, cv=5, scoring="accuracy")
            results[name] = {
                "mean_accuracy": round(scores.mean(), 3),
                "std":           round(scores.std(), 3),
            }
        return results

    def predict(self, features: dict):
        """
        Predict risk for a feature dict.
        Returns risk score, label, probabilities, top drivers.
        """
        if not self.trained:
            self.train()

        x = np.array([[features[f] for f in FEATURES]])

        # Primary model prediction
        primary_model = self.models[self.primary]
        proba = primary_model.predict_proba(x)[0]
        pred_class = int(np.argmax(proba))
        risk_score = float(
            0.0 * proba[0] + 0.33 * proba[1] + 0.66 * proba[2] + 1.0 * proba[3]
        )

        # Ensemble vote
        votes = []
        for model in self.models.values():
            votes.append(int(model.predict(x)[0]))
        from collections import Counter
        ensemble_class = Counter(votes).most_common(1)[0][0]

        # Feature importance (from GB model)
        gb = self.models["Gradient Boosting"]
        importances = gb.feature_importances_
        feat_imp = sorted(
            zip(FEATURES, importances),
            key=lambda x: x[1], reverse=True
        )
        top_features = feat_imp[:3]

        # Recommendations based on top drivers
        recs = _get_recommendations(top_features, pred_class)

        return {
            "risk_score":    round(risk_score, 3),
            "risk_label":    LABELS[ensemble_class],
            "risk_class":    ensemble_class,
            "probabilities": {LABELS[i]: round(float(p), 3) for i, p in enumerate(proba)},
            "top_features":  [(f, round(float(imp), 3)) for f, imp in top_features],
            "recommendations": recs,
            "ensemble_votes": {LABELS[v]: votes.count(v) for v in set(votes)},
        }

    def predict_event(self, event_name: str):
        """Predict risk for a named historical event using its preset profile."""
        if event_name not in EVENT_PROFILES:
            return None
        profile = {k: v for k, v in EVENT_PROFILES[event_name].items()
                   if k in FEATURES}
        result = self.predict(profile)
        result["narrative"] = EVENT_PROFILES[event_name].get("narrative", "")
        result["event_name"] = event_name
        return result

    def predict_timeline(self, event_name: str, days=300):
        """
        Simulate how risk evolves day-by-day for an event.
        Adds noise and a recovery trend after peak.
        """
        if not self.trained:
            self.train()

        profile = EVENT_PROFILES.get(event_name, EVENT_PROFILES["COVID-19 Pandemic"])
        base = {k: v for k, v in profile.items() if k in FEATURES}
        duration = {
            "COVID-19 Pandemic":       200,
            "Suez Canal Blockage":     50,
            "1973 Oil Embargo":        150,
            "Semiconductor Shortage":  180,
            "Russia-Ukraine Sanctions":250,
            "Taiwan Strait Scenario":  100,
            "Korean War":              180,
            "World War II":            280,
        }.get(event_name, 150)

        timeline = []
        for day in range(days):
            # Risk decays after disruption ends
            if day < duration:
                decay = 1.0
            else:
                days_past = day - duration
                decay = max(0.1, 1.0 - (days_past / 80))

            # Add realistic noise
            noisy = {}
            for f in FEATURES:
                val = base[f] * decay
                val += np.random.normal(0, 0.03)
                noisy[f] = float(np.clip(val, 0, 1))

            result = self.predict(noisy)
            timeline.append({
                "day":        day,
                "risk_score": result["risk_score"],
                "risk_label": result["risk_label"],
            })

        return timeline


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def _get_recommendations(top_features, risk_class):
    recs_map = {
        "conflict_intensity": [
            "Activate war-risk insurance on all shipments",
            "Pre-position safety stock outside conflict zones",
            "Establish alternative routing through neutral corridors",
        ],
        "trade_concentration": [
            "Immediately qualify secondary suppliers in different regions",
            "Increase safety stock for concentrated-source inputs by 60%+",
            "Consider near-shoring or friend-shoring critical components",
        ],
        "energy_dependency": [
            "Hedge energy costs via futures contracts",
            "Audit which production nodes are most energy-intensive",
            "Develop contingency plans for fuel rationing scenarios",
        ],
        "chokepoint_exposure": [
            "Map all routes through Suez, Hormuz, Malacca — quantify exposure",
            "Pre-negotiate Cape of Good Hope rerouting with carriers",
            "Consider air freight contracts for highest-value low-weight goods",
        ],
        "sanctions_active": [
            "Conduct sanctions compliance audit across all suppliers",
            "Identify and replace any sanctioned-country inputs immediately",
            "Establish letter-of-credit facilities in neutral banking jurisdictions",
        ],
        "gdp_shock": [
            "Stress-test demand forecasts under -20%, -40% scenarios",
            "Renegotiate supplier contracts to include volume flexibility clauses",
            "Build cash reserves to weather receivables delays",
        ],
        "lead_time_variability": [
            "Switch to demand-driven ordering with shorter replenishment cycles",
            "Increase safety stock by 1.5× current levels",
            "Add real-time shipment tracking across all tier-1 suppliers",
        ],
        "political_instability": [
            "Monitor political risk indices (GDELT, ACLED) for early warning",
            "Diversify away from single-government-dependent supply nodes",
            "Establish local stockpiles in politically stable buffer countries",
        ],
    }

    recs = []
    for feat, _ in top_features:
        recs.extend(recs_map.get(feat, [])[:2])

    if risk_class >= 3:
        recs.insert(0, "🚨 CRITICAL: Activate full supply chain continuity plan immediately")
    elif risk_class >= 2:
        recs.insert(0, "⚠️ HIGH RISK: Convene supply chain risk committee within 48 hours")

    return recs[:5]


# ─────────────────────────────────────────────
# SINGLETON — import this in dashboard
# ─────────────────────────────────────────────
_model_instance = None
import threading
_model_lock = threading.Lock()

def get_model() -> GeopoliticalRiskModel:
    """
    Return trained singleton model.

    Thread-safe: on Streamlit Cloud (and similar multi-session deployments),
    this module-level singleton is shared across every concurrent user's
    session in the same process. Without a lock, two sessions hitting this
    on first load at the same time could both enter GradientBoosting/
    RandomForest .fit() on the *same* estimator objects simultaneously,
    corrupting internal state mid-fit (e.g. sklearn's self.n_classes_ being
    read as a list by one thread right as another thread has just collapsed
    it to a scalar) — this is what causes intermittent
    "TypeError: 'int' object is not subscriptable" crashes on deploy.
    """
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:  # re-check inside the lock
                _model_instance = GeopoliticalRiskModel()
                _model_instance.train()
    return _model_instance


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Training ML Risk Model...")
    model = get_model()

    print("\nCross-validation accuracy:")
    cv = model.cross_validate()
    for name, scores in cv.items():
        print(f"  {name}: {scores['mean_accuracy']:.1%} ± {scores['std']:.1%}")

    print("\n" + "═"*60)
    print("Event Risk Predictions:")
    print("═"*60)

    for event_name in EVENT_PROFILES:
        result = model.predict_event(event_name)
        print(f"\n{event_name}")
        print(f"  Risk Score : {result['risk_score']:.3f}")
        print(f"  Label      : {result['risk_label']}")
        print(f"  Top Drivers: {', '.join(f for f, _ in result['top_features'])}")
        print(f"  Narrative  : {result['narrative'][:100]}...")
