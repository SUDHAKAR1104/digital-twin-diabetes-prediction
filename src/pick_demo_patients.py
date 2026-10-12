import pandas as pd

d = pd.read_csv("data/synthetic_patients.csv", parse_dates=["timestamp"])
d["hour"] = d["timestamp"].dt.hour
d["day"] = d["timestamp"].dt.date

g = d.groupby("patient_id")
s = pd.DataFrame({
    "hba1c": g["hba1c"].first(),
    "spike_rate": g["spike_2h"].mean(),
    "avg_steps": g["steps"].mean(),
})
# share of this patient's spike rows that fall in the evening (18:00-23:59)
eve = d[d["hour"] >= 18].groupby("patient_id")["spike_2h"].mean()
s["evening_spike_rate"] = eve

stable = s.sort_values("spike_rate").head(1)
dinner = s[s["spike_rate"] > 0.10].sort_values("evening_spike_rate", ascending=False).head(1)
walker = s[(s["spike_rate"] > 0.10)].sort_values("avg_steps", ascending=False).head(1)

print("STABLE (lowest spike rate):\n", stable.round(3), "\n")
print("DINNER SPIKER (highest evening spike rate):\n", dinner.round(3), "\n")
print("WALKER (spikes, but most active):\n", walker.round(3))
