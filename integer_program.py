import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
from datetime import timedelta, date

np.random.seed(42)

hospital_df = pd.read_csv("Data Files/Hospital Locations.csv")
bc_df = pd.read_csv("Data Files/Blood Center Locations.csv")
time_df = pd.read_csv("Data Files/Travel Times.csv", index_col=0)

"""
Blood compatibility matrix
"""
# Define the blood types
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Define the blood compatibility matrix
compatibility_matrix = [
    [1, 0, 0, 0, 1, 0, 1, 0],  # A+
    [1, 1, 0, 0, 1, 1, 0, 0],  # A-
    [0, 0, 1, 0, 1, 0, 1, 0],  # B+
    [0, 0, 1, 1, 1, 1, 0, 0],  # B-
    [0, 0, 0, 0, 1, 0, 0, 0],  # AB+
    [0, 0, 0, 0, 1, 1, 0, 0],  # AB-
    [1, 0, 1, 0, 1, 0, 1, 0],  # O+
    [1, 1, 1, 1, 1, 1, 1, 1]  # O-
]

# Create a pandas dataframe from the compatibility matrix
compatibility_df = pd.DataFrame(compatibility_matrix, index=blood_groups, columns=blood_groups)

"""
Blood center inventory capacities
"""
b_centers = ['Hospital Duchess Of Kent', 'Hospital Melaka', 'Hospital Miri',
             'Hospital Pulau Pinang', 'Hospital Queen Elizabeth II',
             'Hospital Raja Perempuan Zainab II',
             'Hospital Raja Permaisuri Bainun', 'Hospital Seberang Jaya',
             'Hospital Seri Manjung', 'Hospital Sibu',
             'Hospital Sultan Haji Ahmad Shah', 'Hospital Sultanah Aminah',
             'Hospital Sultanah Bahiyah', 'Hospital Sultanah Nora Ismail',
             'Hospital Sultanah Nur Zahirah', 'Hospital Taiping',
             'Hospital Tawau', 'Hospital Tengku Ampuan Afzan',
             'Hospital Tengku Ampuan Rahimah', 'Hospital Tuanku Jaafar',
             'Hospital Umum Sarawak', 'Pusat Darah Negara']

capacities = [462.0,
              1578.0,
              513.0,
              1580.0,
              1911.0,
              1490.0,
              1979.0,
              919.0,
              493.0,
              540.0,
              708.0,
              1647.0,
              3480.0,
              718.0,
              1576.0,
              869.0,
              505.0,
              911.0,
              1696.0,
              938.0,
              1093.0,
              9134.0]

capacities = [np.ceil(num / 100) * 100 for num in capacities]

"""
Demand center inventory capacities
"""
dc_capacities = [243.0, 13.0, 332.0, 445.0, 83.0, 10.0, 115.0, 201.0, 135.0, 40.0, 47.0, 48.0, 408.0,
                 81.0, 129.0, 30.0, 62.0, 41.0, 318.0, 135.0, 77.0, 339.0, 76.0, 78.0, 577.0, 209.0,
                 427.0, 133.0, 710.0, 21.0, 239.0, 44.0, 14.0, 344.0, 35.0, 348.0, 219.0, 245.0, 336.0,
                 48.0, 152.0, 15.0, 37.0, 231.0, 84.0, 141.0, 230.0, 383.0, 179.0, 601.0, 443.0, 38.0,
                 28.0, 85.0, 392.0, 61.0, 152.0, 445.0, 583.0, 624.0, 129.0, 227.0, 48.0, 77.0, 68.0,
                 422.0, 136.0, 0.0, 130.0, 46.0, 296.0, 271.0, 72.0, 121.0, 182.0, 37.0, 230.0, 25.0,
                 97.0, 130.0, 456.0, 314.0, 277.0, 258.0, 127.0, 58.0, 141.0, 48.0, 258.0, 191.0, 126.0,
                 498.0, 19.0, 100.0, 36.0, 354.0, 47.0, 215.0, 164.0, 298.0, 292.0, 403.0, 471.0, 601.0,
                 106.0, 186.0, 654.0, 121.0, 146.0, 146.0, 202.0, 438.0, 206.0, 174.0, 373.0, 101.0,
                 548.0, 14.0, 15.0, 57.0, 125.0, 226.0, 153.0, 166.0, 244.0, 293.0, 108.0, 239.0, 691.0,
                 232.0, 214.0, 6.0, 60.0, 82.0, 28.0, 356.0, 347.0, 101.0, 68.0, 206.0, 173.0, 292.0,
                 127.0, 420.0, 526.0, 577.0, 102.0]

dc_capacities = [np.ceil(num / 100) * 100 if num != 0 else 100.0 for num in dc_capacities]

hospital_names = pd.read_csv("Data Files/Hospital Locations.csv", usecols=["Hospital"])
hospital_capacity = {hosp: dc for (hosp, dc) in zip(hospital_names["Hospital"], dc_capacities)}

"""
Donations data
"""

blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

blood_probs = [0.2454, 0.0035, 0.2956, 0.0064,
               0.0651, 0.0014, 0.3776, 0.0050]

blood_dist = {k: v for k, v in zip(blood_groups, blood_probs)}

donations_df = pd.read_csv(
    "https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/donations_facility.csv",
    usecols=["date", "hospital", "daily", "blood_a", "blood_b", "blood_o", "blood_ab",
             "type_apheresis_plasma", "type_apheresis_platelet", "type_wholeblood", "location_centre",
             "location_mobile"],
    parse_dates=["date"])

rh_probs = {}

for blood_type in ['A', 'B', 'AB', 'O']:
    positive_count = blood_dist[f"{blood_type}+"]
    negative_count = blood_dist[f"{blood_type}-"]
    total_count = positive_count + negative_count

    rh_probs[f"{blood_type}+"] = positive_count / total_count
    rh_probs[f"{blood_type}-"] = negative_count / total_count

for blood_type in ['a', 'b', 'o', 'ab']:
    blood_positive_col = f"blood_{blood_type}+"
    blood_negative_col = f"blood_{blood_type}-"

    donations_df[blood_positive_col] = [np.sum(np.random.choice(a=[1, 0],
                                                                size=size,
                                                                p=[rh_probs[f"{blood_type.upper()}+"],
                                                                   rh_probs[f"{blood_type.upper()}-"]]))
                                        for size in donations_df[f"blood_{blood_type}"]]

    donations_df[blood_negative_col] = donations_df[f"blood_{blood_type}"] - donations_df[blood_positive_col]

"""
Demand simulation
"""
state_pop = {'Selangor': 7.04,
             'Johor': 4.02,
             'Sabah': 3.39,
             'Perak': 2.52,
             'Sarawak': 2.47,
             'Kedah': 2.17,
             'Kuala Lumpur': 1.96,
             'Kelantan': 1.83,
             'Pulau Pinang': 1.74,
             'Pahang': 1.61,
             'Negeri Sembilan': 1.21,
             'Terengganu': 1.19,
             'Melaka': 1.01,
             'Perlis': 0.29,
             'Putrajaya': 0.12,
             'Labuan': 0.10,
             'Malacca': 1.01,
             'Penang': 1.8}

area_counts = hospital_df.groupby("Area").count()
hosp_count = {}

for state in state_pop:
    try:
        hosp_count[state] = area_counts.loc[state].values[0]
    except KeyError:
        hosp_count[state] = 0

scaled_pop = {state: (hosp_count[state] and state_pop[state] / hosp_count[state] or 0) for state in state_pop}

hosp_pops = {row.Hospital: np.random.exponential(scaled_pop[row.Area] + 0.2) for row in hospital_df.itertuples()}

"""
Generate scenarios
"""


def generate_random_date_sets(start_date, end_date, d, n):
    date_range = end_date - start_date
    if date_range.days < d:
        raise ValueError("Date range is smaller than the number of days in a set.")

    date_sets = []
    while len(date_sets) < n:
        random_start = start_date + timedelta(days=np.random.randint(0, date_range.days - d))
        random_end = random_start + timedelta(days=d - 1)
        date_set = pd.date_range(start=random_start, end=random_end)
        date_sets.append(date_set)

    return date_sets


start_date = pd.to_datetime('2006-01-01')
end_date = pd.to_datetime('2023-01-01')
d = 3
n_runs = 5
random_date_sets = generate_random_date_sets(start_date, end_date, d, n_runs)

J = list(b_centers)
T = list(range(1, d + 1))
G = blood_groups

c = 6
p = 110
s = 35
r = 34.32*1.1
h = 16.24
M = 24 * 60
V = 50
F = gp.tupledict({(i, j): compatibility_df.loc[i, j] for i in G for j in G})
C_j = gp.tupledict(zip(J, capacities))
I = list(range(1, 2 * len(b_centers) + 1))
N = gp.tupledict({j: 2 for j in J})

runtimes = []
obj_values = []
count = 0
for dates in random_date_sets:
    print(f"Iteration #{count}")
    count += 1

    start_time = time.time()

    K = list(np.random.choice(hospital_df["Hospital"], replace=False, size=50))

    tau = gp.tupledict({(j, k): time_df.loc[j, k] for j in J for k in K})
    C_k = gp.tupledict({k: hospital_capacity[k] for k in K})

    donations_df1 = donations_df[donations_df.date.isin(dates)].copy(deep=True)

    rows = []
    for date in dates:
        for hospital in K:
            reqs = np.random.poisson(hosp_pops[hospital] * 5) * np.random.randint(5, 9)
            rows.append([date, hospital, reqs])

    requests_df1 = pd.DataFrame(rows, columns=["date", "hospital", "requests"])
    requests_df1["date"] = pd.to_datetime(requests_df1["date"])

    rows = []
    for req in requests_df1["requests"]:
        req_types = pd.Series(np.random.choice(a=blood_groups, size=req, p=blood_probs))
        rows.append(req_types.value_counts())

    blood_group_reqs = pd.DataFrame(rows, columns=blood_groups).reset_index(drop=True)
    blood_group_reqs = blood_group_reqs.fillna(0).astype(int)

    requests_df1 = pd.concat([requests_df1, blood_group_reqs], axis=1)

    d = gp.tupledict({(t, g, k): requests_df1.loc[(requests_df1["hospital"] == k) &
                                                  (requests_df1["date"] == dates[t - 1]), g].values[0]
                      for t in T[1:] for g in G for k in K})
    n = gp.tupledict({(t, g, j): donations_df1.loc[(donations_df1["hospital"] == j) &
                                                   (donations_df1["date"] == dates[t - 1]),
                                                   f"blood_{g.lower()}"].values[0]
                      for t in T for g in G for j in J})

    model = gp.Model()

    x = model.addVars(T[:-1], T[:-1], G, J, K, vtype=gp.GRB.INTEGER, lb=0, name='x')
    b = model.addVars(T[:-1], T[1:], G, G, K, vtype=gp.GRB.INTEGER, lb=0, name='b')
    v = model.addVars(T[:-1], I, J, K, vtype=gp.GRB.BINARY, name='v')
    y = model.addVars(T, T, G, J, vtype=gp.GRB.INTEGER, lb=0, name='y')
    z = model.addVars(T[:-1], T[1:], G, K, vtype=gp.GRB.INTEGER, lb=0, name='z')
    w = model.addVars(T, G, J, vtype=gp.GRB.INTEGER, lb=0, name='w')
    u = model.addVars(T[1:], G, K, vtype=gp.GRB.INTEGER, lb=0, name='u')

    """
    Objective function
    """
    obj = gp.quicksum(r * tau[j, k] * v[t, i, j, k] for t in T[:-1] for i in I for j in J for k in K) + \
          gp.quicksum(
              h * x[t_prime, t, g, j, k] for t_prime in T[:-1] for t in T[t_prime - 1:-1] for g in G for j in J for k in
              K) + \
          gp.quicksum(c * w[t, g, j] for t in T for g in G for j in J) + \
          gp.quicksum(p * u[t, g, k] for t in T[1:] for g in G for k in K)
    model.setObjective(obj, GRB.MINIMIZE)

    """
    Constraints
    """

    # Inventory capacity
    for j in J:
        for t in T:
            model.addConstr(gp.quicksum(y[t_prime, t, g, j] for g in G for t_prime in T[:t]) <= C_j[j])

    for k in K:
        for t in T[1:]:
            model.addConstr(gp.quicksum(z[t_prime, t, g, k] for g in G for t_prime in T[:t - 1]) <= C_k[k])

    # Blood perishability
    for g in G:
        for j in J:
            for t in T:
                if t > s:
                    model.addConstr(w[t, g, j] == y[t - s - 1, t - 1, g, j])
                else:
                    model.addConstr(w[t, g, j] == 0)

    # Blood center inventory
    for g in G:
        for j in J:
            for t in T:
                for t_prime in T:
                    if t == t_prime:
                        model.addConstr(y[t_prime, t, g, j] == n[t, g, j])
                    elif t_prime < t <= t_prime + s:
                        model.addConstr(y[t_prime, t, g, j] ==
                                        (y[t_prime, t - 1, g, j] - gp.quicksum(x[t_prime, t - 1, g, j, k] for k in K)))
                    else:
                        model.addConstr(y[t_prime, t, g, j] == 0)

    # Demand center inventory
    for g in G:
        for k in K:
            for t in T[2:]:
                for t_prime in T[:-1]:
                    if t_prime < t <= t_prime + s:
                        model.addConstr(z[t_prime, t, g, k] ==
                                        (z[t_prime, t - 1, g, k] +
                                         gp.quicksum(x[t_prime, t - 1, g, j, k] for j in J) -
                                         gp.quicksum(b[t_prime, t - 1, g, g_prime, k] for g_prime in G)))
                    else:
                        model.addConstr(z[t_prime, t, g, k] == 0)

    t = 2
    for g in G:
        for k in K:
            for t_prime in T[:-1]:
                if t_prime < t <= t_prime + s:
                    model.addConstr(z[t_prime, t, g, k] ==
                                    gp.quicksum(x[t_prime, t - 1, g, j, k] for j in J))
                else:
                    model.addConstr(z[t_prime, t, g, k] == 0)

    t = T[-1]
    for g in G:
        for k in K:
            for t_prime in T[:-1]:
                if t_prime < t <= t_prime + s:
                    model.addConstr((z[t_prime, t, g, k] >=
                                     gp.quicksum(b[t_prime, t, g, g_prime, k] for g_prime in G)))

    # Unmet demand
    for g_prime in G:
        for k in K:
            for t in T[1:]:
                model.addConstr(u[t, g_prime, k] == (d[t, g_prime, k] -
                                                     gp.quicksum(b[t_prime, t, g, g_prime, k]
                                                                 for g in G for t_prime in T[t - s:t - 1])))
    # Maximum travel time
    for g in G:
        for j in J:
            for k in K:
                for t_prime in T[:-1]:
                    for t in T[:-1]:
                        if tau[j, k] > M:
                            model.addConstr(x[t_prime, t, g, j, k] == 0)

    # Vehicle capacity
    for j in J:
        for k in K:
            for t in T[:-1]:
                model.addConstr(gp.quicksum(x[t_prime, t, g, j, k] for g in G for t_prime in T[:t]) <=
                                gp.quicksum(V * v[t, i, j, k] for i in I))
    # One trip per vehicle per time period
    for i in I:
        for t in T[:-1]:
            model.addConstr(gp.quicksum(v[t, i, j, k] for j in J for k in K) <= 1)

    # Blood center vehicles
    for j in J:
        for t in T[:-1]:
            model.addConstr(gp.quicksum(v[t, i, j, k] for i in I for k in K) <= N[j])

    # Blood group compatibility
    for g in G:
        for g_prime in G:
            for k in K:
                for t in T[1:]:
                    for t_prime in T[:-1]:
                        if F[g, g_prime] == 0:
                            model.addConstr(b[t_prime, t, g, g_prime, k] == 0)

    model.setParam('TimeLimit', 3 * 60)  # Set time limit to 3 minutes per model run

    end_time = time.time()
    creation_time = end_time - start_time
    print("Model Creation:", creation_time, "seconds")

    model.optimize()

    rt = model.runtime
    runtimes.append(rt)
    print("RUNTIME:", rt)

    ov = model.objVal
    obj_values.append(ov)
    print("OBJECTIVE VALUE:", ov)

print(runtimes)
print(np.mean(runtimes))

print(obj_values)
print(np.mean(obj_values))
