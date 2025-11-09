# 🧠 ECDSA Nonce Optimization Framework – Simulated Annealing + ML Refinement

This script implements a **hybrid optimization engine** designed for advanced analysis of
ECDSA signatures, focusing on recovering potential nonce values (`k`) that minimize the
variation in recovered private key candidates (`d`).  

It combines **simulated annealing (SA)**, **hill climbing**, **parallel processing**, and
**machine learning prediction (XGBoost)** to iteratively improve guesses for `k`
based on how consistent resulting `d` values are across signatures.

──────────────────────────────────────────────────────────────
⚙️  PURPOSE

Given multiple ECDSA signatures sharing the same private key, the goal is to find a `k`
value such that all signatures yield identical recovered private keys using:

d = ((s * k - z) * r⁻¹) mod n


If all computed `d` values match (error = 0), the correct `k` (and thus `d`) is found.

──────────────────────────────────────────────────────────────
📚  COMPONENTS OVERVIEW

### 1️⃣ Signature Dataset
A list of observed ECDSA signatures, each containing:
- `r` — ECDSA signature component (integer)
- `s` — ECDSA signature component (integer)
- `z` — message hash (integer)

All values must be integers derived from hex strings.

---

### 2️⃣ Cryptographic Core
```python
@lru_cache(maxsize=None)
def recover_d(r, s, z, k):
    inv_r = inverse_mod(r, n)
    d = ((s * k - z) % n) * inv_r % n
    return d if 1 < d < n else None


→ Computes a candidate private key d for given (r, s, z, k).

3️⃣ Objective Function

Measures how well a k candidate fits all signatures:

error = Σ |d_i - d_j|


Lower error means more consistent d values across all signatures.
error = 0 implies a valid k (all d identical).

4️⃣ Local Hill Climbing

Performs fine-grained search around a given k:

Starts with a step size (step_init)

Moves ±step to find better (lower-error) candidates

Reduces step size if no improvement found

Efficient for quick local refinement of simulated annealing results.

5️⃣ Simulated Annealing (SA)

Probabilistic global search algorithm:

Starts at high temperature T_init

Randomly perturbs k using a Gaussian offset

Accepts worse solutions with decreasing probability as T cools

Continues until T_min or a perfect match (error=0) is found

This helps avoid getting trapped in local minima.

6️⃣ Machine Learning Refinement (XGBoost)

After each optimization cycle, historical (k, error) pairs are recorded.
An XGBoost regression model predicts the next likely candidate region for k:

model = xgb.XGBRegressor()
model.fit(log(k), log(error))
predicted_k = candidate near lowest predicted error


→ This self-learning loop guides future searches more efficiently.

7️⃣ Parallel Execution

Multiple independent simulated annealing workers run in parallel using
ProcessPoolExecutor, each starting from a different randomized initial k.

Best results are merged after all workers finish.

──────────────────────────────────────────────────────────────
🚀 HOW IT WORKS

Define or import multiple ECDSA signatures (r, s, z).

Initialize search parameters (temperature, alpha, etc.).

Run parallel simulated annealing.

Apply hill climbing for local refinement.

Log results and train ML model on search history.

Use model prediction to choose the next starting k.

Repeat until error == 0 (indicating consistent d).

──────────────────────────────────────────────────────────────
⚙️ MAIN PARAMETERS

Parameter	Description	Example Value
T_init	Initial temperature for SA	1e60
T_min	Minimum temperature (stop condition)	1
alpha	Cooling rate per iteration	0.995
max_iter	Max iterations per annealing cycle	10000
num_workers	Number of parallel SA processes	4
step_init	Initial hill climb step size	10**6
min_step	Smallest allowed refinement step	1

──────────────────────────────────────────────────────────────
🧮 OUTPUT

During execution, you’ll see detailed progress logs like:

Iter 0: k = 1.92e+65, error = 3.82e+65, T = 1.00e+60
Hill Climb Iter 4: k = 1.83e+65, error = 4.7e+63, step = 1000000
Predicted best candidate k = 1.82e+65 based on history.


If successful:

✅ Found candidate k with error = 0
🔑 Recovered private key d = 0x12f4a3c9...


──────────────────────────────────────────────────────────────
📦 DEPENDENCIES

Install via pip:

pip install ecdsa numpy xgboost


──────────────────────────────────────────────────────────────
⚠️ DISCLAIMER

This project is for educational and cryptographic research only.
It explores optimization and AI-assisted search algorithms on cryptographic
data structures.
It must not be used for unauthorized key recovery or cryptanalysis of
real-world cryptocurrency systems.

──────────────────────────────────────────────────────────────
✅ SUMMARY

This script merges:

Cryptographic ECDSA analysis

Simulated Annealing global search

Hill climbing local refinement

Parallel multiprocessing

Machine learning prediction feedback (XGBoost)

The result is an adaptive, multi-strategy optimizer for exploring how
numeric search methods can approach cryptographic parameter consistency.

BTC donation address: bc1q4nyq7kr4nwq6zw35pg0zl0k9jmdmtmadlfvqhr
