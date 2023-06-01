# Optimizing Blood Supply Distribution

The data files and code files required to run an integer program and reinforcement learning model using Proximal Policy Optimization (PPO) to optimize the distribution of blood in a national blood supply system. The case study applied here is on the Malaysian national blood banking system. 

## Data
* The [Data Files](https://github.com/ameenaelagha/blood-supply-chain/tree/main/Data%20Files) directory contains the geocodes for blood centers and hospitals, as well as the driving time between each blood center-hospital pair for 22 main blood centers and 147 hospitals in Malaysia.
* The [Instance Data](https://github.com/ameenaelagha/blood-supply-chain/tree/main/Instance%20Data) directory includes donation data and request data for three scenarios: small, medium, and large. The donation data is subset from the public [Malaysian Ministry of Health data](https://github.com/MoH-Malaysia/data-darah-public/blob/main/donations_facility.csv). The Rh types for each set of donations are simulated, as are the demand datasets.
* The [Data Preprocessing](https://github.com/ameenaelagha/blood-supply-chain/blob/main/Data%20Preprocessing.ipynb) notebook shows the steps for preparing the data for input to both models.

## Models
* [integer_program.py](https://github.com/ameenaelagha/blood-supply-chain/blob/main/integer_program.py) contains the the Python code for the integer program, implemented using GurobiPy. The code is designed to generate `n_runs` sample instances of duration `d` and compute the average objective function value and runtime for each configuration. 
