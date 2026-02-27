# Airflow Lab 1 - Customer Segmentation Pipeline (GMM Clustering)

A machine learning workflow using Apache Airflow and Docker that automates customer segmentation using Gaussian Mixture Model (GMM) clustering and determines the optimal number of clusters using the elbow method on BIC scores.

---

## Project Structure

```
airflow_lab1/
├── config/
├── dags/
│   ├── data/
│   │   ├── file.csv          # Credit card customer dataset
│   │   └── test.csv          # Test sample (BALANCE, PURCHASES, CREDIT_LIMIT)
│   ├── model/
│   │   └── model.sav         # Saved GMM models (k=1..20)
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py            # ML functions (GMM)
│   └── airflow.py            # DAG definition (TaskFlow API)
├── logs/                     # Airflow task logs
├── plugins/
├── working_data/
├── .env                      # Airflow UID config
├── pyproject.toml            # uv project config
└── docker-compose.yaml       # Docker services config
```

---

## Dataset

Credit card customer data with 18 features including:

| Feature | Description |
|---|---|
| `BALANCE` | Balance amount left in account |
| `PURCHASES` | Amount of purchases made |
| `CREDIT_LIMIT` | Credit limit of the card |
| `CASH_ADVANCE` | Cash advance amount |
| `PAYMENTS` | Amount paid by customer |
| ... | 13 more features |

Features used for clustering: `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Apache Airflow 2.9.2 | Workflow orchestration |
| Docker & Docker Compose | Containerized environment |
| PostgreSQL 13 | Airflow metadata database |
| Redis 7.2 | Celery message broker |
| scikit-learn | GMM clustering + MinMaxScaler |
| kneed | Elbow method on BIC curve |
| pandas | Data loading and preprocessing |
| uv | Local Python environment management |

---

## ML Pipeline

The DAG `Airflow_Lab1` consists of 4 sequential tasks:

```
task_load_data → task_data_preprocessing → task_build_save_model → task_load_model_elbow
```

### Task 1: `task_load_data`
- Reads `file.csv` from `dags/data/`
- Serializes the DataFrame using `pickle` + `base64` encoding (JSON-safe for XCom)
- Returns base64-encoded string

### Task 2: `task_data_preprocessing`
- Deserializes the data
- Drops rows with NaN values
- Selects features: `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`
- Scales features using `MinMaxScaler`
- Returns scaled data as base64-encoded string

### Task 3: `task_build_save_model`
- Fits GMM for n_components = 1 to 20
- Computes BIC score for each k (lower = better fit)
- Saves all models to `dags/model/model.sav`
- Returns BIC scores as JSON-safe list

### Task 4: `task_load_model_elbow`
- Loads saved GMM models
- Applies `KneeLocator` on BIC curve to find optimal number of clusters
- Predicts cluster for test sample in `test.csv`
- Prints and returns the predicted cluster

**Why GMM over KMeans?**
GMM is a probabilistic model that assigns soft cluster memberships, making it more flexible for clusters of varying shapes and sizes — better suited for real-world customer segmentation.

**Why BIC instead of SSE?**
GMM doesn't have inertia/SSE. BIC (Bayesian Information Criterion) penalizes model complexity and is the natural equivalent for finding the optimal number of components in a GMM.

---

## DAG Design

The DAG uses the **TaskFlow API** (decorator style) — the modern recommended approach for Airflow 2.x+:

```python
@dag(dag_id='Airflow_Lab1', schedule=None, catchup=False, ...)
def airflow_lab1():

    @task
    def task_load_data(): ...

    @task
    def task_data_preprocessing(data): ...

    @task
    def task_build_save_model(data): ...

    @task
    def task_load_model_elbow(bic_scores): ...

    data         = task_load_data()
    preprocessed = task_data_preprocessing(data)
    bic_scores   = task_build_save_model(preprocessed)
    task_load_model_elbow(bic_scores)
```

**XCom data passing:** Data between tasks is passed as `base64`-encoded pickle strings (JSON-safe), avoiding XCom size issues.

---

## Setup & Installation

### Prerequisites
- Docker Desktop (4GB+ RAM allocated)
- Python 3.8+
- uv (Python package manager)

### Step 1: Clone and navigate to project
```bash
cd airflow_lab1
```

### Step 2: Set up local Python environment
```bash
uv venv
source .venv/bin/activate
uv add pandas scikit-learn kneed apache-airflow
```

> `apache-airflow` is optional locally — only needed for IDE autocomplete. The pipeline runs inside Docker.

### Step 3: Data files
- `file.csv` and `test.csv` are already provided in `dags/data/` from the course GitHub repo.
- No additional data generation needed.

### Step 4: Configure environment
```bash
echo "AIRFLOW_UID=50000" > .env
```

### Step 5: Fetch Docker Compose file
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'
```

### Step 6: Key docker-compose.yaml changes made
- Disabled example DAGs: `AIRFLOW__CORE__LOAD_EXAMPLES: 'false'`
- Added required packages: `_PIP_ADDITIONAL_REQUIREMENTS: pandas scikit-learn kneed`
- Set `airflow-init` packages to empty: `_PIP_ADDITIONAL_REQUIREMENTS: ''`
- Changed credentials: `airflow2/airflow2`
- Changed postgres port to `5433:5432` (5432 was occupied locally)
- Added `working_data` volume mount
- Added `networks: default` to all services including `airflow-cli` and `flower`
- Added custom Docker network at bottom: `networks: default: driver: bridge`

### Step 7: Initialize and start Airflow
```bash
# Initialize database and create admin user (first time only)
docker compose up airflow-init

# Start all services in background
docker compose up -d
```

### Step 8: Access Airflow UI
- URL: http://localhost:8080
- Username: `airflow2`
- Password: `airflow2`

---

## Running the Pipeline

1. Open http://localhost:8080
2. Find **Airflow_Lab1** in the DAGs list (may take ~30 seconds to appear)
3. Click the **Trigger DAG** ▶️ button
4. Monitor tasks turning green in the **Graph** tab
5. Click `task_load_model_elbow` → **Logs** to see the result

---

## Viewing Results

Navigate to: **DAG → Graph → task_load_model_elbow → Logs**

```
INFO - k=1, BIC=XXXX
INFO - k=2, BIC=XXXX
...
INFO - Optimal no. of clusters (BIC elbow): X
INFO - Predicted cluster for test sample: X
```

---

## Updating the Pipeline

Since `dags/` is mounted as a Docker volume, any changes to `lab.py` or `airflow.py` are **instantly reflected** inside the container — no restart needed. Just:

1. Edit `dags/src/lab.py` locally
2. Trigger the DAG again in the UI

---

## Stopping Airflow

```bash
docker compose down
```

To remove all volumes and start fresh:
```bash
docker compose down -v
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Port 5432 already in use | Changed postgres port to `5433:5432` |
| Port 8080 already in use | Stop other running Airflow instances via `docker compose down` |
| DAG not showing in UI | Wait ~30 seconds for scheduler to pick it up |
| `airflow-init` exit code 1 (pip as root) | Set `_PIP_ADDITIONAL_REQUIREMENTS: ''` in airflow-init service |
| Containers can't find `postgres` hostname | Add custom network to all services in docker-compose.yaml |
| `ModuleNotFoundError: src.lab` | Ensure `dags/src/__init__.py` exists |
| XCom size errors | Use `base64` + `pickle` encoding for large data objects |

---

## Author

**Ankit Kumar Sahoo**
Northeastern University - Data Analytics Engineering