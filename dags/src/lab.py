import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import pickle
import os
import base64


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns base64-encoded string.
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes data, drops NaNs, selects features, scales with MinMax,
    and returns base64-encoded pickled array.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    scaler = MinMaxScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    serialized = pickle.dumps(clustering_data_scaled)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Fits GMM for n_components = 1..20, saves the best model (lowest BIC),
    and returns BIC values as a JSON-safe list.
    """
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    bic_scores = []
    models = {}

    for k in range(1, 21):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            max_iter=300
        )
        gmm.fit(data)
        bic = gmm.bic(data)
        bic_scores.append(bic)
        models[k] = gmm
        print(f"k={k}, BIC={bic:.2f}")

    # Save all models
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(models, f)

    print(f"Models saved to {output_path}")
    return bic_scores  # JSON-safe list


def load_model_elbow(filename: str, bic_scores: list):
    """
    Loads saved GMM models, finds optimal n_components via elbow on BIC curve,
    and returns cluster prediction for test.csv.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    with open(output_path, "rb") as f:
        models = pickle.load(f)

    # Find elbow on BIC curve (decreasing = lower BIC is better)
    k_range = list(range(1, 21))
    kl = KneeLocator(k_range, bic_scores, curve="convex", direction="decreasing")
    optimal_k = kl.elbow or 1  # fallback to 1 if no elbow found
    print(f"Optimal no. of clusters (BIC elbow): {optimal_k}")

    # Use best model
    best_model = models[optimal_k]

    # Predict on test.csv
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = best_model.predict(test_df)[0]
    print(f"Predicted cluster for test sample: {pred}")

    return int(pred)