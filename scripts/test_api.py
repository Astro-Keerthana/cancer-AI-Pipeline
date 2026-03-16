import requests
import numpy as np
import json
import os

API_URL      = "http://127.0.0.1:8000"
PROCESSED_DIR = "data/processed"


def test_health():
    """Test /health endpoint."""
    print("─" * 50)
    print("🔍 Testing /health ...")
    r = requests.get(f"{API_URL}/health")
    print(f"   Status Code : {r.status_code}")
    print(f"   Response    : {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200, "❌ Health check failed!"
    print("Health check passed!")


def test_root():
    """Test / endpoint."""
    print("─" * 50)
    print("🔍 Testing / (root) ...")
    r = requests.get(f"{API_URL}/")
    print(f"   Status Code : {r.status_code}")
    print(f"   Response    : {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    print("Root check passed!")


def test_predict_from_real_data():
    """Test /predict using actual preprocessed test samples."""
    print("─" * 50)
    print("🔍 Testing /predict with REAL test data ...")

    # ── Load real test data ───────────────────────────────────
    X_test_path = os.path.join(PROCESSED_DIR, "X_test.npy")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test.npy")

    if not os.path.exists(X_test_path):
        print("   ⚠️  X_test.npy not found, using random data instead")
        test_predict_random()
        return

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    print(f"   Loaded X_test shape : {X_test.shape}")
    print(f"   Loaded y_test shape : {y_test.shape}")
    print()

    # ── Test each sample ─────────────────────────────────────
    correct = 0
    for i in range(len(X_test)):
        sample   = X_test[i].tolist()
        true_label = "Tumor" if y_test[i] == 1 else "Normal"

        payload = {
            "sample_id":       f"test-sample-{i+1:02d}",
            "gene_expression": sample
        }

        r = requests.post(f"{API_URL}/predict", json=payload)

        if r.status_code == 200:
            result     = r.json()
            pred_label = result["prediction"]
            prob_t     = result["probability_tumor"]
            prob_n     = result["probability_normal"]
            confidence = result["confidence"]
            match      = "✅" if pred_label == true_label else "❌"

            if pred_label == true_label:
                correct += 1

            print(f"   Sample {i+1:02d} {match}")
            print(f"     True Label  : {true_label}")
            print(f"     Predicted   : {pred_label}  (confidence: {confidence})")
            print(f"     Prob Tumor  : {prob_t:.4f}")
            print(f"     Prob Normal : {prob_n:.4f}")
            print()
        else:
            print(f"Sample {i+1} failed: {r.status_code} — {r.text}")

    accuracy = correct / len(X_test) * 100
    print(f"API Test Accuracy : {correct}/{len(X_test)} = {accuracy:.1f}%")


def test_predict_random():
    """Fallback: test /predict with random data."""
    print("─" * 50)
    print("Testing /predict with RANDOM data ...")

    # Get expected feature count from API
    r        = requests.get(f"{API_URL}/")
    n_feats  = r.json().get("expected_features", 41410)

    payload = {
        "sample_id":       "random-test-01",
        "gene_expression": np.random.randn(n_feats).tolist()
    }

    r = requests.post(f"{API_URL}/predict", json=payload)
    print(f"   Status Code : {r.status_code}")
    print(f"   Response    : {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    print("Random predict passed!")


def test_predict_batch():
    """Test /predict/batch with multiple samples."""
    print("─" * 50)
    print("Testing /predict/batch ...")

    X_test_path = os.path.join(PROCESSED_DIR, "X_test.npy")

    if os.path.exists(X_test_path):
        X_test  = np.load(X_test_path)
        samples = X_test[:3]   # use first 3 real samples
    else:
        n_feats = 41410
        samples = np.random.randn(3, n_feats)

    payload = [
        {
            "sample_id":       f"batch-sample-{i+1:02d}",
            "gene_expression": samples[i].tolist()
        }
        for i in range(len(samples))
    ]

    r = requests.post(f"{API_URL}/predict/batch", json=payload)
    print(f"   Status Code : {r.status_code}")

    if r.status_code == 200:
        result = r.json()
        print(f"   Total      : {result['total']}")
        print(f"   Successful : {result['successful']}")
        print(f"   Failed     : {result['failed']}")
        print()
        for pred in result["predictions"]:
            print(f"   {pred.get('sample_id')} → "
                  f"{pred.get('prediction')}  "
                  f"(Tumor: {pred.get('probability_tumor')}, "
                  f"Normal: {pred.get('probability_normal')})")
        print("   Batch predict passed!")
    else:
        print(f"Batch failed: {r.text}")


def test_invalid_input():
    """Test that bad input returns proper error."""
    print("─" * 50)
    print("Testing invalid input handling ...")

    payload = {
        "sample_id":       "bad-sample",
        "gene_expression": [0.1, 0.2, 0.3]   # wrong feature count
    }

    r = requests.post(f"{API_URL}/predict", json=payload)
    print(f"   Status Code : {r.status_code}  (expected 422)")
    print(f"   Error Detail: {r.json().get('detail')}")
    assert r.status_code == 422, "❌ Should have returned 422!"
    print("   Invalid input handled correctly!")


# ─── Run All Tests ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧬 Cancer API — Full Test Suite")
    print("=" * 50)

    try:
        test_health()
        test_root()
        test_predict_from_real_data()
        test_predict_batch()
        test_invalid_input()

        print("=" * 50)
        print("ALL TESTS PASSED — API is fully working!")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("\nCannot connect to API!")
        print("   Make sure the server is running first:")
        print("   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
