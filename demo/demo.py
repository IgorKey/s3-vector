"""
End-to-end demo for S3 Vector Store PoC.

Usage:
    1. Start services: docker compose up -d
    2. Run demo: python -m demo.demo

Demonstrates:
    - Creating a collection
    - Batch loading vectors
    - Similarity search
    - Metadata filtering
    - Search with payload
"""

import sys
import time

import httpx

from demo.sample_data import generate_chunks, generate_query_vector

BASE_URL = "http://localhost:8000"
COLLECTION = "confluence-kb"
DIMENSION = 1536


def main():
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    # Health check
    print("=== Health Check ===")
    resp = client.get("/health")
    print(f"Status: {resp.json()}")
    assert resp.status_code == 200

    # Create collection
    print(f"\n=== Creating collection '{COLLECTION}' ===")
    resp = client.post(
        "/collections",
        json={
            "name": COLLECTION,
            "dimension": DIMENSION,
            "distance_metric": "cosine",
        },
    )
    if resp.status_code == 409:
        print("Collection already exists, continuing...")
    else:
        assert resp.status_code == 201
        print(f"Created: {resp.json()}")

    # Load vectors
    chunks = generate_chunks()
    print(f"\n=== Loading {len(chunks)} vectors (batch) ===")

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        resp = client.post(
            f"/collections/{COLLECTION}/vectors:batchPut",
            json={"vectors": batch},
        )
        assert resp.status_code == 200
        print(f"  Batch {i // batch_size + 1}: {resp.json()}")

    # Stats
    print("\n=== Stats ===")
    resp = client.get("/stats")
    print(f"Stats: {resp.json()}")

    # Search
    query_vector = generate_query_vector()
    print("\n=== Similarity Search (top 5) ===")
    start = time.monotonic()
    resp = client.post(
        f"/collections/{COLLECTION}/search",
        json={
            "query_vector": query_vector,
            "top_k": 5,
            "include_payload": True,
            "include_metadata": True,
        },
    )
    elapsed = (time.monotonic() - start) * 1000
    assert resp.status_code == 200
    results = resp.json()["results"]

    for i, r in enumerate(results):
        print(f"\n  #{i + 1} | id: {r['id']} | score: {r['score']:.4f}")
        if r.get("metadata"):
            print(f"       title: {r['metadata'].get('title', 'N/A')}")
            print(f"       space: {r['metadata'].get('space', 'N/A')}")
        if r.get("payload"):
            text = r["payload"].get("text", "")
            print(f"       text: {text[:80]}...")
    print(f"\n  Search latency: {elapsed:.1f}ms")

    # Search with filter
    print("\n=== Search with filter (space=OPS) ===")
    resp = client.post(
        f"/collections/{COLLECTION}/search",
        json={
            "query_vector": query_vector,
            "top_k": 3,
            "include_metadata": True,
            "filter": {"space": "OPS"},
        },
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    for i, r in enumerate(results):
        print(f"  #{i + 1} | id: {r['id']} | score: {r['score']:.4f}")
        if r.get("metadata"):
            print(f"       title: {r['metadata'].get('title', 'N/A')}")

    # Get single vector
    first_id = chunks[0]["id"]
    print(f"\n=== Get vector '{first_id}' ===")
    resp = client.get(f"/collections/{COLLECTION}/vectors/{first_id}")
    assert resp.status_code == 200
    vec_data = resp.json()
    print(f"  id: {vec_data['id']}")
    print(f"  vector dims: {len(vec_data['vector'])}")
    print(f"  metadata: {vec_data.get('metadata')}")

    print("\n=== Demo complete! ===")
    print(
        "\nTo test restart recovery:"
        "\n  1. docker compose restart vector-api"
        "\n  2. python -m demo.demo  (run search again)"
    )


if __name__ == "__main__":
    main()
