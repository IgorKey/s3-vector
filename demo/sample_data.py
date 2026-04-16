"""Generate synthetic document chunks with random embeddings for demo."""

import numpy as np

DIMENSION = 1536

DOCUMENTS = [
    {
        "document_id": "doc-1",
        "source": "confluence",
        "space": "ENG",
        "title": "How to create a Jira issue",
        "chunks": [
            "To create a Jira issue, navigate to the project board and click 'Create'.",
            "Fill in the summary, description, and assignee fields. Choose the issue type.",
            "Set the priority and sprint. Add labels if needed. Click 'Create' to submit.",
        ],
    },
    {
        "document_id": "doc-2",
        "source": "confluence",
        "space": "ENG",
        "title": "Git branching strategy",
        "chunks": [
            "We use trunk-based development with short-lived feature branches.",
            "Branch names follow the pattern: feature/JIRA-123-short-description.",
            "All branches must pass CI before merging. Squash merge is preferred.",
        ],
    },
    {
        "document_id": "doc-3",
        "source": "confluence",
        "space": "OPS",
        "title": "Incident response process",
        "chunks": [
            "When an incident is detected, create a Slack channel #inc-YYYY-MM-DD-title.",
            "Assign an incident commander and communicate status every 15 minutes.",
            "After resolution, write a post-mortem within 48 hours.",
        ],
    },
    {
        "document_id": "doc-4",
        "source": "confluence",
        "space": "ENG",
        "title": "Code review guidelines",
        "chunks": [
            "All code changes require at least one approval before merging.",
            "Focus on correctness, readability, and test coverage in reviews.",
            "Be constructive in comments. Suggest improvements, don't just criticize.",
        ],
    },
    {
        "document_id": "doc-5",
        "source": "confluence",
        "space": "OPS",
        "title": "Deployment pipeline",
        "chunks": [
            "Our CI/CD pipeline runs on GitHub Actions with staging and production environments.",
            "Staging deploys automatically on merge to main. Production requires manual approval.",
            "Rollbacks are automated — revert the merge commit and the pipeline handles the rest.",
        ],
    },
]


def generate_chunks():
    """Generate chunks with synthetic embeddings."""
    np.random.seed(42)
    chunks = []
    for doc in DOCUMENTS:
        for i, text in enumerate(doc["chunks"]):
            vec = np.random.randn(DIMENSION).astype(np.float32)
            vec = vec / np.linalg.norm(vec)

            chunks.append(
                {
                    "id": f"{doc['document_id']}-chunk-{i}",
                    "vector": vec.tolist(),
                    "metadata": {
                        "document_id": doc["document_id"],
                        "source": doc["source"],
                        "space": doc["space"],
                        "title": doc["title"],
                        "chunk_no": i,
                    },
                    "payload": {"text": text},
                }
            )
    return chunks


def generate_query_vector():
    """Generate a query vector (simulating 'how to create an issue')."""
    np.random.seed(42)
    vec = np.random.randn(DIMENSION).astype(np.float32)
    noise = np.random.randn(DIMENSION).astype(np.float32) * 0.1
    vec = vec + noise
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()
