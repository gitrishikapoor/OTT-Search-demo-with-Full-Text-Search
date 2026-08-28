# Cloud Spanner Multi-Modal OTT Search Demo

A production-grade demonstration of **Multi-Modal Search & Graph Discovery** built on Google Cloud Spanner. This project showcases how to implement high-performance, real-time search capabilities including **Full-Text Search (FTS)**, **Vector/Semantic Search**, **Hybrid Search (FTS + Vector)**, and **Property Graph Queries (using GQL)** inside a single, scalable database engine.

---

## 🚀 Features

- **Full-Text Search (FTS):** Tokenized, sub-string, and n-gram lexical queries across multiple attributes (titles, synopses, cast, audio, subtitles).
- **Semantic / Vector Search:** Dense 768-dimensional vector matching utilizing cosine similarity over Title + Synopsis descriptors.
- **Hybrid Search:** Intelligent combining of lexical FTS matches and semantic vector distances using custom rank-fusion scoring.
- **Spanner Property Graph (GQL):** Leverages Spanner's native Property Graph engine to execute multi-hop relational path-finding (e.g., cast-to-genre relationships, co-viewer affinities). *Experience this in action inside any movie's detailed deep-dive modal!*
- **Interactive UI:** Embedded, responsive React + Tailwind CSS dashboard providing instant search tabs, filters (Audio, Rating, Tier), and deep-dive metadata modals.

---

## 📂 Repository Contents

- `app.py`: FastAPI backend serving the search endpoints and hosting the embedded single-page app (SPA).
- `seed_db.py`: Complete database bootstrap script to populate schema tables, build relationships, and generate mock metadata.
- `schema.ddl`: Complete Cloud Spanner database schema defining base entities, full-text search indexes, and the Property Graph topology.
- `index.html`: Responsive single-page frontend application.
- `requirements.txt`: Python dependency definitions.
- `Dockerfile`: Minimal, optimized Docker build configuration based on `python:3.11-slim`.

---

## 🛠️ Local Developer Setup

### 1. Prerequisites
- **Python:** Version 3.11 or higher.
- **Google Cloud SDK (`gcloud`):** Authenticated with a Google Cloud Project.
- **Cloud Spanner:** A running Spanner instance and database.

### 2. Install Dependencies
Clone this repository, navigate to the directory, create a virtual environment, and install the required packages:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Define your target Spanner connection properties. The scripts will automatically load these:

```bash
export SPANNER_INSTANCE="your-spanner-instance-id"
export SPANNER_DATABASE="your-spanner-database-id"

# (Optional) If running locally with a specific service account key:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### 4. Create the Database Schema
Apply the DDL commands defined in `schema.ddl` to your Cloud Spanner database. You can do this via the Google Cloud Console, or via the `gcloud` CLI:

```bash
gcloud spanner databases ddl update $SPANNER_DATABASE \
    --instance=$SPANNER_INSTANCE \
    --ddl-file=schema.ddl
```

### 5. Seed Mock Data & Pre-computed Embeddings
Run the seeding script to populate the tables and establish the graph relationships:

```bash
python seed_db.py
```

### 6. Launch the FastAPI Server
Run the local development server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```
Open your browser and navigate to `http://localhost:8080` to experience the search interface!

---

## 🐳 Run with Docker

You can package and run the application locally or in any containerized environment:

```bash
# Build the Docker image
docker build -t ott-search-demo .

# Run the container
docker run -p 8080:8080 \
  -e SPANNER_INSTANCE="your-spanner-instance-id" \
  -e SPANNER_DATABASE="your-spanner-database-id" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/key.json" \
  -v /local/path/to/key.json:/app/key.json \
  ott-search-demo
```

---

## ☁️ Deploy to Google Cloud Run

To host the application publicly on Google Cloud Run with a single command:

```bash
gcloud run deploy ott-search-demo \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars="SPANNER_INSTANCE=your-spanner-instance-id,SPANNER_DATABASE=your-spanner-database-id"
```
