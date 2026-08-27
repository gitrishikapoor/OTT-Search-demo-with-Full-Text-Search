# Multi-Modal OTT Search Demo & Walkthrough Guide

Welcome to the **Cloud Spanner Multi-Modal OTT Search Application**. This guide explains the core business use case, system architecture, product features, and provides a step-by-step walkthrough to demonstrate the application's unique multi-modal capabilities.

---

## 📽️ The Business Use Case

In modern Over-The-Top (OTT) streaming platforms (such as Netflix, Disney+, Prime Video), search is the primary driver of user engagement and content discovery. Standard keyword search is no longer sufficient. Users expect:
1. **Typo Resilience:** Resilient search that handles misspellings.
2. **Semantic / Contextual Understanding:** Ability to find titles by searching concepts (e.g., "artificial intelligence rebel" should return *The Matrix* or *Terminator*).
3. **Cross-Table Intelligence:** Natural language search that references actors and their roles (e.g., "salman as a police officer" should map Salman Khan to *Dabangg*).
4. **Graph-based Recommendation:** Finding deeply connected titles based on co-viewer behavior, shared casts, or franchise universes without heavy batch-processed machine learning pipelines.

### The Challenge
Traditionally, engineering this requires building, operating, and synchronizing a complex mesh of separate databases:
- A relational DB for catalog transactions.
- A search engine (e.g., Elasticsearch) for full-text search.
- A vector database (e.g., Pinecone) for semantic matching.
- A graph database (e.g., Neo4j) for relational recommendations.

### The Solution: Cloud Spanner Multi-Modal Engine
This application demonstrates how **Google Cloud Spanner** consolidates all four paradigms into a **single, globally scalable database**:
- **Relational storage** with high transactional integrity.
- **Native Full-Text Search (FTS)** using built-in tokenizers and search indexes.
- **Native Vector Search** using built-in vector distance functions (`COSINE_DISTANCE`).
- **Native Property Graph Queries (GQL)** using built-in graph relational topologies.

---

## 🏗️ System Architecture

The following diagram illustrates how the single-page application (SPA) communicates with the FastAPI backend, which harnesses the full power of Cloud Spanner to serve FTS, vector, and graph results in sub-millisecond latencies:

![OTT Search Architecture](/usr/local/google/home/rikapoor/.gemini/antigravity-cli/brain/5ceb687c-9ebd-42ae-bf82-bb82510af211/ott_search_architecture_1787645681118.png)

---

## 🔍 Core Features & Technical Deep-Dive

### 1. Full-Text Search (FTS)
Spanner uses tokenized indexes to support lexical searches. 
* **Substring & N-grams:** Supports partial string matching, making it highly resilient to incomplete inputs.
* **Storing Clause:** Utilizes the `STORING` parameter in search indexes so Spanner can serve catalog metadata (such as poster URLs, ratings, and release years) directly from the index itself without executing expensive base table joins.

### 2. Vector / Semantic Search
The application maps both content descriptions and user queries into dense 768-dimensional float arrays.
* **Deterministic Matching:** Uses a secure, deterministic hashing vectorizer (`hashlib.sha256`) to ensure reproducible embeddings.
* **Cosine Distance:** Executes high-speed vector distance matching directly within Spanner using `COSINE_DISTANCE()`.

### 3. Hybrid Search (FTS + Vector)
This represents the state-of-the-art in search technology. It executes FTS and Vector search in parallel Common Table Expressions (CTEs), normalizes their scores, and merges the results.
* **Reciprocal Rank Fusion (RRF):** Blends the precision of exact-word keyword matches with the conceptual breadth of semantic searches.
* **Zero-Term Browse Fallback:** If the search terms are entirely extracted for metadata filtering (leaving `target_q` empty), the engine bypasses vector calculations and runs an optimized catalog browser query to prevent noise.

### 4. Spanner Property Graphs (GQL)
Uses Spanner's brand-new native property graph model (`OttKnowledgeGraph`):
* **No Joins Needed:** Translates highly complex multi-table joins into clear, elegant GQL paths like:
  `MATCH (s:Title)-[e:FANS_ALSO_WATCHED]->(t:Title) WHERE s.ContentId = @contentId`
* **Real-time Recommendations:** Generates deeply relevant recommendations based on cast crossovers, micro-genre loops, and user affinity clusters.

---

## 🚶 Step-by-Step Demonstration Walkthrough

Follow these steps to demonstrate the application's features to anyone experiencing the system for the first time.

### 🏁 Step 1: Accessing the Dashboard
1. Open your browser and navigate to the deployed URL:
   👉 **[https://ott-search-demo-213791137710.us-central1.run.app](https://ott-search-demo-213791137710.us-central1.run.app)**
2. Notice the sleek, dark-themed Netflix-style user interface with a global search bar, search-mode tabs (Full-Text Search vs. Hybrid Query vs. Property Graph), and a side filter panel.

---

### 🔍 Step 2: Test Full-Text Search Resiliency
This scenario demonstrates Spanner's substring matching and lexical index retrieval.

1. **Click the "Full-Text Search" tab.**
2. **Search for:** `idiot`
3. **Observe:** *3 Idiots* immediately surfaces.
4. **Search for:** `andaz`
5. **Observe:** *Andaz Apna Apna* surfaces. Spanner easily finds titles using sub-strings and stored index matches.

---

### 🧠 Step 3: Test Semantic and Concept-based Search
This scenario showcases how semantic vector search can understand the *meaning* of a query even if none of the words match the title or description exactly.

1. **Click the "Hybrid Search" tab.**
2. **Search for:** `space time travel sci-fi`
3. **Observe:** *Interstellar* appears as the top result. Spanner's vector comparison successfully maps "space", "time travel", and "sci-fi" to *Interstellar*'s synopsis (which contains terms like "wormhole", "astronaut", "dimensions") even though the word "sci-fi" or "time travel" is not in the movie's database columns.

---

### 🕵️ Step 4: Test Actor-Concept Cross-Table Querying
This scenario demonstrates our custom Intent Parser which separates actor/talent terms from descriptive concepts, queries Spanner's `People` table to fetch the ID on the fly, and applies it as a strict structured filter on the search result.

1. **Stay on the "Hybrid Search" tab.**
2. **Search for:** `salman as a police officer`
3. **Observe:**
   - The application parses out the actor `"salman"` and dynamically resolves it to Salman Khan's database ID (`p-salman`).
   - It filters for movies starring Salman Khan, while using `"police officer"` as the search prompt.
   - **Dabangg** (where Salman stars as police officer Chulbul Pandey) surfaces as the top rank, while movies starring other police officers are strictly filtered out.

4. **Search for:** `aamir khan`
5. **Observe:**
   - The search engine dynamically extracts Aamir Khan, resolves his ID to `p-aamikhan` on startup, and displays all his titles (*3 Idiots*, *Andaz Apna Apna*) with outstanding ranking.

---

### 🎛️ Step 5: Test Synchronized Global Left-Pane Filters
Showcase how left-pane filters apply seamlessly and instantly to both FTS and Hybrid search modes.

1. **Search for:** `comedy`
2. **Apply Left-Pane Filters:**
   - Set **Age Rating** to `PG-13`.
   - Set **Access Tier** to `SVOD_PREMIUM`.
   - Set **Audio Track Language** to `Hindi`.
3. **Observe:** The active tab immediately filters down the results to only match Hindi PG-13 SVOD Premium comedy titles, showing flawless integration between structured SQL filters and unstructured text/vector matches.

---

### 🕸️ Step 6: Test Spanner Property Graph (GQL) Discovery
This is the showstopper scenario demonstrating Spanner's native graph database capabilities.

1. **Click on "3 Idiots" to open its detailed modal.**
2. **Look at the "Fans Also Watched" section at the bottom.**
3. **Observe:** A carousel of recommended movies appears.
4. **How it works:** When you open the modal, the backend executes a fast GQL graph query:
   ```sql
   GRAPH OttKnowledgeGraph
   MATCH (s:Title)-[e:FANS_ALSO_WATCHED]->(t:Title)
   WHERE s.ContentId = @contentId
   RETURN t.ContentId, t.PrimaryTitle, t.PosterUrl, e.AffinityScore
   ORDER BY e.AffinityScore DESC
   ```
   This retrieves connected nodes in the graph in real-time without performing any SQL JOIN statements!

---

> [!TIP]
> **Key Architectural Takeaway:**
> Every single one of these actions — from tokenized text indexes to cosine vector distance and property graph node-matching — was served by **one single Cloud Spanner database instance**. This drastically reduces operational overhead, eliminates data synchronization pipelines, and ensures 100% real-time transactional consistency across all discovery channels.
