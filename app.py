import hashlib
import random
import datetime
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import spanner

app = FastAPI(
    title="Cloud Spanner Multi-Modal OTT Search API",
    description="Backend API powering Full-Text, Hybrid (Vector + Lexical), and Hyper-Graph Discovery Searches",
    version="1.0.0"
)

# Enable CORS so frontend React app can communicate with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# Connect to Spanner using configurable environment variables
SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE", "your-spanner-instance-id")
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE", "your-spanner-database-id")

spanner_client = spanner.Client()
instance = spanner_client.instance(SPANNER_INSTANCE)
database = instance.database(SPANNER_DATABASE)

# ============================================================================
# EMBEDDING GENERATOR (Deterministic, matches seed_db.py exactly)
# ============================================================================
def get_embedding(text: str) -> list[float]:
    hash_val = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = random.Random(hash_val)
    
    vec = []
    for _ in range(768):
        vec.append(rng.gauss(0, 1))
        
    genres = ["scifi", "space", "crime", "drama", "anime", "action", "thriller", "mystery", "fantasy", "history"]
    matched_genres = [g for g in genres if g in text.lower()]
    for g in matched_genres:
        g_hash = int(hashlib.sha256(g.encode('utf-8')).hexdigest(), 16) % (2**32)
        g_rng = random.Random(g_hash)
        for i in range(768):
            vec[i] += 1.5 * g_rng.gauss(0, 1)
            
    sq_sum = sum(x*x for x in vec)
    norm = sq_sum ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

# ============================================================================
# API MODELS & ROUTERS
# ============================================================================

class TitleDetail(BaseModel):
    ContentId: str
    PrimaryTitle: str
    ContentType: str
    ReleaseYear: int
    AgeRating: str
    PosterUrl: Optional[str]
    ImdbRating: Optional[float]
    AccessTier: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading index.html</h1><p>{str(e)}</p>", status_code=500)

# Endpoint to get list of titles for seeding selection/dropdowns
@app.get("/api/titles")
def get_all_titles(limit: int = 100):
    query = "SELECT ContentId, PrimaryTitle, ReleaseYear, PosterUrl, PopularityScore FROM Titles ORDER BY PopularityScore DESC LIMIT @limit"
    try:
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query,
                params={"limit": limit},
                param_types={"limit": spanner.param_types.INT64}
            )
            titles = []
            for row in results:
                titles.append({
                    "ContentId": row[0],
                    "PrimaryTitle": row[1],
                    "ReleaseYear": row[2],
                    "PosterUrl": row[3],
                    "PopularityScore": row[4]
                })
            return {"success": True, "titles": titles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spanner query error: {str(e)}")

# Get single Title details for popup modal
@app.get("/api/titles/{content_id}")
def get_title_details(content_id: str):
    query = """
    SELECT 
        ContentId, PrimaryTitle, OriginalTitle, ContentType, ReleaseYear, AgeRating,
        DurationMins, SeasonsCount, Synopsis, Tagline, PosterUrl, BannerUrl,
        TrailerUrl, ImdbRating, PopularityScore, AccessTier, AudioLanguages,
        SubtitleLanguages, QualityProfiles
    FROM Titles
    WHERE ContentId = @content_id
    """
    try:
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query,
                params={"content_id": content_id},
                param_types={"content_id": spanner.param_types.STRING}
            )
            rows = list(results)
            if not rows:
                raise HTTPException(status_code=404, detail="Title not found")
            
            row = rows[0]
            # Fetch genres as well
            genre_query = """
            SELECT g.Name, g.Slug
            FROM TitleGenres tg
            JOIN Genres g ON tg.GenreId = g.GenreId
            WHERE tg.ContentId = @content_id
            """
            genre_results = snapshot.execute_sql(
                genre_query,
                params={"content_id": content_id},
                param_types={"content_id": spanner.param_types.STRING}
            )
            genres = [{"Name": r[0], "Slug": r[1]} for r in genre_results]

            # Fetch cast/talent
            talent_query = """
            SELECT p.FullName, tt.Role, tt.CharacterName, p.ProfileImageUrl
            FROM TitleTalent tt
            JOIN People p ON tt.PersonId = p.PersonId
            WHERE tt.ContentId = @content_id
            ORDER BY tt.BillingOrder ASC
            """
            talent_results = snapshot.execute_sql(
                talent_query,
                params={"content_id": content_id},
                param_types={"content_id": spanner.param_types.STRING}
            )
            talent = [{"FullName": r[0], "Role": r[1], "CharacterName": r[2], "ProfileImageUrl": r[3]} for r in talent_results]

            return {
                "success": True,
                "title": {
                    "ContentId": row[0],
                    "PrimaryTitle": row[1],
                    "OriginalTitle": row[2],
                    "ContentType": row[3],
                    "ReleaseYear": row[4],
                    "AgeRating": row[5],
                    "DurationMins": row[6],
                    "SeasonsCount": row[7],
                    "Synopsis": row[8],
                    "Tagline": row[9],
                    "PosterUrl": row[10],
                    "BannerUrl": row[11],
                    "TrailerUrl": row[12],
                    "ImdbRating": row[13],
                    "PopularityScore": row[14],
                    "AccessTier": row[15],
                    "AudioLanguages": row[16],
                    "SubtitleLanguages": row[17],
                    "QualityProfiles": row[18],
                    "Genres": genres,
                    "Talent": talent
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spanner error: {str(e)}")

# ============================================================================
# TAB 1: ADVANCED FULL-TEXT SEARCH (FTS)
# ============================================================================
def preprocess_fts_query(q: str) -> str:
    if not q:
        return ""
    cleaned = "".join([c if c.isalnum() or c.isspace() else " " for c in q])
    return " ".join(cleaned.strip().split())
def get_substituted_fts_query(q, age_rating, access_tier, audio_lang, limit):
    escaped_q = q.replace("'", "''")
    escaped_q_lower = escaped_q.lower()
    search_q = f"'{escaped_q}'"
    
    escaped_age = age_rating.replace("'", "''") if age_rating else None
    age_f = f"'{escaped_age}'" if escaped_age else "NULL"
    
    escaped_access = access_tier.replace("'", "''") if access_tier else None
    access_f = f"'{escaped_access}'" if escaped_access else "NULL"
    
    escaped_audio = audio_lang.replace("'", "''") if audio_lang else None
    audio_f = f"'{escaped_audio}'" if escaped_audio else "NULL"
    lim = str(limit)
    
    return f"""@{{OPTIMIZER_VERSION=6}}
WITH MatchedTitles AS (
    -- Direct FTS Match on Titles
    SELECT 
        t.ContentId,
        COALESCE(SCORE(t.SearchTokens, {search_q}), 0.1) AS MatchScore
    FROM Titles AS t
    WHERE SEARCH(t.SearchTokens, {search_q})

    UNION ALL

    -- Substring Title Match
    SELECT 
        t.ContentId,
        1.1 AS MatchScore
    FROM Titles AS t
    WHERE SEARCH_SUBSTRING(t.PrimaryTitleNgramTokens, {search_q})

    UNION ALL

    -- Genre Match
    SELECT 
        tg.ContentId,
        1.5 AS MatchScore
    FROM TitleGenres tg
    JOIN Genres g ON tg.GenreId = g.GenreId
    WHERE SEARCH(g.SearchTokens, {search_q})

    UNION ALL

    -- Talent Match
    SELECT 
        tt.ContentId,
        1.2 AS MatchScore
    FROM TitleTalent tt
    JOIN People p ON tt.PersonId = p.PersonId
    WHERE SEARCH(p.SearchTokens, {search_q})

    UNION ALL

    -- Franchise Match
    SELECT 
        tf.ContentId,
        1.3 AS MatchScore
    FROM TitleFranchise tf
    JOIN Franchises f ON tf.FranchiseId = f.FranchiseId
    WHERE SEARCH(f.SearchTokens, {search_q})

    UNION ALL

    -- Alias Match
    SELECT 
        ta.ContentId,
        1.4 AS MatchScore
    FROM TitleAliases ta
    WHERE SEARCH(ta.SearchTokens, {search_q})
),
AggregatedScores AS (
    SELECT 
        ContentId,
        MAX(MatchScore) AS FinalScore
    FROM MatchedTitles
    GROUP BY ContentId
)
SELECT 
    t.ContentId,
    t.PrimaryTitle,
    t.ContentType,
    t.ReleaseYear,
    t.AgeRating,
    t.PosterUrl,
    t.ImdbRating,
    t.AccessTier,
    agg.FinalScore AS RelevanceScore,
    t.Synopsis
FROM Titles AS t
JOIN AggregatedScores agg ON t.ContentId = agg.ContentId
WHERE ({age_f} IS NULL OR t.AgeRating = {age_f})
  AND ({access_f} IS NULL OR t.AccessTier = {access_f})
  AND ({audio_f} IS NULL OR {audio_f} IN UNNEST(t.AudioLanguages))
ORDER BY RelevanceScore DESC, t.PopularityScore DESC
LIMIT {lim};"""

@app.get("/api/search/fts")
def search_fts(
    q: str = Query(..., description="Full-Text Search query"),
    age_rating: Optional[str] = Query(None, description="Age rating filter"),
    access_tier: Optional[str] = Query(None, description="Access tier filter"),
    audio_lang: Optional[str] = Query(None, description="Audio language filter"),
    limit: int = Query(20, description="Results limit")
):
    query = """
    @{OPTIMIZER_VERSION=6}
    WITH MatchedTitles AS (
        -- Direct FTS Match
        SELECT 
            t.ContentId,
            COALESCE(SCORE(t.SearchTokens, @searchQuery), 0.1) AS MatchScore
        FROM Titles AS t
        WHERE SEARCH(t.SearchTokens, @searchQuery)

        UNION ALL

        -- Substring Title Match
        SELECT 
            t.ContentId,
            1.1 AS MatchScore
        FROM Titles AS t
        WHERE SEARCH_SUBSTRING(t.PrimaryTitleNgramTokens, @searchQuery)

        UNION ALL

        -- Genre Match
        SELECT 
            tg.ContentId,
            1.5 AS MatchScore
        FROM TitleGenres tg
        JOIN Genres g ON tg.GenreId = g.GenreId
        WHERE SEARCH(g.SearchTokens, @searchQuery)


        UNION ALL


        -- Talent Match
        SELECT 
            tt.ContentId,
            1.2 AS MatchScore
        FROM TitleTalent tt
        JOIN People p ON tt.PersonId = p.PersonId
        WHERE SEARCH(p.SearchTokens, @searchQuery)


        UNION ALL


        -- Franchise Match
        SELECT 
            tf.ContentId,
            1.3 AS MatchScore
        FROM TitleFranchise tf
        JOIN Franchises f ON tf.FranchiseId = f.FranchiseId
        WHERE SEARCH(f.SearchTokens, @searchQuery)


        UNION ALL


        -- Alias Match
        SELECT 
            ta.ContentId,
            1.4 AS MatchScore
        FROM TitleAliases ta
        WHERE SEARCH(ta.SearchTokens, @searchQuery)
    ),
    AggregatedScores AS (
        SELECT 
            ContentId,
            MAX(MatchScore) AS FinalScore
        FROM MatchedTitles
        GROUP BY ContentId
    )
    SELECT 
        t.ContentId,
        t.PrimaryTitle,
        t.ContentType,
        t.ReleaseYear,
        t.AgeRating,
        t.PosterUrl,
        t.ImdbRating,
        t.AccessTier,
        agg.FinalScore AS RelevanceScore,
        t.Synopsis
    FROM Titles AS t
    JOIN AggregatedScores agg ON t.ContentId = agg.ContentId
    WHERE (@ageRatingFilter IS NULL OR t.AgeRating = @ageRatingFilter)
      AND (@accessTierFilter IS NULL OR t.AccessTier = @accessTierFilter)
      AND (@audioLangFilter IS NULL OR @audioLangFilter IN UNNEST(t.AudioLanguages))
    ORDER BY RelevanceScore DESC, t.PopularityScore DESC
    LIMIT @limit
    """
    
    processed_q = preprocess_fts_query(q)
    
    params = {
        "searchQuery": processed_q if processed_q else q,
        "ageRatingFilter": age_rating,
        "accessTierFilter": access_tier,
        "audioLangFilter": audio_lang,
        "limit": limit
    }
    param_types = {
        "searchQuery": spanner.param_types.STRING,
        "ageRatingFilter": spanner.param_types.STRING,
        "accessTierFilter": spanner.param_types.STRING,
        "audioLangFilter": spanner.param_types.STRING,
        "limit": spanner.param_types.INT64
    }

    try:
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(query, params=params, param_types=param_types)
            data = []
            for r in results:
                # Calculate simple snippet highlight on our own in case SNIPPET() function fails
                synopsis = r[9]
                highlighted_syn = synopsis
                # Simple backend keyword highlighting to ensure high visual fidelity
                for term in q.split():
                    if len(term) > 2:
                        highlighted_syn = highlighted_syn.replace(term, f"<mark class='bg-yellow-500/30 text-white px-1 rounded'>{term}</mark>")
                
                data.append({
                    "ContentId": r[0],
                    "PrimaryTitle": r[1],
                    "ContentType": r[2],
                    "ReleaseYear": r[3],
                    "AgeRating": r[4],
                    "PosterUrl": r[5],
                    "ImdbRating": r[6],
                    "AccessTier": r[7],
                    "RelevanceScore": r[8],
                    "HighlightedSynopsis": highlighted_syn
                })
            substituted_q = get_substituted_fts_query(processed_q if processed_q else q, age_rating, access_tier, audio_lang, limit)
            return {
                "success": True, 
                "query": q, 
                "results": data,
                "executed_queries": {
                    "parameterized": query.strip(),
                    "substituted": substituted_q.strip()
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spanner FTS error: {str(e)}")

# ============================================================================
# TAB 2: HYBRID SEARCH (FTS + VECTOR SIMILARITY)
# ============================================================================
# Global feature flag for quick rollback capability. Set to False to disable
# semantic intent parsing and metadata filtering globally.
INTENT_PARSING_ENABLED = True

# Talent keyword mappings used by the intent parser.
# Initialized with a static fallback map, and populated dynamically on startup
# from the database to ensure 100% synchronization and prevent typo issues.
talent_keywords = {
    "salman khan": "p-salman",
    "salman": "p-salman",
    "shah rukh khan": "p-srk",
    "shah rukh": "p-srk",
    "srk": "p-srk",
    "aamir khan": "p-aamikhan",
    "aamir": "p-aamikhan",
    "christopher nolan": "p-nolan",
    "nolan": "p-nolan",
    "leonardo dicaprio": "p-dicaprio",
    "dicaprio": "p-dicaprio",
    "cillian murphy": "p-murphy",
    "cillian": "p-murphy",
    "murphy": "p-murphy",
    "robert downey": "p-rdj",
    "rdj": "p-rdj",
    "amitabh bachchan": "p-amitabh",
    "amitabh": "p-amitabh",
    "zendaya": "p-zendaya",
    "chalamet": "p-chalamet",
    "timothee chalamet": "p-chalamet",
}

@app.on_event("startup")
def load_talent_keywords():
    global talent_keywords
    try:
        query = "SELECT PersonId, FullName, KnownAs FROM People"
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(query)
            new_keywords = {}
            for row in results:
                person_id, full_name, known_as = row
                
                # Normalize full name
                fn_lower = full_name.lower().strip()
                new_keywords[fn_lower] = person_id
                
                # Split first and last name
                parts = fn_lower.split()
                if len(parts) > 1:
                    first_name = parts[0]
                    last_name = parts[-1]
                    if first_name not in ["bobby", "al", "tom"]:
                        new_keywords[first_name] = person_id
                    if last_name not in ["khan", "murphy", "bale", "singh", "bobb", "bobby", "lee"]:
                        new_keywords[last_name] = person_id
                        
                # Normalize KnownAs alias
                if known_as:
                    ka_lower = known_as.lower().strip()
                    new_keywords[ka_lower] = person_id
            
            # Merge with default fallback list
            talent_keywords.update(new_keywords)
            print(f"Successfully loaded {len(talent_keywords)} talent keyword mappings dynamically from Spanner!")
    except Exception as e:
        print(f"Warning: Could not pre-load talent keywords dynamically: {str(e)}")

def parse_search_intent(q: str):
    """
    Parses a raw natural language query to extract structured metadata signals.
    Returns: (cleaned_query_text, detected_lang_code, detected_content_type, detected_genre_id)
    """
    if not q:
        return "", None, None, None
        
    q_lower = q.lower()
    
    # 1. Detect language keyword
    detected_lang = None
    lang_keywords = {
        "hindi": "hi",
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "telugu": "te",
        "tamil": "ta",
        "japanese": "ja"
    }
    for keyword, code in lang_keywords.items():
        if keyword in q_lower:
            detected_lang = code
            q_lower = q_lower.replace(keyword, "").strip()
            
    # 2. Detect content type keyword
    detected_type = None
    type_keywords = {
        "tv shows": "TV_SERIES",
        "tv show": "TV_SERIES",
        "tv series": "TV_SERIES",
        "series": "TV_SERIES",
        "shows": "TV_SERIES",
        "show": "TV_SERIES",
        "movies": "MOVIE",
        "movie": "MOVIE"
    }
    for keyword, val in type_keywords.items():
        if keyword in q_lower:
            detected_type = val
            q_lower = q_lower.replace(keyword, "").strip()
            
    # 3. Detect genre keyword (g-comedy, etc.)
    detected_genre = None
    genre_keywords = {
        "comedy": "g-comedy",
        "humor": "g-comedy",
        "comic": "g-comedy",
        "funny": "g-comedy",
        "action": "g-action",
        "adventure": "g-action",
        "anime": "g-anime",
        "animation": "g-anime",
        "animated": "g-anime",
        "crime": "g-crime",
        "mystery": "g-crime",
        "detective": "g-crime",
        "drama": "g-drama",
        "emotional": "g-drama",
        "fantasy": "g-fantasy",
        "myth": "g-fantasy",
        "mythology": "g-fantasy",
        "history": "g-history",
        "historical": "g-history",
        "biography": "g-history",
        "biopic": "g-history",
        "sci-fi": "g-scifi",
        "scifi": "g-scifi",
        "space": "g-scifi",
        "alien": "g-scifi",
        "thriller": "g-thriller",
        "suspense": "g-thriller",
        "psychological": "g-thriller"
    }
    for keyword, genre_id in genre_keywords.items():
        if keyword in q_lower:
            detected_genre = genre_id
            
    # 4. Detect talent (actors, directors) keyword
    detected_talent = None
    for keyword, person_id in talent_keywords.items():
        if keyword in q_lower:
            detected_talent = person_id
            q_lower = q_lower.replace(keyword, "").strip()
            break

    remaining_q = " ".join(q_lower.split()).strip()
    return remaining_q, detected_lang, detected_type, detected_genre, detected_talent

@app.get("/api/search/hybrid")
def search_hybrid(
    q: str = Query(..., description="Hybrid search query text"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Weight between Lexical (alpha=1.0) and Vector (alpha=0.0)"),
    limit: int = Query(20, description="Results limit"),
    intent_parsing: bool = Query(True, description="Enable semantic intent parsing & metadata filtering"),
    age_rating: Optional[str] = Query(None, description="Age rating filter"),
    access_tier: Optional[str] = Query(None, description="Access tier filter"),
    audio_lang: Optional[str] = Query(None, description="Audio language filter")
):
    # Determine if intent parsing is active (both global backend flag and API query parameter must be True)
    use_intent_parsing = INTENT_PARSING_ENABLED and intent_parsing
    
    if use_intent_parsing:
        processed_q = preprocess_fts_query(q)
        target_q, detected_lang, detected_type, detected_genre, detected_talent = parse_search_intent(processed_q if processed_q else q)
    else:
        # Rollback path: set all filters to None and use the original processed query directly
        processed_q = preprocess_fts_query(q)
        target_q = processed_q if processed_q else q
        detected_lang = None
        detected_type = None
        detected_genre = None
        detected_talent = None

    # Define the single, native Spanner hybrid query utilizing ML.PREDICT for the embedding vector with metadata filters:
    hybrid_query = """
    @{OPTIMIZER_VERSION=6}
    WITH QueryVector AS (
        SELECT embeddings.values AS vec
        FROM ML.PREDICT(
          MODEL TextEmbeddingModel,
          (SELECT @queryText AS content)
        )
    ),
    FtsMatched AS (
        -- Direct FTS Match
        SELECT 
            t.ContentId,
            COALESCE(SCORE(t.SearchTokens, @queryText), 0.1) AS MatchScore
        FROM Titles AS t
        WHERE SEARCH(t.SearchTokens, @queryText)

        UNION ALL

        -- Substring Title Match
        SELECT 
            t.ContentId,
            1.1 AS MatchScore
        FROM Titles AS t
        WHERE SEARCH_SUBSTRING(t.PrimaryTitleNgramTokens, @queryText)

        UNION ALL

        -- Genre Match
        SELECT 
            tg.ContentId,
            1.5 AS MatchScore
        FROM TitleGenres tg
        JOIN Genres g ON tg.GenreId = g.GenreId
        WHERE SEARCH(g.SearchTokens, @queryText)

        UNION ALL

        -- Talent Match
        SELECT 
            tt.ContentId,
            1.2 AS MatchScore
        FROM TitleTalent tt
        JOIN People p ON tt.PersonId = p.PersonId
        WHERE SEARCH(p.SearchTokens, @queryText)

        UNION ALL

        -- Franchise Match
        SELECT 
            tf.ContentId,
            1.3 AS MatchScore
        FROM TitleFranchise tf
        JOIN Franchises f ON tf.FranchiseId = f.FranchiseId
        WHERE SEARCH(f.SearchTokens, @queryText)

        UNION ALL

        -- Alias Match
        SELECT 
            ta.ContentId,
            1.4 AS MatchScore
        FROM TitleAliases ta
        WHERE SEARCH(ta.SearchTokens, @queryText)
    ),
    AggregatedFts AS (
        SELECT 
            ContentId,
            MAX(MatchScore) AS FtsScore
        FROM FtsMatched
        GROUP BY ContentId
    ),
    VectorMatches AS (
        SELECT 
            t.ContentId, 
            COSINE_DISTANCE(t.Embedding, qv.vec) AS VectorDistance
        FROM Titles t, QueryVector qv
        WHERE (@contentTypeFilter IS NULL OR t.ContentType = @contentTypeFilter)
          AND (@audioLangFilter IS NULL OR @audioLangFilter IN UNNEST(t.AudioLanguages))
          AND (@genreFilter IS NULL OR EXISTS (
              SELECT 1 FROM TitleGenres tg 
              WHERE tg.ContentId = t.ContentId AND tg.GenreId = @genreFilter
          ))
          AND (@ageRatingFilter IS NULL OR t.AgeRating = @ageRatingFilter)
          AND (@accessTierFilter IS NULL OR t.AccessTier = @accessTierFilter)
          AND (@talentFilter IS NULL OR EXISTS (
              SELECT 1 FROM TitleTalent tt 
              WHERE tt.ContentId = t.ContentId AND tt.PersonId = @talentFilter
          ))
        ORDER BY VectorDistance ASC
        LIMIT 100
    ),
    Combined AS (
        SELECT 
            COALESCE(v.ContentId, f.ContentId) AS ContentId,
            COALESCE(f.FtsScore, 0.0) AS FtsScore,
            COALESCE(v.VectorDistance, 1.0) AS VectorDistance
        FROM VectorMatches v
        FULL OUTER JOIN AggregatedFts f ON v.ContentId = f.ContentId
    )
    SELECT 
        t.ContentId, 
        t.PrimaryTitle, 
        t.ContentType, 
        t.ReleaseYear, 
        t.PosterUrl, 
        t.ImdbRating, 
        t.Synopsis,
        c.FtsScore,
        c.VectorDistance,
        (@alpha * LEAST(c.FtsScore / 5.0, 1.0)) + ((1.0 - @alpha) * (1.0 - c.VectorDistance)) AS BlendedScore
    FROM Combined c
    JOIN Titles t ON c.ContentId = t.ContentId
    WHERE (@contentTypeFilter IS NULL OR t.ContentType = @contentTypeFilter)
      AND (@audioLangFilter IS NULL OR @audioLangFilter IN UNNEST(t.AudioLanguages))
      AND (@genreFilter IS NULL OR EXISTS (
          SELECT 1 FROM TitleGenres tg 
          WHERE tg.ContentId = t.ContentId AND tg.GenreId = @genreFilter
      ))
      AND (@ageRatingFilter IS NULL OR t.AgeRating = @ageRatingFilter)
      AND (@accessTierFilter IS NULL OR t.AccessTier = @accessTierFilter)
      AND (@talentFilter IS NULL OR EXISTS (
          SELECT 1 FROM TitleTalent tt 
          WHERE tt.ContentId = t.ContentId AND tt.PersonId = @talentFilter
      ))
    ORDER BY BlendedScore DESC
    LIMIT @limit
    """

    try:
        hybrid_results = []
        
        # Decide which query to run based on whether there's target text left to search
        if use_intent_parsing and not target_q:
            active_query = """
            SELECT 
                t.ContentId, 
                t.PrimaryTitle, 
                t.ContentType, 
                t.ReleaseYear, 
                t.PosterUrl, 
                t.ImdbRating, 
                t.Synopsis,
                0.0 AS FtsScore,
                0.5 AS VectorDistance,
                1.0 AS BlendedScore
            FROM Titles t
            WHERE (@contentTypeFilter IS NULL OR t.ContentType = @contentTypeFilter)
              AND (@audioLangFilter IS NULL OR @audioLangFilter IN UNNEST(t.AudioLanguages))
              AND (@genreFilter IS NULL OR EXISTS (
                  SELECT 1 FROM TitleGenres tg 
                  WHERE tg.ContentId = t.ContentId AND tg.GenreId = @genreFilter
              ))
              AND (@ageRatingFilter IS NULL OR t.AgeRating = @ageRatingFilter)
              AND (@accessTierFilter IS NULL OR t.AccessTier = @accessTierFilter)
              AND (@talentFilter IS NULL OR EXISTS (
                  SELECT 1 FROM TitleTalent tt 
                  WHERE tt.ContentId = t.ContentId AND tt.PersonId = @talentFilter
              ))
            ORDER BY t.ImdbRating DESC, t.PopularityScore DESC
            LIMIT @limit
            """
        else:
            active_query = hybrid_query

        with database.snapshot() as snapshot:
            rows = snapshot.execute_sql(
                active_query,
                params={
                    "queryText": target_q,
                    "contentTypeFilter": detected_type,
                    "audioLangFilter": audio_lang if audio_lang else detected_lang,
                    "genreFilter": detected_genre,
                    "ageRatingFilter": age_rating,
                    "accessTierFilter": access_tier,
                    "talentFilter": detected_talent,
                    "alpha": alpha,
                    "limit": 50
                },
                param_types={
                    "queryText": spanner.param_types.STRING,
                    "contentTypeFilter": spanner.param_types.STRING,
                    "audioLangFilter": spanner.param_types.STRING,
                    "genreFilter": spanner.param_types.STRING,
                    "ageRatingFilter": spanner.param_types.STRING,
                    "accessTierFilter": spanner.param_types.STRING,
                    "talentFilter": spanner.param_types.STRING,
                    "alpha": spanner.param_types.FLOAT64,
                    "limit": spanner.param_types.INT64
                }
            )
            hybrid_results = list(rows)

        blended_list = []
        for r in hybrid_results:
            content_id, title, c_type, year, poster, rating, synopsis, fts_score, vec_dist, blended_score = r
            
            sem_sim = 1.0 - vec_dist  # Similarity score (1.0 is identical)
            similarity_pct = int(sem_sim * 100)
            lexical_score = round(fts_score, 2)
            
            if not target_q:
                match_explanation = f"Metadata Browse Match! (Filtered by extracted criteria)"
                match_type = "metadata"
            elif fts_score > 0.0 and sem_sim > 0.8:
                match_explanation = f"Perfect Hybrid Match! (Keyword score: {lexical_score}, Semantic similarity: {similarity_pct}%)"
                match_type = "hybrid"
            elif sem_sim > 0.8:
                match_explanation = f"Semantic Concept Match! (Similarity: {similarity_pct}%)"
                match_type = "semantic"
            else:
                match_explanation = f"Lexical Keyword Match! (FTS score: {lexical_score})"
                match_type = "lexical"

            blended_list.append({
                "ContentId": content_id,
                "PrimaryTitle": title,
                "ContentType": c_type,
                "ReleaseYear": year,
                "PosterUrl": poster,
                "ImdbRating": rating,
                "Synopsis": synopsis,
                "RawFtsScore": fts_score,
                "SemanticSimilarity": sem_sim,
                "RrfScore": blended_score,
                "BlendedScore": blended_score,
                "Explanation": match_explanation,
                "MatchType": match_type
            })

        display_q = target_q if target_q else q
        search_escaped = display_q.replace("'", "''")
        
        # Format substituted variables for the SQL text representation shown in the UI
        type_sub = f"'{detected_type}'" if detected_type else "NULL"
        effective_lang = audio_lang if audio_lang else detected_lang
        lang_sub = f"'{effective_lang}'" if effective_lang else "NULL"
        genre_sub = f"'{detected_genre}'" if detected_genre else "NULL"
        age_sub = f"'{age_rating}'" if age_rating else "NULL"
        tier_sub = f"'{access_tier}'" if access_tier else "NULL"
        talent_sub = f"'{detected_talent}'" if detected_talent else "NULL"
        
        parameterized_combined_query = """@{OPTIMIZER_VERSION=6}
WITH QueryVector AS (
    SELECT embeddings.values AS vec
    FROM ML.PREDICT(
      MODEL TextEmbeddingModel,
      (SELECT @queryText AS content)
    )
),
FtsMatched AS (
    -- Direct FTS Match
    SELECT 
        t.ContentId,
        COALESCE(SCORE(t.SearchTokens, @queryText), 0.1) AS MatchScore
    FROM Titles AS t
    WHERE SEARCH(t.SearchTokens, @queryText)

    UNION ALL

    -- Substring Title Match
    SELECT 
        t.ContentId,
        1.1 AS MatchScore
    FROM Titles AS t
    WHERE SEARCH_SUBSTRING(t.PrimaryTitleNgramTokens, @queryText)

    UNION ALL

    -- Genre Match
    SELECT 
        tg.ContentId,
        1.5 AS MatchScore
    FROM TitleGenres tg
    JOIN Genres g ON tg.GenreId = g.GenreId
    WHERE SEARCH(g.SearchTokens, @queryText)

    UNION ALL

    -- Talent Match
    SELECT 
        tt.ContentId,
        1.2 AS MatchScore
    FROM TitleTalent tt
    JOIN People p ON tt.PersonId = p.PersonId
    WHERE SEARCH(p.SearchTokens, @queryText)

    UNION ALL

    -- Franchise Match
    SELECT 
        tf.ContentId,
        1.3 AS MatchScore
    FROM TitleFranchise tf
    JOIN Franchises f ON tf.FranchiseId = f.FranchiseId
    WHERE SEARCH(f.SearchTokens, @queryText)

    UNION ALL

    -- Alias Match
    SELECT 
        ta.ContentId,
        1.4 AS MatchScore
    FROM TitleAliases ta
    WHERE SEARCH(ta.SearchTokens, @queryText)
),
AggregatedFts AS (
    SELECT 
        ContentId,
        MAX(MatchScore) AS FtsScore
    FROM FtsMatched
    GROUP BY ContentId
),
VectorMatches AS (
    SELECT 
        t.ContentId, 
        COSINE_DISTANCE(t.Embedding, qv.vec) AS VectorDistance
    FROM Titles t, QueryVector qv
    WHERE (@contentTypeFilter IS NULL OR t.ContentType = @contentTypeFilter)
      AND (@audioLangFilter IS NULL OR @audioLangFilter IN UNNEST(t.AudioLanguages))
      AND (@genreFilter IS NULL OR EXISTS (
          SELECT 1 FROM TitleGenres tg 
          WHERE tg.ContentId = t.ContentId AND tg.GenreId = @genreFilter
      ))
      AND (@ageRatingFilter IS NULL OR t.AgeRating = @ageRatingFilter)
      AND (@accessTierFilter IS NULL OR t.AccessTier = @accessTierFilter)
      AND (@talentFilter IS NULL OR EXISTS (
          SELECT 1 FROM TitleTalent tt 
          WHERE tt.ContentId = t.ContentId AND tt.PersonId = @talentFilter
      ))
    ORDER BY VectorDistance ASC
    LIMIT 100
),
Combined AS (
    SELECT 
        COALESCE(v.ContentId, f.ContentId) AS ContentId,
        COALESCE(f.FtsScore, 0.0) AS FtsScore,
        COALESCE(v.VectorDistance, 1.0) AS VectorDistance
    FROM VectorMatches v
    FULL OUTER JOIN AggregatedFts f ON v.ContentId = f.ContentId
)
SELECT 
    t.ContentId, 
    t.PrimaryTitle, 
    t.ContentType, 
    t.ReleaseYear, 
    t.PosterUrl, 
    t.ImdbRating, 
    t.Synopsis,
    c.FtsScore,
    c.VectorDistance,
    (@alpha * LEAST(c.FtsScore / 5.0, 1.0)) + ((1.0 - @alpha) * (1.0 - c.VectorDistance)) AS BlendedScore
FROM Combined c
JOIN Titles t ON c.ContentId = t.ContentId
WHERE (@contentTypeFilter IS NULL OR t.ContentType = @contentTypeFilter)
  AND (@audioLangFilter IS NULL OR @audioLangFilter IN UNNEST(t.AudioLanguages))
  AND (@genreFilter IS NULL OR EXISTS (
      SELECT 1 FROM TitleGenres tg 
      WHERE tg.ContentId = t.ContentId AND tg.GenreId = @genreFilter
  ))
  AND (@ageRatingFilter IS NULL OR t.AgeRating = @ageRatingFilter)
  AND (@accessTierFilter IS NULL OR t.AccessTier = @accessTierFilter)
  AND (@talentFilter IS NULL OR EXISTS (
      SELECT 1 FROM TitleTalent tt 
      WHERE tt.ContentId = t.ContentId AND tt.PersonId = @talentFilter
  ))
ORDER BY BlendedScore DESC
LIMIT @limit"""

        substituted_combined_query = f"""@{{OPTIMIZER_VERSION=6}}
WITH QueryVector AS (
    SELECT embeddings.values AS vec
    FROM ML.PREDICT(
      MODEL TextEmbeddingModel,
      (SELECT '{search_escaped}' AS content)
    )
),
FtsMatched AS (
    -- Direct FTS Match
    SELECT 
        t.ContentId,
        COALESCE(SCORE(t.SearchTokens, '{search_escaped}'), 0.1) AS MatchScore
    FROM Titles AS t
    WHERE SEARCH(t.SearchTokens, '{search_escaped}')

    UNION ALL

    -- Substring Title Match
    SELECT 
        t.ContentId,
        1.1 AS MatchScore
    FROM Titles AS t
    WHERE SEARCH_SUBSTRING(t.PrimaryTitleNgramTokens, '{search_escaped}')

    UNION ALL

    -- Genre Match
    SELECT 
        tg.ContentId,
        1.5 AS MatchScore
    FROM TitleGenres tg
    JOIN Genres g ON tg.GenreId = g.GenreId
    WHERE SEARCH(g.SearchTokens, '{search_escaped}')

    UNION ALL

    -- Talent Match
    SELECT 
        tt.ContentId,
        1.2 AS MatchScore
    FROM TitleTalent tt
    JOIN People p ON tt.PersonId = p.PersonId
    WHERE SEARCH(p.SearchTokens, '{search_escaped}')

    UNION ALL

    -- Franchise Match
    SELECT 
        tf.ContentId,
        1.3 AS MatchScore
    FROM TitleFranchise tf
    JOIN Franchises f ON tf.FranchiseId = f.FranchiseId
    WHERE SEARCH(f.SearchTokens, '{search_escaped}')

    UNION ALL

    -- Alias Match
    SELECT 
        ta.ContentId,
        1.4 AS MatchScore
    FROM TitleAliases ta
    WHERE SEARCH(ta.SearchTokens, '{search_escaped}')
),
AggregatedFts AS (
    SELECT 
        ContentId,
        MAX(MatchScore) AS FtsScore
    FROM FtsMatched
    GROUP BY ContentId
),
VectorMatches AS (
    SELECT 
        t.ContentId, 
        COSINE_DISTANCE(t.Embedding, qv.vec) AS VectorDistance
    FROM Titles t, QueryVector qv
    WHERE ({type_sub} IS NULL OR t.ContentType = {type_sub})
      AND ({lang_sub} IS NULL OR {lang_sub} IN UNNEST(t.AudioLanguages))
      AND ({genre_sub} IS NULL OR EXISTS (
          SELECT 1 FROM TitleGenres tg 
          WHERE tg.ContentId = t.ContentId AND tg.GenreId = {genre_sub}
      ))
      AND ({age_sub} IS NULL OR t.AgeRating = {age_sub})
      AND ({tier_sub} IS NULL OR t.AccessTier = {tier_sub})
      AND ({talent_sub} IS NULL OR EXISTS (
          SELECT 1 FROM TitleTalent tt 
          WHERE tt.ContentId = t.ContentId AND tt.PersonId = {talent_sub}
      ))
    ORDER BY VectorDistance ASC
    LIMIT 100
),
Combined AS (
    SELECT 
        COALESCE(v.ContentId, f.ContentId) AS ContentId,
        COALESCE(f.FtsScore, 0.0) AS FtsScore,
        COALESCE(v.VectorDistance, 1.0) AS VectorDistance
    FROM VectorMatches v
    FULL OUTER JOIN AggregatedFts f ON v.ContentId = f.ContentId
)
SELECT 
    t.ContentId, 
    t.PrimaryTitle, 
    t.ContentType, 
    t.ReleaseYear, 
    t.PosterUrl, 
    t.ImdbRating, 
    t.Synopsis,
    c.FtsScore,
    c.VectorDistance,
    ({alpha} * LEAST(c.FtsScore / 5.0, 1.0)) + ({(1.0 - alpha)} * (1.0 - c.VectorDistance)) AS BlendedScore
FROM Combined c
JOIN Titles t ON c.ContentId = t.ContentId
WHERE ({type_sub} IS NULL OR t.ContentType = {type_sub})
  AND ({lang_sub} IS NULL OR {lang_sub} IN UNNEST(t.AudioLanguages))
  AND ({genre_sub} IS NULL OR EXISTS (
      SELECT 1 FROM TitleGenres tg 
      WHERE tg.ContentId = t.ContentId AND tg.GenreId = {genre_sub}
  ))
  AND ({age_sub} IS NULL OR t.AgeRating = {age_sub})
  AND ({tier_sub} IS NULL OR t.AccessTier = {tier_sub})
  AND ({talent_sub} IS NULL OR EXISTS (
      SELECT 1 FROM TitleTalent tt 
      WHERE tt.ContentId = t.ContentId AND tt.PersonId = {talent_sub}
  ))
ORDER BY BlendedScore DESC
LIMIT {limit}"""
        
        if use_intent_parsing and not target_q:
            parameterized_combined_query = """SELECT 
    t.ContentId, 
    t.PrimaryTitle, 
    t.ContentType, 
    t.ReleaseYear, 
    t.PosterUrl, 
    t.ImdbRating, 
    t.Synopsis,
    0.0 AS FtsScore,
    0.5 AS VectorDistance,
    1.0 AS BlendedScore
FROM Titles t
WHERE (@contentTypeFilter IS NULL OR t.ContentType = @contentTypeFilter)
  AND (@audioLangFilter IS NULL OR @audioLangFilter IN UNNEST(t.AudioLanguages))
  AND (@genreFilter IS NULL OR EXISTS (
      SELECT 1 FROM TitleGenres tg 
      WHERE tg.ContentId = t.ContentId AND tg.GenreId = @genreFilter
  ))
  AND (@ageRatingFilter IS NULL OR t.AgeRating = @ageRatingFilter)
  AND (@accessTierFilter IS NULL OR t.AccessTier = @accessTierFilter)
  AND (@talentFilter IS NULL OR EXISTS (
      SELECT 1 FROM TitleTalent tt 
      WHERE tt.ContentId = t.ContentId AND tt.PersonId = @talentFilter
  ))
ORDER BY t.ImdbRating DESC, t.PopularityScore DESC
LIMIT @limit"""

            substituted_combined_query = f"""SELECT 
    t.ContentId, 
    t.PrimaryTitle, 
    t.ContentType, 
    t.ReleaseYear, 
    t.PosterUrl, 
    t.ImdbRating, 
    t.Synopsis,
    0.0 AS FtsScore,
    0.5 AS VectorDistance,
    1.0 AS BlendedScore
FROM Titles t
WHERE ({type_sub} IS NULL OR t.ContentType = {type_sub})
  AND ({lang_sub} IS NULL OR {lang_sub} IN UNNEST(t.AudioLanguages))
  AND ({genre_sub} IS NULL OR EXISTS (
      SELECT 1 FROM TitleGenres tg 
      WHERE tg.ContentId = t.ContentId AND tg.GenreId = {genre_sub}
  ))
  AND ({age_sub} IS NULL OR t.AgeRating = {age_sub})
  AND ({tier_sub} IS NULL OR t.AccessTier = {tier_sub})
  AND ({talent_sub} IS NULL OR EXISTS (
      SELECT 1 FROM TitleTalent tt 
      WHERE tt.ContentId = t.ContentId AND tt.PersonId = {talent_sub}
  ))
ORDER BY t.ImdbRating DESC, t.PopularityScore DESC
LIMIT {limit}"""

        return {
            "success": True, 
            "query": q, 
            "results": blended_list[:limit],
            "executed_queries": {
                "parameterized_fts": parameterized_combined_query.strip(),
                "substituted_fts": substituted_combined_query.strip(),
                "parameterized_vector": parameterized_combined_query.strip(),
                "substituted_vector": substituted_combined_query.strip()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spanner Hybrid Search error: {str(e)}")

# ============================================================================
# TAB 3: HYPER-HYBRID DISCOVERY (SPANNER GQL PROPERTY GRAPH TRAVERSAL)
# ============================================================================
@app.get("/api/search/hyper-graph")
def search_hyper_graph(
    seed_id: str = Query(..., description="ContentId of seed movie to explore from"),
    limit: int = Query(25, description="Max traversal results to return")
):
    # Perform clean GQL matches traversing our interconnected Spanner property graph:
    # 1. Hop 1: Titles recommended directly via Co-watch affinity
    # 2. Hop 2: Titles connected through creative people (actors/directors who participated in both)
    # 3. Hop 2: Titles connected through shared franchise clusters (e.g. Star Wars, Nolanverse)
    
    co_watch_query = """
    SELECT 
        TargetTitleId,
        TargetTitleName,
        CAST(NULL AS STRING) AS TargetPoster,
        'FANS_ALSO_WATCHED' AS ConnectionType,
        'Direct Recommendation' AS ConnectionDetail,
        1 AS HopDistance,
        AffinityStrength
    FROM GRAPH_TABLE(OttKnowledgeGraph
      MATCH (seed:Title)-[rel:FANS_ALSO_WATCHED]-(target:Title)
      WHERE seed.ContentId = @seed_id AND seed.ContentId != target.ContentId
      COLUMNS (
        target.ContentId AS TargetTitleId,
        target.PrimaryTitle AS TargetTitleName,
        target.PopularityScore AS AffinityStrength
      )
    )
    LIMIT @limit
    """

    talent_query = """
    SELECT 
        TargetTitleId,
        TargetTitleName,
        CAST(NULL AS STRING) AS TargetPoster,
        'PARTICIPATED_IN' AS ConnectionType,
        ConnectionDetail,
        2 AS HopDistance,
        AffinityStrength
    FROM GRAPH_TABLE(OttKnowledgeGraph
      MATCH (seed:Title)-[:PARTICIPATED_IN]-(p:Person)-[:PARTICIPATED_IN]-(target:Title)
      WHERE seed.ContentId = @seed_id AND seed.ContentId != target.ContentId
      COLUMNS (
        target.ContentId AS TargetTitleId,
        target.PrimaryTitle AS TargetTitleName,
        p.FullName AS ConnectionDetail,
        target.PopularityScore AS AffinityStrength
      )
    )
    LIMIT @limit
    """

    franchise_query = """
    SELECT 
        TargetTitleId,
        TargetTitleName,
        CAST(NULL AS STRING) AS TargetPoster,
        'BELONGS_TO_FRANCHISE' AS ConnectionType,
        ConnectionDetail,
        2 AS HopDistance,
        AffinityStrength
    FROM GRAPH_TABLE(OttKnowledgeGraph
      MATCH (seed:Title)-[:BELONGS_TO_FRANCHISE]-(f:Franchise)-[:BELONGS_TO_FRANCHISE]-(target:Title)
      WHERE seed.ContentId = @seed_id AND seed.ContentId != target.ContentId
      COLUMNS (
        target.ContentId AS TargetTitleId,
        target.PrimaryTitle AS TargetTitleName,
        f.Name AS ConnectionDetail,
        target.PopularityScore AS AffinityStrength
      )
    )
    LIMIT @limit
    """

    params = {"seed_id": seed_id, "limit": limit}
    param_types = {"seed_id": spanner.param_types.STRING, "limit": spanner.param_types.INT64}

    try:
        nodes = {}
        edges = []
        
        # Populate seed node first
        with database.snapshot(multi_use=True) as snapshot:
            seed_res = snapshot.execute_sql(
                "SELECT PrimaryTitle, PosterUrl, PopularityScore, ContentType FROM Titles WHERE ContentId = @seed_id",
                params={"seed_id": seed_id},
                param_types={"seed_id": spanner.param_types.STRING}
            )
            seed_rows = list(seed_res)
            if not seed_rows:
                raise HTTPException(status_code=404, detail="Seed title not found")
            
            seed_title, seed_poster, seed_pop, seed_type = seed_rows[0]
            nodes[seed_id] = {
                "id": seed_id,
                "label": seed_title,
                "poster": seed_poster,
                "type": "seed",
                "val": 30
            }

            # Query Co-watch direct edges
            co_watch_res = snapshot.execute_sql(co_watch_query, params=params, param_types=param_types)
            for r in co_watch_res:
                target_id, target_name, target_poster, conn_type, detail, hops, strength = r
                if target_id not in nodes:
                    nodes[target_id] = {
                        "id": target_id,
                        "label": target_name,
                        "poster": None,
                        "type": "title",
                        "val": 20
                    }
                edges.append({
                    "from": seed_id,
                    "to": target_id,
                    "label": "Fans also watched",
                    "color": "#3b82f6",  # Blue for co-watch
                    "arrows": "to"
                })

            # Query Talent connection edges (Titles sharing directors or actors)
            talent_res = snapshot.execute_sql(talent_query, params=params, param_types=param_types)
            for r in talent_res:
                target_id, target_name, target_poster, conn_type, detail, hops, strength = r
                # Create intermediate Person node
                person_id = f"inter-p-{detail.replace(' ', '-').lower()}"
                if person_id not in nodes:
                    nodes[person_id] = {
                        "id": person_id,
                        "label": detail,
                        "type": "person",
                        "val": 15
                    }
                if target_id not in nodes:
                    nodes[target_id] = {
                        "id": target_id,
                        "label": target_name,
                        "poster": None,
                        "type": "title",
                        "val": 20
                    }
                # Connect Seed -> Person -> Target
                edge1 = {"from": seed_id, "to": person_id, "label": "Directed/Cast", "color": "#a855f7"}
                edge2 = {"from": person_id, "to": target_id, "label": "Directed/Cast", "color": "#a855f7"}
                if edge1 not in edges:
                    edges.append(edge1)
                if edge2 not in edges:
                    edges.append(edge2)

            # Query Franchise connection edges
            franchise_res = snapshot.execute_sql(franchise_query, params=params, param_types=param_types)
            for r in franchise_res:
                target_id, target_name, target_poster, conn_type, detail, hops, strength = r
                # Create intermediate Franchise node
                franchise_node_id = f"inter-f-{detail.replace(' ', '-').lower()}"
                if franchise_node_id not in nodes:
                    nodes[franchise_node_id] = {
                        "id": franchise_node_id,
                        "label": detail,
                        "type": "franchise",
                        "val": 18
                    }
                if target_id not in nodes:
                    nodes[target_id] = {
                        "id": target_id,
                        "label": target_name,
                        "poster": None,
                        "type": "title",
                        "val": 20
                    }
                # Connect Seed -> Franchise -> Target
                edge1 = {"from": seed_id, "to": franchise_node_id, "label": "Belongs To", "color": "#eab308"}
                edge2 = {"from": franchise_node_id, "to": target_id, "label": "Belongs To", "color": "#eab308"}
                if edge1 not in edges:
                    edges.append(edge1)
                if edge2 not in edges:
                    edges.append(edge2)

            # Fetch PosterUrls for all target Title nodes in python
            matched_title_ids = [nid for nid in nodes.keys() if nid.startswith("t-")]
            if matched_title_ids:
                posters_res = snapshot.execute_sql(
                    "SELECT ContentId, PosterUrl FROM Titles WHERE ContentId IN UNNEST(@ids)",
                    params={"ids": matched_title_ids},
                    param_types={"ids": spanner.param_types.Array(spanner.param_types.STRING)}
                )
                poster_map = {row[0]: row[1] for row in posters_res}
                for nid in matched_title_ids:
                    if nid in nodes:
                        nodes[nid]["poster"] = poster_map.get(nid)
                    edges.append(edge1)
                if edge2 not in edges:
                    edges.append(edge2)

        sub_co_watch = co_watch_query.replace("@seed_id", f"'{seed_id}'").replace("@limit", str(limit))
        sub_talent = talent_query.replace("@seed_id", f"'{seed_id}'").replace("@limit", str(limit))
        sub_franchise = franchise_query.replace("@seed_id", f"'{seed_id}'").replace("@limit", str(limit))

        return {
            "success": True,
            "seed": {
                "id": seed_id,
                "label": seed_title,
                "poster": seed_poster,
                "popularity": seed_pop,
                "type": seed_type
            },
            "graph": {
                "nodes": list(nodes.values()),
                "edges": edges
            },
            "executed_queries": {
                "parameterized_co_watch": co_watch_query.strip(),
                "substituted_co_watch": sub_co_watch.strip(),
                "parameterized_talent": talent_query.strip(),
                "substituted_talent": sub_talent.strip(),
                "parameterized_franchise": franchise_query.strip(),
                "substituted_franchise": sub_franchise.strip()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spanner Hyper-Graph error: {str(e)}")

class QueryRequest(BaseModel):
    sql: str

@app.post("/api/query")
def execute_custom_query(req: QueryRequest):
    try:
        with database.snapshot() as snapshot:
            result = snapshot.execute_sql(req.sql)
            rows = []
            columns = []
            for row in result:
                if not columns:
                    columns = [field.name for field in result.metadata.row_type.fields]
                serializable_row = []
                for val in row:
                    if isinstance(val, (datetime.datetime, datetime.date)):
                        serializable_row.append(val.isoformat())
                    elif isinstance(val, bytes):
                        serializable_row.append(val.hex())
                    elif isinstance(val, list):
                        serializable_row.append(str(val))
                    else:
                        serializable_row.append(val)
                rows.append(serializable_row)
            
            return {
                "success": True,
                "columns": columns if columns else ["Result"],
                "rows": rows[:100]
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
