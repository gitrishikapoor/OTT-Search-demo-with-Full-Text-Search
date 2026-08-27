-- ============================================================================
-- 1. BASE ENTITY TABLES
-- ============================================================================

-- Primary Content Table (Movies, Series, Documentaries, Live Events)
CREATE TABLE Titles (
    ContentId STRING(64) NOT NULL,
    PrimaryTitle STRING(255) NOT NULL,
    OriginalTitle STRING(255),
    ContentType STRING(32) NOT NULL, -- 'MOVIE', 'TV_SERIES', 'DOCUMENTARY', 'SPECIAL'
    ReleaseYear INT64 NOT NULL,
    AgeRating STRING(16) NOT NULL,    -- 'G', 'PG', 'PG-13', 'R', 'NC-17', 'TV-MA'
    DurationMins INT64,               -- Null for TV series, populated for movies
    SeasonsCount INT64,               -- Null for movies, populated for series
    Synopsis STRING(MAX) NOT NULL,
    Tagline STRING(512),
    PosterUrl STRING(1024),
    BannerUrl STRING(1024),
    TrailerUrl STRING(1024),
    ImdbRating FLOAT64,
    PopularityScore FLOAT64 NOT NULL, -- Base popularity index (0.0 - 100.0)
    AccessTier STRING(32) NOT NULL,   -- 'FREE_AVOD', 'SVOD_STANDARD', 'SVOD_PREMIUM', 'TVOD_RENT'
    AudioLanguages ARRAY<STRING(32)>, -- ['en', 'es', 'fr', 'hi', 'ja']
    SubtitleLanguages ARRAY<STRING(32)>,-- ['en', 'es', 'fr', 'de', 'zh']
    AudioLanguageNames ARRAY<STRING(64)>, -- ['English', 'Spanish', 'French', 'Hindi', 'Japanese']
    SubtitleLanguageNames ARRAY<STRING(64)>,-- ['English', 'Spanish', 'French', 'German', 'Chinese']
    QualityProfiles ARRAY<STRING(16)>,-- ['HD', '4K_UHD', 'HDR10', 'DOLBY_VISION', 'DOLBY_ATMOS']
    Embedding ARRAY<FLOAT64>(vector_length=>768), -- Semantic embedding of Title + Synopsis + Tags
    CreatedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
    UpdatedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
    
    -- Generated TOKENLIST columns for Search
    SearchTokens TOKENLIST AS (
        TOKENIZE_FULLTEXT(ARRAY[
            PrimaryTitle, 
            COALESCE(OriginalTitle, ''), 
            Synopsis, 
            COALESCE(Tagline, ''),
            ARRAY_TO_STRING(AudioLanguages, ' '),
            ARRAY_TO_STRING(SubtitleLanguages, ' '),
            COALESCE(ARRAY_TO_STRING(AudioLanguageNames, ' '), ''),
            COALESCE(ARRAY_TO_STRING(SubtitleLanguageNames, ' '), '')
        ])
    ) HIDDEN,
    
    PrimaryTitleNgramTokens TOKENLIST AS (
        TOKENIZE_SUBSTRING(PrimaryTitle, ngram_size_min => 3, ngram_size_max => 10)
    ) HIDDEN
) PRIMARY KEY (ContentId);

-- Talent & Crew (Actors, Directors, Writers, Producers)
CREATE TABLE People (
    PersonId STRING(64) NOT NULL,
    FullName STRING(255) NOT NULL,
    KnownAs STRING(255),
    PrimaryRole STRING(64) NOT NULL, -- 'ACTOR', 'DIRECTOR', 'WRITER', 'SHOWRUNNER'
    Bio STRING(MAX),
    BirthYear INT64,
    ProfileImageUrl STRING(1024),
    PopularityScore FLOAT64,
    Embedding ARRAY<FLOAT64>(vector_length=>768),
    
    -- Generated TOKENLIST column for cast & crew search
    SearchTokens TOKENLIST AS (
        TOKENIZE_FULLTEXT(ARRAY[FullName, COALESCE(KnownAs, ''), COALESCE(Bio, '')])
    ) HIDDEN
) PRIMARY KEY (PersonId);

-- Content to Talent Mapping
CREATE TABLE TitleTalent (
    ContentId STRING(64) NOT NULL,
    PersonId STRING(64) NOT NULL,
    Role STRING(64) NOT NULL,        -- 'DIRECTOR', 'LEAD_ACTOR', 'SUPPORTING_ACTOR', 'WRITER'
    CharacterName STRING(255),
    BillingOrder INT64 NOT NULL
) PRIMARY KEY (ContentId, PersonId, Role),
  INTERLEAVE IN PARENT Titles ON DELETE CASCADE;

-- Genres & Micro-Genres
CREATE TABLE Genres (
    GenreId STRING(64) NOT NULL,
    Name STRING(64) NOT NULL,
    Slug STRING(64) NOT NULL,
    Description STRING(512),
    
    -- Generated TOKENLIST column for genre search
    SearchTokens TOKENLIST AS (
        TOKENIZE_FULLTEXT(Name)
    ) HIDDEN
) PRIMARY KEY (GenreId);

-- Title to Genre Mapping
CREATE TABLE TitleGenres (
    ContentId STRING(64) NOT NULL,
    GenreId STRING(64) NOT NULL
) PRIMARY KEY (ContentId, GenreId),
  INTERLEAVE IN PARENT Titles ON DELETE CASCADE;

-- Franchises / Universes (e.g., Marvel Cinematic Universe, Star Wars, Nolanverse)
CREATE TABLE Franchises (
    FranchiseId STRING(64) NOT NULL,
    Name STRING(255) NOT NULL,
    Description STRING(MAX),
    
    -- Generated TOKENLIST column for franchise search
    SearchTokens TOKENLIST AS (
        TOKENIZE_FULLTEXT(ARRAY[Name, COALESCE(Description, '')])
    ) HIDDEN
) PRIMARY KEY (FranchiseId);

CREATE TABLE TitleFranchise (
    ContentId STRING(64) NOT NULL,
    FranchiseId STRING(64) NOT NULL,
    ChronologicalOrder INT64
) PRIMARY KEY (ContentId, FranchiseId),
  INTERLEAVE IN PARENT Titles ON DELETE CASCADE;

-- Search Synonyms, Aliases, and Common Misspellings
CREATE TABLE TitleAliases (
    AliasId STRING(64) NOT NULL,
    ContentId STRING(64) NOT NULL,
    AliasText STRING(255) NOT NULL,
    AliasType STRING(32) NOT NULL, -- 'TRANSLITERATION', 'ACRONYM', 'COMMON_TYPO', 'REGIONAL_TITLE'
    
    -- Generated TOKENLIST column for synonyms & alias search
    SearchTokens TOKENLIST AS (
        TOKENIZE_FULLTEXT(AliasText)
    ) HIDDEN
) PRIMARY KEY (ContentId, AliasId),
  INTERLEAVE IN PARENT Titles ON DELETE CASCADE;

-- User Watch Affinity / Co-watch Matrix for Graph Search
CREATE TABLE WatchAffinity (
    SourceContentId STRING(64) NOT NULL,
    TargetContentId STRING(64) NOT NULL,
    AffinityScore FLOAT64 NOT NULL, -- Co-view probability weight (0.0 to 1.0)
    SharedAudienceCount INT64 NOT NULL
) PRIMARY KEY (SourceContentId, TargetContentId);

-- ============================================================================
-- 2. FULL-TEXT SEARCH (FTS) SEARCH INDEXES
-- ============================================================================

-- Comprehensive Search Index on Titles
CREATE SEARCH INDEX TitlesSearchIndex ON Titles (
    SearchTokens
)
STORING (
    ContentType,
    ReleaseYear,
    AgeRating,
    DurationMins,
    PosterUrl,
    PopularityScore,
    ImdbRating,
    AccessTier,
    AudioLanguages,
    SubtitleLanguages
)
OPTIONS (
    sort_order_sharding = true
);

-- Search Index for Instant N-gram / Substring Title Matching
CREATE SEARCH INDEX TitlesNgramIndex ON Titles (
    PrimaryTitleNgramTokens
)
STORING (ReleaseYear, PosterUrl, PopularityScore);

-- Search Index for Genres Full-Text Search
CREATE SEARCH INDEX GenresSearchIndex ON Genres (
    SearchTokens
) OPTIONS (sort_order_sharding = true);

-- Search Index for People Full-Text Search (Cast, Crew, Directors)
CREATE SEARCH INDEX PeopleSearchIndex ON People (
    SearchTokens
) OPTIONS (sort_order_sharding = true);

-- Search Index for Franchises Full-Text Search
CREATE SEARCH INDEX FranchisesSearchIndex ON Franchises (
    SearchTokens
) OPTIONS (sort_order_sharding = true);

-- Search Index for TitleAliases Full-Text Search
CREATE SEARCH INDEX TitleAliasesSearchIndex ON TitleAliases (
    SearchTokens
) OPTIONS (sort_order_sharding = true);

-- ============================================================================
-- 3. SPANNER PROPERTY GRAPH DEFINITION (GQL)
-- ============================================================================

CREATE PROPERTY GRAPH OttKnowledgeGraph
  NODE TABLES (
    Titles 
      LABEL Title 
      PROPERTIES (ContentId, PrimaryTitle, ContentType, ReleaseYear, AgeRating, PopularityScore, ImdbRating),
    People 
      LABEL Person 
      PROPERTIES (PersonId, FullName, PrimaryRole),
    Genres 
      LABEL Genre 
      PROPERTIES (GenreId, Name),
    Franchises 
      LABEL Franchise 
      PROPERTIES (FranchiseId, Name)
  )
  EDGE TABLES (
    TitleTalent 
      SOURCE KEY (ContentId) REFERENCES Titles (ContentId)
      DESTINATION KEY (PersonId) REFERENCES People (PersonId)
      LABEL PARTICIPATED_IN
      PROPERTIES (Role, CharacterName),
    TitleGenres 
      SOURCE KEY (ContentId) REFERENCES Titles (ContentId)
      DESTINATION KEY (GenreId) REFERENCES Genres (GenreId)
      LABEL HAS_GENRE,
    TitleFranchise 
      SOURCE KEY (ContentId) REFERENCES Titles (ContentId)
      DESTINATION KEY (FranchiseId) REFERENCES Franchises (FranchiseId)
      LABEL BELONGS_TO_FRANCHISE,
    WatchAffinity 
      SOURCE KEY (SourceContentId) REFERENCES Titles (ContentId)
      DESTINATION KEY (TargetContentId) REFERENCES Titles (ContentId)
      LABEL FANS_ALSO_WATCHED
      PROPERTIES (AffinityScore, SharedAudienceCount)
  );
