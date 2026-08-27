import datetime
import hashlib
import random
from google.cloud import spanner

# ============================================================================
# DETERMINISTIC SEMANTIC-LIKE EMBEDDING GENERATOR
# ============================================================================
def get_embedding(text: str) -> list[float]:
    # Use hash-based deterministic seed
    hash_val = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = random.Random(hash_val)
    
    # Start with standard normal distribution (using Box-Muller transform)
    vec = []
    for _ in range(768):
        vec.append(rng.gauss(0, 1))
        
    # Add genre centroid bias to create distinct vector clusters (so cosine similarity is meaningful)
    genres = ["scifi", "space", "crime", "drama", "anime", "action", "thriller", "mystery", "fantasy", "history", "indian", "bollywood", "comedy", "comic"]
    matched_genres = [g for g in genres if g in text.lower()]
    for g in matched_genres:
        g_hash = int(hashlib.sha256(g.encode('utf-8')).hexdigest(), 16) % (2**32)
        g_rng = random.Random(g_hash)
        for i in range(768):
            vec[i] += 1.8 * g_rng.gauss(0, 1)
            
    # Normalize to unit length (unit vector dot product = cosine similarity)
    sq_sum = sum(x*x for x in vec)
    norm = sq_sum ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

# ============================================================================
# SEED DATA DEFINITIONS
# ============================================================================

GENRES_DATA = [
    ("g-scifi", "Sci-Fi & Space", "scifi", "Futuristic, space travel, high-concept technologies, and alien worlds."),
    ("g-crime", "Crime & Mystery", "crime", "Heists, detectives, legal battles, and underground syndicates."),
    ("g-drama", "Drama", "drama", "Character-driven narratives, emotional depth, and realistic conflicts."),
    ("g-anime", "Anime & Animation", "anime", "Beautiful hand-drawn and digital animated masterpieces from across the globe."),
    ("g-action", "Action & Adventure", "action", "High-adrenaline sequences, death-defying stunts, and epic journeys."),
    ("g-thriller", "Psychological Thriller", "thriller", "Mind-bending plots, high tension, and psychological battles."),
    ("g-fantasy", "Fantasy & Myth", "fantasy", "Magic systems, historical mythologies, sword & sorcery, and legendary creatures."),
    ("g-history", "Historical & Biography", "history", "Based on true events, historic figures, and epic chronicles of the past."),
    ("g-comedy", "Comedy & Humor", "comedy", "Laugh-out-loud comedies, witty humor, situational parodies, and fun family-friendly comic series.")
]

FRANCHISES_DATA = [
    ("f-mcu", "Marvel Cinematic Universe", "The massive, interconnected universe of superhero titles starting with Iron Man."),
    ("f-starwars", "Star Wars", "A galaxy far, far away, telling the story of Jedi, Sith, and galactic civil wars."),
    ("f-nolanverse", "Nolanverse", "The cinematic masterpieces of Christopher Nolan, known for intellectual depth and high visual fidelity."),
    ("f-lotr", "Middle-earth", "The legendary epic fantasy world created by J.R.R. Tolkien, directed by Peter Jackson."),
    ("f-spyuniverse", "YRF Spy Universe", "The highly popular interconnected Indian spy thriller universe featuring Tiger, Pathaan, and Kabir.")
]

PEOPLE_DATA = [
    # Directors
    ("p-nolan", "Christopher Nolan", "Chris Nolan", "DIRECTOR", "Known for cerebral, non-linear, and visually spectacular films.", 1970, "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=150", 98.5),
    ("p-villeneuve", "Denis Villeneuve", "Denis Villeneuve", "DIRECTOR", "Master of modern atmosphere, sound design, and world-building in sci-fi.", 1967, "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=150", 95.0),
    ("p-scorsese", "Martin Scorsese", "Marty", "DIRECTOR", "Legendary director of crime epics, psychological character studies, and dramas.", 1942, "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=150", 97.0),
    ("p-jackson", "Peter Jackson", "Peter Jackson", "DIRECTOR", "Acclaimed New Zealand director who brought Middle-earth to life.", 1961, "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=150", 93.0),
    ("p-miyazaki", "Hayao Miyazaki", "Hayao Miyazaki", "DIRECTOR", "Co-founder of Studio Ghibli and master animator.", 1941, "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=150", 96.5),
    ("p-lucas", "George Lucas", "George Lucas", "DIRECTOR", "Creator of Star Wars and pioneer of modern digital effects.", 1944, "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=150", 91.0),
    ("p-rajamouli", "S. S. Rajamouli", "Rajamouli", "DIRECTOR", "Visionary Indian film director known for his high-budget, action-packed, larger-than-life epic action blockbusters.", 1973, "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=150", 96.0),
    
    # Actors
    ("p-murphy", "Cillian Murphy", "Cillian Murphy", "ACTOR", "Frequent Nolan collaborator with striking blue eyes and deep intensity.", 1976, "https://images.unsplash.com/photo-1554080353-a576cf803bda?w=150", 94.2),
    ("p-bale", "Christian Bale", "Christian Bale", "ACTOR", "Method actor known for extreme physical transformations and intense dedication.", 1974, "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=150", 96.0),
    ("p-dicaprio", "Leonardo DiCaprio", "Leo", "ACTOR", "One of the most acclaimed actors of his generation, leading global blockbusters.", 1974, "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=150", 99.0),
    ("p-chalamet", "Timothée Chalamet", "Timo", "ACTOR", "Talented young star leading modern sci-fi and dramatic epics.", 1995, "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=150", 95.5),
    ("p-zendaya", "Zendaya", "Zendaya", "ACTOR", "Award-winning actress and global fashion and screen icon.", 1996, "https://images.unsplash.com/photo-1542751371-adc38448a05e?w=150", 97.0),
    ("p-deniro", "Robert De Niro", "Bobby D", "ACTOR", "Iconic actor who defined generations of dramatic and crime characters.", 1943, "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=150", 98.0),
    ("p-pacino", "Al Pacino", "Al Pacino", "ACTOR", "Legendary powerhouse actor famous for intense, charismatic performances.", 1940, "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=150", 97.8),
    ("p-hamill", "Mark Hamill", "Mark Hamill", "ACTOR", "The definitive Luke Skywalker and master voice actor.", 1951, "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=150", 89.5),
    ("p-gosling", "Ryan Gosling", "Ryan Gosling", "ACTOR", "Charismatic star of dramatic, romantic, and dystopian films.", 1980, "https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=150", 95.0),
    ("p-robbie", "Margot Robbie", "Margot Robbie", "ACTOR", "Versatile actress and producer leading global culture-defining films.", 1990, "https://images.unsplash.com/photo-1513151233558-d860c5398176?w=150", 98.2),
    ("p-reeves", "Keanu Reeves", "The One", "ACTOR", "Beloved star of groundbreaking sci-fi and action franchises.", 1964, "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=150", 94.0),
    ("p-srk", "Shah Rukh Khan", "King Khan", "ACTOR", "One of the world's biggest movie stars, known as the King of Bollywood with massive global popularity.", 1965, "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=150", 99.2),
    ("p-aamikhan", "Aamir Khan", "Mr. Perfectionist", "ACTOR", "Acclaimed Indian superstar and producer known for character-driven, highly influential films.", 1965, "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=150", 97.5),
    ("p-nawaz", "Nawazuddin Siddiqui", "Nawaz", "ACTOR", "Intense, critically acclaimed Indian actor known for raw, gritty realism and powerful crime performances.", 1974, "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=150", 92.0),
    ("p-deepika", "Deepika Padukone", "Deepika", "ACTOR", "Leading Indian actress who has achieved global success in major action, drama, and fantasy blockbusters.", 1986, "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=150", 97.8),
    ("p-akshay", "Akshay Kumar", "Akki", "ACTOR", "Iconic Bollywood action star and king of physical comedy.", 1967, "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150", 96.0),
    ("p-paresh", "Paresh Rawal", "Babu Bhaiya", "ACTOR", "Legendary Indian character actor famous for his comedic genius.", 1955, "https://images.unsplash.com/photo-1628157582853-a796fa650a6a?w=150", 94.0),
    ("p-salman", "Salman Khan", "Bhai", "ACTOR", "One of the most commercially successful Indian superstars, known as the Bhai of Bollywood with massive action, drama, and comedy blockbuster franchises.", 1965, "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150", 99.0)
]

# Fields: ContentId, PrimaryTitle, OriginalTitle, ContentType, ReleaseYear, AgeRating, DurationMins, SeasonsCount, Synopsis, Tagline, PosterUrl, BannerUrl, TrailerUrl, ImdbRating, PopularityScore, AccessTier, AudioLanguages, SubtitleLanguages, QualityProfiles
TITLES_DATA = [
    # Nolan Sci-Fi/Drama
    ("t-interstellar", "Interstellar", "Interstellar", "MOVIE", 2014, "PG-13", 169, None, 
     "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival on a dying Earth.", 
     "Mankind was born on Earth. It was never meant to die here.", 
     "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&q=80", "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=800&q=80", "https://www.youtube.com/embed/zSWdZVtXT7E", 
     8.7, 98.0, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es", "fr", "de"], ["4K_UHD", "HDR10", "DOLBY_ATMOS"]),
     
    ("t-inception", "Inception", "Inception", "MOVIE", 2010, "PG-13", 148, None, 
     "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.", 
     "Your mind is the scene of the crime.", 
     "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=400&q=80", "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=800&q=80", "https://www.youtube.com/embed/YoHD9XEInc0", 
     8.8, 97.5, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es", "zh"], ["HD", "DOLBY_VISION"]),
     
    ("t-tenet", "Tenet", "Tenet", "MOVIE", 2020, "PG-13", 150, None, 
     "Armed with only one word, Tenet, and fighting for the survival of the entire world, a Protagonist journeys through a twilight world of international espionage on a mission that will unfold in something beyond real time.", 
     "Time runs out.", 
     "https://images.unsplash.com/photo-1496568818309-53d7c7753022?w=400&q=80", "https://images.unsplash.com/photo-1496568818309-53d7c7753022?w=800&q=80", "https://www.youtube.com/embed/LdOM0x0XDwM", 
     7.3, 85.0, "SVOD_PREMIUM", ["en", "es", "fr", "ja"], ["en", "es", "fr", "de", "zh"], ["4K_UHD", "HDR10"]),
     
    ("t-oppenheimer", "Oppenheimer", "Oppenheimer", "MOVIE", 2023, "R", 180, None, 
     "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II.", 
     "The world forever changes.", 
     "https://images.unsplash.com/photo-1461360228754-6e81c478b882?w=400&q=80", "https://images.unsplash.com/photo-1461360228754-6e81c478b882?w=800&q=80", "https://www.youtube.com/embed/uYPbbksJxIg", 
     8.9, 99.5, "TVOD_RENT", ["en", "es", "fr", "hi"], ["en", "es", "hi"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-dunkirk", "Dunkirk", "Dunkirk", "MOVIE", 2017, "PG-13", 106, None, 
     "Allied soldiers from Belgium, the British Commonwealth and Empire, and France are surrounded by the German Army and evacuated during a fierce battle in World War II.", 
     "When 400,000 men couldn't get home, home came for them.", 
     "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400&q=80", "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&q=80", "https://www.youtube.com/embed/F-eMt3GrSL8", 
     7.8, 88.0, "SVOD_STANDARD", ["en", "fr", "de"], ["en", "es"], ["4K_UHD", "HDR10"]),

    ("t-prestige", "The Prestige", "The Prestige", "MOVIE", 2006, "PG-13", 130, None, 
     "After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other.", 
     "Are you watching closely?", 
     "https://images.unsplash.com/photo-1509248961158-e54f6934749c?w=400&q=80", "https://images.unsplash.com/photo-1509248961158-e54f6934749c?w=800&q=80", "https://www.youtube.com/embed/o4gHCmTQDVI", 
     8.5, 92.0, "SVOD_STANDARD", ["en", "es"], ["en", "es", "zh"], ["HD"]),

    ("t-memento", "Memento", "Memento", "MOVIE", 2000, "R", 113, None, 
     "A man with short-term memory loss attempts to track down his wife's murderer through a complex, reverse-chronological puzzle.", 
     "Some memories are best forgotten.", 
     "https://images.unsplash.com/photo-1554080353-a576cf803bda?w=400&q=80", "https://images.unsplash.com/photo-1554080353-a576cf803bda?w=800&q=80", "https://www.youtube.com/embed/4CV41hoyS8A", 
     8.4, 89.0, "FREE_AVOD", ["en"], ["en", "es"], ["HD"]),

    # Villeneuve Sci-Fi
    ("t-dune1", "Dune: Part One", "Dune", "MOVIE", 2021, "PG-13", 155, None, 
     "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset, the spice melange on the desert planet Arrakis.", 
     "Beyond fear, destiny awaits.", 
     "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=400&q=80", "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=800&q=80", "https://www.youtube.com/embed/n9xhJrPXY4g", 
     8.0, 95.0, "SVOD_STANDARD", ["en", "es", "fr", "hi"], ["en", "es", "fr", "de"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-dune2", "Dune: Part Two", "Dune: Part Two", "MOVIE", 2024, "PG-13", 166, None, 
     "Paul Atreides unites with Chani and the Fremen while seeking revenge against the conspirators who destroyed his family, striving to prevent a terrible future.", 
     "Long live the fighters.", 
     "https://images.unsplash.com/photo-1547234935-80c7145ec969?w=400&q=80", "https://images.unsplash.com/photo-1547234935-80c7145ec969?w=800&q=80", "https://www.youtube.com/embed/Way9Dexny3w", 
     8.6, 99.0, "TVOD_RENT", ["en", "es", "ja", "hi"], ["en", "es", "ja"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-bladerunner2049", "Blade Runner 2049", "Blade Runner 2049", "MOVIE", 2017, "R", 164, None, 
     "A new blade runner, LAPD Officer K, unearths a long-buried secret that has the potential to plunge what's left of society into chaos.", 
     "The key to the future is finally unearthed.", 
     "https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=400&q=80", "https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=800&q=80", "https://www.youtube.com/embed/gCcx85zbxz4", 
     8.0, 93.0, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es", "fr", "de"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-arrival", "Arrival", "Arrival", "MOVIE", 2016, "PG-13", 116, None, 
     "A linguist works with the military to communicate with alien lifeforms who have arrived on Earth, discovering a mind-bending secret about time.", 
     "Why are they here?", 
     "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=400&q=80", "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=800&q=80", "https://www.youtube.com/embed/AMgyWT075KY", 
     7.9, 91.0, "SVOD_STANDARD", ["en", "es"], ["en", "es", "zh"], ["HD", "DOLBY_VISION"]),

    ("t-sicario", "Sicario", "Sicario", "MOVIE", 2015, "R", 121, None, 
     "An idealistic FBI agent is enlisted by a government task force to aid in the escalating war against drugs at the border area between the U.S. and Mexico.", 
     "In Mexico, Sicario means hitman.", 
     "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&q=80", "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=800&q=80", "https://www.youtube.com/embed/G8Hok9ih55g", 
     7.7, 87.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["HD"]),

    # Batman & DC Universe
    ("t-darkknight", "The Dark Knight", "The Dark Knight", "MOVIE", 2008, "PG-13", 152, None, 
     "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.", 
     "Why So Serious?", 
     "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=400&q=80", "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=800&q=80", "https://www.youtube.com/embed/EXeTwQWrcwY", 
     9.0, 99.0, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es", "zh"], ["4K_UHD", "HDR10"]),

    ("t-batmanbegins", "Batman Begins", "Batman Begins", "MOVIE", 2005, "PG-13", 140, None, 
     "After training with his mentor, Batman begins his fight to free crime-ridden Gotham City from corruption.", 
     "The legend begins.", 
     "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=400&q=80", "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=800&q=80", "https://www.youtube.com/embed/neY2xiriUNo", 
     8.2, 90.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["HD"]),

    ("t-darkknightrises", "The Dark Knight Rises", "The Dark Knight Rises", "MOVIE", 2012, "PG-13", 164, None, 
     "Eight years after the Joker's reign of anarchy, Batman is forced from his exile with the help of the enigmatic Selina Kyle to defend Gotham from the brutal terrorist Bane.", 
     "The Legend Ends.", 
     "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=400&q=80", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&q=80", "https://www.youtube.com/embed/g8evyE9TuYk", 
     8.4, 94.0, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es", "de"], ["4K_UHD", "HDR10"]),

    ("t-thebatman", "The Batman", "The Batman", "MOVIE", 2022, "PG-13", 176, None, 
     "On his second year of fighting crime, Batman uncovers corruption in Gotham City that connects to his own family while facing a serial killer known as the Riddler.", 
     "Unmask the truth.", 
     "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?w=400&q=80", "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?w=800&q=80", "https://www.youtube.com/embed/mqq_HMC_Nof", 
     7.8, 92.0, "SVOD_PREMIUM", ["en", "es", "ja"], ["en", "es", "zh"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-joker", "Joker", "Joker", "MOVIE", 2019, "R", 122, None, 
     "A mentally troubled stand-up comedian embarks on a downward spiral that leads to the creation of an iconic criminal mastermind in Gotham.", 
     "Put on a happy face.", 
     "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=400&q=80", "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=800&q=80", "https://www.youtube.com/embed/zAGVQLH_XxY", 
     8.4, 95.5, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es", "zh"], ["4K_UHD", "DOLBY_VISION"]),

    # Scorsese Crime & Drama
    ("t-irishman", "The Irishman", "The Irishman", "MOVIE", 2019, "R", 209, None, 
     "An old truck driver reflects on his past as a hitman for a mob family, and his connection to the disappearance of labor leader Jimmy Hoffa.", 
     "His painting. His life.", 
     "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&q=80", "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=800&q=80", "https://www.youtube.com/embed/WHXXBs0YtU8", 
     7.8, 86.0, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-goodfellas", "Goodfellas", "Goodfellas", "MOVIE", 1990, "R", 145, None, 
     "The story of Henry Hill and his life in the mob, relationship with his wife Karen, and mob partners Jimmy Conway and Tommy DeVito.", 
     "As far back as I can remember, I always wanted to be a gangster.", 
     "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=400&q=80", "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=800&q=80", "https://www.youtube.com/embed/2ilzidi_J8Q", 
     8.7, 94.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["HD"]),

    ("t-killersflower", "Killers of the Flower Moon", "Killers of the Flower Moon", "MOVIE", 2023, "R", 206, None, 
     "Members of the Osage tribe in northeastern Oklahoma are murdered under mysterious circumstances in the 1920s, sparking a major F.B.I. investigation.", 
     "Based on a true conspiracy of greed.", 
     "https://images.unsplash.com/photo-1533105079780-92b9be482077?w=400&q=80", "https://images.unsplash.com/photo-1533105079780-92b9be482077?w=800&q=80", "https://www.youtube.com/embed/EP34YtfsOnI", 
     7.6, 92.5, "TVOD_RENT", ["en", "es"], ["en", "es"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-godfather", "The Godfather", "The Godfather", "MOVIE", 1972, "R", 175, None, 
     "Don Vito Corleone, head of a mafia family, decides to hand over his empire to his youngest son Michael, triggering a bloody gang war.", 
     "An offer you can't refuse.", 
     "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400&q=80", "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=800&q=80", "https://www.youtube.com/embed/UaVTIH8ujFM", 
     9.2, 98.5, "SVOD_STANDARD", ["en", "it", "es"], ["en", "es"], ["4K_UHD", "HDR10"]),

    # Anime & Animation
    ("t-spiritedaway", "Spirited Away", "Sen to Chihiro no Kamikakushi", "MOVIE", 2001, "PG", 125, None, 
     "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, where humans are changed into beasts.", 
     "Nothing that happens is ever forgotten, even if you can't remember it.", 
     "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=400&q=80", "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=800&q=80", "https://www.youtube.com/embed/ByXuk9QqQMC", 
     8.6, 96.0, "SVOD_STANDARD", ["ja", "en", "es"], ["en", "es", "fr"], ["HD"]),

    ("t-princessmononoke", "Princess Mononoke", "Mononoke-hime", "MOVIE", 1997, "PG-13", 134, None, 
     "On a journey to find the cure for a Tatarigami's curse, Ashitaka finds himself in the middle of a war between the forest gods and Tatara, a mining colony.", 
     "The Fate of the World Rests on His Courage.", 
     "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400&q=80", "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800&q=80", "https://www.youtube.com/embed/4OiM_ABiZg4", 
     8.4, 91.5, "SVOD_STANDARD", ["ja", "en", "es"], ["en", "es"], ["HD"]),

    ("t-spiderverse", "Spider-Man: Into the Spider-Verse", "Spider-Man: Into the Spider-Verse", "MOVIE", 2018, "PG", 117, None, 
     "Teen Miles Morales becomes the Spider-Man of his universe and must join with five spider-powered individuals from other dimensions to stop a threat for all realities.", 
     "Anyone can wear the mask.", 
     "https://images.unsplash.com/photo-1569003339405-ea396a5a8a90?w=400&q=80", "https://images.unsplash.com/photo-1569003339405-ea396a5a8a90?w=800&q=80", "https://www.youtube.com/embed/g4HbzQFUp3A", 
     8.4, 96.5, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es", "fr"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    # Lord of the Rings Epic Fantasy
    ("t-lotr1", "The Lord of the Rings: The Fellowship of the Ring", "The Fellowship of the Ring", "MOVIE", 2001, "PG-13", 178, None, 
     "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.", 
     "One Ring to rule them all.", 
     "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400&q=80", "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800&q=80", "https://www.youtube.com/embed/V75dMMIW2B4", 
     8.8, 98.0, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es"], ["4K_UHD", "HDR10", "DOLBY_ATMOS"]),

    ("t-lotr2", "The Lord of the Rings: The Two Towers", "The Two Towers", "MOVIE", 2002, "PG-13", 179, None, 
     "While Frodo and Sam edge closer to Mordor with the help of the shifty Gollum, the divided fellowship makes a stand against Sauron's new ally, Saruman, and his hordes of Isengard.", 
     "The battle for Middle-earth has begun.", 
     "https://images.unsplash.com/photo-1535663116191-4e1b8bbfbdf6?w=400&q=80", "https://images.unsplash.com/photo-1535663116191-4e1b8bbfbdf6?w=800&q=80", "https://www.youtube.com/embed/LbfMDwc4azU", 
     8.8, 97.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["4K_UHD", "HDR10", "DOLBY_ATMOS"]),

    ("t-lotr3", "The Lord of the Rings: The Return of the King", "The Return of the King", "MOVIE", 2003, "PG-13", 201, None, 
     "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring.", 
     "The Eye of the Enemy is moving.", 
     "https://images.unsplash.com/photo-1505438531158-2e598f44af31?w=400&q=80", "https://images.unsplash.com/photo-1505438531158-2e598f44af31?w=800&q=80", "https://www.youtube.com/embed/r5X-hFf6Bwo", 
     9.0, 98.5, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es"], ["4K_UHD", "HDR10", "DOLBY_ATMOS"]),

    # Star Wars Space Opera
    ("t-starwars4", "Star Wars: A New Hope", "Star Wars: Episode IV - A New Hope", "MOVIE", 1977, "PG", 121, None, 
     "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire's world-destroying battle station, while also attempting to rescue Princess Leia from the mysterious Darth Vader.", 
     "A long time ago in a galaxy far, far away...", 
     "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=400&q=80", "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=800&q=80", "https://www.youtube.com/embed/vZ78yosqiM8", 
     8.6, 95.0, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es", "fr"], ["4K_UHD", "HDR10"]),

    ("t-starwars5", "Star Wars: The Empire Strikes Back", "Star Wars: Episode V - The Empire Strikes Back", "MOVIE", 1980, "PG", 124, None, 
     "After the Rebels are brutally overpowered by the Empire on the ice planet Hoth, Luke Skywalker begins Jedi training with Yoda, while his friends are pursued by Darth Vader.", 
     "The Adventure Continues...", 
     "https://images.unsplash.com/photo-1483921020237-2ff51e8e4b22?w=400&q=80", "https://images.unsplash.com/photo-1483921020237-2ff51e8e4b22?w=800&q=80", "https://www.youtube.com/embed/JNwNXF9Y6kY", 
     8.7, 96.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["4K_UHD", "HDR10"]),

    ("t-starwars6", "Star Wars: Return of the Jedi", "Star Wars: Episode VI - Return of the Jedi", "MOVIE", 1983, "PG", 131, None, 
     "After a daring mission to rescue Han Solo from Jabba the Hutt, the Rebels dispatch to Endor to destroy a second Death Star, while Luke struggles to help Darth Vader back from the dark side.", 
     "The Triumph of the Force.", 
     "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400&q=80", "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=800&q=80", "https://www.youtube.com/embed/5U1OtvT9MyE", 
     8.3, 92.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["4K_UHD", "HDR10"]),

    # Matrix Sci-Fi Cyberpunk
    ("t-matrix", "The Matrix", "The Matrix", "MOVIE", 1999, "R", 136, None, 
     "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence.", 
     "Free your mind.", 
     "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=400&q=80", "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=800&q=80", "https://www.youtube.com/embed/m8e-FF8MsqU", 
     8.7, 96.0, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es", "zh"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-matrixreload", "The Matrix Reloaded", "The Matrix Reloaded", "MOVIE", 2003, "R", 138, None, 
     "Neo and the rebel leaders estimate that they have 72 hours before 250,000 Sentinels discover Zion and destroy it, requiring him to go deeper into the Matrix.", 
     "Free your mind.", 
     "https://images.unsplash.com/photo-1515879218367-8466d910aaa4?w=400&q=80", "https://images.unsplash.com/photo-1515879218367-8466d910aaa4?w=800&q=80", "https://www.youtube.com/embed/hMb9z8mN8vU", 
     7.2, 78.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["HD"]),

    # Hit Series / Shows (TV_SERIES)
    ("t-breakingbad", "Breaking Bad", "Breaking Bad", "TV_SERIES", 2008, "TV-MA", None, 5, 
     "A chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine with a former student in order to secure his family's future.", 
     "Change the Equation.", 
     "https://images.unsplash.com/photo-1532187863486-abf9d39d66e8?w=400&q=80", "https://images.unsplash.com/photo-1532187863486-abf9d39d66e8?w=800&q=80", "https://www.youtube.com/embed/HhesaQXLuRY", 
     9.5, 99.5, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es", "fr", "zh"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-bettercallsaul", "Better Call Saul", "Better Call Saul", "TV_SERIES", 2015, "TV-MA", None, 6, 
     "The trials and tribulations of criminal lawyer Jimmy McGill in the years leading up to his fateful representation of Walter White in Breaking Bad.", 
     "Lying is what we do best.", 
     "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?w=400&q=80", "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?w=800&q=80", "https://www.youtube.com/embed/hKcxOdZ8GjU", 
     9.0, 97.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-succession", "Succession", "Succession", "TV_SERIES", 2018, "TV-MA", None, 4, 
     "The Roy family is known for controlling the biggest media and entertainment company in the world. However, their world changes when their father steps down.", 
     "Waystar Royco. Business is War.", 
     "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=400&q=80", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&q=80", "https://www.youtube.com/embed/OzY2q27p2zM", 
     8.9, 95.5, "SVOD_PREMIUM", ["en", "es"], ["en", "es"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-severance", "Severance", "Severance", "TV_SERIES", 2022, "TV-MA", None, 1, 
     "Mark leads a team of office workers whose memories have been surgically divided between their work and personal lives, beginning a journey to uncover the dark conspiracy.", 
     "Please dial down your expectations.", 
     "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400&q=80", "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&q=80", "https://www.youtube.com/embed/xkT2Z99I9K0", 
     8.7, 93.0, "SVOD_PREMIUM", ["en", "es", "fr"], ["en", "es"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-blackmirror", "Black Mirror", "Black Mirror", "TV_SERIES", 2011, "TV-MA", None, 6, 
     "An anthology series exploring a twisted, high-tech multiverse where humanity's greatest innovations and darkest instincts collide.", 
     "The future is bright.", 
     "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=400&q=80", "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&q=80", "https://www.youtube.com/embed/V0XOApF5nLU", 
     8.7, 94.0, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es", "zh"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-strangerthings", "Stranger Things", "Stranger Things", "TV_SERIES", 2016, "TV-14", None, 4, 
     "When a young boy vanishes, a small town uncovers a mystery involving secret government experiments, terrifying supernatural forces and one strange little girl with powers.", 
     "One summer can change everything.", 
     "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=400&q=80", "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=800&q=80", "https://www.youtube.com/embed/b9EkMc79ZSU", 
     8.7, 98.0, "SVOD_STANDARD", ["en", "es", "fr", "ja"], ["en", "es", "ja"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-cyberpunk", "Cyberpunk: Edgerunners", "Cyberpunk: Edgerunners", "TV_SERIES", 2022, "TV-MA", None, 1, 
     "A street kid trying to survive in a technology and body modification-obsessed city of the future decides to stay alive by becoming an edgerunner mercenary.", 
     "Choose your upgrades wisely.", 
     "https://images.unsplash.com/photo-1542751371-adc38448a05e?w=400&q=80", "https://images.unsplash.com/photo-1542751371-adc38448a05e?w=800&q=80", "https://www.youtube.com/embed/JtqIas3bYhg", 
     8.3, 90.0, "SVOD_STANDARD", ["ja", "en", "es"], ["en", "es", "fr"], ["HD", "DOLBY_VISION"]),

    ("t-arcane", "Arcane", "Arcane: League of Legends", "TV_SERIES", 2021, "TV-14", None, 2, 
     "Set in the utopian region of Piltover and the oppressed underground of Zaun, the story follows the origins of two iconic League champions-and the power that will tear them apart.", 
     "Every legend has a beginning.", 
     "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=400&q=80", "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=800&q=80", "https://www.youtube.com/embed/fXmAurh012s", 
     9.0, 97.5, "SVOD_STANDARD", ["en", "es", "ja", "fr"], ["en", "es", "ja"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-chernobyl", "Chernobyl", "Chernobyl", "TV_SERIES", 2019, "TV-MA", None, 1, 
     "In April 1986, an explosion at the Chernobyl Nuclear Power Plant in the Union of Soviet Socialist Republics becomes one of the world's worst man-made catastrophes.", 
     "What is the cost of lies?", 
     "https://images.unsplash.com/photo-1485081661826-f6d702410a51?w=400&q=80", "https://images.unsplash.com/photo-1485081661826-f6d702410a51?w=800&q=80", "https://www.youtube.com/embed/s9APLXM9Ei8", 
     9.4, 96.0, "SVOD_PREMIUM", ["en", "ru"], ["en", "es", "fr"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-lastofus", "The Last of Us", "The Last of Us", "TV_SERIES", 2023, "TV-MA", None, 1, 
     "After a global pandemic destroys civilization, a hardened survivor takes charge of a 14-year-old girl who may be humanity's last hope for survival in a cordyceps post-apocalyptic world.", 
     "When you're lost in the darkness, look for the light.", 
     "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=400&q=80", "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=800&q=80", "https://www.youtube.com/embed/uLtkt8BonwM", 
     8.8, 97.0, "SVOD_PREMIUM", ["en", "es", "pt"], ["en", "es"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-theboys", "The Boys", "The Boys", "TV_SERIES", 2019, "TV-MA", None, 4, 
     "A group of vigilantes set out to take down corrupt superheroes who abuse their superpowers and corporate backing.", 
     "Never meet your heroes.", 
     "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=400&q=80", "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=800&q=80", "https://www.youtube.com/embed/M1BhREpk_94", 
     8.7, 95.0, "SVOD_STANDARD", ["en", "es", "pt", "hi"], ["en", "es"], ["4K_UHD", "HDR10", "DOLBY_ATMOS"]),

    # More Movies to reach 50+
    ("t-mulan", "Mulan", "Mulan", "MOVIE", 1998, "G", 88, None, 
     "To save her father from death in the Imperial Army, a young Chinese maiden secretly goes in his place and becomes one of China's greatest heroines in the process.", 
     "The flower that blooms in adversity is the most rare and beautiful of all.", 
     "https://images.unsplash.com/photo-1508013861974-9f63471c68e5?w=400&q=80", "https://images.unsplash.com/photo-1508013861974-9f63471c68e5?w=800&q=80", "https://www.youtube.com/embed/KK8Fm_p5uE4", 
     7.6, 84.0, "SVOD_STANDARD", ["en", "zh"], ["en", "es"], ["HD"]),

    ("t-barbie", "Barbie", "Barbie", "MOVIE", 2023, "PG-13", 114, None, 
     "Stereotypical Barbie experiences a full-on existential crisis and must travel to the real world with Ken to discover the truth about herself.", 
     "She's everything. He's just Ken.", 
     "https://images.unsplash.com/photo-1513151233558-d860c5398176?w=400&q=80", "https://images.unsplash.com/photo-1513151233558-d860c5398176?w=800&q=80", "https://www.youtube.com/embed/pBk4NYhWNMM", 
     6.9, 94.0, "SVOD_STANDARD", ["en", "es", "ja"], ["en", "es"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-lalaland", "La La Land", "La La Land", "MOVIE", 2016, "PG-13", 128, None, 
     "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.", 
     "Here's to the fools who dream.", 
     "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400&q=80", "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=800&q=80", "https://www.youtube.com/embed/0pdqf4P9MB8", 
     8.0, 91.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["HD", "DOLBY_ATMOS"]),

    ("t-firstman", "First Man", "First Man", "MOVIE", 2018, "PG-13", 141, None, 
     "A look at the life of the legendary astronaut, Neil Armstrong, and the legendary space mission that led him to become the first man to walk on the Moon in 1969.", 
     "An experience that will change your perspective.", 
     "https://images.unsplash.com/photo-1447433589675-4aaa569f3e05?w=400&q=80", "https://images.unsplash.com/photo-1447433589675-4aaa569f3e05?w=800&q=80", "https://www.youtube.com/embed/2Mcll-P8S9Y", 
     7.3, 80.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["4K_UHD", "HDR10"]),

    ("t-pulpfiction", "Pulp Fiction", "Pulp Fiction", "MOVIE", 1994, "R", 154, None, 
     "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.", 
     "Just because you are a character doesn't mean that you have character.", 
     "https://images.unsplash.com/photo-1585647347483-22b66260dfff?w=400&q=80", "https://images.unsplash.com/photo-1585647347483-22b66260dfff?w=800&q=80", "https://www.youtube.com/embed/s7EdQ4FqbhY", 
     8.9, 96.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["HD"]),

    ("t-django", "Django Unchained", "Django Unchained", "MOVIE", 2012, "R", 165, None, 
     "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner.", 
     "Life, liberty, and the pursuit of vengeance.", 
     "https://images.unsplash.com/photo-1533105079780-92b9be482077?w=400&q=80", "https://images.unsplash.com/photo-1533105079780-92b9be482077?w=800&q=80", "https://www.youtube.com/embed/0fUCuvbWKD0", 
     8.4, 94.5, "SVOD_STANDARD", ["en", "de"], ["en", "es"], ["HD"]),

    ("t-solaris", "Solaris", "Solyaris", "MOVIE", 1972, "PG", 167, None, 
     "A psychologist is sent to a space station orbiting a mysterious ocean planet to investigate the emotional collapse of its scientific crew.", 
     "A journey deep into the mind.", 
     "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=400&q=80", "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=800&q=80", "https://www.youtube.com/embed/8I_V7F9NloE", 
     8.0, 75.0, "FREE_AVOD", ["ru", "en"], ["en", "es", "fr"], ["HD"]),

    ("t-truedetective", "True Detective", "True Detective", "TV_SERIES", 2014, "TV-MA", None, 4, 
     "Seasonal anthology series in which police investigations unearth the personal and professional secrets of those involved, both within and outside the law.", 
     "Touch darkness and darkness touches you.", 
     "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&q=80", "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=800&q=80", "https://www.youtube.com/embed/fVQXyMcNJL4", 
     8.9, 94.0, "SVOD_PREMIUM", ["en", "es"], ["en", "es"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-avengersinf", "Avengers: Infinity War", "Avengers: Infinity War", "MOVIE", 2018, "PG-13", 149, None, 
     "The Avengers and their allies must be willing to sacrifice all in an attempt to defeat the powerful Thanos before his blitz of devastation and ruin puts an end to the universe.", 
     "An entire universe. Once and for all.", 
     "https://images.unsplash.com/photo-1535663116191-4e1b8bbfbdf6?w=400&q=80", "https://images.unsplash.com/photo-1535663116191-4e1b8bbfbdf6?w=800&q=80", "https://www.youtube.com/embed/6ZfuNTqbHE8", 
     8.4, 98.0, "SVOD_STANDARD", ["en", "es", "fr"], ["en", "es"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-avengersend", "Avengers: Endgame", "Avengers: Endgame", "MOVIE", 2019, "PG-13", 181, None, 
     "After the devastating events of Avengers: Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions.", 
     "Part of the journey is the end.", 
     "https://images.unsplash.com/photo-1461360228754-6e81c478b882?w=400&q=80", "https://images.unsplash.com/photo-1461360228754-6e81c478b882?w=800&q=80", "https://www.youtube.com/embed/TcMBFSGVi1c", 
     8.4, 98.5, "SVOD_STANDARD", ["en", "es", "fr", "ja"], ["en", "es"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-ironman", "Iron Man", "Iron Man", "MOVIE", 2008, "PG-13", 126, None, 
     "After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique weaponized suit of armor to fight evil.", 
     "Heroes aren't born. They're built.", 
     "https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=400&q=80", "https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=800&q=80", "https://www.youtube.com/embed/8hYlB38asDY", 
     7.9, 93.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["4K_UHD", "HDR10"]),

    # ============================================================================
    # INDIAN MOVIES & SHOWS (Seeding all requested genres beautifully!)
    # ============================================================================
    ("t-rrr", "RRR", "RRR", "MOVIE", 2022, "R", 187, None, 
     "A fearless Indian revolutionary and an officer in the British army, who are close friends, join forces to lead an epic rebellion against the tyrannical British Raj in 1920s India.", 
     "Rise, Roar, Revolt.", 
     "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=400&q=80", "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=800&q=80", "https://www.youtube.com/embed/NgBoMJy386M", 
     7.8, 95.0, "SVOD_STANDARD", ["te", "hi", "en", "ta", "ml"], ["en", "es", "fr"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-3idiots", "3 Idiots", "3 Idiots", "MOVIE", 2009, "PG-13", 170, None, 
     "Two friends search for their long-lost companion. They revisit their college days and recall the memories of their friend who inspired them to think differently, even as the world called them idiots.", 
     "All is Well.", 
     "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=400&q=80", "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=800&q=80", "https://www.youtube.com/embed/K0eDlFX9GMc", 
     8.4, 93.0, "FREE_AVOD", ["hi", "en"], ["en", "es", "zh", "ja"], ["HD"]),

    ("t-drishyam", "Drishyam", "Drishyam", "MOVIE", 2013, "PG-13", 160, None, 
     "A common cable TV operator goes to extreme lengths, crafting a perfect alibi to protect his family from a rigorous police investigation when they accidentally commit a crime.", 
     "Visuals can be deceiving.", 
     "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&q=80", "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=800&q=80", "https://www.youtube.com/embed/AuuX2j143EE", 
     8.3, 89.0, "SVOD_STANDARD", ["ml", "hi", "ta", "te"], ["en", "fr"], ["HD"]),

    ("t-sacredgames", "Sacred Games", "Sacred Games", "TV_SERIES", 2018, "TV-MA", None, 2, 
     "A link in their pasts leads an honest police officer in Mumbai to a fugitive gang boss Ganesh Gaitonde, whose cryptic warning spurs the officer on a quest to save Mumbai from cataclysmic destruction.", 
     "Kalyug has begun.", 
     "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=400&q=80", "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=800&q=80", "https://www.youtube.com/embed/w-Xe8gLBc5I", 
     8.6, 91.5, "SVOD_PREMIUM", ["hi", "en", "ta"], ["en", "es", "fr"], ["4K_UHD", "DOLBY_VISION"]),

    ("t-swades", "Swades", "Swades", "MOVIE", 2004, "PG", 189, None, 
     "A successful Indian scientist working at NASA returns to an underprivileged village in rural India to search for his childhood nanny and in the process, finds his roots and true purpose.", 
     "We, the People.", 
     "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=400&q=80", "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=800&q=80", "https://www.youtube.com/embed/fD3zI-6v8B0", 
     8.2, 88.0, "FREE_AVOD", ["hi"], ["en", "es"], ["HD"]),

    ("t-kalki", "Kalki 2898 AD", "Kalki 2898 AD", "MOVIE", 2024, "PG-13", 181, None, 
     "A modern avatar of the legendary Vishnu, a Hindu god, is believed to have descended to earth in a post-apocalyptic dystopian futuristic world to protect the last remaining city.", 
     "The battle of ages begins.", 
     "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&q=80", "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=800&q=80", "https://www.youtube.com/embed/kQw4w9WgXcQ", 
     7.6, 94.0, "SVOD_PREMIUM", ["te", "hi", "ta", "ml", "en"], ["en", "es"], ["4K_UHD", "DOLBY_VISION", "DOLBY_ATMOS"]),

    ("t-ramayana", "Ramayana: The Legend of Prince Rama", "Ramayana: The Legend of Prince Rama", "MOVIE", 1992, "G", 135, None, 
     "An award-winning anime adaptation of the ancient Hindu epic Ramayana, recounting the exile of Prince Rama, his wife Sita, and his glorious battle against the demon king Ravana.", 
     "The timeless anime legend.", 
     "https://images.unsplash.com/photo-1608155686393-8fdd966d784d?w=400&q=80", "https://images.unsplash.com/photo-1608155686393-8fdd966d784d?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     9.1, 87.0, "FREE_AVOD", ["ja", "hi", "en"], ["en"], ["HD"]),

    ("t-herapheri", "Hera Pheri", "Hera Pheri", "MOVIE", 2000, "PG", 156, None, 
     "Three unemployed, eccentric men Baburao, Raju, and Shyam look for answers to all their financial problems, but find themselves in the middle of a hilarious kidnapping and ransom case.", 
     "The ultimate comic laugh riot.", 
     "https://images.unsplash.com/photo-1514306191717-452ec28c7814?w=400&q=80", "https://images.unsplash.com/photo-1514306191717-452ec28c7814?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     8.2, 95.0, "FREE_AVOD", ["hi"], ["en"], ["HD"]),

    ("t-andazapna", "Andaz Apna Apna", "Andaz Apna Apna", "MOVIE", 1994, "PG", 160, None, 
     "Two lazy slackers Amar and Prem compete to win the heart of a wealthy heiress, landing in hilarious comic situations with a local gangster named Crime Master Gogo.", 
     "Two rivals, one comedy of errors.", 
     "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=400&q=80", "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     8.0, 91.0, "FREE_AVOD", ["hi"], ["en"], ["HD"]),

    ("t-munnabhai", "Munna Bhai M.B.B.S.", "Munna Bhai M.B.B.S.", "MOVIE", 2003, "PG-13", 156, None, 
     "A lovable Mumbai gangster sets out to fulfill his father's dream of becoming a doctor, using his own unique form of comedy, warmth, and compassion.", 
     "Laughter is the best medicine.", 
     "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=400&q=80", "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     8.1, 94.0, "SVOD_STANDARD", ["hi"], ["en"], ["HD"]),

    ("t-hangover", "The Hangover", "The Hangover", "MOVIE", 2009, "R", 100, None, 
     "Three buddies wake up from a wild bachelor party in Las Vegas with no memory of the previous night, a tiger in their bathroom, and the groom missing.", 
     "Some guys just can't handle Vegas.", 
     "https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=400&q=80", "https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     7.7, 96.0, "SVOD_STANDARD", ["en", "es"], ["en", "es"], ["HD"]),

    ("t-bajrangi", "Bajrangi Bhaijaan", "Bajrangi Bhaijaan", "MOVIE", 2015, "PG", 163, None, 
     "An Indian man with a magnanimous heart, Pavan, undertakes the task to reunite a young mute Pakistani girl, Munni, with her family in her homeland.", 
     "A journey of love and humanity beyond borders.", 
     "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=400&q=80", "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     8.1, 98.0, "SVOD_STANDARD", ["hi", "ur"], ["en", "es"], ["HD"]),

    ("t-tiger", "Ek Tha Tiger", "Ek Tha Tiger", "MOVIE", 2012, "PG-13", 132, None, 
     "A raw agent code-named Tiger is sent to Dublin to observe an Indian scientist, but falls in love with his caretaker Zoya, who has her own secret identity.", 
     "The ultimate action-packed spy romance.", 
     "https://images.unsplash.com/photo-1496568818309-53d7c7753022?w=400&q=80", "https://images.unsplash.com/photo-1496568818309-53d7c7753022?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     7.3, 95.0, "SVOD_STANDARD", ["hi", "en"], ["en", "fr"], ["HD"]),

    ("t-dabangg", "Dabangg", "Dabangg", "MOVIE", 2010, "PG-13", 126, None, 
     "A corrupt but fearless police officer named Chulbul Pandey faces challenges from his family, local gangsters, and politicians, handling them with witty comic one-liners and bold action.", 
     "Fearless, funny, and absolutely unstoppable.", 
     "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=400&q=80", "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=800&q=80", "https://www.youtube.com/embed/dQw4w9WgXcQ", 
     7.2, 94.0, "FREE_AVOD", ["hi"], ["en"], ["HD"])
]

# Title to Talent map (ContentId, PersonId, Role, CharacterName, BillingOrder)
TITLE_TALENT_DATA = [
    # Interstellar
    ("t-interstellar", "p-nolan", "DIRECTOR", None, 1),
    ("t-interstellar", "p-chalamet", "SUPPORTING_ACTOR", "Young Tom", 4),
    
    # Inception
    ("t-inception", "p-nolan", "DIRECTOR", None, 1),
    ("t-inception", "p-dicaprio", "LEAD_ACTOR", "Cobb", 2),
    ("t-inception", "p-murphy", "SUPPORTING_ACTOR", "Robert Fischer", 3),
    
    # Tenet
    ("t-tenet", "p-nolan", "DIRECTOR", None, 1),
    
    # Oppenheimer
    ("t-oppenheimer", "p-nolan", "DIRECTOR", None, 1),
    ("t-oppenheimer", "p-murphy", "LEAD_ACTOR", "J. Robert Oppenheimer", 2),
    ("t-oppenheimer", "p-deniro", "SUPPORTING_ACTOR", "President Truman (Simulated)", 10),
    
    # Dunkirk
    ("t-dunkirk", "p-nolan", "DIRECTOR", None, 1),
    ("t-dunkirk", "p-murphy", "SUPPORTING_ACTOR", "Shivering Soldier", 2),
    
    # Prestige
    ("t-prestige", "p-nolan", "DIRECTOR", None, 1),
    ("t-prestige", "p-bale", "LEAD_ACTOR", "Alfred Borden", 2),
    
    # Memento
    ("t-memento", "p-nolan", "DIRECTOR", None, 1),
    
    # Dune: Part One
    ("t-dune1", "p-villeneuve", "DIRECTOR", None, 1),
    ("t-dune1", "p-chalamet", "LEAD_ACTOR", "Paul Atreides", 2),
    ("t-dune1", "p-zendaya", "SUPPORTING_ACTOR", "Chani", 3),
    
    # Dune: Part Two
    ("t-dune2", "p-villeneuve", "DIRECTOR", None, 1),
    ("t-dune2", "p-chalamet", "LEAD_ACTOR", "Paul Atreides", 2),
    ("t-dune2", "p-zendaya", "LEAD_ACTOR", "Chani", 3),
    
    # Blade Runner 2049
    ("t-bladerunner2049", "p-villeneuve", "DIRECTOR", None, 1),
    ("t-bladerunner2049", "p-gosling", "LEAD_ACTOR", "Officer K", 2),
    
    # Arrival
    ("t-arrival", "p-villeneuve", "DIRECTOR", None, 1),
    
    # Sicario
    ("t-sicario", "p-villeneuve", "DIRECTOR", None, 1),
    
    # Dark Knight
    ("t-darkknight", "p-nolan", "DIRECTOR", None, 1),
    ("t-darkknight", "p-bale", "LEAD_ACTOR", "Bruce Wayne / Batman", 2),
    ("t-darkknight", "p-murphy", "SUPPORTING_ACTOR", "Scarecrow", 5),
    
    # Batman Begins
    ("t-batmanbegins", "p-nolan", "DIRECTOR", None, 1),
    ("t-batmanbegins", "p-bale", "LEAD_ACTOR", "Bruce Wayne / Batman", 2),
    ("t-batmanbegins", "p-murphy", "SUPPORTING_ACTOR", "Scarecrow", 4),
    
    # Dark Knight Rises
    ("t-darkknightrises", "p-nolan", "DIRECTOR", None, 1),
    ("t-darkknightrises", "p-bale", "LEAD_ACTOR", "Bruce Wayne / Batman", 2),
    ("t-darkknightrises", "p-murphy", "SUPPORTING_ACTOR", "Scarecrow (Cameo)", 6),
    
    # The Batman
    ("t-thebatman", "p-gosling", "SUPPORTING_ACTOR", "GCPD Officer (Simulated)", 10),
    
    # Joker
    ("t-joker", "p-deniro", "SUPPORTING_ACTOR", "Murray Franklin", 2),
    
    # Irishman
    ("t-irishman", "p-scorsese", "DIRECTOR", None, 1),
    ("t-irishman", "p-deniro", "LEAD_ACTOR", "Frank Sheeran", 2),
    ("t-irishman", "p-pacino", "LEAD_ACTOR", "Jimmy Hoffa", 3),
    
    # Goodfellas
    ("t-goodfellas", "p-scorsese", "DIRECTOR", None, 1),
    ("t-goodfellas", "p-deniro", "LEAD_ACTOR", "Jimmy Conway", 2),
    
    # Killers of Flower Moon
    ("t-killersflower", "p-scorsese", "DIRECTOR", None, 1),
    ("t-killersflower", "p-dicaprio", "LEAD_ACTOR", "Ernest Burkhart", 2),
    ("t-killersflower", "p-deniro", "LEAD_ACTOR", "William Hale", 3),
    
    # Godfather
    ("t-godfather", "p-scorsese", "SUPPORTING_ACTOR", "Advisor (Simulated)", 10),
    ("t-godfather", "p-pacino", "LEAD_ACTOR", "Michael Corleone", 2),
    
    # Spirited Away
    ("t-spiritedaway", "p-miyazaki", "DIRECTOR", None, 1),
    
    # Princess Mononoke
    ("t-princessmononoke", "p-miyazaki", "DIRECTOR", None, 1),
    
    # Spider-Verse
    ("t-spiderverse", "p-miyazaki", "SUPPORTING_ACTOR", "Animator (Simulated)", 10),
    
    # Lord of the Rings
    ("t-lotr1", "p-jackson", "DIRECTOR", None, 1),
    ("t-lotr2", "p-jackson", "DIRECTOR", None, 1),
    ("t-lotr3", "p-jackson", "DIRECTOR", None, 1),
    
    # Star Wars
    ("t-starwars4", "p-lucas", "DIRECTOR", None, 1),
    ("t-starwars4", "p-hamill", "LEAD_ACTOR", "Luke Skywalker", 2),
    ("t-starwars5", "p-lucas", "DIRECTOR", None, 1),
    ("t-starwars5", "p-hamill", "LEAD_ACTOR", "Luke Skywalker", 2),
    ("t-starwars6", "p-lucas", "DIRECTOR", None, 1),
    ("t-starwars6", "p-hamill", "LEAD_ACTOR", "Luke Skywalker", 2),
    
    # Matrix
    ("t-matrix", "p-reeves", "LEAD_ACTOR", "Neo", 1),
    ("t-matrixreload", "p-reeves", "LEAD_ACTOR", "Neo", 1),
    
    # Series
    ("t-breakingbad", "p-cranston_simulated", "LEAD_ACTOR", "Walter White", 1),
    ("t-bettercallsaul", "p-odom_simulated", "LEAD_ACTOR", "Jimmy McGill", 1),
    ("t-succession", "p-cox_simulated", "LEAD_ACTOR", "Logan Roy", 1),
    ("t-severance", "p-scott_simulated", "LEAD_ACTOR", "Mark Scout", 1),
    ("t-blackmirror", "p-anthology_simulated", "DIRECTOR", None, 1),
    ("t-strangerthings", "p-duffer_simulated", "DIRECTOR", None, 1),
    ("t-cyberpunk", "p-trigger_simulated", "DIRECTOR", None, 1),
    ("t-arcane", "p-fortiche_simulated", "DIRECTOR", None, 1),
    ("t-chernobyl", "p-harris_simulated", "LEAD_ACTOR", "Valery Legasov", 1),
    ("t-lastofus", "p-pascal_simulated", "LEAD_ACTOR", "Joel Miller", 1),
    ("t-theboys", "p-urban_simulated", "LEAD_ACTOR", "Billy Butcher", 1),
    
    ("t-mulan", "p-miyazaki", "SUPPORTING_ACTOR", "Advisor (Simulated)", 15),
    ("t-barbie", "p-robbie", "LEAD_ACTOR", "Barbie", 1),
    ("t-lalaland", "p-gosling", "LEAD_ACTOR", "Sebastian", 1),
    ("t-firstman", "p-gosling", "LEAD_ACTOR", "Neil Armstrong", 1),
    ("t-pulpfiction", "p-tarantino_simulated", "DIRECTOR", None, 1),
    ("t-django", "p-dicaprio", "LEAD_ACTOR", "Calvin Candie", 2),
    ("t-solaris", "p-tarkovsky_simulated", "DIRECTOR", None, 1),
    ("t-truedetective", "p-mcconaughey_simulated", "LEAD_ACTOR", "Rust Cohle", 1),
    
    ("t-avengersinf", "p-robbie", "SUPPORTING_ACTOR", "Cameo (Simulated)", 20),
    ("t-avengersend", "p-robbie", "SUPPORTING_ACTOR", "Cameo (Simulated)", 20),
    ("t-ironman", "p-lucas", "SUPPORTING_ACTOR", "Technician (Simulated)", 20),

    # ============================================================================
    # INDIAN TALENT WIRED TO TITLES
    # ============================================================================
    ("t-rrr", "p-rajamouli", "DIRECTOR", None, 1),
    ("t-rrr", "p-deepika", "SUPPORTING_ACTOR", "Freedom Fighter (Special App.)", 5),
    ("t-3idiots", "p-aamikhan", "LEAD_ACTOR", "Rancho / Phunsukh Wangdu", 1),
    ("t-drishyam", "p-nawaz", "SUPPORTING_ACTOR", "Investigating IG Officer", 3),
    ("t-sacredgames", "p-nawaz", "LEAD_ACTOR", "Ganesh Gaitonde (Gang Lord)", 1),
    ("t-swades", "p-srk", "LEAD_ACTOR", "Mohan Bhargava (NASA Scientist)", 1),
    ("t-kalki", "p-deepika", "LEAD_ACTOR", "Sumathi / Sum-80", 2),
    
    # Comedy talent connections
    ("t-herapheri", "p-akshay", "LEAD_ACTOR", "Raju", 1),
    ("t-herapheri", "p-paresh", "LEAD_ACTOR", "Baburao Ganpatrao Apte", 2),
    ("t-andazapna", "p-aamikhan", "LEAD_ACTOR", "Amar Manohar", 1),
    ("t-andazapna", "p-salman", "LEAD_ACTOR", "Prem Bhopali", 2),
    ("t-bajrangi", "p-salman", "LEAD_ACTOR", "Pawan Kumar Chaturvedi", 1),
    ("t-tiger", "p-salman", "LEAD_ACTOR", "Tiger / Avinash Singh Rathore", 1),
    ("t-dabangg", "p-salman", "LEAD_ACTOR", "Chulbul Pandey", 1)
]

# Title Genres Mapping (ContentId, GenreId)
TITLE_GENRES_DATA = [
    ("t-interstellar", "g-scifi"), ("t-interstellar", "g-drama"),
    ("t-inception", "g-scifi"), ("t-inception", "g-thriller"),
    ("t-tenet", "g-scifi"), ("t-tenet", "g-thriller"),
    ("t-oppenheimer", "g-drama"), ("t-oppenheimer", "g-history"),
    ("t-dunkirk", "g-action"), ("t-dunkirk", "g-history"),
    ("t-prestige", "g-drama"), ("t-prestige", "g-mystery"),
    ("t-memento", "g-thriller"), ("t-memento", "g-mystery"),
    ("t-dune1", "g-scifi"), ("t-dune1", "g-action"),
    ("t-dune2", "g-scifi"), ("t-dune2", "g-action"),
    ("t-bladerunner2049", "g-scifi"), ("t-bladerunner2049", "g-thriller"),
    ("t-arrival", "g-scifi"), ("t-arrival", "g-mystery"),
    ("t-sicario", "g-crime"), ("t-sicario", "g-thriller"),
    ("t-darkknight", "g-action"), ("t-darkknight", "g-crime"),
    ("t-batmanbegins", "g-action"), ("t-batmanbegins", "g-crime"),
    ("t-darkknightrises", "g-action"), ("t-darkknightrises", "g-crime"),
    ("t-thebatman", "g-crime"), ("t-thebatman", "g-mystery"),
    ("t-joker", "g-drama"), ("t-joker", "g-crime"),
    ("t-irishman", "g-crime"), ("t-irishman", "g-drama"),
    ("t-goodfellas", "g-crime"), ("t-goodfellas", "g-drama"),
    ("t-killersflower", "g-crime"), ("t-killersflower", "g-history"),
    ("t-godfather", "g-crime"), ("t-godfather", "g-drama"),
    ("t-spiritedaway", "g-anime"), ("t-spiritedaway", "g-fantasy"),
    ("t-princessmononoke", "g-anime"), ("t-princessmononoke", "g-fantasy"),
    ("t-spiderverse", "g-anime"), ("t-spiderverse", "g-action"),
    ("t-lotr1", "g-fantasy"), ("t-lotr1", "g-action"),
    ("t-lotr2", "g-fantasy"), ("t-lotr2", "g-action"),
    ("t-lotr3", "g-fantasy"), ("t-lotr3", "g-action"),
    ("t-starwars4", "g-scifi"), ("t-starwars4", "g-action"),
    ("t-starwars5", "g-scifi"), ("t-starwars5", "g-action"),
    ("t-starwars6", "g-scifi"), ("t-starwars6", "g-action"),
    ("t-matrix", "g-scifi"), ("t-matrix", "g-action"),
    ("t-matrixreload", "g-scifi"), ("t-matrixreload", "g-action"),
    ("t-breakingbad", "g-crime"), ("t-breakingbad", "g-drama"),
    ("t-bettercallsaul", "g-crime"), ("t-bettercallsaul", "g-drama"),
    ("t-succession", "g-drama"),
    ("t-severance", "g-thriller"), ("t-severance", "g-scifi"),
    ("t-blackmirror", "g-scifi"), ("t-blackmirror", "g-thriller"),
    ("t-strangerthings", "g-scifi"), ("t-strangerthings", "g-mystery"),
    ("t-cyberpunk", "g-anime"), ("t-cyberpunk", "g-scifi"),
    ("t-arcane", "g-anime"), ("t-arcane", "g-fantasy"),
    ("t-chernobyl", "g-drama"), ("t-chernobyl", "g-history"),
    ("t-lastofus", "g-action"), ("t-lastofus", "g-drama"),
    ("t-theboys", "g-action"), ("t-theboys", "g-scifi"),
    ("t-mulan", "g-anime"), ("t-mulan", "g-fantasy"),
    ("t-barbie", "g-fantasy"), ("t-barbie", "g-drama"),
    ("t-lalaland", "g-drama"),
    ("t-firstman", "g-history"), ("t-firstman", "g-drama"),
    ("t-pulpfiction", "g-crime"), ("t-pulpfiction", "g-thriller"),
    ("t-django", "g-action"), ("t-django", "g-drama"),
    ("t-solaris", "g-scifi"), ("t-solaris", "g-mystery"),
    ("t-truedetective", "g-crime"), ("t-truedetective", "g-mystery"),
    ("t-avengersinf", "g-action"), ("t-avengersinf", "g-scifi"),
    ("t-avengersend", "g-action"), ("t-avengersend", "g-scifi"),
    ("t-ironman", "g-action"), ("t-ironman", "g-scifi"),

    # ============================================================================
    # INDIAN GENRES WIRE (Seeding all requested genres beautifully!)
    # ============================================================================
    ("t-rrr", "g-action"), ("t-rrr", "g-history"),
    ("t-3idiots", "g-drama"),
    ("t-drishyam", "g-thriller"), ("t-drishyam", "g-crime"),
    ("t-sacredgames", "g-crime"), ("t-sacredgames", "g-thriller"), ("t-sacredgames", "g-drama"),
    ("t-swades", "g-drama"), ("t-swades", "g-history"),
    ("t-kalki", "g-scifi"), ("t-kalki", "g-fantasy"), ("t-kalki", "g-action"),
    ("t-ramayana", "g-anime"), ("t-ramayana", "g-fantasy"),
    
    # Comedy connections
    ("t-herapheri", "g-comedy"),
    ("t-andazapna", "g-comedy"),
    ("t-munnabhai", "g-comedy"),
    ("t-hangover", "g-comedy"),
    ("t-3idiots", "g-comedy"),
    ("t-bajrangi", "g-drama"),
    ("t-bajrangi", "g-comedy"),
    ("t-tiger", "g-action"),
    ("t-tiger", "g-thriller"),
    ("t-dabangg", "g-action"),
    ("t-dabangg", "g-comedy")
]

# Title Franchise Mapping (ContentId, FranchiseId, ChronologicalOrder)
TITLE_FRANCHISE_DATA = [
    # Star Wars
    ("t-starwars4", "f-starwars", 4),
    ("t-starwars5", "f-starwars", 5),
    ("t-starwars6", "f-starwars", 6),
    
    # Nolanverse Batman
    ("t-batmanbegins", "f-nolanverse", 1),
    ("t-darkknight", "f-nolanverse", 2),
    ("t-darkknightrises", "f-nolanverse", 3),
    
    # LOTR
    ("t-lotr1", "f-lotr", 1),
    ("t-lotr2", "f-lotr", 2),
    ("t-lotr3", "f-lotr", 3),
    
    # MCU
    ("t-ironman", "f-mcu", 1),
    ("t-avengersinf", "f-mcu", 19),
    ("t-avengersend", "f-mcu", 22),
    ("t-tiger", "f-spyuniverse", 1)
]

# Title Aliases / Synonyms / Typos (AliasId, ContentId, AliasText, AliasType)
TITLE_ALIASES_DATA = [
    ("a-st4", "t-strangerthings", "ST4", "ACRONYM"),
    ("a-st", "t-strangerthings", "ST", "ACRONYM"),
    ("a-interstelar", "t-interstellar", "Interstelar", "COMMON_TYPO"),
    ("a-incepton", "t-inception", "Incepton", "COMMON_TYPO"),
    ("a-dk", "t-darkknight", "DK", "ACRONYM"),
    ("a-tdk", "t-darkknight", "TDK", "ACRONYM"),
    ("a-mononoke-en", "t-princessmononoke", "Princess Mononoke", "REGIONAL_TITLE"),
    ("a-rrr-exp", "t-rrr", "Rise Roar Revolt", "REGIONAL_TITLE"),
    ("a-3id", "t-3idiots", "Three Idiots", "REGIONAL_TITLE"),
    ("a-hp", "t-herapheri", "Hera Pheri", "REGIONAL_TITLE"),
    ("a-aaa", "t-andazapna", "Andaz Apna Apna", "REGIONAL_TITLE"),
    ("a-gogo", "t-andazapna", "Crime Master Gogo", "REGIONAL_TITLE"),
    ("a-babu", "t-herapheri", "Babu Bhaiya", "REGIONAL_TITLE"),
    ("a-bb", "t-bajrangi", "Bhaijaan", "REGIONAL_TITLE"),
    ("a-chulbul", "t-dabangg", "Chulbul Pandey", "REGIONAL_TITLE"),
    ("a-ett", "t-tiger", "Ek Tha Tiger", "REGIONAL_TITLE")
]

# Watch Affinity co-watch matrix (SourceContentId, TargetContentId, AffinityScore, SharedAudienceCount)
WATCH_AFFINITY_DATA = [
    ("t-oppenheimer", "t-darkknight", 0.85, 24500),
    ("t-darkknight", "t-oppenheimer", 0.85, 24500),
    
    ("t-interstellar", "t-arrival", 0.92, 45000),
    ("t-arrival", "t-interstellar", 0.92, 45000),
    
    ("t-dune1", "t-dune2", 0.98, 98000),
    ("t-dune2", "t-dune1", 0.98, 98000),
    
    ("t-breakingbad", "t-bettercallsaul", 0.95, 87000),
    ("t-bettercallsaul", "t-breakingbad", 0.95, 87000),
    
    ("t-lotr1", "t-lotr2", 0.99, 120000),
    ("t-lotr2", "t-lotr3", 0.99, 122000),
    ("t-lotr3", "t-lotr1", 0.99, 118000),
    
    ("t-spiritedaway", "t-princessmononoke", 0.91, 35000),
    ("t-princessmononoke", "t-spiritedaway", 0.91, 35000),

    # Wire Indian affinities
    ("t-rrr", "t-kalki", 0.88, 54000),
    ("t-kalki", "t-rrr", 0.88, 54000),
    ("t-3idiots", "t-swades", 0.90, 42000),
    ("t-swades", "t-3idiots", 0.90, 42000),
    ("t-drishyam", "t-sacredgames", 0.86, 29000),
    ("t-sacredgames", "t-drishyam", 0.86, 29000)
]

# ============================================================================
# LANGUAGE FULL-TEXT TRANSLATIONS HELPERS
# ============================================================================
LANG_MAP = {
    'hi': 'Hindi',
    'ur': 'Urdu',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'ru': 'Russian',
    'te': 'Telugu',
    'ta': 'Tamil',
    'ml': 'Malayalam'
}

def get_language_names(codes):
    if not codes:
        return []
    return [LANG_MAP.get(c.lower(), c) for c in codes]

import os

SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE", "your-spanner-instance-id")
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE", "your-spanner-database-id")

# ============================================================================
# DATABASE SEEDING ROUTINE
# ============================================================================
def seed_database():
    spanner_client = spanner.Client()
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)
    
    print("Database connection opened. Starting complete re-seeding...")
    
    print("Inserting Genres...")
    genres_values = []
    for g in GENRES_DATA:
        genres_values.append([g[0], g[1], g[2], g[3]])
        
    with database.snapshot() as snapshot:
        pass # Force connection
        
    with database.batch() as batch:
        batch.insert_or_update(
            table='Genres',
            columns=['GenreId', 'Name', 'Slug', 'Description'],
            values=genres_values
        )
    print("Genres inserted.")
    
    print("Inserting Franchises...")
    with database.batch() as batch:
        batch.insert_or_update(
            table='Franchises',
            columns=['FranchiseId', 'Name', 'Description'],
            values=FRANCHISES_DATA
        )
    print("Franchises inserted.")
    
    print("Inserting People...")
    people_values = []
    for p in PEOPLE_DATA:
        text_to_embed = f"{p[1]} ({p[3]}). {p[4] if p[4] else ''}"
        emb = get_embedding(text_to_embed)
        people_values.append([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], emb])
        
    with database.batch() as batch:
        batch.insert_or_update(
            table='People',
            columns=['PersonId', 'FullName', 'KnownAs', 'PrimaryRole', 'Bio', 'BirthYear', 'ProfileImageUrl', 'PopularityScore', 'Embedding'],
            values=people_values
        )
    print("People inserted.")
    
    print("Inserting Titles...")
    titles_values = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for t in TITLES_DATA:
        text_to_embed = f"{t[1]} {t[8]} {t[9] if t[9] else ''}"
        emb = get_embedding(text_to_embed)
        audio_names = get_language_names(t[16])
        sub_names = get_language_names(t[17])
        titles_values.append([
            t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15], 
            t[16], t[17], audio_names, sub_names, t[18], emb, now, now
        ])
        
    with database.batch() as batch:
        batch.insert_or_update(
            table='Titles',
            columns=[
                'ContentId', 'PrimaryTitle', 'OriginalTitle', 'ContentType', 'ReleaseYear', 'AgeRating', 'DurationMins', 'SeasonsCount', 'Synopsis', 'Tagline', 'PosterUrl', 'BannerUrl', 'TrailerUrl', 'ImdbRating', 'PopularityScore', 'AccessTier', 
                'AudioLanguages', 'SubtitleLanguages', 'AudioLanguageNames', 'SubtitleLanguageNames', 'QualityProfiles', 'Embedding', 'CreatedAt', 'UpdatedAt'
            ],
            values=titles_values
        )
    print("Titles inserted.")
    
    print("Inserting TitleTalent...")
    with database.batch() as batch:
        batch.insert_or_update(
            table='TitleTalent',
            columns=['ContentId', 'PersonId', 'Role', 'CharacterName', 'BillingOrder'],
            values=TITLE_TALENT_DATA
        )
    print("TitleTalent inserted.")
    
    print("Inserting TitleGenres...")
    with database.batch() as batch:
        batch.insert_or_update(
            table='TitleGenres',
            columns=['ContentId', 'GenreId'],
            values=TITLE_GENRES_DATA
        )
    print("TitleGenres inserted.")
    
    print("Inserting TitleFranchise...")
    with database.batch() as batch:
        batch.insert_or_update(
            table='TitleFranchise',
            columns=['ContentId', 'FranchiseId', 'ChronologicalOrder'],
            values=TITLE_FRANCHISE_DATA
        )
    print("TitleFranchise inserted.")
    
    print("Inserting TitleAliases...")
    with database.batch() as batch:
        batch.insert_or_update(
            table='TitleAliases',
            columns=['AliasId', 'ContentId', 'AliasText', 'AliasType'],
            values=TITLE_ALIASES_DATA
        )
    print("TitleAliases inserted.")
    
    print("Inserting WatchAffinity...")
    with database.batch() as batch:
        batch.insert_or_update(
            table='WatchAffinity',
            columns=['SourceContentId', 'TargetContentId', 'AffinityScore', 'SharedAudienceCount'],
            values=WATCH_AFFINITY_DATA
        )
    print("WatchAffinity inserted.")
    
    print("Database seeding completed successfully! Seeded 55+ distinct, highly polished movie/show entities with relatable visual resources.")

if __name__ == '__main__':
    seed_database()
