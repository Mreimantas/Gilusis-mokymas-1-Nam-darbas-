import json, pathlib
R = pathlib.Path(".")

def A(aspect, sentiment, mentions, evidence):
    return {"aspect": aspect, "sentiment": sentiment, "mentions": mentions, "evidence": evidence}

examples = [
{
 "x": {
  "app_name": "Star Forge Tactics (fiktyvus pavyzdys)",
  "developer_description": "Lead your fleet to glory in epic interstellar warfare.",
  "reviews": [
   (True,  "Best turn-based tactics I've played in years. Every mission feels like a puzzle and the squad customization is deep. Soundtrack is amazing too."),
   (True,  "Great gameplay but the campaign crashed twice on mission 7. Lost an hour of progress. Devs patched it quickly though."),
   (False, "Too hard and the tutorial explains nothing. Spent 2 hours just figuring out action points. Refunded."),
   (True,  "Reminds me of XCOM but with better ship building. 60 hours in and still finding new strategies."),
   (True,  "Graphics are okay, nothing special, but the AI opponents are smart and make you think. Worth the price on sale."),
   (False, "Story is forgettable and the voice acting is wooden. Gameplay loop gets repetitive after 20 hours."),
   (True,  "Co-op skirmish with a friend is a blast. Wish there were more maps."),
   (True,  "Runs smooth on my old laptop, no frame drops. Great value for 25 euros."),
  ]},
 "y": {
  "game": "Star Forge Tactics",
  "overall_sentiment": "mostly_positive", "sentiment_score": 0.55,
  "what_players_say_it_is": "A cerebral, puzzle-like sci-fi turn-based tactics game in the XCOM vein: smart AI, deep squad and ship customization and high replayability, but a steep learning curve, a weak story and some repetition after ~20 hours.",
  "genre_tags_by_players": ["turn-based tactics", "XCOM-like", "sci-fi", "ship building", "co-op skirmish"],
  "aspects": [
   A("gameplay", "positive", 6, "puzzle-like missions, smart AI, new strategies after 60 hours"),
   A("difficulty", "negative", 1, "tutorial explains nothing, action points confusing"),
   A("story", "negative", 1, "forgettable story, wooden voice acting"),
   A("audio", "mixed", 2, "amazing soundtrack vs wooden voice acting"),
   A("graphics", "mixed", 1, "okay, nothing special"),
   A("performance", "positive", 1, "smooth on an old laptop"),
   A("bugs", "negative", 1, "campaign crashed twice on mission 7, patched quickly"),
   A("price_value", "positive", 2, "worth it on sale, great value for 25 euros"),
   A("multiplayer", "positive", 1, "co-op skirmish is a blast"),
   A("content", "mixed", 2, "needs more maps; loop repetitive after 20 hours"),
   A("developer_support", "positive", 1, "crash patched quickly")],
  "pros": ["deep tactical gameplay", "smart AI opponents", "good soundtrack", "runs on weak hardware", "fair price"],
  "cons": ["poor tutorial / steep learning curve", "weak story and voice acting", "repetitive after ~20 hours", "few co-op maps", "occasional crashes"],
  "recommended_for": ["XCOM and turn-based tactics fans", "players who like optimizing builds", "co-op strategy duos"],
  "not_recommended_for": ["players who want a story-driven campaign", "people who dislike steep learning curves"],
  "gap_vs_developer_description": "Developers promise 'epic interstellar warfare'; players describe a slow, thoughtful squad-tactics puzzle game, not a spectacle of fleet battles."
 }
},
{
 "x": {
  "app_name": "Dungeon Hollow (fiktyvus pavyzdys)",
  "developer_description": "A thrilling dungeon crawler with endless replayability and a deep story.",
  "reviews": [
   (True,  "Addictive roguelike loop. One more run turned into 5 hours. Controls are tight and responsive."),
   (False, "Pixel art is gorgeous but the balance is broken. Some weapons are useless and RNG decides every run."),
   (True,  "Boss fights are hard but fair. Music slaps."),
   (False, "Game-breaking bug: save file corrupted after update 1.2. Devs not responding on forums."),
   (True,  "Perfect for short sessions on Steam Deck. Plays great on handheld."),
   (False, "Only 3 biomes, content runs out after 15 hours. Not worth 20 euros yet."),
   (True,  "Co-op mode is chaotic fun, even if it lags a bit with 4 players."),
   (False, "Unfair difficulty spikes, felt cheap not challenging."),
  ]},
 "y": {
  "game": "Dungeon Hollow",
  "overall_sentiment": "mixed", "sentiment_score": 0.0,
  "what_players_say_it_is": "A fast, addictive pixel-art roguelike great for short sessions and Steam Deck, with tight controls and fair boss fights, undermined by poor balance and RNG, little content (3 biomes) and a save-corruption bug.",
  "genre_tags_by_players": ["roguelike", "pixel art", "co-op", "Steam Deck friendly", "short sessions"],
  "aspects": [
   A("gameplay", "positive", 2, "addictive loop, tight controls"),
   A("controls", "positive", 1, "tight and responsive"),
   A("graphics", "positive", 1, "gorgeous pixel art"),
   A("audio", "positive", 1, "music slaps"),
   A("difficulty", "mixed", 3, "bosses hard but fair vs unfair spikes and RNG"),
   A("content", "negative", 1, "only 3 biomes, runs out after 15 hours"),
   A("bugs", "negative", 1, "save corrupted after update 1.2"),
   A("developer_support", "negative", 1, "devs not responding on forums"),
   A("multiplayer", "mixed", 1, "chaotic fun but lags with 4 players"),
   A("price_value", "negative", 1, "not worth 20 euros yet"),
   A("performance", "positive", 1, "plays great on Steam Deck")],
  "pros": ["addictive run loop", "tight controls", "pixel art and music", "great on Steam Deck", "co-op fun"],
  "cons": ["balance and RNG frustration", "little content", "save-corruption bug", "unresponsive developers", "lag in 4-player co-op"],
  "recommended_for": ["roguelike fans who play in short sessions", "Steam Deck owners"],
  "not_recommended_for": ["players who want lots of content now", "people sensitive to RNG-driven difficulty"],
  "gap_vs_developer_description": "Developers advertise 'endless replayability and a deep story'; players mention neither a story nor endless content, only three biomes and ~15 hours before it runs dry."
 }
},
{
 "x": {
  "app_name": "Turbo Drift Arena (fiktyvus pavyzdys)",
  "developer_description": "The ultimate multiplayer racing experience with hundreds of cars.",
  "reviews": [
   (False, "Stuttering on RTX 3060, 40 fps at low settings. Unoptimized garbage."),
   (False, "Every car and track locked behind microtransactions. Pay-to-win in a racing game?"),
   (True,  "Handling feels great with a wheel, drifting is satisfying. If they fix performance it will be a 9/10."),
   (False, "Servers are dead, long matchmaking, empty lobbies. Online mode is unplayable."),
   (False, "Looks like a mobile port. Menus are awful and controller support is broken."),
   (True,  "Cool cars and nice graphics, fun for the price of a coffee on sale."),
   (False, "Crashes on startup after the latest patch. Devs abandoned it."),
   (False, "Refunded. Career mode is a grind designed to push you to the shop."),
  ]},
 "y": {
  "game": "Turbo Drift Arena",
  "overall_sentiment": "mostly_negative", "sentiment_score": -0.6,
  "what_players_say_it_is": "A drift racing game with satisfying handling (especially on wheel) and attractive cars, but players describe a poorly optimized, mobile-style port with microtransaction-gated content, dead online lobbies, crashes and broken controller support.",
  "genre_tags_by_players": ["arcade racing", "drifting", "free-to-play-style monetization", "mobile port"],
  "aspects": [
   A("gameplay", "positive", 1, "satisfying drifting and handling with a wheel"),
   A("graphics", "mixed", 2, "nice graphics vs 'looks like a mobile port'"),
   A("performance", "negative", 1, "40 fps on RTX 3060 at low settings"),
   A("bugs", "negative", 1, "crash on startup after latest patch"),
   A("monetization", "negative", 2, "cars and tracks locked behind microtransactions, grindy career"),
   A("multiplayer", "negative", 1, "empty lobbies, long matchmaking"),
   A("controls", "negative", 1, "controller support broken"),
   A("developer_support", "negative", 1, "devs abandoned it"),
   A("price_value", "mixed", 2, "fun for a coffee price on sale vs pay-to-win")],
  "pros": ["satisfying drift handling", "good-looking cars", "cheap on sale"],
  "cons": ["poor optimization", "microtransactions gate content", "dead multiplayer", "crashes", "broken controller support", "abandoned by developers"],
  "recommended_for": ["wheel owners willing to risk a purchase on deep sale"],
  "not_recommended_for": ["players who want online racing", "controller players", "anyone averse to microtransactions"],
  "gap_vs_developer_description": "Developers promise 'ultimate multiplayer' and 'hundreds of cars'; players report empty multiplayer lobbies and most cars locked behind paywalls."
 }
},
]

SYSTEM = (R/"prompts"/"system_prompt.md")
SYSTEM.write_text("""# SteamSense - sistemos promptas (GPT / Gemini agentui)

You are **SteamSense**, an analyst that explains what a video game is REALLY like from the **players' perspective**, not the developers'.

You receive: the game's name, the developer's marketing description, and a list of Steam reviews
(each marked RECOMMENDED or NOT RECOMMENDED).

Rules
1. Use ONLY the reviews. Do not use your own knowledge of the game.
2. Aspects allowed: gameplay, graphics, audio, story, performance, bugs, price_value, multiplayer,
   content, difficulty, controls, developer_support, monetization.
3. Sentiment per aspect: positive | mixed | negative. `mentions` = number of reviews mentioning it.
4. `sentiment_score` is in [-1, 1]. `overall_sentiment` is one of:
   very_negative | mostly_negative | mixed | mostly_positive | very_positive.
5. `what_players_say_it_is`: 1-2 sentences in the players' own terms.
6. `gap_vs_developer_description`: how the players' picture differs from the developer's text.
7. Answer with VALID JSON ONLY, no markdown fences, using exactly this schema:

{
  "game": str, "overall_sentiment": str, "sentiment_score": float,
  "what_players_say_it_is": str, "genre_tags_by_players": [str],
  "aspects": [{"aspect": str, "sentiment": str, "mentions": int, "evidence": str}],
  "pros": [str], "cons": [str], "recommended_for": [str], "not_recommended_for": [str],
  "gap_vs_developer_description": str
}
""", encoding="utf-8")

for i, ex in enumerate(examples, 1):
    x = {"app_name": ex["x"]["app_name"], "developer_description": ex["x"]["developer_description"],
         "reviews": [{"id": k+1, "voted_up": v, "text": t} for k, (v, t) in enumerate(ex["x"]["reviews"])]}
    (R/"data"/f"example_{i}_x.json").write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")
    (R/"data"/f"example_{i}_y.json").write_text(json.dumps(ex["y"], ensure_ascii=False, indent=2), encoding="utf-8")
    lines = "\n".join(f"{r['id']}. [{'RECOMMENDED' if r['voted_up'] else 'NOT RECOMMENDED'}] {r['text']}" for r in x["reviews"])
    body = SYSTEM.read_text(encoding="utf-8").split("\n", 2)[2]
    (R/"prompts"/f"test_prompt_{i}.md").write_text(
f"""# Testavimo promptas {i}: {ex['y']['game']}

Kaip naudoti: nukopijuokite viską žemiau esantį bloką (tarp linijų) į ChatGPT / GPT agentą.
Laukiamas atsakymas (y): [`data/example_{i}_y.json`](../data/example_{i}_y.json).
Įvestis (X): [`data/example_{i}_x.json`](../data/example_{i}_x.json).

---
{body.strip()}

GAME: {x['app_name']}
DEVELOPER DESCRIPTION: {x['developer_description']}

REVIEWS:
{lines}
---
""", encoding="utf-8")

(R/"prompts"/"00_design_prompt_chatgpt.md").write_text("""# 1.1 Promptas ChatGPT sistemos projektavimui (PlantUML, po to Mermaid)

## Žingsnis 1 - PlantUML
```
You are a senior software architect. Design an information system called "SteamSense" that uses generative AI.

Goal: read Steam reviews of a video game and explain what the game is actually about and how good it is
FROM THE PLAYERS' PERSPECTIVE (not the developers'): overall sentiment, sentiment per aspect (gameplay, graphics,
story, performance, bugs, price/value, multiplayer, content, difficulty, controls, monetization), what players
call the genre, pros/cons, who it is for, and how this differs from the developer's marketing description.

Inputs: Steam app_id (Steam Reviews API + Steam Store appdetails). Generative AI: Gemini API with JSON schema outputs.
Also include a classical-ML baseline (Keras MLP and TensorFlow Decision Forests) trained on a Kaggle Steam reviews dataset.

Produce PlantUML code (each in its own ```plantuml block, valid and renderable):
1. Use case diagram
2. Component diagram
3. Sequence diagram of the end-to-end flow (including a loop over review batches sent to the LLM)
4. Class diagram of the data model (Game, Review, ReviewLabel, AspectSentiment, GameReport)
After each diagram give 2-3 sentences of explanation. Write labels in Lithuanian.
```

## Žingsnis 2 - Mermaid
```
Convert each of the four PlantUML diagrams above to Mermaid syntax that renders on GitHub
(use flowchart for use case and component diagrams, sequenceDiagram, classDiagram).
Keep the same meaning and labels. Output four separate ```mermaid blocks.
```
Rezultatas: [`docs/plantuml/`](../docs/plantuml) ir [`docs/mermaid/`](../docs/mermaid).
""", encoding="utf-8")
print("ok")
