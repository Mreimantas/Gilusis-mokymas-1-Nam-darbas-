# %% [markdown]
# # 🎮 SteamSense: žaidimas žaidėjų akimis
# **Gemini API prototipas (užduotis 1.2.2)**
#
# ## Sistemos apžvalga
# Kūrėjų aprašymas parduoda svajonę, o žaidėjų apžvalgos parodo realybę. **SteamSense** agentas:
#
# 1. gauna Steam `app_id`;
# 2. atsisiunčia žaidimo aprašymą (kūrėjų pusė) ir realias žaidėjų apžvalgas (Steam viešasis API, raktas nereikalingas);
# 3. **Gemini** kiekvienai apžvalgai priskiria nuotaiką ir paminėtus aspektus (gameplay, graphics, performance, ...) - struktūrizuotas JSON;
# 4. `pandas` agreguoja rezultatus;
# 5. **Gemini** sintezuoja galutinę ataskaitą: *ką žaidėjai sako, kas tai per žaidimas*, privalumai/trūkumai, kam tinka, kuo žaidėjų vaizdas skiriasi nuo kūrėjų aprašymo;
# 6. vizualizuojame ir **įvertiname** agentą (Gemini nuotaika vs. Steam „Recommended / Not recommended“ balsas).
#
# ```
# app_id → Steam API → valymas → Gemini (paketais) → pandas → Gemini sintezė → grafikai + JSON
# ```
# **Paruošimas:** Colab kairėje `🔑 Secrets` pridėkite `GEMINI_API_KEY` (raktą gaunate https://aistudio.google.com/apikey) ir įjunkite „Notebook access“.

# %% [markdown]
# ## 1. Aplinka ir Gemini klientas
# `google-genai` yra oficialus Gemini SDK. `pydantic` naudojame aprašyti **JSON schemą**, kurią Gemini privalo atitikti (`response_schema`). Taip negauname laisvo teksto, kurį sunku apdoroti.

# %%
!pip -q install google-genai pydantic pandas matplotlib seaborn requests

# %%
import os, json, time, textwrap, collections
from typing import Literal, List
import requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

try:
    from google.colab import userdata
    API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    import getpass
    API_KEY = os.environ.get("GEMINI_API_KEY") or getpass.getpass("Gemini API key: ")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"   # greitas ir pigus; sudėtingesnei sintezei galima rinktis gemini-2.5-pro
sns.set_theme(style="whitegrid")
print("Klientas paruoštas, modelis:", MODEL)

# %% [markdown]
# ## 2. Duomenų modelis (JSON schemos)
# * `ReviewLabel` - vienos apžvalgos žymėjimas (1 žingsnis, daug kartų).
# * `GameReport` - galutinė ataskaita apie žaidimą (2 žingsnis, vieną kartą).
#
# Tai atitinka klasių diagramą iš `docs/`.

# %%
ASPECTS = ["gameplay", "graphics", "audio", "story", "performance", "bugs", "price_value",
           "multiplayer", "content", "difficulty", "controls", "developer_support", "monetization"]
Aspect = Literal["gameplay", "graphics", "audio", "story", "performance", "bugs", "price_value",
                 "multiplayer", "content", "difficulty", "controls", "developer_support", "monetization"]
Sent3 = Literal["positive", "mixed", "negative"]

class AspectMention(BaseModel):
    aspect: Aspect
    sentiment: Sent3

class ReviewLabel(BaseModel):
    id: int
    sentiment: Sent3
    aspects: List[AspectMention]

class AspectSummary(BaseModel):
    aspect: Aspect
    sentiment: Sent3
    mentions: int
    evidence: str = Field(description="trumpas pagrindimas iš apžvalgų")

class GameReport(BaseModel):
    overall_sentiment: Literal["very_negative", "mostly_negative", "mixed", "mostly_positive", "very_positive"]
    sentiment_score: float = Field(description="nuo -1 iki 1")
    what_players_say_it_is: str
    genre_tags_by_players: List[str]
    aspects: List[AspectSummary]
    pros: List[str]
    cons: List[str]
    recommended_for: List[str]
    not_recommended_for: List[str]
    gap_vs_developer_description: str

# %% [markdown]
# ## 3. Duomenų gavimas iš Steam
# * `appdetails` - žaidimo pavadinimas ir **kūrėjų** aprašymas.
# * `appreviews` - žaidėjų apžvalgos (`voted_up` = žaidėjas rekomenduoja ar ne; tai bus mūsų „tikroji“ etiketė vertinimui).
#
# Nustatykite `APP_ID`. Pavyzdžiai: `570` Dota 2, `1245620` Elden Ring, `1091500` Cyberpunk 2077, `413150` Stardew Valley.

# %%
APP_ID = 1091500      # <-- keiskite
N_REVIEWS = 150       # kiek apžvalgų analizuoti (didesnis skaičius = daugiau API užklausų)
LANG = "english"

def fetch_game_info(appid):
    r = requests.get("https://store.steampowered.com/api/appdetails", params={"appids": appid, "l": "english"}, timeout=20)
    d = r.json()[str(appid)]["data"]
    return {"name": d["name"], "developer_description": d.get("short_description", "")}

def fetch_reviews(appid, n=150, language="english"):
    out, cursor = [], "*"
    while len(out) < n:
        r = requests.get(f"https://store.steampowered.com/appreviews/{appid}",
                         params=dict(json=1, filter="all", language=language, review_type="all",
                                     purchase_type="all", num_per_page=100, cursor=cursor), timeout=20)
        j = r.json()
        if not j.get("reviews"):
            break
        for rv in j["reviews"]:
            out.append({"voted_up": rv["voted_up"], "text": rv["review"],
                        "votes_up": rv["votes_up"],
                        "playtime_h": rv["author"].get("playtime_forever", 0) / 60})
        cursor = j["cursor"]
        time.sleep(0.5)
    return out[:n]

try:
    game = fetch_game_info(APP_ID)
    raw = fetch_reviews(APP_ID, N_REVIEWS, LANG)
    SOURCE = "Steam API (gyvi duomenys)"
except Exception as e:
    print("Steam API nepasiekiamas:", e, "→ naudojami pavyzdiniai duomenys iš data/example_1_x.json")
    ex = json.load(open("data/example_1_x.json"))
    game = {"name": ex["app_name"], "developer_description": ex["developer_description"]}
    raw = [{"voted_up": r["voted_up"], "text": r["text"], "votes_up": 0, "playtime_h": 0} for r in ex["reviews"]]
    SOURCE = "data/example_1_x.json"

print(SOURCE, "|", game["name"], "|", len(raw), "apžvalgų")
print("Kūrėjų aprašymas:", game["developer_description"])

# %% [markdown]
# ## 4. Valymas
# Pašaliname labai trumpas apžvalgas („10/10“, „gg“), nes jose nėra aspektų, išpjauname per ilgas ir sunumeruojame, kad vėliau galėtume susieti Gemini atsakymą su originalia apžvalga.

# %%
df = pd.DataFrame(raw)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df["text"].str.split().str.len() >= 5].reset_index(drop=True)
df["text_short"] = df["text"].str.slice(0, 700)
df["id"] = df.index + 1
print(len(df), "apžvalgų po valymo; Steam rekomenduoja:", round(df.voted_up.mean() * 100, 1), "%")
df.head()

# %% [markdown]
# ## 5. 1 agento žingsnis: nuotaika ir aspektai kiekvienai apžvalgai
# Siunčiame apžvalgas **paketais po 25** (mažiau užklausų, pigiau). `temperature=0.1` - norime stabilių, o ne kūrybingų atsakymų. `response_schema=List[ReviewLabel]` priverčia Gemini grąžinti galiojantį JSON.

# %%
LABEL_PROMPT = """You label Steam game reviews. For each review return:
- id (same as given)
- sentiment of the review toward the game: positive | mixed | negative
- aspects the reviewer actually discusses, each with its own sentiment.
Allowed aspects: {aspects}. Do not invent aspects that are not mentioned.

REVIEWS:
{reviews}"""

def label_batch(batch: pd.DataFrame, retries=3):
    txt = "\n".join(f"[{r.id}] {r.text_short}" for r in batch.itertuples())
    prompt = LABEL_PROMPT.format(aspects=", ".join(ASPECTS), reviews=txt)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   response_schema=List[ReviewLabel], temperature=0.1))
            return [x.model_dump() for x in resp.parsed]
        except Exception as e:
            print("Klaida, kartojame:", str(e)[:120]); time.sleep(2 * (attempt + 1))
    return []

labels = []
for i in range(0, len(df), 25):
    labels += label_batch(df.iloc[i:i + 25])
    print(f"{min(i + 25, len(df))}/{len(df)}", end="  ")

lab_df = pd.DataFrame(labels).rename(columns={"sentiment": "gemini_sentiment"})
df = df.merge(lab_df, on="id", how="left")
print("\nPažymėta:", df.gemini_sentiment.notna().sum())
df[["id", "voted_up", "gemini_sentiment", "aspects"]].head()

# %% [markdown]
# ## 6. Agregavimas (pandas)
# Išskleidžiame aspektus į lentelę: kiek kartų kiekvienas aspektas paminėtas ir kokia nuotaika.

# %%
rows = []
for r in df.dropna(subset=["aspects"]).itertuples():
    for a in r.aspects:
        rows.append({"id": r.id, "aspect": a["aspect"], "sentiment": a["sentiment"]})
asp = pd.DataFrame(rows)
asp_tab = (asp.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
              .reindex(columns=["positive", "mixed", "negative"], fill_value=0))
asp_tab["total"] = asp_tab.sum(axis=1)
asp_tab = asp_tab.sort_values("total", ascending=False)
asp_tab

# %% [markdown]
# ## 7. 2 agento žingsnis: galutinė ataskaita
# Gemini gauna: agreguotą statistiką, **naudingiausias** apžvalgas (daugiausia `votes_up`), ilgiausiai žaidusių nuomones ir kūrėjų aprašymą. Prašome atsakyti tik iš apžvalgų ir atskirai nurodyti, kuo žaidėjų vaizdas skiriasi nuo kūrėjų.

# %%
top = pd.concat([df.sort_values("votes_up", ascending=False).head(12),
                 df.sort_values("playtime_h", ascending=False).head(8)]).drop_duplicates("id")
top_txt = "\n".join(f"- [{'REC' if r.voted_up else 'NOT REC'}, {r.playtime_h:.0f}h] {r.text_short[:400]}" for r in top.itertuples())

SYN_PROMPT = f"""You are SteamSense. Explain what the game "{game['name']}" is REALLY like from the PLAYERS' perspective.
Use ONLY the data below, not your own knowledge of the game.

DEVELOPER DESCRIPTION: {game['developer_description']}

AGGREGATED ASPECT STATS (counts of reviews):
{asp_tab.to_string()}

SHARE OF RECOMMENDED REVIEWS: {df.voted_up.mean():.0%}

MOST HELPFUL / LONGEST-PLAYTIME REVIEWS:
{top_txt}

Fill the schema. `gap_vs_developer_description` must contrast the players' picture with the developer text."""

resp = client.models.generate_content(
    model=MODEL, contents=SYN_PROMPT,
    config=types.GenerateContentConfig(response_mime_type="application/json",
                                       response_schema=GameReport, temperature=0.2))
report = resp.parsed
print(f"🎮 {game['name']}  ({SOURCE})")
print("Bendras vertinimas:", report.overall_sentiment, f"({report.sentiment_score:+.2f})")
print("\nKą sako žaidėjai:\n" + textwrap.fill(report.what_players_say_it_is, 100))
print("\nŽanro žymės:", ", ".join(report.genre_tags_by_players))
print("\n✅ Privalumai:", *report.pros, sep="\n  - ")
print("\n❌ Trūkumai:", *report.cons, sep="\n  - ")
print("\nTinka:", "; ".join(report.recommended_for))
print("Netinka:", "; ".join(report.not_recommended_for))
print("\n↔ Kūrėjai vs žaidėjai:\n" + textwrap.fill(report.gap_vs_developer_description, 100))

# %% [markdown]
# ## 8. Vizualizacija
# 1. Aspektų nuotaika (sukrauta juosta) - kas žaidėjams svarbiausia ir kas džiugina / piktina.
# 2. Bendra nuotaika: Gemini vs Steam balsai.
# 3. Aspektų pasiskirstymas pagal žaidimo trukmę (ar naujokai ir veteranai skundžiasi skirtingai?).

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
colors = {"positive": "#4caf50", "mixed": "#ffc107", "negative": "#e53935"}
asp_tab.head(10)[["positive", "mixed", "negative"]].plot.barh(stacked=True, ax=ax[0],
        color=[colors[c] for c in ["positive", "mixed", "negative"]])
ax[0].invert_yaxis(); ax[0].set_title("Aspektų nuotaika (kiek apžvalgų)"); ax[0].set_xlabel("paminėjimai")

cmp = pd.DataFrame({
    "Gemini": df.gemini_sentiment.value_counts(normalize=True).reindex(["positive", "mixed", "negative"], fill_value=0),
    "Steam balsas": pd.Series({"positive": df.voted_up.mean(), "mixed": 0, "negative": 1 - df.voted_up.mean()})})
cmp.plot.bar(ax=ax[1], color=["#5c6bc0", "#90a4ae"]); ax[1].set_title("Nuotaikos dalis: Gemini vs Steam")
ax[1].set_ylabel("dalis"); ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()

# %%
df["playtime_bucket"] = pd.cut(df.playtime_h, [-1, 5, 30, 100, 1e6], labels=["<5h", "5-30h", "30-100h", ">100h"])
if df.playtime_h.sum() > 0:
    pt = df.groupby("playtime_bucket", observed=True).voted_up.mean().mul(100)
    pt.plot.bar(color="#5c6bc0", figsize=(6, 3.5), title="% rekomenduojančių pagal žaidimo laiką")
    plt.ylabel("%"); plt.xticks(rotation=0); plt.show()
else:
    print("Pavyzdiniuose duomenyse nėra žaidimo trukmės.")

# %% [markdown]
# ## 9. Agento įvertinimas
# Steam `voted_up` yra nemokamos „tikrosios“ etiketės. Palyginame jas su Gemini nuotaika (`mixed` laikome neutralia ir ją praleidžiame). Tai nėra tobulas matas (žmogus gali parašyti kritišką tekstą ir vis tiek rekomenduoti), bet gerai parodo, ar agentas supranta tekstą.

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
ev = df[df.gemini_sentiment.isin(["positive", "negative"])]
y_true = ev.voted_up.map({True: "positive", False: "negative"}); y_pred = ev.gemini_sentiment
print("Tikslumas (be mixed):", round(accuracy_score(y_true, y_pred), 3), "| nenaudota 'mixed':", len(df) - len(ev))
print(classification_report(y_true, y_pred, zero_division=0))
cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Gemini +", "Gemini -"], yticklabels=["Steam +", "Steam -"])
plt.title("Steam balsas vs Gemini"); plt.show()

# %% [markdown]
# ## 10. Rezultatų išsaugojimas
# Išsaugome įvestį (X) ir išvestį (y) į `data/`. Taip gaunate papildomų pavyzdžių GitHub saugyklai (reikalavimas 1.2.1).

# %%
os.makedirs("data", exist_ok=True)
x_out = {"app_name": game["name"], "developer_description": game["developer_description"],
         "reviews": [{"id": int(r.id), "voted_up": bool(r.voted_up), "text": r.text_short} for r in df.head(10).itertuples()]}
y_out = {"game": game["name"], **report.model_dump()}
json.dump(x_out, open(f"data/real_{APP_ID}_x.json", "w"), ensure_ascii=False, indent=2)
json.dump(y_out, open(f"data/real_{APP_ID}_y.json", "w"), ensure_ascii=False, indent=2)
print("Išsaugota: data/real_%s_x.json ir data/real_%s_y.json" % (APP_ID, APP_ID))

# %% [markdown]
# ## 11. Išvados ir apribojimai
# * Gemini su JSON schema patikimai paverčia nestruktūrizuotą tekstą lentele, o antras žingsnis iš lentelės padaro suprantamą ataskaitą.
# * Apribojimai: ironija ir memai („10/10 would crash again“); tik anglų kalba pagal numatytuosius nustatymus (`LANG` galima keisti); analizuojama imtis, o ne visos apžvalgos; modelis gali neatpažinti retų aspektų.
# * Tolesni žingsniai: kelių žaidimų palyginimas, laiko eilutės (kaip nuotaika keitėsi po patch'ų), RAG su visomis apžvalgomis.
