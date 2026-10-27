# 🎮 SteamSense: žaidimas žaidėjų akimis

**Tema:** Sentimento analizė apie vaizdo žaidimus „Steam“ platformoje.
**Idėja:** kūrėjų aprašymas parduoda svajonę, o Steam apžvalgos parodo realybę. Sistema skaito žaidėjų apžvalgas ir paaiškina, **koks žaidimas yra žaidėjų akimis**: bendrą nuotaiką, nuotaiką pagal aspektus (gameplay, graphics, performance, bugs, price, multiplayer...), žanrą žaidėjų žodžiais, privalumus ir trūkumus, kam žaidimas tinka ir kuo žaidėjų vaizdas skiriasi nuo kūrėjų aprašymo.

**Komanda:** _Eimantas Bimba_, _Mindaugas Moisiejus_ · **Dalykas / grupė:** _Gilusis Mokymas naudojant TensorFlow DIPf-24_

| Dalis | Kur | Kas |
|---|---|---|
| 1.1 | [`prompts/00_design_prompt_chatgpt.md`](prompts/00_design_prompt_chatgpt.md), [`docs/`](docs) | ChatGPT promptas, PlantUML ir Mermaid diagramos |
| 1.2.1 | [`data/`](data), [`prompts/`](prompts) | 3 End-to-End pavyzdžiai (X, y) ir testavimo promptai |
| 1.2.2 | [`notebooks/02_gemini_steam_agent.ipynb`](notebooks/02_gemini_steam_agent.ipynb) | Gemini API agento prototipas Colab'e |
| 1.3 | [`notebooks/03_tabular_tf_tfdf.ipynb`](notebooks/03_tabular_tf_tfdf.ipynb) | TensorFlow ir TF Decision Forests su Kaggle Steam apžvalgomis |
| 1.4 | [`notebooks/04_cnn.ipynb`](notebooks/04_cnn.ipynb) | CNN: žanras iš viršelio |
| 1.5 | [`notebooks/05_transfer_learning.ipynb`](notebooks/05_transfer_learning.ipynb) | Transfer learning (MobileNetV2) |

### 1.2.2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mreimantas/Gilusis-mokymas-1-Nam-darbas-/blob/main/notebooks/02_gemini_steam_agent.ipynb)

### 1.3
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mreimantas/Gilusis-mokymas-1-Nam-darbas-/blob/main/notebooks/03_tabular_tf_tfdf.ipynb)

### 1.4
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mreimantas/Gilusis-mokymas-1-Nam-darbas-/blob/main/notebooks/04_cnn.ipynb)

### 1.5
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mreimantas/Gilusis-mokymas-1-Nam-darbas-/blob/main/notebooks/05_transfer_learning.ipynb)

## Architektūra (Mermaid)

### Panaudojimo atvejai
```mermaid
flowchart LR
  U([Vartotojas])
  S([Steam Web API])
  G([Gemini API])
  subgraph SteamSense
    UC1(Įvesti Steam app_id)
    UC2(Gauti žaidimo apžvalgas)
    UC3(Klasifikuoti nuotaiką ir aspektus)
    UC4(Sugeneruoti ataskaitą: žaidimas žaidėjų akimis)
    UC5(Palyginti su kūrėjų aprašymu)
    UC6(Vizualizuoti rezultatus)
  end
  U --> UC1 & UC4 & UC6
  UC1 -. include .-> UC2
  UC4 -. include .-> UC3
  UC5 -. extend .-> UC4
  UC2 --> S
  UC3 --> G
  UC4 --> G
```

### Komponentai
```mermaid
flowchart TB
  subgraph Sources[Duomenų šaltiniai]
    SR[Steam Reviews API]
    SD[Steam Store API appdetails]
    KG[Kaggle Steam Reviews]
  end
  subgraph App[SteamSense]
    ING[Duomenų surinkimas] --> PRE[Valymas ir paketavimas]
    PRE --> AG[Gemini agentas: aspektai ir nuotaika]
    AG --> AGG[Agregavimas pandas]
    AGG --> SYN[Gemini sintezė: GameReport JSON]
    SYN --> VIZ[Vizualizacija matplotlib]
    ML[Klasikiniai ML: TF MLP, TF-DF]
  end
  GEM{{Gemini API}}
  DATA[(data/ X,y)]
  SR --> ING
  SD --> ING
  KG --> ML
  AG <--> GEM
  SYN <--> GEM
  SYN --> DATA
  ML -. baseline palyginimas .-> AG
```

### Sekų diagrama (End-to-End srautas)
```mermaid
sequenceDiagram
  actor V as Vartotojas
  participant UI as Colab / UI
  participant ST as Steam API
  participant GM as Gemini API
  V->>UI: app_id
  UI->>ST: appdetails(app_id)
  ST-->>UI: pavadinimas, kūrėjų aprašymas
  UI->>ST: appreviews(app_id, cursor)
  ST-->>UI: N apžvalgų + voted_up
  UI->>UI: valymas, paketai po 25
  loop kiekvienam paketui
    UI->>GM: prompt + schema ReviewLabel[]
    GM-->>UI: JSON nuotaika + aspektai
  end
  UI->>UI: agregavimas pandas
  UI->>GM: statistika + top apžvalgos + kūrėjų aprašymas
  GM-->>UI: GameReport JSON
  UI->>V: ataskaita + grafikai
```

### Duomenų modelis
```mermaid
classDiagram
  class Game { +int app_id +str name +str developer_description }
  class Review { +int id +str text +bool voted_up +int votes_up +float playtime_h }
  class ReviewLabel { +int review_id +str sentiment }
  class AspectSentiment { +str aspect +str sentiment +int mentions +str evidence }
  class GameReport {
    +str overall_sentiment
    +float sentiment_score
    +str what_players_say_it_is
    +list genre_tags_by_players
    +list pros
    +list cons
    +list recommended_for
    +list not_recommended_for
    +str gap_vs_developer_description
  }
  Game "1" -- "*" Review
  Review "1" -- "1" ReviewLabel
  ReviewLabel "1" -- "*" AspectSentiment
  Game "1" -- "1" GameReport
  GameReport o-- AspectSentiment
```

PlantUML versijos: [`docs/plantuml/`](docs/plantuml) (atidaryti per https://www.plantuml.com/plantuml).

## Kaip paleisti
1. Gaukite raktą https://aistudio.google.com/apikey → Colab `🔑 Secrets` → `GEMINI_API_KEY`.
2. Atidarykite `notebooks/02_gemini_steam_agent.ipynb` Colab'e, pakeiskite `APP_ID` ir paleiskite visas ląsteles.
3. 03 reikia Kaggle prieigos (`kagglehub.login()`); 04 ir 05 reikia GPU (T4).

## Apribojimai
Ironija ir memai, daugiausia analizuojama anglų kalba, imtis nėra visos apžvalgos, LLM atsakymai nedeterministiniai (naudojame `temperature` 0,1-0,2 ir JSON schemą).
