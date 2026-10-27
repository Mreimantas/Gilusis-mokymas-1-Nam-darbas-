# Darbo ataskaitos šablonas (kas parašyta, ką aiškinti gynimo metu)

## 1.1 Tema ir diagramos
Tema: SteamSense. Naudotas promptas: `prompts/00_design_prompt_chatgpt.md` (du žingsniai: PlantUML, paskui Mermaid). Pateiktos 4 diagramos: panaudojimo atvejų, komponentų, sekų, klasių. **Ką pasakyti:** kodėl Mermaid (GitHub jį renderina tiesiogiai README), kuo skiriasi PlantUML sintaksė.

## 1.2 End-to-End sistema
* Duomenys: Steam Reviews API (apžvalgos + `voted_up`) ir Store API (kūrėjų aprašymas).
* DI dalis: Gemini su **JSON schema** (struktūrizuota išvestis), du etapai (žymėjimas paketais, paskui sintezė).
* Vertinimas: Gemini nuotaika vs. Steam balsas (accuracy, confusion matrix).
* **Ką pasakyti:** kodėl du etapai (pigiau ir patikimiau nei viena milžiniška užklausa), kodėl žemas `temperature`, kodėl paketai po 25.

## 1.3 Tabuliniai modeliai
Kaggle `andrewmvd/steam-reviews`. Požymiai: stilistiniai + aspektų raktažodžiai + 300 dažniausių žodžių. Modeliai: LogReg (baseline), Keras MLP, TF-DF RF ir GBT. **Ką pasakyti:** kodėl standartizuojame MLP, bet ne medžiams; ką rodo požymių svarba; kodėl žodynas mokomas tik iš train.

## 1.4 CNN
Duomenys surinkti patys (SteamSpy + Steam CDN), 4 žanrai. Aprašyti sluoksniai (Conv2D, MaxPool, GAP, Dropout, softmax), augmentacija, EarlyStopping. **Ką pasakyti:** kodėl train/val/test, kodėl permokymas matomas grafike.

## 1.5 Transfer learning
MobileNetV2 (ImageNet). Dvi fazės: feature extraction → fine-tuning (LR 1e-5). Variantas B: įterpiniai + LogReg + t-SNE. **Ką pasakyti:** kodėl `preprocess_input`, kodėl `training=False` bazei, kodėl perkompiliuojame prieš fine-tuning.

## Svarbu
Prieš įteikiant: paleiskite kiekvieną notebook'ą Colab'e, kad išliktų **išvestys** (grafikai, lentelės), ir įrašykite savo gautus skaičius į README.
