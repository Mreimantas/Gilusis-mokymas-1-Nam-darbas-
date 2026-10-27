# %% [markdown]
# # 📊 Steam apžvalgų nuotaika: TensorFlow ir TensorFlow Decision Forests
# **Užduotis 1.3** - AdventureWorks DW pavyzdžių perdarymas su verslo duomenimis iš Kaggle.
#
# ## Ką daro AdventureWorks Colab'ai (ir ką darome mes)
# | Etapas | AdventureWorks | Mūsų Steam versija |
# |---|---|---|
# | Duomenys | DW lentelė su klientų / pardavimų atributais | Kaggle „Steam Reviews“: apžvalgos tekstas, `review_score` (+1 / -1), `review_votes`, žaidimas |
# | Tikslas (y) | pvz. ar klientas pirks / klasė | `label` = 1, jei žaidėjas **rekomenduoja** žaidimą |
# | Požymiai (X) | skaitiniai ir kategoriniai stulpeliai | ilgis, šauktukai, balsai, **aspektų raktažodžių skaičiai** + dažniausi žodžiai |
# | Modelis 1 | TensorFlow / Keras MLP | tas pats |
# | Modelis 2 | TF Decision Forests (Random Forest, GBT) | tas pats |
# | Vertinimas | accuracy, ROC, požymių svarba | tas pats + palyginimas su paprasta logistine regresija |
#
# **Kodėl du modelių tipai?** Neuroniniai tinklai reikalauja normalizuotų skaitinių požymių ir ilgiau mokosi, o sprendimų miškai tiesiogiai dirba su lentele, nereikalauja normalizavimo ir duoda **požymių svarbą**, kurią lengva interpretuoti verslui.

# %%
!pip -q install kagglehub tensorflow_decision_forests

# %%
import os, re, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf, tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED); sns.set_theme(style="whitegrid")
print("TF", tf.__version__, "| TF-DF", tfdf.__version__)

# %% [markdown]
# ## 1. Duomenų įkėlimas iš Kaggle
# Naudojame `andrewmvd/steam-reviews` (milijonai apžvalgų). Failas didelis, todėl skaitome **dalimis** ir iš kiekvienos dalies paimame 3 % atsitiktinę imtį (kitaip gautume tik pirmų žaidimų apžvalgas).
#
# Prieš paleidžiant: https://www.kaggle.com/settings → *Create New Token*, o Colab'e `kagglehub.login()` paprašys vartotojo vardo ir rakto. Jei duomenų rinkinio pavadinimas Kaggle pasikeitė, pakeiskite `DATASET`. Kitas variantas: `najzeko/steam-reviews-dataset`.

# %%
import kagglehub
DATASET = "andrewmvd/steam-reviews"
try:
    path = kagglehub.dataset_download(DATASET)
except Exception:
    kagglehub.login(); path = kagglehub.dataset_download(DATASET)
csv_path = glob.glob(os.path.join(path, "*.csv"))[0]
print(csv_path, round(os.path.getsize(csv_path) / 1e6), "MB")

parts = []
for chunk in pd.read_csv(csv_path, chunksize=200_000):
    parts.append(chunk.sample(frac=0.03, random_state=SEED))
raw = pd.concat(parts, ignore_index=True)
print(raw.shape); raw.head()

# %% [markdown]
# ## 2. Stulpelių suderinimas ir etiketė
# Rinkinys naudoja `review_score` = 1 / -1. Jei jūsų versijoje stulpeliai kitaip vadinasi, pakeiskite žemiau esantį žodyną.

# %%
RENAME = {"review_text": "text", "review_score": "score", "review_votes": "votes", "app_name": "game", "app_id": "app_id"}
df = raw.rename(columns={k: v for k, v in RENAME.items() if k in raw.columns})
df = df.dropna(subset=["text", "score"])
df["text"] = df["text"].astype(str)
df = df[df.text.str.len() > 3]
df["label"] = (df["score"] > 0).astype(int)
df["votes"] = pd.to_numeric(df.get("votes", 0), errors="coerce").fillna(0)
print(df.shape, "\nKlasių balansas:\n", df.label.value_counts(normalize=True).round(3))

# %% [markdown]
# ## 3. Žvalgomoji analizė (EDA)
# Svarbu pažiūrėti **klasių disbalansą** (Steam apžvalgos dažniausiai teigiamos) ir ar tekstų ilgis susijęs su nuotaika.

# %%
df["n_chars"] = df.text.str.len()
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
df.label.value_counts().sort_index().plot.bar(ax=ax[0], color=["#e53935", "#4caf50"]); ax[0].set_title("Klasės (0 = ne, 1 = taip)")
sns.boxplot(data=df[df.n_chars < 1500], x="label", y="n_chars", ax=ax[1], showfliers=False); ax[1].set_title("Teksto ilgis pagal klasę")
df.groupby("game").label.agg(["mean", "size"]).query("size > 200")["mean"].plot.hist(ax=ax[2], bins=15)
ax[2].set_title("Žaidimų rekomendacijų dalies pasiskirstymas")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Požymių inžinerija (feature engineering)
# Tekstą paverčiame **lentele**, kaip AdventureWorks DW:
# * **Stilistiniai**: simbolių / žodžių skaičius, šauktukai, klaustukai, didžiųjų raidžių dalis;
# * **Aspektų raktažodžiai** (susieja su Gemini projektu): kiek žodžių iš kiekvienos temos (gameplay, performance, bugs, price, story, multiplayer, monetization) yra apžvalgoje;
# * **Dažniausi žodžiai**: 300 dažniausių žodžių (dvejetainiai požymiai).
# Visi stulpelių pavadinimai gauna prefiksą (`w_`), nes TF-DF reikalauja tvarkingų pavadinimų.

# %%
ASPECT_WORDS = {
  "kw_gameplay": r"gameplay|mechanic|combat|puzzle|level|quest|fun|addict|boring|repetitive|grind",
  "kw_graphics": r"graphic|visual|art|beautiful|ugly|animation|texture",
  "kw_perf":     r"fps|lag|stutter|optimi[sz]|performance|frame|crash|freez",
  "kw_bugs":     r"bug|glitch|broken|fix|patch|issue",
  "kw_price":    r"price|worth|refund|money|cheap|expensive|sale|\$|euro|value",
  "kw_story":    r"story|plot|character|writing|narrative|ending|lore",
  "kw_multi":    r"multiplayer|co-?op|online|server|lobby|matchmaking|friends",
  "kw_monet":    r"dlc|microtransaction|pay.?to.?win|p2w|loot ?box|cash ?grab",
}
def build_features(d):
    t = d.text
    X = pd.DataFrame(index=d.index)
    X["n_chars"] = t.str.len(); X["n_words"] = t.str.split().str.len()
    X["n_excl"] = t.str.count("!"); X["n_quest"] = t.str.count(r"\?")
    X["caps_ratio"] = t.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    X["votes"] = d.votes
    low = t.str.lower()
    for k, pat in ASPECT_WORDS.items():
        X[k] = low.str.count(pat)
    return X

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df.label, random_state=SEED)

cv = CountVectorizer(max_features=300, stop_words="english", binary=True, min_df=20)
cv.fit(train_df.text)                      # žodynas mokomas TIK iš train, kad nebūtų nutekėjimo
def make_table(d):
    X = build_features(d)
    W = pd.DataFrame(cv.transform(d.text).toarray(), index=d.index,
                     columns=["w_" + re.sub(r"\W", "_", w) for w in cv.get_feature_names_out()])
    X = pd.concat([X, W], axis=1); X["label"] = d.label.values
    return X
tr, va, te = make_table(train_df), make_table(val_df), make_table(test_df)
FEATURES = [c for c in tr.columns if c != "label"]
print("Lentelė:", tr.shape, "| požymių:", len(FEATURES)); tr.iloc[:3, :14]

# %% [markdown]
# ## 5. Bazinis modelis: logistinė regresija
# Be baseline neįmanoma pasakyti, ar sudėtingi modeliai kažką duoda.

# %%
scaler = StandardScaler().fit(tr[FEATURES])
lr = LogisticRegression(max_iter=1000).fit(scaler.transform(tr[FEATURES]), tr.label)
p_lr = lr.predict_proba(scaler.transform(te[FEATURES]))[:, 1]
res = {"LogReg": (accuracy_score(te.label, p_lr > .5), roc_auc_score(te.label, p_lr))}
print("LogReg  acc=%.3f  AUC=%.3f" % res["LogReg"])

# %% [markdown]
# ## 6. TensorFlow / Keras MLP
# * Įėjimas: standartizuoti požymiai (neuroniniai tinklai jautrūs skalei).
# * `Dense(128) → Dropout → Dense(64) → Dropout → Dense(1, sigmoid)`: du paslėpti sluoksniai su ReLU, `Dropout` mažina permokymą, `sigmoid` duoda tikimybę.
# * `binary_crossentropy` - standartinė dviejų klasių klaida; `EarlyStopping` sustabdo mokymą, kai validacija nebegerėja.
# * `class_weight` kompensuoja disbalansą.

# %%
Xtr, Xva, Xte = [scaler.transform(d[FEATURES]).astype("float32") for d in (tr, va, te)]
cw_pos = (tr.label == 0).sum() / (tr.label == 1).sum()
class_weight = {0: 1.0, 1: float(cw_pos)}
mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURES),)),
    tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),  tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")])
mlp.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
mlp.summary()
hist = mlp.fit(Xtr, tr.label.values, validation_data=(Xva, va.label.values), epochs=40, batch_size=256,
               class_weight=class_weight, verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

# %%
h = pd.DataFrame(hist.history)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
h[["loss", "val_loss"]].plot(ax=ax[0], title="Klaida"); h[["auc", "val_auc"]].plot(ax=ax[1], title="AUC")
plt.show()
p_mlp = mlp.predict(Xte, verbose=0).ravel()
res["Keras MLP"] = (accuracy_score(te.label, p_mlp > .5), roc_auc_score(te.label, p_mlp))
print("MLP  acc=%.3f  AUC=%.3f" % res["Keras MLP"])

# %% [markdown]
# ## 7. TensorFlow Decision Forests
# * **Random Forest**: daug nepriklausomų medžių, galutinis balsavimas. Atsparus permokymui, mažai hiperparametrų.
# * **Gradient Boosted Trees**: medžiai mokosi paeiliui, kiekvienas taiso ankstesnių klaidas. Dažnai tiksliausias tabuliniams duomenims.
# * Nereikia normalizacijos ir nereikia perrašyti lentelės: `pd_dataframe_to_tf_dataset`.

# %%
to_ds = lambda d: tfdf.keras.pd_dataframe_to_tf_dataset(d, label="label")
train_ds, val_ds, test_ds = to_ds(tr), to_ds(va), to_ds(te)

rf = tfdf.keras.RandomForestModel(num_trees=300, verbose=0); rf.fit(train_ds)
gbt = tfdf.keras.GradientBoostedTreesModel(num_trees=300, verbose=0); gbt.fit(train_ds, validation_data=val_ds)
for name, m in [("TF-DF Random Forest", rf), ("TF-DF GBT", gbt)]:
    p = m.predict(test_ds, verbose=0).ravel()
    res[name] = (accuracy_score(te.label, p > .5), roc_auc_score(te.label, p))
    print(name, "acc=%.3f AUC=%.3f" % res[name])

# %% [markdown]
# ## 8. Modelių palyginimas ir ROC kreivės

# %%
cmp = pd.DataFrame(res, index=["accuracy", "ROC AUC"]).T.sort_values("ROC AUC", ascending=False)
display(cmp.round(3))
probs = {"LogReg": p_lr, "Keras MLP": p_mlp,
         "TF-DF RF": rf.predict(test_ds, verbose=0).ravel(), "TF-DF GBT": gbt.predict(test_ds, verbose=0).ravel()}
plt.figure(figsize=(6, 5))
for n, p in probs.items():
    fpr, tpr, _ = roc_curve(te.label, p); plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(te.label, p):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC"); plt.show()

best = probs["TF-DF GBT"] > .5
sns.heatmap(confusion_matrix(te.label, best), annot=True, fmt="d", cmap="Blues"); plt.title("GBT: painiavos matrica"); plt.show()
print(classification_report(te.label, best))

# %% [markdown]
# ## 9. Požymių svarba: kas lemia, ar žaidimas rekomenduojamas?
# Tai pagrindinė sprendimų miškų nauda verslui: matome, kurie požymiai (ir kurios temos) labiausiai skiria teigiamas apžvalgas nuo neigiamų.

# %%
vi = gbt.make_inspector().variable_importances()
key = "SUM_SCORE" if "SUM_SCORE" in vi else list(vi.keys())[0]
imp = pd.DataFrame([(f[0].name, s) for f, s in vi[key]], columns=["feature", "importance"]).head(20)
plt.figure(figsize=(7, 6)); sns.barplot(data=imp, y="feature", x="importance", color="#5c6bc0")
plt.title("TOP 20 požymių (GBT, %s)" % key); plt.show()

# %% [markdown]
# ## 10. Išvados
# * Palyginkite lentelę 8 skyriuje: ar sudėtingi modeliai pranoksta logistinę regresiją? Dažniausiai TF-DF GBT ≈ MLP > LogReg, o skirtumai nedideli, nes požymiai yra tik „maišas žodžių“ ir raktažodžiai.
# * Šie modeliai **nesupranta konteksto** („not good“ vs „good“), todėl LLM (Gemini projektas `02_...`) gali duoti prasmingesnį rezultatą, ypač aspektų lygyje.
# * Pagerinimai: TF-IDF / embeddings kaip požymiai, hiperparametrų paieška (`tfdf.tuner`), kryžminė validacija.
