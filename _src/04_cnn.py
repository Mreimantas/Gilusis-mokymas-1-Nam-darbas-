# %% [markdown]
# # 🧠 CNN: Steam žaidimo žanras iš viršelio paveikslėlio
# **Užduotis 1.4** - konvoliucinio tinklo (CNN) Colab'o perdarymas su savo duomenimis.
#
# ## Idėja
# Mūsų projektas atsako į klausimą „koks tai žaidimas?“. Tekstą analizuoja Gemini, o čia tą patį klausimą sprendžiame **iš vaizdo**: ar CNN pagal Steam parduotuvės viršelį (`header.jpg`) atpažins žanrą (Racing / Strategy / Sports / RPG)?
#
# ## Kaip veikia CNN (tai, ką reikia paaiškinti gynimo metu)
# | Sluoksnis | Ką daro |
# |---|---|
# | `Rescaling(1/255)` | pikselių reikšmes iš 0-255 sumažina į 0-1, kad mokymas būtų stabilesnis |
# | `RandomFlip / RandomZoom / RandomContrast` | **duomenų augmentacija**: mokymo metu kiekvieną kartą šiek tiek pakeičia paveikslėlį, taip tinklas mažiau įsimena konkrečius vaizdus |
# | `Conv2D(filtrai, 3x3)` | slenka mažu langu (filtru) per vaizdą ir ieško raštų: kraštų, spalvų, vėliau formų (automobilis, kardas, kamuolys) |
# | `MaxPooling2D` | iš kiekvieno 2x2 langelio palieka didžiausią reikšmę: mažina dydį, daro atpažinimą atsparesnį poslinkiams |
# | `GlobalAveragePooling2D` | kiekvieną požymių žemėlapį suvidurkina iki vieno skaičiaus (mažiau parametrų nei `Flatten`) |
# | `Dense + Dropout` | klasifikatorius; `Dropout` atsitiktinai išjungia dalį neuronų, kovoja su permokymu |
# | `Dense(softmax)` | tikimybė kiekvienam žanrui, suma = 1 |
#
# **Prieš paleidžiant:** `Runtime → Change runtime type → T4 GPU`.

# %% [markdown]
# ## 1. Duomenų rinkinio sukūrimas
# Duomenis renkame patys iš viešų šaltinių: **SteamSpy API** pateikia žaidimus pagal žanrą, o paveikslėlius siunčiame iš Steam CDN. Pasirenkame populiariausius žaidimus (jų viršeliai tvarkingi) ir išmetame tuos, kurie priklauso keliems pasirinktiems žanrams. Taip klasės neprieštarauja viena kitai.

# %%
#@@INCLUDE _dataset_cell.txt

# %% [markdown]
# ## 2. Skaidymas ir `tf.data` vamzdynas
# 70 % mokymui, 15 % validacijai (hiperparametrams ir ankstyvam stabdymui), 15 % galutiniam testui, kurio mokymo metu **nematome**. `stratify` išlaiko vienodą klasių proporciją visose dalyse.

# %%
#@@INCLUDE _split_cell.txt

# %% [markdown]
# ## 3. Modelis
# Trys konvoliuciniai blokai (32 → 64 → 128 filtrų): ankstyvi sluoksniai mato paprastus raštus, gilesni sudėtingesnius.

# %%
aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomZoom(0.1),
                           tf.keras.layers.RandomContrast(0.1)], name="augmentation")
def conv_block(f):
    return [tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D()]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMG_SIZE + (3,)),
    aug, tf.keras.layers.Rescaling(1 / 255),
    *conv_block(32), *conv_block(64), *conv_block(128),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(GENRES), activation="softmax")])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %% [markdown]
# ## 4. Mokymas
# `EarlyStopping` sustabdo, kai validacijos klaida nustoja mažėti, o `ReduceLROnPlateau` sumažina mokymosi greitį, kai pasiekiama „plynaukštė“.

# %%
cbs = [tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
       tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
hist = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=cbs)

# %%
h = pd.DataFrame(hist.history)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
h[["loss", "val_loss"]].plot(ax=ax[0], title="Klaida"); h[["accuracy", "val_accuracy"]].plot(ax=ax[1], title="Tikslumas")
plt.show()

# %% [markdown]
# ## 5. Vertinimas testo aibėje
# Painiavos matrica parodo, kuriuos žanrus tinklas painioja. Tikėtina, kad Racing ir Sports atskiria geriau (aiškūs vaizdiniai motyvai), o Strategy ir RPG painioja (panaši fantazijos estetika).

# %%
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Testo tikslumas: {test_acc:.3f} (atsitiktinis spėjimas: {1/len(GENRES):.2f})")
y_pred = np.concatenate([model.predict(x, verbose=0).argmax(1) for x, _ in test_ds])
y_true = np.concatenate([y.numpy() for _, y in test_ds])
print(classification_report(y_true, y_pred, target_names=GENRES))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=GENRES, yticklabels=GENRES)
plt.xlabel("Prognozė"); plt.ylabel("Tikra"); plt.show()
CNN_TEST_ACC = test_acc

# %% [markdown]
# ## 6. Ką tinklas „mato“: filtrų aktyvacijos
# Parodome pirmojo konvoliucinio sluoksnio išėjimą vienam paveikslėliui, o šviesios sritys rodo, kur filtras „aptiko“ savo raštą.

# %%
x0 = imgs[:1]
first_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][0]
feat = tf.keras.Model(model.inputs, first_conv.output)(x0).numpy()[0]   # modelis pats atlieka Rescaling
plt.figure(figsize=(12, 4)); plt.subplot(1, 5, 1); plt.imshow(x0[0].numpy().astype("uint8")); plt.axis("off"); plt.title("originalas")
for i in range(4):
    plt.subplot(1, 5, i + 2); plt.imshow(feat[..., i], cmap="viridis"); plt.axis("off"); plt.title(f"filtras {i}")
plt.show()

# %% [markdown]
# ## 7. Išvados
# * CNN iš nulio su ~700 paveikslėlių pasiekia gerokai geriau nei atsitiktinį spėjimą, bet duomenų mažai, todėl tikslumas ribotas (galite pamatyti didelį atotrūkį tarp `accuracy` ir `val_accuracy`: permokymas).
# * Kitas žingsnis `05_transfer_learning.ipynb`: tą pačią užduotį sprendžiame su iš anksto apmokytu tinklu.
