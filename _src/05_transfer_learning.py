# %% [markdown]
# # 🔁 Transfer learning: Steam žanras iš viršelio su MobileNetV2
# **Užduotis 1.5** - Transfer learning Colab'ų perdarymas su savo duomenimis.
#
# ## Kas yra transfer learning?
# `MobileNetV2` jau išmokytas ties **ImageNet** (1,4 mln. nuotraukų, 1000 klasių) ir moka atpažinti kraštus, tekstūras, objektus. Mes **nemokome iš nulio**, o pakartotinai panaudojame šias žinias:
#
# 1. **Feature extraction (išskyrimas)**: užšaldome visą MobileNetV2 (`trainable=False`) ir apmokome tik naują viršutinį klasifikatorių.
# 2. **Fine-tuning (smulkus derinimas)**: atšaldome paskutinius ~30 pagrindinio tinklo sluoksnių ir tęsiame mokymą su **labai mažu** mokymosi greičiu (1e-5), kad nesugadintume jau išmoktų požymių.
# 3. **Variantas B (embeddings)**: iš MobileNetV2 paimame požymių vektorius kaip „įterpinius“ ir ant jų apmokome paprastą logistinę regresiją, o `t-SNE` atvaizduojame, kaip žanrai išsidėsto erdvėje.
#
# Svarbu: kiekvienas ImageNet modelis turi savo **`preprocess_input`** (MobileNetV2 tikisi reikšmių [-1, 1]), todėl jo negalima keisti į paprastą `/255`.
#
# **Prieš paleidžiant:** `Runtime → Change runtime type → T4 GPU`. Jei jau paleidote `04_cnn`, duomenys bus perpanaudoti (tas pats `/content/steam_genres`).

# %%
#@@INCLUDE _dataset_cell.txt

# %%
#@@INCLUDE _split_cell.txt

# %% [markdown]
# ## 1. Modelis: MobileNetV2 + naujas klasifikatorius
# `include_top=False` pašalina originalų 1000 klasių sluoksnį. Mūsų viršus: `GlobalAveragePooling2D → Dropout → Dense(softmax)`.

# %%
base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomZoom(0.1)], name="augmentation")
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = aug(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)          # training=False svarbu: BatchNorm sluoksniai lieka „inferencijos“ režime
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(GENRES), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Mokomų parametrų:", sum(int(np.prod(w.shape)) for w in model.trainable_weights),
      "| iš viso:", model.count_params())

# %% [markdown]
# ## 2. Fazė 1: feature extraction
# Mokosi tik galutinis `Dense` sluoksnis, todėl tai greita ir veikia net su mažu duomenų kiekiu.

# %%
cbs = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
h1 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cbs)
acc1 = model.evaluate(test_ds, verbose=0)[1]; print(f"Testas po feature extraction: {acc1:.3f}")

# %% [markdown]
# ## 3. Fazė 2: fine-tuning
# Atšaldome paskutinius 30 MobileNetV2 sluoksnių ir **iš naujo kompiliuojame** su mažesniu mokymosi greičiu (kitaip pakeitimai nesuveiks).

# %%
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cbs)
acc2 = model.evaluate(test_ds, verbose=0)[1]; print(f"Testas po fine-tuning: {acc2:.3f}")

# %%
H = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)], ignore_index=True)
plt.figure(figsize=(8, 4)); plt.plot(H.accuracy, label="train"); plt.plot(H.val_accuracy, label="val")
plt.axvline(len(h1.history["loss"]) - 0.5, color="k", ls="--", label="fine-tuning pradžia"); plt.legend()
plt.title("Transfer learning: tikslumas"); plt.xlabel("epocha"); plt.show()

y_pred = np.concatenate([model.predict(x, verbose=0).argmax(1) for x, _ in test_ds])
y_true = np.concatenate([y.numpy() for _, y in test_ds])
print(classification_report(y_true, y_pred, target_names=GENRES))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=GENRES, yticklabels=GENRES)
plt.xlabel("Prognozė"); plt.ylabel("Tikra"); plt.show()

# %% [markdown]
# ## 4. Variantas B: įterpiniai + logistinė regresija + t-SNE
# Gauname 1280 skaičių vektorių kiekvienam paveikslėliui. Jei žanrai vektorių erdvėje sudaro grupes, tai rodo, kad ImageNet požymiai tinka mūsų užduočiai.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
extractor = tf.keras.Sequential([tf.keras.layers.Input(shape=IMG_SIZE + (3,)),
    tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input), base, tf.keras.layers.GlobalAveragePooling2D()])
emb = lambda ds: np.concatenate([extractor.predict(x, verbose=0) for x, _ in ds])
E_tr, E_te = emb(make_ds(p_tr, y_tr)), emb(test_ds)
clf = LogisticRegression(max_iter=2000).fit(E_tr, y_tr)
acc_emb = clf.score(E_te, y_true); print(f"Įterpiniai + LogReg testo tikslumas: {acc_emb:.3f}")
Z = TSNE(n_components=2, random_state=SEED, perplexity=25).fit_transform(E_te)
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=[GENRES[i] for i in y_true], palette="tab10"); plt.title("t-SNE (testo aibė)"); plt.show()

# %% [markdown]
# ## 5. Palyginimas ir išvados
# Jei `04_cnn` buvo paleistas toje pačioje sesijoje, kintamasis `CNN_TEST_ACC` bus užpildytas; kitu atveju įrašykite reikšmę ranka.

# %%
try: cnn = CNN_TEST_ACC
except NameError: cnn = float("nan")
res = pd.DataFrame({"modelis": ["CNN iš nulio", "MobileNetV2 feature extraction", "MobileNetV2 + fine-tuning", "Įterpiniai + LogReg"],
                    "testo tikslumas": [cnn, acc1, acc2, acc_emb]}).set_index("modelis")
display(res.round(3)); res.plot.barh(legend=False, color="#5c6bc0", title="Testo tikslumas"); plt.show()

# %% [markdown]
# * Transfer learning paprastai gerokai pranoksta CNN iš nulio, nes mažos imties problema sprendžiama panaudojant ImageNet žinias.
# * Fine-tuning kartais duoda tik nedidelį pagerėjimą arba net permoko, todėl būtinas mažas LR ir stebėjimas pagal `val_accuracy`.
# * Apribojimas: viršelis yra marketingas (kaip ir kūrėjų aprašymas!). Tai dera su projekto mintimi: žanrą iš viršelio spėti sunku, o žaidėjų apžvalgos (Gemini notebook'as) parodo, koks žaidimas yra iš tikrųjų.
