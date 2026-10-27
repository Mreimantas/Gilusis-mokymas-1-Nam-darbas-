# 1.1 Promptas ChatGPT sistemos projektavimui (PlantUML, po to Mermaid)

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
