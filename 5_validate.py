from comet import download_model, load_from_checkpoint
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ref = pd.read_csv("validation_sentences/validation_sentences_pl.csv")['phrase_pl'].tolist()
src = pd.read_csv("validation_sentences/validation_sentences_en.csv")['phrase_en'].tolist()

translations = {}

for file_name in os.listdir('translation_validation'):
    translation = pd.read_csv(os.path.join('translation_validation', file_name))['phrase'].tolist()
    translations[file_name.split('_')[1]] = []
    for src_sentence, mt_sentence, ref_sentence in zip(src, translation, ref):
        translations[file_name.split('_')[1]].append({"src": src_sentence, "mt": mt_sentence, "ref": ref_sentence})


model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
scores_dict = {}
mean_scores = {}
for model_name, data in translations.items():
    scores = comet_model.predict(data)
    scores_dict[model_name] = scores['scores']
    mean_scores[model_name] = np.mean(scores['scores'])

    df = pd.DataFrame({
        "src": src,
        "mt": [d["mt"] for d in data],
        "ref": ref,
        "comet_score": scores['scores']
    })
    out_path = f"comet_validation/validation_with_comet_{model_name}"
    df.to_csv(out_path, index=False)
    print(f"Saved COMET scores for {model_name} to {out_path}; mean score: {mean_scores[model_name]:.4f}")

model_names = [name.split('.')[0] for name in mean_scores.keys()]
scores = list(mean_scores.values())

sorted_data = sorted(zip(model_names, scores), key=lambda x: x[0])
sorted_model_names, sorted_scores = zip(*sorted_data)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(sorted_model_names), y=list(sorted_scores), palette="viridis")

plt.xlabel("Model Name")
plt.ylabel("Mean COMET Score")
plt.title("Mean COMET Score per Model")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y')
plt.savefig("comet_validation/mean_comet_scores_per_model.png")
plt.show()
