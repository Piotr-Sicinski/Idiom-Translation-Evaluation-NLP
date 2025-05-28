import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

evaluation_folder = "evaluation_out"

df = pd.read_csv("evaluation_results.csv")

if not os.path.exists(evaluation_folder):
    os.makedirs(evaluation_folder)

model_avg_correctness = df.groupby("model")["result"].mean()
print("Average correctness rate for each model:")
print(model_avg_correctness)

idiom_avg_correctness = df.groupby("idiom_id")["result"].mean()
print("\nAverage correctness rate for each idiom:")
print(idiom_avg_correctness)

correctness_results = model_avg_correctness.to_dict()
idiom_correctness_results = idiom_avg_correctness.to_dict()

final_results = {
    "model_correctness": correctness_results,
    "idiom_correctness": idiom_correctness_results
}

with open(os.path.join(evaluation_folder, "correctness_results.json"), "w") as json_file:
    json.dump(final_results, json_file, indent=4)

idiom_correctness = df.groupby(["idiom_id", "model"])["result"].mean().reset_index()
idiom_correctness.rename(columns={"result": "correctness_rate"}, inplace=True)

models = df["model"].unique()

sns.set_theme(style="whitegrid")

bins = 10
correctness_range = (0, 1)

for model in models:
    model_data = idiom_correctness[idiom_correctness["model"] == model]

    plt.figure(figsize=(10, 6))
    sns.histplot(model_data["correctness_rate"], bins=bins, kde=True, color='blue', edgecolor='black')

    plt.xlim(correctness_range)

    plt.title(f"Distribution of Correctness Rate for {model}", fontsize=16)
    plt.xlabel("Correctness Rate", fontsize=14)
    plt.ylabel("Number of Idioms", fontsize=14)

    plt.savefig(os.path.join(evaluation_folder, f"correctness_distribution_{model}.png"))
    plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(idiom_avg_correctness, bins=bins, kde=True, color='green', edgecolor='black')

plt.xlim(correctness_range)

plt.title("Distribution of Idiom Correctness Rates", fontsize=16)
plt.xlabel("Correctness Rate", fontsize=14)
plt.ylabel("Number of Idioms", fontsize=14)

plt.savefig(os.path.join(evaluation_folder, "idiom_correctness_distribution.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=model_avg_correctness.index, y=model_avg_correctness.values, palette="viridis")
plt.title("Mean Idiom Correctness for Each Model", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Mean Correctness", fontsize=14)
# plt.ylim(0, 1)
plt.grid(axis='x')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(evaluation_folder, "mean_correctness_bar_chart.png"))
plt.close()

print("Plots saved")
