import os
import csv
import json
import asyncio
import pandas as pd
from typing import List
from openai import AsyncOpenAI

with open("api-keys.json") as f:
    config = json.load(f)

client = AsyncOpenAI(api_key=config["api_key"])

input_original = "idiom_sentences.csv"
translation_folder = "translation_out"
model_files = [f for f in os.listdir(translation_folder) if f.endswith(".csv")]

originals = {}
with open(input_original, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  # skip header
    for row in reader:
        phrase_id, idiom_id, phrase = row
        originals[(phrase_id, idiom_id)] = phrase.strip()


async def judge_translation(original: str, translated: str) -> int:
    prompt = f"""
    You are a bilingual assistant that focuses on idiom translation correctness. Check if the English sentence is a correct translation of the Polish sentence.

    Polish: {original}
    English: {translated}

    Answer with "1" if the translation is correct and faithful.
    Answer with "0" if it is incorrect or significantly flawed.
    Only reply with a single digit: 1 or 0.
    """

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    return 1 if content == "1" else 0


async def evaluate_all():
    results = []
    tasks = []

    for file in model_files:
        model_name = file.replace("translations_", "").replace(".csv", "")
        path = os.path.join(translation_folder, file)

        print(f"Evaluating: {model_name}")

        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        # Create a list of tasks to evaluate translations concurrently
        for phrase_id, idiom_id, translated in rows:
            original = originals.get((phrase_id, idiom_id), None)
            if not original:
                raise ValueError(f"Original sentence not found for {phrase_id}-{idiom_id}")

            tasks.append(evaluate_translation(phrase_id, idiom_id, translated, original, model_name, results))

    await asyncio.gather(*tasks)

    # Sort results by model, idiom_id, phrase_id
    sorted_results = sorted(results, key=lambda x: (x['model'], x['idiom_id'], x['phrase_id']))

    df = pd.DataFrame(sorted_results)
    df.to_csv("evaluation_results.csv", index=False, encoding="utf-8")
    print("Saved evaluation_results.csv")


async def evaluate_translation(phrase_id, idiom_id, translated, original, model_name, results):
    try:
        score = await judge_translation(original, translated)
        results.append({
            "phrase_id": phrase_id,
            "idiom_id": idiom_id,
            "model": model_name,
            "result": score
        })
        print(f"[{model_name}] & {phrase_id}-{idiom_id} -> {score}")
    except Exception as e:
        print(f"Error {model_name} {phrase_id}-{idiom_id}: {e}")

if __name__ == "__main__":
    asyncio.run(evaluate_all())
