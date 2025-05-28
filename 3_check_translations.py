import os
import csv
import json
import asyncio
import pandas as pd
from typing import List
import google.generativeai as genai

with open("api-keys.json") as f:
    config = json.load(f)

genai.configure(api_key=config["gemini_api_key"])
model = genai.GenerativeModel("gemini-1.5-pro-latest")

input_original = "eng_idiom_sentences.csv"
translation_folder = "pol_translations"
model_files = [f for f in os.listdir(translation_folder) if f.endswith(".csv")]

originals = {}
with open(input_original, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  # skip header
    for row in reader:
        phrase_id, idiom_id, phrase = row
        originals[(phrase_id, idiom_id)] = phrase.strip()


async def judge_translation(original: str, translated: str) -> int:
    # prompt = f"""
    # You are a bilingual assistant that focuses on idiom translation correctness. Check if the English sentence is a correct translation of the Polish sentence.

    # Polish: {original}
    # English: {translated}

    # Answer with "1" if the translation is correct and faithful.
    # Answer with "0" if it is incorrect or significantly flawed.
    # Only reply with a single digit: 1 or 0.
    # """

    prompt = f"""
You are a bilingual assistant that focuses on idiom translation correctness. Check if the Polish sentence is a correct translation of the English sentence.

English: 
{original}

Polish: 
{translated}

Answer with "1" if the translation is correct and faithful.
Answer with "0" if it is incorrect, literal or significantly flawed.
Only reply with a single digit: 1 or 0.
    """

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        return 1 if content == "1" else 0
    except Exception as e:
        print(f"Error in Gemini API: {e}")
        return 0


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

        for phrase_id, idiom_id, translated in rows:
            original = originals.get((phrase_id, idiom_id), None)
            if not original:
                raise ValueError(f"Original sentence not found for {phrase_id}-{idiom_id}")

            tasks.append(evaluate_translation(phrase_id, idiom_id, translated, original, model_name, results))

    await asyncio.gather(*tasks)

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
