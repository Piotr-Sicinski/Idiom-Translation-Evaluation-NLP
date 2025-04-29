from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import csv
import json
import asyncio
from typing import List, Dict

from transformers import pipeline
import torch

from openai import AsyncOpenAI

with open("api-keys.json") as f:
    config = json.load(f)

gpt_client = AsyncOpenAI(api_key=config["api_key"])

input_csv = "idiom_sentences.csv"
output_folder = "translation_out"
os.makedirs(output_folder, exist_ok=True)


class LLM:
    name: str

    async def translate(self, text: str) -> str:
        raise NotImplementedError


class APITranslator(LLM):
    name = "gpt-3.5-turbo"

    async def translate(self, text: str) -> str:
        prompt = f"Translate the following Polish sentence to natural English:\n\n{text}"
        response = await gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


class HuggingFaceTranslator(LLM):
    def __init__(self, name: str, model_name: str, src_lang=None, tgt_lang=None):
        self.name = name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if src_lang and tgt_lang:
            self.translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            self.translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

    async def translate(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.translator, text)
        return result[0]["translation_text"]


TRANSLATORS: Dict[str, LLM] = {
    "gpt-3.5-turbo": APITranslator(),
    "opus-mt-pl-en": HuggingFaceTranslator("opus-mt-pl-en", "Helsinki-NLP/opus-mt-pl-en"),
    "nllb-600M": HuggingFaceTranslator(
        "nllb-600M",
        "facebook/nllb-200-distilled-600M",
        src_lang="pol_Latn",
        tgt_lang="eng_Latn"
    ),
}


async def translate_with_model(model: LLM, input_rows: List[List[str]]):
    output_rows = []
    for row in input_rows:
        phrase_id, idiom_id, sentence = row
        try:
            translation = await model.translate(sentence)
            output_rows.append([phrase_id, idiom_id, translation])
            print(f"[{model.name}] & {sentence} -> {translation}")
        except Exception as e:
            print(f"[{model.name}] Error: {e} — {sentence}")
    return output_rows


async def main():
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # skip header
        input_rows = [row for row in reader]

    for model_name, model in TRANSLATORS.items():
        print(f"\nTranslating using {model_name}")
        translated_rows = await translate_with_model(model, input_rows)

        output_path = os.path.join(output_folder, f"translations_{model_name}.csv")
        with open(output_path, "w", newline='', encoding="utf-8") as outcsv:
            writer = csv.writer(outcsv, quoting=csv.QUOTE_ALL)
            writer.writerow(["phrase_id", "idiom_id", "phrase"])
            writer.writerows(translated_rows)

        print(f"Saved: {output_path}")

if __name__ == "__main__":
    print("Using models:")
    for name in TRANSLATORS.keys():
        print(f" - {name}")
    asyncio.run(main())
