from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os
import csv
import json
import asyncio
from typing import List, Dict
import httpx
import tempfile

from transformers import pipeline
import torch

from openai import AsyncOpenAI
import argparse
import argostranslate.package
import argostranslate.translate
import requests

from googletrans import Translator as GoogleTranslatorClient

with open("api-keys.json") as f:
    config = json.load(f)

gpt_client = AsyncOpenAI(api_key=config["api_key"])
google_api_key = config["google_translate_api_key"]

class LLM:
    name: str

    async def translate(self, text: str) -> str:
        raise NotImplementedError


class APITranslator(LLM):
    name = "gpt-3.5-turbo"

    async def translate(self, text: str) -> str:
        prompt = f"Translate the following English sentence to natural Polish:\n\n{text}"
        response = await gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    

class DeepSeekTranslator(LLM):
    name = "deepseek-chat"

    def __init__(self):
        self.api_key = config["deepseek_api_key"]
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    async def translate(self, text: str) -> str:
        prompt = f"Translate the following English sentence to natural Polish:\n\n{text}. Give me only translation. Do not add comments"
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("DeepSeek API error:", e)
            raise e
        


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
    

class GoogleTranslateAPITranslator(LLM):
    name = "google-translate-api"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://translation.googleapis.com/language/translate/v2"

    async def translate(self, text: str) -> str:
        params = {
            "q": text,
            "source": "en",
            "target": "pl",
            "format": "text",
            "key": self.api_key
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            return data["data"]["translations"][0]["translatedText"]
        
class MBart50Translator(LLM):
    def __init__(self):
        self.name = "mbart50"
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.src_lang = "en_XX"
        self.tgt_lang = "pl_PL"

    async def translate(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._translate_sync, text)

    def _translate_sync(self, text: str) -> str:
        self.tokenizer.src_lang = self.src_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
class ArgosTranslateTranslator(LLM):
    name = "argos-translate"

    def __init__(self):
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        installed_languages = argostranslate.translate.get_installed_languages()
        installed_codes = [lang.code for lang in installed_languages]

        if "en" not in installed_codes or "pl" not in installed_codes:
            for pkg in available_packages:
                if pkg.from_code == "en" and pkg.to_code == "pl":
                    print("Installing Argos Translate package en->pl...")
                    download_path = pkg.download()
                    argostranslate.package.install_from_path(download_path)
                    break

        self.installed_languages = argostranslate.translate.get_installed_languages()
        self.from_lang = next(filter(lambda l: l.code == "en", self.installed_languages))
        self.to_lang = next(filter(lambda l: l.code == "pl", self.installed_languages))
        self.translator = self.from_lang.get_translation(self.to_lang)

    async def translate(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.translator.translate, text)
        return result

class Madlad400Translator(LLM):
    def __init__(self, max_length=200):
        self.name = "madlad400-3b-mt"
        self.model_name = "google/madlad400-3b-mt"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.src_lang = "eng"
        self.tgt_lang = "<2pl>" 
        self.max_length = max_length

    async def translate(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._translate_sync, text)

    def _translate_sync(self, text: str) -> str:
        prefix = f"{self.tgt_lang} "
        input_text = prefix + text

        encoded = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        generated_tokens = self.model.generate(
            **encoded,
            max_length=120,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )

        decoded = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        if prefix in decoded:
            parts = decoded.split(prefix)
            cleaned = parts[1].strip()
            return cleaned
        else:
            return decoded

    

TRANSLATORS: Dict[str, LLM] = {
    "google-translate-api": GoogleTranslateAPITranslator(google_api_key),
    "gpt-3.5-turbo": APITranslator(),
    "nllb-600M": HuggingFaceTranslator(
        "nllb-600M",
        "facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="pol_Latn"
    ),
    "madlad400-3b-mt": HuggingFaceTranslator(
        name="madlad400-3b-mt",
        model_name="google/madlad400-3b-mt",
        src_lang="eng_Latn",
        tgt_lang="pol_Latn"
    ),
    "mbart50": MBart50Translator(),
    "deepseek-chat": DeepSeekTranslator(),
    "argos-translate" : ArgosTranslateTranslator(),
    "madlad400-3b-mt": Madlad400Translator(max_length=200),
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


async def main(args):

    with open(args.input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)
        input_rows = [row for row in reader]

    for model_name, model in TRANSLATORS.items():
        print(f"\nTranslating using {model_name}")
        translated_rows = await translate_with_model(model, input_rows)

        output_path = os.path.join(args.output_folder, f"translations_{model_name}.csv")
        with open(output_path, "w", newline='', encoding="utf-8") as outcsv:
            writer = csv.writer(outcsv, quoting=csv.QUOTE_ALL)
            writer.writerow(["phrase_id", "idiom_id", "phrase"])
            writer.writerows(translated_rows)

        print(f"Saved: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Translate Polish idiom sentences.")
    parser.add_argument(
        "-i", "--input-csv",
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output-folder",
        required=True,
        help="Directory to save the translation output."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    print("Using models:")
    for name in TRANSLATORS.keys():
        print(f" - {name}")
    asyncio.run(main(args))
