import json
import csv
import asyncio
import openai
from openai import AsyncOpenAI

with open("api-keys.json") as f:
    config = json.load(f)

client = AsyncOpenAI(api_key=config["api_key"])

input_csv = "idioms.csv"
output_csv = "idiom_sentences.csv"

# Limit the number of concurrent requests to avoid hitting the API rate limit
semaphore = asyncio.Semaphore(100)
phrase_counter = 1
phrase_lock = asyncio.Lock()


async def generate_examples(idiom_id, idiom):
    global phrase_counter

    prompt = f"""
Dla frazeologizmu "{idiom}", wygeneruj trzy różne zdania z jego użyciem.
Każde zdanie powinno być inne i zawierać frazeologizm w kontekście. Użyj frazeologizmu w formie, w jakiej jest używany w języku polskim. Nie używaj cudzysłowów ani nawiasów. Nie dodawaj żadnych dodatkowych informacji ani wyjaśnień.
Napisz zdania jedno pod drugim.
"""

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",  # "gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            content = response.choices[0].message.content
            sentences = [s.strip(" -1234567890.").strip() for s in content.split('\n') if s.strip()]

            result = []
            async with phrase_lock:
                global phrase_counter
                for sentence in sentences[:3]:
                    result.append([phrase_counter, idiom_id, sentence])
                    phrase_counter += 1

            return result

        except Exception as e:
            print(f"Error with idiom '{idiom}': {e}")
            return []


async def main():
    tasks = []

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            idiom_id, idiom = int(row[0]), row[1].strip()
            tasks.append(generate_examples(idiom_id, idiom))

    all_results = await asyncio.gather(*tasks)

    results = [row for group in all_results for row in group]
    results.sort(key=lambda x: x[1])  # Sorting by phrase_id

    with open(output_csv, 'w', newline='', encoding='utf-8') as outcsv:
        writer = csv.writer(outcsv, quoting=csv.QUOTE_ALL)
        writer.writerow(["phrase_id", "idiom_id", "phrase"])
        writer.writerows(results)

    print("Sentenses generated.")

if __name__ == "__main__":
    asyncio.run(main())
