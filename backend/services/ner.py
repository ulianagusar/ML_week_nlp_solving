import spacy
from pymorphy2 import MorphAnalyzer

# # Load spaCy model for Russian NER
# nlp = spacy.load("ru_core_news_md")

# def normalize_words(words):
#     """Normalize words using pymorphy2 (lemmatization)."""
#     morph = MorphAnalyzer()
#     return list({morph.parse(word)[0].normal_form for word in words})

# def get_name(mes):
#     """Extracts person names (PER) from the text using spaCy."""
#     doc = nlp(mes)
#     names = {ent.text for ent in doc.ents if ent.label_ == "PER"}
#     return normalize_words(names)

# def get_location(mes):
#     """Extracts locations (LOC) from the text using spaCy."""
#     doc = nlp(mes)
#     locations = {ent.text for ent in doc.ents if ent.label_ == "LOC"}
#     return normalize_words(locations)

# def get_weapons(mes, weapons_list_path="weapon.txt"):
#     """Checks for weapon names in the text using a predefined list from 'weapon.txt'."""
#     try:
#         with open(weapons_list_path, "r", encoding="utf-8") as file:
#             weapons = {line.strip().lower() for line in file}
#         return [w for w in weapons if w in mes.lower()]
#     except FileNotFoundError:
#         print("Warning: weapons.txt not found. Returning an empty list.")
#         return []


def get_name(mes):

    return "name"

def get_location(mes):

    return "location"

def get_weapons(mes):

    return "weapons"


