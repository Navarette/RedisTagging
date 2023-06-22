import numpy as np
import openai
from pypdf import PdfReader
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import requests
from bs4 import BeautifulSoup
import re
import redis

import spacy
from spacy.matcher import PhraseMatcher
from pdfminer.high_level import extract_text
import PyPDF2

# pip install PyPDF2==1.26.0
# pip install spacy
# pip install pdfminer.six
# python -m spacy download en_core_web_sm
# pip install openai
# pip install openai
# pip install redis
# pip install requests
# pip install BeautifulSoup
# pip install re
# pip install pypdf

taxonomy = [
    # Programming Languages
    "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "Swift", "Rust", "PHP",
    # Technology Companies
    "Tesla", "Apple", "Microsoft", "Google", "Amazon", "Facebook", "IBM", "Intel", "Nvidia", "Samsung",

    # News and Media
    "Breaking news", "Headlines", "Journalism", "Broadcast", "Press release", "Media coverage",
    "Editorial", "Op-ed", "Investigative reporting", "Public relations",

    # Scientific Fields
    "Physics", "Chemistry", "Biology", "Astronomy", "Geology", "Mathematics", "Psychology",
    "Sociology", "Anthropology", "Economics",

    # Arts and Entertainment
    "Film", "Music", "Literature", "Theater", "Visual arts", "Dance", "Television", "Gaming",
    "Photography", "Fashion",

    # Sports
    "Soccer", "Basketball", "Tennis", "Golf", "Baseball", "Athletics", "Swimming", "Cricket",
    "Rugby", "Volleyball",

    # Global Issues
    "Climate change", "Human rights", "Poverty", "Education", "Healthcare", "Sustainability",
    "Gender equality", "Social justice", "Conflict resolution", "Immigration",

    # Business and Finance
    "Entrepreneurship", "Startups", "Investments", "Stock market", "Financial planning", "Banking",
    "Insurance", "Marketing", "E-commerce", "Supply chain",

    # Travel and Tourism
    "Destinations", "Hotels", "Airlines", "Vacation packages", "Sightseeing", "Adventure travel",
    "Sustainable tourism", "Travel insurance", "Cultural experiences", "Backpacking",

    # Health and Wellness
    "Nutrition", "Exercise", "Mental health", "Yoga", "Meditation", "Alternative medicine",
    "Wellness retreats", "Healthy living", "Stress management", "Self-care",

    # Smartphone-related terms
    "Smartphone", "Mobile device", "Operating system", "Android", "iOS", "Windows Phone",
    "UI (User Interface)", "Touchscreen", "Processor", "RAM (Random Access Memory)",

    # Additional terms
    "Artificial Intelligence", "Machine Learning", "Data Science", "Big Data", "Cloud Computing",
    "Internet of Things (IoT)", "Cybersecurity", "Virtual Reality (VR)", "Augmented Reality (AR)",
    "Blockchain", "Cryptocurrency", "Robotics", "Automation", "DevOps", "Software Development",
    "Web Development", "Mobile App Development", "User Experience (UX)", "User Interface (UI)",
    "Human-Computer Interaction (HCI)", "Data Visualization", "Natural Language Processing (NLP)",
    "Computer Vision", "Information Retrieval", "Algorithm", "Data Mining", "Internet", "Social Media",
    "Digital Marketing", "Content Creation", "Gaming Industry", "Sports Industry", "Healthcare Industry",
    "Education Industry", "Environmental Conservation", "Renewable Energy", "Sustainable Development",
    "Social Entrepreneurship", "Workplace Diversity", "Globalization", "Cultural Diversity",
    "Consumer Behavior", "Supply Chain Management", "Financial Technology (Fintech)",
    "Cryptocurrency Exchange", "Online Payment Systems", "Digital Transformation", "Customer Relationship Management (CRM)", "Online Advertising",
    "Search Engine Optimization (SEO)", "Influencer Marketing", "Travel Destinations", "Adventure Sports",
    "Food and Beverage Industry", "Fashion Industry", "Art Galleries", "Music Festivals",
    "Book Publishing", "Online Streaming", "Fitness and Exercise Equipment", "Mental Wellness Apps",
    "Organic Food", "Vegan Lifestyle", "Sustainable Fashion", "Eco-Tourism", "Adventure Travel",
    "Beach Holidays", "Historical Sites", "Wildlife Conservation"

]

file_paths = [
    '/workspace/RedisTagging/pdf/art.pdf',
    '/workspace/RedisTagging/pdf/smartphone.pdf'
]

library = []

r = redis.Redis(host='localhost', port=6379, db=0)

# Store the terms in the Redis table
for term in taxonomy:
    r.hset("taxonomy", term, 1)

# Retrieve all the terms from the table
all_terms = r.hkeys("taxonomy")

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in reader.pages:
            text += page.extractText()
    return text.encode('utf-8').decode('utf-8')

# to save the file path and its tags
def tag_pdf(file_path, terms):
    # Estrazione del testo dal PDF
    text = extract_text_from_pdf(file_path)

    # Caricamento del modello linguistico
    nlp = spacy.load('en_core_web_sm')

    # Elaborazione del testo
    doc = nlp(text)

    # Creazione del PhraseMatcher per verificare frasi con parole della tassonomia
    matcher = PhraseMatcher(nlp.vocab)
    for term in terms:
        term_str = term.decode('utf-8')  # Converti il termine da byte a stringa
        matcher.add("Taxonomy", [nlp.make_doc(term_str.lower())])

    # Trova le corrispondenze con la tassonomia nel testo
    matches = matcher(doc)

    # Ottieni i tag - utilizza una lista e un set per evitare duplicati
    tags = list(set(doc[start:end].text.lower() for _, start, end in matches))

    return tags

for file_path in file_paths:
    tags = tag_pdf(file_path, all_terms)
    library.append({'file_path': file_path, 'tags': tags})

for item in library:
    print("File Path:", item['file_path'])
    print("Tags:", item['tags'])
    print()



