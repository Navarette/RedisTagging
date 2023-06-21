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

INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
# distance metric for the vectors (ex. COSINE, IP, L2)
DISTANCE_METRIC = "COSINE"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""


def __init__(self):
    # Connect to Redis
    self.redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD
    )
