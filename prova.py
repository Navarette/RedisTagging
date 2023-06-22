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


import openai
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



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


class DataService():

    def __init__(self):
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD
        )

    def drop_redis_data(self, index_name: str = INDEX_NAME):
        try:
            self.redis_client.flushdb()
            print('Index dropped')
        except:
            # Index doees not exist
            print('Index does not exist')

    def load_data_to_redis(self, embeddings):
        # Constants
        vector_dim = len(embeddings[0]['vector'])  # length of the vectors
        
		# Initial number of vectors
        vector_number = len(embeddings)

        # Define RediSearch fields
        text = TextField(name="text")
        text_embedding = VectorField("vector",
                                     "FLAT", {
                                         "TYPE": "FLOAT32",
                                         "DIM": vector_dim,
                                         "DISTANCE_METRIC": "COSINE",
                                         "INITIAL_CAP": vector_number,
                                     }
                                     )
        fields = [text, text_embedding]

        # Check if index exists
        try:
            self.redis_client.ft(INDEX_NAME).info()
            print("Index already exists")
        except:
            # Create RediSearch Index
            self.redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(
                    prefix=[PREFIX], index_type=IndexType.HASH)
            )

        for embedding in embeddings:
            key = f"{PREFIX}:{str(embedding['id'])}"
            embedding["vector"] = np.array(
                embedding["vector"], dtype=np.float32).tobytes()
            self.redis_client.hset(key, mapping=embedding)
        print(
            f"Loaded {self.redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")


    def remove_newlines(self,text):
        text = open(text, "r", encoding="UTF-8").read()
        text = text.replace('\n', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('  ', ' ')
        text = text.replace('  ', ' ')
        return text



    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int = 550):
        # Read data from pdf file and split it into chunks
        reader = PdfReader(pdf_path)
        chunks = []
        for page in reader.pages:
            text_page = page.extract_text()
            chunks.extend([text_page[i:i+chunk_length].replace('\n', '')
                          for i in range(0, len(text_page), chunk_length)])

        # Create embeddings
        response = openai.Embedding.create(
            model='text-embedding-ada-002', input=chunks)
        return [{'id': value['index'], 'vector':value['embedding'], 'text':chunks[value['index']]} for value in response['data']]

    def txt_to_embeddings(self, text, chunk_length: int = 250):
        # Read data from pdf file and split it into chunks
        text = open(text, "r")
        text = text.read()

        chunks = []
        
        chunks.extend([text[i:i+chunk_length].replace('\n', '')
                       for i in range(0, len(text), chunk_length)])

        # Create embeddings
        response = openai.Embedding.create(
            model='text-embedding-ada-002', input=chunks)
        return [{'id': value['index'], 'vector':value['embedding'], 'text':chunks[value['index']]} for value in response['data']]
    
    def url_to_embeddings(self, url, chunk_length: int = 250):
        # Read data from pdf file and split it into chunks

        def rimuovi_contenuto_angolare(testo):
            pattern = r"<.*?>"  # Pattern per cercare "<" seguito da qualsiasi carattere, incluso il newline, fino a ">"
            testo_senza_angolari = re.sub(pattern, "", testo)  # Rimuovi i match del pattern dal testo
            return testo_senza_angolari




        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        body = soup.find('body')
        content = str(soup.find_all("p"))
    
        text = rimuovi_contenuto_angolare(content)


        chunks = []
        
        chunks.extend([text[i:i+chunk_length].replace('\n', '')
                       for i in range(0, len(text), chunk_length)])

        # Create embeddings
        response = openai.Embedding.create(
            model='text-embedding-ada-002', input=chunks)
        return [{'id': value['index'], 'vector':value['embedding'], 'text':chunks[value['index']]} for value in response['data']]

    def search_redis(self,
                     user_query: str,
                     index_name: str = "embeddings-index",
                     vector_field: str = "vector",
                     return_fields: list = ["text", "vector_score"],
                     hybrid_fields="*",
                     k: int = 20,
                     print_results: bool = False,
                     ):
        # Creates embedding vector from user query
        embedded_query = openai.Embedding.create(input=user_query,
                                                 model="text-embedding-ada-002",
                                                 )["data"][0]['embedding']
        # Prepare the Query
        base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
        query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict = {"vector": np.array(
            embedded_query).astype(dtype=np.float32).tobytes()}
        # perform vector search
        results = self.redis_client.ft(index_name).search(query, params_dict)
        if print_results:
            for i, doc in enumerate(results.docs):
                score = 1 - float(doc.vector_score)
                print(f"{i}. {doc.text} (Score: {round(score ,3) })")
        return [doc['text'] for doc in results.docs]


class IntentService():
     def __init__(self):
        pass
     
     def get_intent(self, user_question: str):
         # call the openai ChatCompletion endpoint
         response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=[
               {"role": "user", "content": f'Extract the keywords from the following question: {user_question}'+
                 'Do not answer anything else, only the keywords.'}
            ]
         )
         

         tags = (response['choices'][0]['message']['content'])
         result = re.split(r'[,.]', tags.replace(', ', ' '))
         result = [element.strip() for element in result if element.strip()]

         return result



class taggingservice():
    def __init__(self):
        pass

    def extract_text_from_pdf(file_path):
        return extract_text(file_path)


    def tag_pdf(file_path, taxonomy):
        # Extraction of text from PDF
        text = taggingservice.extract_text_from_pdf(file_path)

        # Load the language model
        nlp = spacy.load('en_core_web_sm')

        # Process the text
        doc = nlp(text)

        # Create the PhraseMatcher to check phrases with words in the taxonomy
        matcher = PhraseMatcher(nlp.vocab)
        for phrase in taxonomy:
            # Convert everything to lowercase
            matcher.add("Taxonomy", [nlp(phrase.lower())])

        # Find the matches with the taxonomy in the text
        matches = matcher(doc)

        # Get the tags - use a list and a set to avoid duplicates
        tags = list(set(doc[start:end].text for _, start, end in matches))

        return tags




# to save the file path and its tags
library = []
file_paths = [
    '/workspace/RedisTagging/pdf/MENTAL HEALTH.pdf',
    '/workspace/RedisTagging/pdf/SMARTPHONE BRANDS DESIGN AND BUYING DECISION.pdf',
    '/workspace/RedisTagging/pdf/volleyball_tutorial.pdf'
    ]
for file_path in file_paths:
    # invokes the class to extract tags
    tags = taggingservice.tag_pdf(file_path, taxonomy)
    # saves the tag and the file path to a dictionary
    library.append({'file_path': file_path, 'tags': tags})


# print(library)




question = "what is mental health"
# Get the intent
intents = IntentService().get_intent(question)
print(intents)


printed_file_paths = [] # to print same filepaths only once
for item in library:
    # print(item)
    for intent in intents:
        if intent in item['tags'] and item['file_path'] not in printed_file_paths: #filepath print cotroller
            print(item['file_path'])
            printed_file_paths.append(item['file_path'])



for item in printed_file_paths:
    data = DataService().pdf_to_embeddings(item)
    DataService().load_data_to_redis(data)