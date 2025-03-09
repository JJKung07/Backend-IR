import os
import time
import re
import sqlite3
import Levenshtein
from datetime import datetime, timedelta
from spellchecker import SpellChecker
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch

search_bp = Blueprint('search', __name__)
CORS(search_bp, resources={r"/api/*": {"origins": "*"}})

# Configure Elasticsearch client
es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "Vb0GGDOzMd-khfxnUkYp"),
    ca_certs="http_ca.crt",
    request_timeout=30,  # Changed from 'timeout' to 'request_timeout'
    verify_certs=True,
    ssl_show_warn=False,
    max_retries=3,
    retry_on_timeout=True
)

# SQLite connection helper (if needed for indexing)
DB_PATH = 'resoures/db.db'
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def clean_result(result):
    """Clean R-style vectors and HTML tags from Elasticsearch results"""
    cleaned = result.copy()
    html_pattern = re.compile(r'<[^>]+>')  # Regex to match HTML tags

    def remove_tags(text):
        """Remove HTML tags from text or list of texts"""
        if isinstance(text, list):
            return [html_pattern.sub('', item) for item in text]
        return html_pattern.sub('', str(text))

    def parse_r_vector(text):
        """Parse R-style vectors from strings or lists"""
        if isinstance(text, list):
            text = ' '.join(text)

        # Extract all quoted content using regex
        matches = re.findall(r'"([^"]*)"', text)
        if matches:
            return matches

        # Fallback to comma splitting if no quotes
        return [item.strip() for item in text.split(',')]

    # Clean ingredients
    if 'ingredients' in cleaned:
        if isinstance(cleaned['ingredients'], str):
            cleaned['ingredients'] = parse_r_vector(cleaned['ingredients'])
        elif isinstance(cleaned['ingredients'], list):
            cleaned['ingredients'] = parse_r_vector(' '.join(cleaned['ingredients']))
        # Remove HTML tags from ingredients
        cleaned['ingredients'] = remove_tags(cleaned['ingredients'])

    # Clean highlights
    if 'highlights' in cleaned:
        highlights = cleaned['highlights']
        for field in list(highlights.keys()):
            # Remove HTML tags first
            highlights[field] = remove_tags(highlights[field])

            # Handle R-style vectors in specific fields
            if field in ['RecipeIngredientParts', 'RecipeInstructions']:
                joined = ' '.join(highlights[field])
                parsed = parse_r_vector(joined)
                highlights[field] = parsed

    return cleaned

def correct_typos(query):
    """
    Correct typographical errors in the search query using both
    SpellChecker for individual words and Levenshtein distance
    for common recipe and ingredient terms.
    """
    spell = SpellChecker()
    words = query.lower().split()
    corrected_words = []

    # Load common food-related terms from Elasticsearch
    common_terms = get_common_food_terms()

    for word in words:
        # Skip short words (likely not meaningful or are common words)
        if len(word) <= 2:
            corrected_words.append(word)
            continue

        # First check if any common food term is similar
        closest_term = None
        min_distance = float('inf')

        for term in common_terms:
            distance = Levenshtein.distance(word, term.lower())
            # Only consider terms with a reasonable distance
            if distance < min(3, len(word) // 2) and distance < min_distance:
                min_distance = distance
                closest_term = term

        # If we found a close food term, use it
        if closest_term:
            corrected_words.append(closest_term)
        else:
            # Otherwise use standard spell checker
            correction = spell.correction(word)
            corrected_words.append(correction if correction else word)

    return ' '.join(corrected_words)

def get_common_food_terms():
    """
    Retrieve common food-related terms from the Elasticsearch index
    to improve spell correction for domain-specific vocabulary.
    """
    # This could be cached for performance
    try:
        # Get common ingredients
        ingredient_aggs = es_client.search(
            index="recipes",
            body={
                "size": 0,
                "aggs": {
                    "ingredients": {
                        "terms": {
                            "field": "RecipeIngredientParts.keyword",
                            "size": 1000
                        }
                    }
                }
            }
        )

        # Get common recipe names/keywords
        keyword_aggs = es_client.search(
            index="recipes",
            body={
                "size": 0,
                "aggs": {
                    "keywords": {
                        "terms": {
                            "field": "Keywords.keyword",
                            "size": 1000
                        }
                    }
                }
            }
        )

        # Extract terms from aggregations
        ingredients = [bucket["key"] for bucket in ingredient_aggs["aggregations"]["ingredients"]["buckets"]]
        keywords = [bucket["key"] for bucket in keyword_aggs["aggregations"]["keywords"]["buckets"]]

        # Combine and flatten the list of terms
        common_terms = set()
        for term_list in [ingredients, keywords]:
            for term in term_list:
                # Split multi-word terms and add individual words
                for word in term.split():
                    if len(word) > 2:  # Only add words longer than 2 chars
                        common_terms.add(word)

        return list(common_terms)
    except Exception as e:
        app.logger.error(f"Error retrieving common food terms: {str(e)}")
        return []

def search_in_elasticsearch(query, page=1, size=10):
    """
    Search for recipes in Elasticsearch based on name, ingredients, or cooking process.
    Returns results ranked by relevance.
    """
    from_val = (page - 1) * size

    # Build the search query
    search_body = {
        "from": from_val,
        "size": size,
        "query": {
            "bool": {
                "should": [
                    # Search in recipe name with high boost
                    {"match": {"Name": {"query": query, "boost": 3.0}}},
                    # Search in ingredients
                    {"match": {"RecipeIngredientParts": {"query": query, "boost": 2.0}}},
                    # Search in cooking instructions
                    {"match": {"RecipeInstructions": {"query": query, "boost": 1.5}}},
                    # Search in description
                    {"match": {"Description": {"query": query, "boost": 1.0}}},
                    # Search in keywords
                    {"match": {"Keywords": {"query": query, "boost": 1.0}}},
                    # Fuzzy search for typo tolerance in name
                    {
                        "match": {
                            "Name": {
                                "query": query,
                                "fuzziness": "AUTO",
                                "boost": 1.0
                            }
                        }
                    },
                    # Fuzzy search for typo tolerance in ingredients
                    {
                        "match": {
                            "RecipeIngredientParts": {
                                "query": query,
                                "fuzziness": "AUTO",
                                "boost": 0.8
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "highlight": {
            "fields": {
                "Name": {},
                "RecipeIngredientParts": {},
                "RecipeInstructions": {},
                "Description": {}
            },
            "pre_tags": ["<strong>"],
            "post_tags": ["</strong>"]
        }
    }

    # Execute the search
    result = es_client.search(index="recipes", body=search_body)

    # Process and format the search results
    total_hits = result["hits"]["total"]["value"]
    hits = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]
        score = hit["_score"]

        # Extract highlights if available
        highlights = hit.get("highlight", {})

        # Format the result with raw ingredients string
        recipe = {
            "id": source.get("RecipeId"),
            "name": source.get("Name"),
            "description": source.get("Description", "")[:150] + "..."
            if source.get("Description") and len(source.get("Description")) > 150
            else source.get("Description", ""),
            "image": source.get("Images", "").split(",")[0] if source.get("Images") else None,
            "rating": source.get("AggregatedRating"),
            "reviewCount": source.get("ReviewCount"),
            "cookTime": source.get("CookTime"),
            "ingredients": source.get("RecipeIngredientParts", ""),  # Keep as raw string
            "highlights": highlights,
            "score": score
        }

        # Apply cleaning transformations
        cleaned_recipe = clean_result(recipe)

        # Take first 5 ingredients after cleaning
        cleaned_recipe['ingredients'] = cleaned_recipe['ingredients'][:5]

        hits.append(cleaned_recipe)

    return {
        "hits": hits,
        "total": total_hits
    }

@search_bp.route('/api/search', methods=['GET'])
def search_recipes():
    # Get query parameters
    query = request.args.get('q', '')
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))

    if not query:
        return jsonify({'error': 'Search query is required'}), 400

    # Check for typos and suggest corrections
    original_query = query
    corrected_query = None

    # Only check for corrections if query has more than 2 characters
    if len(query) > 2:
        corrected_query = correct_typos(query)
        # Only set corrected_query if it's different from the original
        if corrected_query == query:
            corrected_query = None

    try:
        # Search in Elasticsearch
        search_results = search_in_elasticsearch(query, page, size)

        # If no results or very few, and we have a corrected query, try the correction
        if corrected_query and (search_results['total'] < 3):
            corrected_results = search_in_elasticsearch(corrected_query, page, size)

            # Return both the original results and suggestion
            return jsonify({
                'results': search_results['hits'],
                'total': search_results['total'],
                'page': page,
                'size': size,
                'suggestion': {
                    'text': corrected_query,
                    'results': corrected_results['hits'],
                    'total': corrected_results['total']
                }
            }), 200

        # Return standard results
        return jsonify({
            'results': search_results['hits'],
            'total': search_results['total'],
            'page': page,
            'size': size,
            'suggestion': {'text': corrected_query} if corrected_query else None
        }), 200

    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'An error occurred during search'}), 500

# adsads
def setup_elasticsearch_index():
    """
    Create and configure the Elasticsearch index for recipes.
    This sets up the appropriate mappings for searching.
    """
    index_name = "recipes"

    # Check if index already exists
    if es_client.indices.exists(index=index_name):
        return

    # Define the mapping for the recipes index
    mapping = {
        "mappings": {
            "properties": {
                "RecipeId": {"type": "keyword"},
                "Name": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "Description": {"type": "text", "analyzer": "english"},
                "RecipeInstructions": {"type": "text", "analyzer": "english"},
                "RecipeIngredientParts": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "RecipeIngredientQuantities": {"type": "text"},
                "Keywords": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "Images": {"type": "text"},
                "AggregatedRating": {"type": "float"},
                "ReviewCount": {"type": "float"},
                "CookTime": {"type": "text"},
                "PrepTime": {"type": "text"},
                "TotalTime": {"type": "text"},
                "RecipeCategory": {"type": "text"},
                "AuthorId": {"type": "keyword"},
                "AuthorName": {"type": "text"}
            }
        },
        "settings": {
            "analysis": {
                "filter": {
                    "english_stop": {
                        "type": "stop",
                        "stopwords": "_english_"
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    },
                    "english_possessive_stemmer": {
                        "type": "stemmer",
                        "language": "possessive_english"
                    }
                },
                "analyzer": {
                    "english": {
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "english_possessive_stemmer",
                            "english_stop",
                            "english_stemmer"
                        ]
                    }
                }
            }
        }
    }

    # Create the index with the mapping
    es_client.indices.create(index=index_name, body=mapping)

def index_recipes_from_sqlite():
    """
    Fetch recipes from SQLite database and index them in Elasticsearch.
    This can be run as a scheduled task or manually.
    """
    # Check and setup the Elasticsearch index
    setup_elasticsearch_index()

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get the total number of recipes
    cursor.execute("SELECT COUNT(*) FROM recipes")
    total_recipes = cursor.fetchone()[0]

    # Batch size for indexing
    batch_size = 1000
    total_batches = (total_recipes + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        offset = batch_num * batch_size

        # Fetch a batch of recipes
        cursor.execute(f"SELECT * FROM recipes LIMIT {batch_size} OFFSET {offset}")
        recipes = cursor.fetchall()

        # Prepare batch for bulk indexing
        bulk_data = []

        for recipe in recipes:
            recipe_dict = dict(recipe)

            # Process recipe for indexing
            # For array fields that are stored as comma-separated strings
            if recipe_dict.get('RecipeIngredientParts'):
                recipe_dict['RecipeIngredientParts'] = recipe_dict['RecipeIngredientParts'].split(',')

            if recipe_dict.get('Keywords'):
                recipe_dict['Keywords'] = recipe_dict['Keywords'].split(',')

            # Add index action
            bulk_data.append(
                {"index": {"_index": "recipes", "_id": recipe_dict['RecipeId']}}
            )
            bulk_data.append(recipe_dict)

        # Execute bulk indexing if we have data
        if bulk_data:
            es_client.bulk(body=bulk_data, refresh=True)

        # Log progress
        print(f"Indexed batch {batch_num + 1}/{total_batches} recipes")

    conn.close()
    print(f"Completed indexing {total_recipes} recipes")

# Command to run the indexing process
@search_bp.cli.command("index-recipes")
def index_recipes_command():
    """Index all recipes from SQLite to Elasticsearch."""
    start_time = time.time()
    index_recipes_from_sqlite()
    elapsed_time = time.time() - start_time
    print(f"Indexing completed in {elapsed_time:.2f} seconds")
