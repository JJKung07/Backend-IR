import os
import time
import re
import sqlite3
import Levenshtein
from datetime import datetime, timedelta
from spellchecker import SpellChecker
from flask import Blueprint, request, jsonify, current_app
from flask_cors import CORS
from elasticsearch import Elasticsearch

search_bp = Blueprint('search', __name__)
CORS(search_bp, resources={r"/api/*": {"origins": "*"}})

# Configure Elasticsearch client
es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "w9WdF24qfMj8bHZtDBwj"),
    ca_certs="D:\Java\IR\Project\Backend\http_ca.crt",
    request_timeout=30,
    verify_certs=True,
    ssl_show_warn=False,
    max_retries=3,
    retry_on_timeout=True
)

# SQLite connection helper
DB_PATH = 'resources/db5.db'

# Global variables for caching common terms
common_terms = []
last_updated = 0


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_r_vector(text):
    """Parse R-style vector strings into Python lists."""
    if not text:
        return []
    matches = re.findall(r'"([^"]*)"', text)
    return matches if matches else [text]


def clean_result(result):
    """Clean R-style vectors and HTML tags from Elasticsearch results"""
    cleaned = result.copy()
    html_pattern = re.compile(r'<[^>]+>')

    def remove_tags(text):
        if isinstance(text, list):
            return [html_pattern.sub('', item) for item in text]
        return html_pattern.sub('', str(text))

    # Handle image arrays
    if 'image' in cleaned and isinstance(cleaned['image'], list):
        cleaned['image'] = cleaned['image'][0] if cleaned['image'] else None

    # Clean ingredients
    if 'ingredients' in cleaned:
        if isinstance(cleaned['ingredients'], str):
            cleaned['ingredients'] = parse_r_vector(cleaned['ingredients'])
        elif isinstance(cleaned['ingredients'], list):
            cleaned['ingredients'] = parse_r_vector(' '.join(cleaned['ingredients']))
        cleaned['ingredients'] = remove_tags(cleaned['ingredients'])

    # Clean highlights
    if 'highlights' in cleaned:
        highlights = cleaned['highlights']
        for field in list(highlights.keys()):
            highlights[field] = remove_tags(highlights[field])
            if field in ['RecipeIngredientParts', 'RecipeInstructions']:
                joined = ' '.join(highlights[field])
                parsed = parse_r_vector(joined)
                highlights[field] = parsed

    return cleaned


def correct_typos(query):
    spell = SpellChecker()
    words = query.lower().split()
    corrected_words = []
    common_terms = get_common_food_terms()

    for word in words:
        if len(word) <= 2:
            corrected_words.append(word)
            continue

        closest_term = None
        min_distance = float('inf')

        for term in common_terms:
            distance = Levenshtein.distance(word, term.lower())
            if distance < min(3, len(word) // 2) and distance < min_distance:
                min_distance = distance
                closest_term = term

        if closest_term:
            corrected_words.append(closest_term)
        else:
            correction = spell.correction(word)
            corrected_words.append(correction if correction else word)

    return ' '.join(corrected_words)


def get_common_food_terms():
    """Fetch and cache common food terms from Elasticsearch, refreshing every hour."""
    global common_terms, last_updated
    if time.time() - last_updated > 3600:  # Refresh every hour
        try:
            ingredient_aggs = es_client.search(
                index="recipes",
                body={"size": 0,
                      "aggs": {"ingredients": {"terms": {"field": "RecipeIngredientParts.keyword", "size": 1000}}}}
            )
            keyword_aggs = es_client.search(
                index="recipes",
                body={"size": 0, "aggs": {"keywords": {"terms": {"field": "Keywords.keyword", "size": 1000}}}}
            )

            ingredients = [bucket["key"] for bucket in ingredient_aggs["aggregations"]["ingredients"]["buckets"]]
            keywords = [bucket["key"] for bucket in keyword_aggs["aggregations"]["keywords"]["buckets"]]

            common_terms = set()
            for term_list in [ingredients, keywords]:
                for term in term_list:
                    for word in term.split():
                        if len(word) > 2:
                            common_terms.add(word.lower())
            common_terms = list(common_terms)
            last_updated = time.time()
        except Exception as e:
            current_app.logger.error(f"Error updating common food terms: {str(e)}")
    return common_terms

def correct_typos(query):
    """Correct typos in the query using cached common food terms and a spell checker."""
    spell = SpellChecker()
    words = query.lower().split()
    corrected_words = []
    common_terms = get_common_food_terms()

    for word in words:
        if len(word) <= 2:
            corrected_words.append(word)
            continue

        closest_term = None
        min_distance = float('inf')

        for term in common_terms:
            distance = Levenshtein.distance(word, term.lower())
            if distance < min(3, len(word) // 2) and distance < min_distance:
                min_distance = distance
                closest_term = term

        if closest_term:
            corrected_words.append(closest_term)
        else:
            correction = spell.correction(word)
            corrected_words.append(correction if correction else word)

    return ' '.join(corrected_words)

def search_in_elasticsearch(query, page=1, size=10, has_rating=False, has_image=False, has_cooktime=False):
    """Search Elasticsearch with enhanced scoring based on ratings and review counts."""
    from_val = (page - 1) * size

    search_body = {
        "from": from_val,
        "size": size,
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"Name": {"query": query, "boost": 3.0}}},
                            {"match": {"RecipeIngredientParts": {"query": query, "boost": 2.0}}},
                            {"match": {"RecipeInstructions": {"query": query, "boost": 1.5}}},
                            {"match": {"Description": {"query": query, "boost": 1.0}}},
                            {"match": {"Keywords": {"query": query, "boost": 1.0}}},
                            {"match": {"Name": {"query": query, "fuzziness": "AUTO", "boost": 1.0}}},
                            {"match": {"RecipeIngredientParts": {"query": query, "fuzziness": "AUTO", "boost": 0.8}}}
                        ],
                        "minimum_should_match": 1,
                        "filter": []
                    }
                },
                "script_score": {
                    "script": {
                        "source": """
                            double rating_factor = 1.0;
                            if (doc['AggregatedRating'].size() > 0 && doc['AggregatedRating'].value > 0) {
                                rating_factor = doc['AggregatedRating'].value / 5.0;
                            }
                            double review_factor = 1.0;
                            if (doc['ReviewCount'].size() > 0) {
                                review_factor = 1 + Math.log(1 + doc['ReviewCount'].value);
                            }
                            return _score * rating_factor * review_factor;
                        """
                    }
                }
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

    # Add filters based on parameters
    if has_rating:
        search_body["query"]["function_score"]["query"]["bool"]["filter"].append({"exists": {"field": "AggregatedRating"}})
    if has_image:
        search_body["query"]["function_score"]["query"]["bool"]["filter"].append({"exists": {"field": "Images"}})
    if has_cooktime:
        search_body["query"]["function_score"]["query"]["bool"]["filter"].append({"exists": {"field": "CookTime"}})

    result = es_client.search(index="recipes", body=search_body)
    total_hits = result["hits"]["total"]["value"]
    hits = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]
        score = hit["_score"]
        highlights = hit.get("highlight", {})

        recipe = {
            "id": hit["_id"],
            "name": source.get("Name"),
            "description": (source.get("Description", "")[:150] + "...") if source.get("Description") else "",
            "image": source.get("Images", ["https://example.com/default-image.jpg"])[0],
            "rating": source.get("AggregatedRating", 0.0),
            "reviewCount": source.get("ReviewCount", 0),
            "cookTime": source.get("CookTime", "Not Specified"),
            "ingredients": source.get("RecipeIngredientParts", []),
            "highlights": highlights,
            "score": score,
            "category": source.get("RecipeCategory"),
            "yield": source.get("RecipeYield", "Unknown")
        }

        cleaned_recipe = clean_result(recipe)
        cleaned_recipe['ingredients'] = cleaned_recipe['ingredients'][:5]
        hits.append(cleaned_recipe)

    return {"hits": hits, "total": total_hits}

@search_bp.route('/api/search', methods=['GET'])
def search_recipes():
    """API endpoint to search for recipes."""
    query = request.args.get('q', '')
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))
    has_rating = request.args.get('has_rating', 'false').lower() == 'true'
    has_image = request.args.get('has_image', 'false').lower() == 'true'
    has_cooktime = request.args.get('has_cooktime', 'false').lower() == 'true'

    if not query:
        return jsonify({'error': 'Search query is required'}), 400

    original_query = query
    corrected_query = None

    if len(query) > 2:
        corrected_query = correct_typos(query)
        if corrected_query and corrected_query == query:
            corrected_query = None

    try:
        search_results = search_in_elasticsearch(query, page, size, has_rating, has_image, has_cooktime)

        if corrected_query and (search_results['total'] < 3):
            corrected_results = search_in_elasticsearch(corrected_query, page, size, has_rating, has_image, has_cooktime)
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

        return jsonify({
            'results': search_results['hits'],
            'total': search_results['total'],
            'page': page,
            'size': size,
            'suggestion': {'text': corrected_query} if corrected_query else None
        }), 200

    except Exception as e:
        current_app.logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'An error occurred during search'}), 500

def setup_elasticsearch_index():
    """Set up the Elasticsearch index with mappings and settings."""
    index_name = "recipes"
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    mapping = {
        "mappings": {
            "properties": {
                "RecipeId": {"type": "keyword"},
                "Name": {"type": "text", "analyzer": "english", "fields": {"keyword": {"type": "keyword"}}},
                "AuthorId": {"type": "keyword"},
                "AuthorName": {"type": "keyword"},
                "CookTime": {"type": "keyword"},
                "PrepTime": {"type": "keyword"},
                "TotalTime": {"type": "keyword"},
                "DatePublished": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM-dd'T'HH:mm:ss||epoch_millis"},
                "Description": {"type": "text", "analyzer": "english"},
                "Images": {"type": "keyword"},
                "RecipeCategory": {"type": "keyword"},
                "Keywords": {"type": "text"},
                "RecipeIngredientQuantities": {"type": "text"},
                "RecipeIngredientParts": {"type": "text", "analyzer": "english"},
                "AggregatedRating": {"type": "float"},
                "ReviewCount": {"type": "integer"},
                "Calories": {"type": "float"},
                "FatContent": {"type": "float"},
                "SaturatedFatContent": {"type": "float"},
                "CholesterolContent": {"type": "float"},
                "SodiumContent": {"type": "float"},
                "CarbohydrateContent": {"type": "float"},
                "FiberContent": {"type": "float"},
                "SugarContent": {"type": "float"},
                "ProteinContent": {"type": "float"},
                "RecipeServings": {"type": "float"},
                "RecipeYield": {"type": "text"},
                "RecipeInstructions": {"type": "text", "analyzer": "english"}
            }
        },
        "settings": {
            "analysis": {
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    "english_stemmer": {"type": "stemmer", "language": "english"},
                    "english_possessive_stemmer": {"type": "stemmer", "language": "possessive_english"}
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

    es_client.indices.create(index=index_name, body=mapping)
    print("Elasticsearch index 'recipes' created.")

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
    print(f"Total recipes to index: {total_recipes}")

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
            # Convert RecipeId to string (Elasticsearch _id must be string)
            recipe_dict['RecipeId'] = str(recipe_dict['RecipeId'])

            # Debug output for the first recipe in each batch
            if not bulk_data:
                print(f"Batch {batch_num + 1} - First recipe ID: {recipe_dict['RecipeId']}")
                print(f"DatePublished: {recipe_dict['DatePublished']}")
                print(f"Images: {recipe_dict['Images']}")
                print(f"Keywords: {recipe_dict['Keywords']}")
                print(f"RecipeIngredientParts: {recipe_dict['RecipeIngredientParts']}")
                print(f"ReviewCount: {recipe_dict['ReviewCount']}")

            # Fix DatePublished format
            if recipe_dict.get('DatePublished'):
                try:
                    # Try common date formats; adjust based on your data
                    dt = datetime.strptime(recipe_dict['DatePublished'], "%Y-%m-%d")  # e.g., "2025-03-23"
                    recipe_dict['DatePublished'] = dt.strftime("%Y-%m-%d")
                except ValueError:
                    try:
                        dt = datetime.strptime(recipe_dict['DatePublished'], "%Y-%m-%d %H:%M:%S")  # e.g., "2025-03-23 12:00:00"
                        recipe_dict['DatePublished'] = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        print(f"Warning: Invalid date for Recipe ID {recipe_dict['RecipeId']}: {recipe_dict['DatePublished']}")
                        recipe_dict['DatePublished'] = None  # Set to None if unparseable

            # Split comma-separated fields safely
            for field in ['Images', 'Keywords', 'RecipeIngredientParts']:
                if recipe_dict.get(field) and isinstance(recipe_dict[field], str):
                    recipe_dict[field] = recipe_dict[field].split(',')
                else:
                    recipe_dict[field] = []  # Default to empty list if null or invalid

            # Convert ReviewCount to integer
            if recipe_dict.get('ReviewCount') is not None:
                try:
                    recipe_dict['ReviewCount'] = int(float(recipe_dict['ReviewCount']))  # REAL to int
                except ValueError:
                    print(f"Warning: Invalid ReviewCount for Recipe ID {recipe_dict['RecipeId']}: {recipe_dict['ReviewCount']}")
                    recipe_dict['ReviewCount'] = 0  # Default to 0 if invalid

            # Add to bulk data
            bulk_data.append({"index": {"_index": "recipes", "_id": recipe_dict['RecipeId']}})
            bulk_data.append(recipe_dict)

        # Execute bulk indexing if we have data
        if bulk_data:
            response = es_client.bulk(body=bulk_data, refresh=True)
            if response["errors"]:
                print(f"Batch {batch_num + 1} failed with errors:")
                for item in response["items"]:
                    if "index" in item and "error" in item["index"]:
                        print(f"Recipe ID {item['index']['_id']}: {item['index']['error']}")
                raise Exception("Bulk indexing failed. Check logs for details.")
            else:
                stats = es_client.cat.indices(index="recipes", format="json")
                doc_count = stats[0]['docs.count'] if stats else "unknown"
                print(f"Indexed batch {batch_num + 1}/{total_batches} ({len(recipes)} recipes), Total docs: {doc_count}")

    conn.close()
    print(f"Completed indexing {total_recipes} recipes")


@search_bp.route('/api/indexes', methods=['GET'])
def get_indexes():
    """API endpoint to list all Elasticsearch indexes."""
    try:
        response = es_client.cat.indices(format="json")
        if not response:
            return jsonify({"message": "No indexes found in Elasticsearch", "indexes": []}), 200
        indexes = [
            {
                "name": index["index"],
                "doc_count": index["docs.count"],
                "size": index["store.size"],
                "status": index["status"]
            }
            for index in response
        ]
        return jsonify({"indexes": indexes}), 200
    except Exception as e:
        current_app.logger.error(f"Failed to retrieve indexes: {str(e)}")
        return jsonify({"error": str(e)}), 500

import click

@click.command("index-recipes")
def index_recipes_command():
    """CLI command to index recipes."""
    start_time = time.time()
    index_recipes_from_sqlite()
    elapsed_time = time.time() - start_time
    print(f"Indexing completed in {elapsed_time:.2f} seconds")

def register_commands(app):
    """Register CLI commands with the Flask app."""
    app.cli.add_command(index_recipes_command)

# Debug endpoint for troubleshooting
@search_bp.route('/api/debug/recipe/<recipe_id>', methods=['GET'])
def debug_recipe(recipe_id):
    """API endpoint to debug a specific recipe by ID."""
    try:
        if not es_client.indices.exists(index="recipes"):
            return jsonify({'error': 'Recipes index not found'}), 404
        result = es_client.get(index="recipes", id=recipe_id)
        return jsonify(result['_source']), 200
    except Exception as e:
        current_app.logger.error(f"Debug failed for recipe {recipe_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500