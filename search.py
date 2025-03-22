import time
import re
import sqlite3
import Levenshtein
from spellchecker import SpellChecker
from flask import Blueprint, request, jsonify, current_app
from flask_cors import CORS
from elasticsearch import Elasticsearch

search_bp = Blueprint('search', __name__)
CORS(search_bp, resources={r"/api/*": {"origins": "*"}})

# Configure Elasticsearch client
es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "Vb0GGDOzMd-khfxnUkYp"),
    ca_certs="http_ca.crt",
    request_timeout=30,
    verify_certs=True,
    ssl_show_warn=False,
    max_retries=3,
    retry_on_timeout=True
)

# SQLite connection helper
DB_PATH = 'Resources/db.db'

# Global variables for caching common terms
common_terms = []
last_updated = 0


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_r_vector(text):
    """Parse R-style vector strings into Python lists."""
    if not text or text.lower() == "character(0)":
        return []

    cleaned = text.replace('c(', '').replace(')', '')
    return [item.strip(' "\'') for item in cleaned.split(', ') if item.strip()]


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
    """Execute Elasticsearch query with proper result cleaning."""
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
                            {"match": {"Keywords": {"query": query, "boost": 1.0}}}
                        ],
                        "filter": []
                    }
                },
                "script_score": {
                    "script": {
                        "source": """
                            double rating_factor = doc['AggregatedRating'].size() > 0 ? 
                                (doc['AggregatedRating'].value / 5.0) : 1.0;
                            double review_factor = doc['ReviewCount'].size() > 0 ? 
                                (1 + Math.log(1 + doc['ReviewCount'].value)) : 1.0;
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

    # Add filters
    filter_conditions = []
    if has_rating:
        filter_conditions.append({"exists": {"field": "AggregatedRating"}})
    if has_image:
        filter_conditions.append({"exists": {"field": "Images"}})
    if has_cooktime:
        filter_conditions.append({"exists": {"field": "CookTime"}})

    if filter_conditions:
        search_body["query"]["function_score"]["query"]["bool"]["filter"] = filter_conditions

    try:
        result = es_client.search(index="recipes", body=search_body)
        total_hits = result["hits"]["total"]["value"]
        hits = []

        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]  # Get the raw score
            highlights = hit.get("highlight", {})

            recipe = {
                "id": hit["_id"],
                "name": source.get("Name"),
                "description": (source.get("Description", "")[:150] + "...") if source.get("Description") else "",
                "image": source.get("Images", []),
                "tags": source.get("Keywords", []),
                "rating": source.get("AggregatedRating", 0.0),
                "reviewCount": source.get("ReviewCount", 0),
                "cookTime": source.get("CookTime", "Not Specified"),
                "ingredients": source.get("RecipeIngredientParts", []),
                "highlights": highlights,
                "score": round(score, 2),  # Keep and format the score
                "category": source.get("RecipeCategory"),
                "yield": source.get("RecipeYield", "Unknown")
            }

            cleaned_recipe = clean_result(recipe)
            cleaned_recipe['ingredients'] = cleaned_recipe['ingredients'][:5]
            hits.append(cleaned_recipe)

        return {"hits": hits, "total": total_hits}

    except Exception as e:
        current_app.logger.error(f"Search error: {str(e)}")
        return {"hits": [], "total": 0}

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

def index_recipes_from_sqlite():
    """Index recipes from SQLite into Elasticsearch."""
    setup_elasticsearch_index()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM recipes")
    recipes = cursor.fetchall()
    bulk_data = []

    for recipe in recipes:
        recipe_dict = dict(recipe)
        if "description" in recipe_dict:
            recipe_dict["Description"] = recipe_dict.pop("description")

        # Parse R-style vectors and handle "character(0)"
        vector_fields = [
            'RecipeIngredientParts', 'Keywords', 'Images',
            'RecipeInstructions', 'RecipeIngredientQuantities'
        ]

        for field in vector_fields:
            if recipe_dict.get(field):
                if recipe_dict[field] == "character(0)":
                    recipe_dict[field] = []
                else:
                    recipe_dict[field] = parse_r_vector(recipe_dict[field])

        # Convert numeric fields
        numeric_fields = [
            'AggregatedRating', 'ReviewCount', 'Calories',
            'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent',
            'SugarContent', 'ProteinContent', 'RecipeServings'
        ]

        for field in numeric_fields:
            value = recipe_dict.get(field)
            if value is not None:
                try:
                    recipe_dict[field] = float(value)
                except (ValueError, TypeError):
                    recipe_dict[field] = None

        bulk_data.append({"index": {"_index": "recipes", "_id": str(recipe_dict['RecipeId'])}})
        bulk_data.append(recipe_dict)

    if bulk_data:
        es_client.bulk(body=bulk_data, refresh=True)

    conn.close()
    print(f"Indexed {len(recipes)} recipes")

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
        result = es_client.get(index="recipes", id=recipe_id)
        return jsonify(result['_source']), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500