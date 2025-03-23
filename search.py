import click
import time
import re
import sqlite3
import Levenshtein
from collections import defaultdict
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
    basic_auth=("elastic", "KEY"),
    ca_certs="PATH_TO_http_ca",
    request_timeout=30,
    verify_certs=True,
    ssl_show_warn=False,
    max_retries=3,
    retry_on_timeout=True
)

DB_PATH = 'resources/db5.db'

# Global variables for caching common terms and bigrams
common_terms = []
common_terms_dict = defaultdict(list)  # For faster lookup by first letter
common_bigrams = set()  # Store bigrams from food terms
last_updated = 0


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_r_vector(text):
    if not text:
        return []
    matches = re.findall(r'"([^"]*)"', text)
    return matches if matches else [text]


def clean_image_link(image_list):
    if not image_list:
        return ""
    joined = ",".join(part.strip() for part in image_list)
    joined = joined.replace('c("', '').replace('")', '').replace('"', '')
    lower_joined = joined.lower()
    jpg_index = lower_joined.find('.jpg')
    if jpg_index != -1:
        joined = joined[:jpg_index + 4]
    if joined == "character(0)" or not joined:
        return ""
    return joined


def clean_result(result):
    cleaned = result.copy()
    html_pattern = re.compile(r'<[^>]+>')

    def remove_tags(text):
        if isinstance(text, list):
            return [html_pattern.sub('', item) for item in text]
        return html_pattern.sub('', str(text))

    if 'image' in cleaned:
        if isinstance(cleaned['image'], list):
            cleaned['image'] = clean_image_link(cleaned['image'])
        else:
            cleaned['image'] = clean_image_link([cleaned['image']])

    if 'ingredients' in cleaned:
        if isinstance(cleaned['ingredients'], str):
            cleaned['ingredients'] = parse_r_vector(cleaned['ingredients'])
        elif isinstance(cleaned['ingredients'], list):
            cleaned['ingredients'] = parse_r_vector(' '.join(cleaned['ingredients']))
        cleaned['ingredients'] = remove_tags(cleaned['ingredients'])

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
    global common_terms, common_terms_dict, common_bigrams, last_updated
    if time.time() - last_updated > 3600:
        try:
            ingredient_aggs = es_client.search(
                index="recipes",
                body={"size": 0,
                      "aggs": {"ingredients": {"terms": {"field": "RecipeIngredientParts.keyword", "size": 2000}}}}
            )
            keyword_aggs = es_client.search(
                index="recipes",
                body={"size": 0, "aggs": {"keywords": {"terms": {"field": "Keywords.keyword", "size": 2000}}}}
            )

            ingredients = [bucket["key"] for bucket in ingredient_aggs["aggregations"]["ingredients"]["buckets"]]
            keywords = [bucket["key"] for bucket in keyword_aggs["aggregations"]["keywords"]["buckets"]]

            common_terms = set()
            common_terms_dict = defaultdict(list)
            common_bigrams = set()

            for term_list in [ingredients, keywords]:
                for term in term_list:
                    words = [w.strip().lower() for w in term.split() if len(w.strip()) > 2]
                    for word in words:
                        common_terms.add(word)
                        common_terms_dict[word[0]].append(word)
                    # Generate bigrams from consecutive words
                    for i in range(len(words) - 1):
                        bigram = f"{words[i]} {words[i + 1]}"
                        common_bigrams.add(bigram)
            common_terms = list(common_terms)
            last_updated = time.time()
        except Exception as e:
            current_app.logger.error(f"Error updating common food terms: {str(e)}")
            common_terms = []
            common_terms_dict = defaultdict(list)
            common_bigrams = set()
    return common_terms_dict


def correct_typos(query):
    spell = SpellChecker(distance=1)  # Faster with reduced distance
    common_terms_local = get_common_food_terms()

    corrected_parts = {"Name": [], "RecipeIngredientParts": [], "RecipeInstructions": []}
    words = query.lower().split()
    i = 0

    while i < len(words):
        # Try bigram correction first
        if i < len(words) - 1:
            bigram = f"{words[i]} {words[i + 1]}"
            prefix1, term1 = (words[i], words[i][4:]) if words[i].startswith("ing:") or words[i].startswith(
                "cook:") else (None, words[i])
            prefix2, term2 = (words[i + 1], words[i + 1][4:]) if words[i + 1].startswith("ing:") or words[
                i + 1].startswith("cook:") else (None, words[i + 1])
            bigram_term = f"{term1} {term2}"

            # Determine target based on prefixes (use first word's prefix if present)
            if prefix1 == "ing:" or prefix2 == "ing:":
                target = "RecipeIngredientParts"
                prefix = "ing:"
            elif prefix1 == "cook:" or prefix2 == "cook:":
                target = "RecipeInstructions"
                prefix = "cook:"
            else:
                target = "Name"
                prefix = None

            # Check if bigram needs correction
            if len(term1) > 2 and len(term2) > 2 and bigram_term not in common_bigrams:
                closest_bigram = None
                min_distance = min(5, (len(term1) + len(term2)) // 2)  # Adjusted for bigram length

                for correct_bigram in common_bigrams:
                    distance = Levenshtein.distance(bigram_term, correct_bigram)
                    if distance < min_distance:
                        min_distance = distance
                        closest_bigram = correct_bigram

                if closest_bigram:
                    word1, word2 = closest_bigram.split()
                    corrected_parts[target].append(f"{prefix}{word1}" if prefix else word1)
                    corrected_parts[target].append(f"{prefix}{word2}" if prefix and not prefix2 else word2)
                    i += 2  # Skip next word since we processed a bigram
                    continue

        # Fallback to unigram correction
        word = words[i]
        prefix = None
        term = word
        if word.startswith("ing:"):
            prefix = "ing:"
            term = word[4:]
            target = "RecipeIngredientParts"
        elif word.startswith("cook:"):
            prefix = "cook:"
            term = word[5:]
            target = "RecipeInstructions"
        else:
            target = "Name"

        if len(term) <= 2:
            corrected_parts[target].append(word)
            i += 1
            continue

        closest_term = None
        min_distance = min(3, len(term) // 2)
        candidates = common_terms_local.get(term[0], [])

        for candidate in candidates:
            distance = Levenshtein.distance(term, candidate)
            if distance < min_distance:
                min_distance = distance
                closest_term = candidate

        if closest_term:
            corrected_term = f"{prefix}{closest_term}" if prefix else closest_term
            corrected_parts[target].append(corrected_term)
        else:
            correction = spell.correction(term)
            if correction and Levenshtein.distance(term, correction) <= 2:
                corrected_term = f"{prefix}{correction}" if prefix else correction
            else:
                corrected_term = word
            corrected_parts[target].append(corrected_term)
        i += 1

    corrected_query = " ".join(
        corrected_parts["Name"] + corrected_parts["RecipeIngredientParts"] + corrected_parts["RecipeInstructions"]
    )
    return corrected_query if corrected_query != query.lower() else query


def parse_query(query):
    terms = {
        "Name": [],
        "RecipeIngredientParts": [],
        "RecipeInstructions": []
    }

    for word in query.split():
        if word.startswith("ing:"):
            terms["RecipeIngredientParts"].append(word[4:])
        elif word.startswith("cook:"):
            terms["RecipeInstructions"].append(word[5:])
        else:
            terms["Name"].append(word)

    return {
        "Name": " ".join(terms["Name"]),
        "RecipeIngredientParts": " ".join(terms["RecipeIngredientParts"]),
        "RecipeInstructions": " ".join(terms["RecipeInstructions"])
    }


def search_in_elasticsearch(query, page=1, size=10, has_rating=False, has_image=False, has_cooktime=False):
    from_val = (page - 1) * size

    parsed_query = parse_query(query)
    name_query = parsed_query["Name"]
    ingredient_query = parsed_query["RecipeIngredientParts"]
    instruction_query = parsed_query["RecipeInstructions"]

    base_boosts = {"Name": 3.0, "RecipeIngredientParts": 2.0, "RecipeInstructions": 1.5}
    emphasis_boost = 2.0

    search_body = {
        "from": from_val,
        "size": size,
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [],
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

    should_clauses = search_body["query"]["function_score"]["query"]["bool"]["should"]

    if name_query:
        should_clauses.extend([
            {"match": {"Name": {"query": name_query, "boost": base_boosts["Name"]}}},
            {"match": {"Name": {"query": name_query, "fuzziness": "AUTO", "boost": 1.0}}}
        ])

    if ingredient_query:
        boost = base_boosts["RecipeIngredientParts"] * emphasis_boost
        should_clauses.extend([
            {"match": {"RecipeIngredientParts": {"query": ingredient_query, "boost": boost}}},
            {"match": {"RecipeIngredientParts": {"query": ingredient_query, "fuzziness": "AUTO", "boost": 0.8}}}
        ])

    if instruction_query:
        boost = base_boosts["RecipeInstructions"] * emphasis_boost
        should_clauses.append(
            {"match": {"RecipeInstructions": {"query": instruction_query, "boost": boost}}}
        )

    if not (name_query or ingredient_query or instruction_query):
        should_clauses.extend([
            {"match": {"Name": {"query": query, "boost": base_boosts["Name"]}}},
            {"match": {"RecipeIngredientParts": {"query": query, "boost": base_boosts["RecipeIngredientParts"]}}},
            {"match": {"RecipeInstructions": {"query": query, "boost": base_boosts["RecipeInstructions"]}}},
            {"match": {"Description": {"query": query, "boost": 1.0}}},
            {"match": {"Keywords": {"query": query, "boost": 1.0}}},
            {"match": {"Name": {"query": query, "fuzziness": "AUTO", "boost": 1.0}}},
            {"match": {"RecipeIngredientParts": {"query": query, "fuzziness": "AUTO", "boost": 0.8}}}
        ])

    if has_rating:
        search_body["query"]["function_score"]["query"]["bool"]["filter"].append(
            {"exists": {"field": "AggregatedRating"}})
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
            "image": clean_image_link(source.get("Images", [])),
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


def setup_elasticsearch_index():
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
                        dt = datetime.strptime(recipe_dict['DatePublished'],
                                               "%Y-%m-%d %H:%M:%S")  # e.g., "2025-03-23 12:00:00"
                        recipe_dict['DatePublished'] = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        print(
                            f"Warning: Invalid date for Recipe ID {recipe_dict['RecipeId']}: {recipe_dict['DatePublished']}")
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
                    print(
                        f"Warning: Invalid ReviewCount for Recipe ID {recipe_dict['RecipeId']}: {recipe_dict['ReviewCount']}")
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
                print(
                    f"Indexed batch {batch_num + 1}/{total_batches} ({len(recipes)} recipes), Total docs: {doc_count}")

    conn.close()
    print(f"Completed indexing {total_recipes} recipes")

@search_bp.route('/api/search', methods=['GET'])
def search_recipes():
    query = request.args.get('q', '')
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))
    has_rating = request.args.get('has_rating', 'false').lower() == 'true'
    has_image = request.args.get('has_image', 'false').lower() == 'true'
    has_cooktime = request.args.get('has_cooktime', 'false').lower() == 'true'

    if not query:
        return jsonify({'error': 'Search query is required'}), 400

    original_query = query
    corrected_query = correct_typos(query) if len(query) > 2 else query

    try:
        search_results = search_in_elasticsearch(corrected_query, page, size, has_rating, has_image, has_cooktime)
        parsed_query = parse_query(corrected_query)

        if corrected_query != original_query and (search_results['total'] < 3):
            corrected_results = search_in_elasticsearch(corrected_query, page, size, has_rating, has_image,
                                                        has_cooktime)
            return jsonify({
                'results': search_results['hits'],
                'total': search_results['total'],
                'page': page,
                'size': size,
                'parsed_query': parsed_query,
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
            'parsed_query': parsed_query,
            'suggestion': {'text': corrected_query} if corrected_query != original_query else None
        }), 200

    except Exception as e:
        current_app.logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'An error occurred during search'}), 500

@search_bp.route('/api/indexes', methods=['GET'])
def get_indexes():
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