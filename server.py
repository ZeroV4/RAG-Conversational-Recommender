#!/usr/bin/env python3
"""Fully schema-driven local RAG conversational recommender."""
import json, mimetypes, os, re, time, urllib.error, urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np
from sentence_transformers import SentenceTransformer
from build_index import MODEL_NAME, DATA_DIR, configuration_digest, format_value, item_text, load_catalogue, load_catalogue_schema, validate_catalogue_schema

ROOT = Path(__file__).resolve().parent
CATALOGUE_PATH, CATALOGUE_SCHEMA_PATH = ROOT / "catalogue.json", ROOT / "catalogue_schema.json"
USER_SCHEMA_PATH, CONTEXT_SCHEMA_PATH = ROOT / "user_model.json", ROOT / "context_model.json"
EMBEDDINGS_PATH, MANIFEST_PATH = DATA_DIR / "item_embeddings.npy", DATA_DIR / "manifest.json"
OLLAMA_URL, GENERATION_MODEL = "http://127.0.0.1:11434/api/chat", "llama3.2:1b"
TOP_K, SEMANTIC_WEIGHT, USER_WEIGHT, CONTEXT_WEIGHT = 4, .60, .25, .15
MIN_QUERY_SIMILARITY, MAX_HISTORY_MESSAGES, TEMPERATURE = .20, 12, .2
SYSTEM_PROMPT = """You are a catalogue-based conversational recommender.
Recommend ONLY items listed under RETRIEVED CANDIDATE ITEMS. Begin a recommendation with `Recommended catalogue item: [ID] Exact title`. Reproduce the ID, title, and description exactly. Present the remaining supplied attributes on separate labelled lines. Do not invent items, attributes, facts, activities, or instructions. Explain the recommendation only from supplied candidate data, user model, context model, match reasons, and scores. If none is suitable, say exactly: I could not find a suitable item in the catalogue. Never combine that refusal with a recommendation. Be concise and acknowledge relevant trade-offs without pressuring the user."""

def load_json(path): return json.loads(path.read_text(encoding="utf-8"))
def terms(value):
    if isinstance(value, list): value = " ".join(str(v) for v in value)
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))
def item_values(item, fields):
    values = []
    for field in fields:
        value = item.get(field); values.extend(value if isinstance(value, list) else [value] if value is not None else [])
    return values

def validate_model_schema(schema, name, catalogue_schema):
    if not isinstance(schema, dict) or not isinstance(schema.get("fields"), list): raise ValueError(f"{name} must contain a fields array.")
    catalogue_fields, seen = {field["id"] for field in catalogue_schema["fields"]}, set()
    allowed_types, allowed_matches = {"text", "number", "select", "tags", "boolean"}, {None, "exact", "ordinal", "token_overlap", "prefer_smaller"}
    allowed_operators = {"maximum", "minimum", "equals"}
    for field in schema["fields"]:
        if not isinstance(field, dict) or not field.get("id"): raise ValueError(f"Every {name} field requires an id.")
        if field["id"] in seen: raise ValueError(f"Duplicate field: {field['id']}.")
        seen.add(field["id"])
        if field.get("type") not in allowed_types: raise ValueError(f"Unsupported type for {field['id']}.")
        if field.get("matching") not in allowed_matches: raise ValueError(f"Unsupported matching method for {field['id']}.")
        unknown = set(field.get("item_fields", [])) - catalogue_fields
        if unknown: raise ValueError(f"{field['id']} refers to undefined catalogue fields: {', '.join(sorted(unknown))}.")
        constraint = field.get("constraint")
        if constraint and constraint.get("item_field") not in catalogue_fields: raise ValueError(f"{field['id']} has a constraint on an undefined catalogue field.")
        if constraint and constraint.get("operator") not in allowed_operators: raise ValueError(f"{field['id']} has an unsupported constraint operator.")
        if field.get("matching") == "ordinal" and not field.get("options"): raise ValueError(f"{field['id']} requires options for ordinal matching.")
        if field.get("type") == "select" and not field.get("options"): raise ValueError(f"{field['id']} requires options for a select control.")
        try: weight = float(field.get("weight", 1))
        except (TypeError, ValueError): raise ValueError(f"{field['id']} has an invalid weight.")
        if weight < 0: raise ValueError(f"{field['id']} weight cannot be negative.")
    return schema

def save_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def validate_model(values, schema, name):
    if not isinstance(values, dict): raise ValueError(f"{name} must be a JSON object.")
    allowed = {field["id"] for field in schema["fields"]}; unknown = set(values) - allowed
    if unknown: raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}.")
    result = {}
    for field in schema["fields"]:
        value, kind = values.get(field["id"], field.get("default")), field["type"]
        if kind in {"text", "select"}:
            if not isinstance(value, str): raise ValueError(f"{field['id']} must be text.")
            value = value.strip()[:500]
        elif kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)): raise ValueError(f"{field['id']} must be a number.")
            if value < field.get("minimum", value) or value > field.get("maximum", value): raise ValueError(f"{field['id']} is outside its permitted range.")
        elif kind == "tags":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value): raise ValueError(f"{field['id']} must be an array of text values.")
            value = [v.strip()[:160] for v in value[:20] if v.strip()]
        elif kind == "boolean" and not isinstance(value, bool): raise ValueError(f"{field['id']} must be true or false.")
        if kind == "select" and value not in field.get("options", []): raise ValueError(f"{field['id']} has an unsupported value.")
        result[field["id"]] = value
    return result

def validate_messages(value):
    if not isinstance(value, list) or not value: raise ValueError("messages must be a non-empty array.")
    result = []
    for message in value[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}: raise ValueError("Invalid message role.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip(): raise ValueError("Message content must be non-empty text.")
        result.append({"role": message["role"], "content": content.strip()[:4000]})
    if result[-1]["role"] != "user": raise ValueError("The final message must be the current user query.")
    return result

def contextualised_query(messages):
    current = messages[-1]["content"]
    if not re.search(r"\b(this|that|it|its|another|one|ones|them|those|these|else|alternative)\b", current, re.I): return current, False
    previous = next((m["content"] for m in reversed(messages[:-1]) if m["role"] == "user"), None)
    return (f"Previous user request: {previous}\nCurrent follow-up: {current}", True) if previous else (current, False)

def previous_ids_to_exclude(messages):
    if not re.search(r"\b(another|alternative|different|else)\b", messages[-1]["content"], re.I): return set()
    return {identifier for message in messages[:-1] if message["role"] == "assistant" for identifier in re.findall(r"\[([^\]\n]+)\]", message["content"])}

def field_score(item, value, field):
    if value in field.get("neutral_values", []) or value in (None, "", []): return None
    actual_values, method = item_values(item, field.get("item_fields", [])), field.get("matching")
    if not actual_values or not method: return None
    if method == "exact":
        wanted = set(value if isinstance(value, list) else [value]); return len(wanted & set(actual_values)) / max(len(wanted), 1)
    if method == "ordinal":
        options, actual = field.get("options", []), str(actual_values[0])
        if value not in options or actual not in options or len(options) < 2: return 0.
        return 1 - abs(options.index(value) - options.index(actual)) / (len(options) - 1)
    if method == "token_overlap":
        wanted = terms(value); return len(wanted & terms(actual_values)) / max(len(wanted), 1)
    if method == "prefer_smaller": return max(0., 1 - float(actual_values[0]) / max(float(value), 1.))
    return None

def model_score(item, values, schema):
    components, reasons = [], []
    for field in schema["fields"]:
        score = field_score(item, values.get(field["id"]), field)
        if score is None: continue
        components.append((score, float(field.get("weight", 1))))
        if score >= .75: reasons.append(f"strong {field['label'].lower()} match")
        elif score > 0: reasons.append(f"partial {field['label'].lower()} match")
    if not components: return .5, reasons
    total = sum(weight for _, weight in components)
    return sum(score * weight for score, weight in components) / total, reasons

def constraint_failures(item, values, schema):
    failures = []
    for field in schema["fields"]:
        rule, expected = field.get("constraint"), values.get(field["id"])
        if not rule or expected in (None, "", []): continue
        actual, operator = item.get(rule.get("item_field")), rule.get("operator")
        if actual is None: continue
        failed = (operator == "maximum" and actual > expected) or (operator == "minimum" and actual < expected) or (operator == "equals" and actual != expected)
        if failed: failures.append(f"{field['label']}: item value {actual}, required {operator} {expected}")
    return failures

class Recommender:
    def __init__(self):
        self.catalogue_schema = load_catalogue_schema(); self.items = load_catalogue(self.catalogue_schema)
        self.user_schema = validate_model_schema(load_json(USER_SCHEMA_PATH), "user_model.json", self.catalogue_schema)
        self.context_schema = validate_model_schema(load_json(CONTEXT_SCHEMA_PATH), "context_model.json", self.catalogue_schema)
        self.identity_field, self.title_field, self.description_field = (self.catalogue_schema[k] for k in ("identity_field", "title_field", "description_field"))
        self.embedding_model = SentenceTransformer(MODEL_NAME); self.ensure_index()
    def replace_models(self, user_schema, context_schema):
        user_schema = validate_model_schema(user_schema, "user_model.json", self.catalogue_schema)
        context_schema = validate_model_schema(context_schema, "context_model.json", self.catalogue_schema)
        save_json_atomic(USER_SCHEMA_PATH, user_schema); save_json_atomic(CONTEXT_SCHEMA_PATH, context_schema)
        self.user_schema, self.context_schema = user_schema, context_schema
    def replace_catalogue(self, catalogue_schema, items):
        catalogue_schema = validate_catalogue_schema(catalogue_schema)
        items = load_catalogue(catalogue_schema, path=_temporary_json(items))
        valid_fields = {field["id"] for field in catalogue_schema["fields"]}
        user_schema = sanitise_model_mappings(self.user_schema, valid_fields)
        context_schema = sanitise_model_mappings(self.context_schema, valid_fields)
        validate_model_schema(user_schema, "user_model.json", catalogue_schema)
        validate_model_schema(context_schema, "context_model.json", catalogue_schema)
        vectors = self.embedding_model.encode([item_text(item, catalogue_schema) for item in items], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        save_json_atomic(CATALOGUE_SCHEMA_PATH, catalogue_schema); save_json_atomic(CATALOGUE_PATH, items)
        save_json_atomic(USER_SCHEMA_PATH, user_schema); save_json_atomic(CONTEXT_SCHEMA_PATH, context_schema)
        self.catalogue_schema, self.items = catalogue_schema, items
        self.user_schema, self.context_schema = user_schema, context_schema
        self.identity_field, self.title_field, self.description_field = (catalogue_schema[k] for k in ("identity_field", "title_field", "description_field"))
        self.embeddings = np.asarray(vectors, dtype=np.float32); DATA_DIR.mkdir(exist_ok=True); np.save(EMBEDDINGS_PATH, self.embeddings)
        self.manifest = {"embedding_model": MODEL_NAME, "dimensions": int(self.embeddings.shape[1]), "item_count": len(items), "item_ids": [self.item_id(item) for item in items], "configuration_sha256": configuration_digest()}
        save_json_atomic(MANIFEST_PATH, self.manifest)
    def item_id(self, item): return str(item[self.identity_field])
    def envelope(self, item, semantic, user_score, context_score, final, reasons):
        excluded = {self.identity_field, self.title_field, self.description_field}
        displays = []
        for field in self.catalogue_schema["fields"]:
            if field["id"] in excluded: continue
            value = item.get(field["id"])
            if value not in (None, "", []):
                displays.append({"field": field["id"], "label": field.get("label", field["id"].replace("_", " ").capitalize()), "value": format_value(value)})
        return {"id": self.item_id(item), "title": str(item[self.title_field]), "description": str(item[self.description_field]), "item": item, "display_fields": displays, "semantic_score": round(float(semantic), 4), "user_score": round(user_score, 4), "context_score": round(context_score, 4), "final_score": round(final, 4), "match_reasons": reasons}
    def ensure_index(self):
        digest, manifest = configuration_digest(), load_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
        ids = [self.item_id(item) for item in self.items]
        stale = not EMBEDDINGS_PATH.exists() or manifest.get("configuration_sha256") != digest or manifest.get("item_ids") != ids
        if stale:
            print("Catalogue or schema changed; creating the vector index automatically.")
            vectors = self.embedding_model.encode([item_text(item, self.catalogue_schema) for item in self.items], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
            DATA_DIR.mkdir(exist_ok=True); np.save(EMBEDDINGS_PATH, np.asarray(vectors, dtype=np.float32))
            manifest = {"embedding_model": MODEL_NAME, "dimensions": int(vectors.shape[1]), "item_count": len(self.items), "item_ids": ids, "configuration_sha256": digest}
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self.manifest, self.embeddings = manifest, np.load(EMBEDDINGS_PATH)
    def retrieve(self, query, user, context):
        additions = []
        for schema, values in ((self.user_schema, user), (self.context_schema, context)):
            additions += [f"{f['label']}: {format_value(values[f['id']])}" for f in schema["fields"] if f.get("include_in_retrieval") and values.get(f["id"]) not in (None, "", [])]
        retrieval_query = "User query: " + query + (("\n" + "\n".join(additions)) if additions else "")
        vector = self.embedding_model.encode(retrieval_query, normalize_embeddings=True, convert_to_numpy=True)
        ranked, filtered = [], []
        for item, semantic in zip(self.items, self.embeddings @ np.asarray(vector, dtype=np.float32)):
            failures = constraint_failures(item, user, self.user_schema) + constraint_failures(item, context, self.context_schema)
            if failures: filtered.append({"id": self.item_id(item), "reasons": failures}); continue
            us, ur = model_score(item, user, self.user_schema); cs, cr = model_score(item, context, self.context_schema)
            final = SEMANTIC_WEIGHT * float(semantic) + USER_WEIGHT * us + CONTEXT_WEIGHT * cs
            ranked.append(self.envelope(item, semantic, us, cs, final, ur + cr))
        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked[:TOP_K], retrieval_query, filtered
    def raw_query_relevance(self, query):
        vector = self.embedding_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = self.embeddings @ np.asarray(vector, dtype=np.float32); index = int(np.argmax(scores))
        return round(float(scores[index]), 4), self.item_id(self.items[index])

def grounding_context(candidates, user, context):
    blocks = ["USER MODEL\n" + json.dumps(user, ensure_ascii=False), "CONTEXT MODEL\n" + json.dumps(context, ensure_ascii=False), "RETRIEVED CANDIDATE ITEMS"]
    for candidate in candidates:
        attributes = "\n".join(f"{entry['label']}: {entry['value']}" for entry in candidate["display_fields"])
        blocks.append(f"[{candidate['id']}] {candidate['title']}\nDescription: {candidate['description']}\n{attributes}\nScores: semantic={candidate['semantic_score']}, user={candidate['user_score']}, context={candidate['context_score']}, final={candidate['final_score']}")
    return "\n\n".join(blocks)

def call_ollama(messages):
    payload = {"model": GENERATION_MODEL, "messages": messages, "stream": False, "options": {"temperature": TEMPERATURE}}
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response: result = json.loads(response.read())
    except urllib.error.HTTPError as error: raise RuntimeError(f"Ollama returned HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    except (urllib.error.URLError, TimeoutError) as error: raise RuntimeError(f"Could not reach Ollama. Start it and install {GENERATION_MODEL}. Details: {error}") from error
    try: answer = result["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError) as error: raise RuntimeError("Ollama returned an unexpected response.") from error
    return answer, payload, result

def validate_grounded_answer(answer, candidates):
    by_id = {item["id"]: item for item in candidates}; cited = set(re.findall(r"\[([^\]\n]+)\]", answer)); refusal = "i could not find a suitable item in the catalogue."
    invalid = sorted(cited - set(by_id)); supported = sorted(identifier for identifier in cited if identifier in by_id and by_id[identifier]["title"].casefold() in answer.casefold() and by_id[identifier]["description"].casefold() in answer.casefold())
    if invalid: return False, {"accepted": False, "reason": "The answer cited IDs outside the retrieved candidates.", "invalid_ids": invalid}
    if cited and refusal in answer.casefold(): return False, {"accepted": False, "reason": "The answer combined a recommendation with a refusal."}
    if not supported: return False, {"accepted": False, "reason": "The answer lacked an exact retrieved ID, title, and description."}
    return True, {"accepted": True, "supported_ids": supported}

def safe_catalogue_response(item):
    lines = [f"Recommended catalogue item: [{item['id']}] {item['title']}", "", f"Description: {item['description']}", ""]
    lines += [f"{entry['label']}: {entry['value']}" for entry in item["display_fields"]]
    lines += [f"Why recommended: {', '.join(item['match_reasons'] or ['highest combined retrieval and personalisation score'])}"]
    return "\n".join(lines)

RECOMMENDER, STARTUP_ERROR = None, None

def _temporary_json(value):
    path = ROOT / ".catalogue-validation.tmp"
    save_json_atomic(path, value)
    return path

def sanitise_model_mappings(schema, valid_fields):
    schema = json.loads(json.dumps(schema))
    for field in schema["fields"]:
        field["item_fields"] = [name for name in field.get("item_fields", []) if name in valid_fields]
        if field.get("constraint", {}).get("item_field") not in valid_fields: field.pop("constraint", None)
    return schema

def start_recommender():
    global RECOMMENDER, STARTUP_ERROR
    try: RECOMMENDER, STARTUP_ERROR = Recommender(), None
    except Exception as error: STARTUP_ERROR = str(error)

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean = unquote(urlsplit(path).path).lstrip("/") or "index.html"; candidate = (ROOT / clean).resolve()
        return str(candidate if candidate == ROOT or ROOT in candidate.parents else ROOT / "__not_found__")
    def end_headers(self): self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff"); super().end_headers()
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/config":
            if not RECOMMENDER: self.send_json(503, {"error": STARTUP_ERROR or "Recommender unavailable."})
            else: self.send_json(200, {"catalogue_schema": RECOMMENDER.catalogue_schema, "catalogue": RECOMMENDER.items, "user_model": RECOMMENDER.user_schema, "context_model": RECOMMENDER.context_schema, "item_count": len(RECOMMENDER.items), "embedding_model": MODEL_NAME, "generation_model": GENERATION_MODEL, "top_k": TOP_K, "minimum_query_similarity": MIN_QUERY_SIMILARITY})
            return
        if path in {"/catalogue.json", "/catalogue_schema.json", "/user_model.json", "/context_model.json"} or path.startswith(("/data/", "/.venv/")): self.send_json(404, {"error": "Not found"}); return
        super().do_GET()
    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/api/catalogue":
            if not RECOMMENDER: self.send_json(503, {"error": STARTUP_ERROR or "Recommender unavailable."}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1000000: raise ValueError("Invalid request body size.")
                data = json.loads(self.rfile.read(length))
                RECOMMENDER.replace_catalogue(data.get("catalogue_schema"), data.get("catalogue"))
            except (ValueError, json.JSONDecodeError, OSError) as error: self.send_json(400, {"error": str(error)}); return
            finally:
                (ROOT / ".catalogue-validation.tmp").unlink(missing_ok=True)
            self.send_json(200, {"message": f"Catalogue saved and {len(RECOMMENDER.items)} item vectors rebuilt.", "catalogue_schema": RECOMMENDER.catalogue_schema, "catalogue": RECOMMENDER.items, "user_model": RECOMMENDER.user_schema, "context_model": RECOMMENDER.context_schema, "item_count": len(RECOMMENDER.items)}); return
        if path == "/api/models":
            if not RECOMMENDER: self.send_json(503, {"error": STARTUP_ERROR or "Recommender unavailable."}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 200000: raise ValueError("Invalid request body size.")
                data = json.loads(self.rfile.read(length))
                user_schema, context_schema = data.get("user_model"), data.get("context_model")
                RECOMMENDER.replace_models(user_schema, context_schema)
            except (ValueError, json.JSONDecodeError, OSError) as error: self.send_json(400, {"error": str(error)}); return
            self.send_json(200, {"message": "Models saved. The generated interface is ready.", "user_model": RECOMMENDER.user_schema, "context_model": RECOMMENDER.context_schema}); return
        if path != "/api/recommend": self.send_json(404, {"error": "Not found"}); return
        if not RECOMMENDER: self.send_json(503, {"error": STARTUP_ERROR or "Recommender unavailable."}); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 200000: raise ValueError("Invalid request body size.")
            data = json.loads(self.rfile.read(length)); user = validate_model(data.get("user_model"), RECOMMENDER.user_schema, "user_model"); context = validate_model(data.get("context_model"), RECOMMENDER.context_schema, "context_model"); messages, debug = validate_messages(data.get("messages")), data.get("debug", False)
            if not isinstance(debug, bool): raise ValueError("debug must be true or false.")
        except (ValueError, json.JSONDecodeError) as error: self.send_json(400, {"error": str(error)}); return
        started = time.perf_counter(); effective, used_history = contextualised_query(messages); raw_similarity, nearest_id = RECOMMENDER.raw_query_relevance(effective)
        config = {"minimum_query_similarity": MIN_QUERY_SIMILARITY, "top_k": TOP_K, "semantic_weight": SEMANTIC_WEIGHT, "user_weight": USER_WEIGHT, "context_weight": CONTEXT_WEIGHT, "temperature": TEMPERATURE, "history_limit": MAX_HISTORY_MESSAGES}
        base_debug = {"configuration": config, "validated_user_model": user, "validated_context_model": context, "retrieval_input": {"current_query": messages[-1]["content"], "effective_query": effective, "used_previous_user_query": used_history}, "raw_query_relevance": {"maximum_similarity": raw_similarity, "nearest_item_id": nearest_id, "accepted": raw_similarity >= MIN_QUERY_SIMILARITY}}
        if raw_similarity < MIN_QUERY_SIMILARITY:
            base_debug["decision"] = "Local out-of-scope refusal; Ollama was not contacted."
            self.send_json(200, {"reply": "This query appears unrelated to the items available in this recommendation catalogue. Please ask for a recommendation connected to the catalogue topics.", "model": "Local relevance gate", "generation_source": "Local out-of-scope refusal", "candidates": [], "grounded": False, "llm_answer_accepted": None, "elapsed_ms": round((time.perf_counter()-started)*1000), "debug": base_debug if debug else None}); return
        candidates, retrieval_query, filtered = RECOMMENDER.retrieve(effective, user, context); excluded = previous_ids_to_exclude(messages)
        if excluded: candidates = [item for item in candidates if item["id"] not in excluded]
        inspection = {**base_debug, "retrieval_query": retrieval_query, "filtered_items": filtered, "excluded_previous_item_ids": sorted(excluded), "ranked_candidates": candidates}
        if not candidates: self.send_json(200, {"reply": "No catalogue items satisfy the current constraints.", "model": "Local constraint gate", "candidates": [], "grounded": False, "debug": inspection if debug else None}); return
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "system", "content": grounding_context(candidates, user, context)}, *messages]
        try:
            answer, request, response = call_ollama(llm_messages); accepted, validation = validate_grounded_answer(answer, candidates); source = "Local LLM"
            if not accepted: answer, source = safe_catalogue_response(candidates[0]), "Validated catalogue fallback"
            inspection.update({"ollama_request": request, "ollama_response": response, "grounding_validation": validation, "displayed_response": answer})
        except RuntimeError as error:
            inspection.update({"ollama_request": {"model": GENERATION_MODEL, "messages": llm_messages}, "ollama_response": {"error": str(error)}}); self.send_json(502, {"error": str(error), "candidates": candidates, "debug": inspection if debug else None}); return
        self.send_json(200, {"reply": answer, "model": response.get("model", GENERATION_MODEL), "generation_source": source, "candidates": candidates, "grounded": True, "llm_answer_accepted": accepted, "elapsed_ms": round((time.perf_counter()-started)*1000), "debug": inspection if debug else None})
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, fmt, *args): print(f"[{self.log_date_time_string()}] {fmt % args}")

if __name__ == "__main__":
    start_recommender(); port = int(os.environ.get("PORT", "8080")); mimetypes.add_type("text/javascript", ".js")
    print(f"RAG Conversational Recommender: http://localhost:{port}\nConfiguration ready: {RECOMMENDER is not None}")
    if STARTUP_ERROR: print(f"Configuration error: {STARTUP_ERROR}")
    print(f"Local LLM: {GENERATION_MODEL} through {OLLAMA_URL}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
