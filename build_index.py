#!/usr/bin/env python3
"""Build the local semantic index from a student-defined catalogue schema."""
import hashlib, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CATALOGUE_PATH, CATALOGUE_SCHEMA_PATH = ROOT / "catalogue.json", ROOT / "catalogue_schema.json"
DATA_DIR = ROOT / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FIELD_TYPES = {"text", "number", "select", "tags", "boolean"}

def load_json(path): return json.loads(path.read_text(encoding="utf-8"))

def validate_catalogue_schema(schema):
    if not isinstance(schema, dict) or not isinstance(schema.get("fields"), list) or not schema["fields"]:
        raise ValueError("catalogue_schema.json must contain a non-empty fields array.")
    field_ids = []
    for field in schema["fields"]:
        if not isinstance(field, dict) or not isinstance(field.get("id"), str) or not field["id"].strip():
            raise ValueError("Every catalogue field requires a non-empty id.")
        if field["id"] in field_ids: raise ValueError(f"Duplicate catalogue field: {field['id']}.")
        if field.get("type") not in FIELD_TYPES: raise ValueError(f"Unsupported catalogue type for {field['id']}.")
        field_ids.append(field["id"])
    for name in ("identity_field", "title_field", "description_field"):
        if schema.get(name) not in field_ids: raise ValueError(f"{name} must name a defined catalogue field.")
    embedding_fields = schema.get("embedding_fields")
    if not isinstance(embedding_fields, list) or not embedding_fields: raise ValueError("embedding_fields must be a non-empty array.")
    unknown = set(embedding_fields) - set(field_ids)
    if unknown: raise ValueError(f"Unknown embedding fields: {', '.join(sorted(unknown))}.")
    return schema

def validate_catalogue_value(value, field, position):
    kind, field_id = field["type"], field["id"]
    if value is None and not field.get("required"): return
    valid = ((kind in {"text", "select"} and isinstance(value, str)) or
             (kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)) or
             (kind == "tags" and isinstance(value, list) and all(isinstance(v, str) for v in value)) or
             (kind == "boolean" and isinstance(value, bool)))
    if not valid: raise ValueError(f"Catalogue item {position}: {field_id} must have type {kind}.")
    if kind == "select" and field.get("options") and value not in field["options"]:
        raise ValueError(f"Catalogue item {position}: unsupported value for {field_id}.")

def load_catalogue_schema(path=CATALOGUE_SCHEMA_PATH): return validate_catalogue_schema(load_json(path))

def load_catalogue(schema=None, path=CATALOGUE_PATH):
    schema, items = schema or load_catalogue_schema(), load_json(path)
    if not isinstance(items, list) or not items: raise ValueError("catalogue.json must contain a non-empty JSON array.")
    definitions = {field["id"]: field for field in schema["fields"]}
    unique_fields = [field["id"] for field in schema["fields"] if field.get("unique")]
    seen = {field_id: set() for field_id in unique_fields}
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict): raise ValueError(f"Catalogue item {position} must be a JSON object.")
        unknown = set(item) - set(definitions)
        if unknown: raise ValueError(f"Catalogue item {position} has undefined fields: {', '.join(sorted(unknown))}.")
        missing = [field_id for field_id, field in definitions.items() if field.get("required") and field_id not in item]
        if missing: raise ValueError(f"Catalogue item {position} is missing: {', '.join(missing)}.")
        for field_id, field in definitions.items():
            if field_id in item: validate_catalogue_value(item[field_id], field, position)
        for field_id in unique_fields:
            marker = json.dumps(item.get(field_id), sort_keys=True, ensure_ascii=False)
            if marker in seen[field_id]: raise ValueError(f"Duplicate value for unique catalogue field {field_id}: {item.get(field_id)}.")
            seen[field_id].add(marker)
    return items

def format_value(value):
    if isinstance(value, list): return ", ".join(str(part) for part in value)
    if isinstance(value, bool): return "yes" if value else "no"
    return str(value)

def item_text(item, schema):
    labels = {field["id"]: field.get("label", field["id"]) for field in schema["fields"]}
    return "\n".join(f"{labels[field_id]}: {format_value(item[field_id])}" for field_id in schema["embedding_fields"] if item.get(field_id) not in (None, "", []))

def configuration_digest():
    digest = hashlib.sha256()
    digest.update(CATALOGUE_PATH.read_bytes()); digest.update(CATALOGUE_SCHEMA_PATH.read_bytes())
    return digest.hexdigest()

def main():
    schema = load_catalogue_schema(); items = load_catalogue(schema)
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([item_text(item, schema) for item in items], normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    DATA_DIR.mkdir(exist_ok=True); np.save(DATA_DIR / "item_embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    identity = schema["identity_field"]
    manifest = {"embedding_model": MODEL_NAME, "dimensions": int(embeddings.shape[1]), "item_count": len(items), "item_ids": [str(item[identity]) for item in items], "configuration_sha256": configuration_digest()}
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Indexed {len(items)} catalogue items as {embeddings.shape[1]}-dimensional vectors.")

if __name__ == "__main__": main()
