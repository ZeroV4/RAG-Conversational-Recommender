# RAG Conversational Recommender System

This is a schema-driven teaching project. Students design the recommendation
data; the supplied software performs the remaining pipeline automatically:

```text
catalogue + user model + context model + query
                    ↓
local embeddings and semantic retrieval
                    ↓
hard constraints and personalised re-ranking
                    ↓
grounded prompt containing only the Top-K candidates
                    ↓
local Llama 3.2 1B through Ollama
                    ↓
conversational recommendation with item citations
```

## Visual model designer

Open the application and select **Design models**. Students can:

- define the catalogue properties and their data types;
- add, edit, or remove candidate items;
- import a JSON array of catalogue items and inspect the inferred schema;
- choose which properties represent the ID, title, and description;
- select the properties used to create semantic embeddings; and
- define the user and context models, their mappings, matching methods,
  weights, and optional constraints.

**Save catalogue and rebuild index** validates the schema and items, writes
`catalogue_schema.json` and `catalogue.json`, and recreates the local item
vectors. **Save and generate interface** validates the user and context models,
writes their JSON files, and immediately rebuilds the visible controls.
Students therefore do not need to edit the JSON files manually.

## The project configuration files

### `catalogue_schema.json`

Define the catalogue contract. Students may choose entirely domain-specific
field names and types. The schema maps the identity, title, and description
fields and selects the fields used for embeddings. All other populated item
fields are displayed automatically and supplied to the local LLM.

### `catalogue.json`

Replace the example items with objects that follow `catalogue_schema.json`.
There are no fixed learning-resource fields: a film catalogue may use `code`,
`name`, `synopsis`, `genres`, and `runtime_minutes`; another domain may define
a completely different structure.

### `user_model.json`

Define relatively stable characteristics and preferences, such as prior
knowledge, preferred format, accessibility needs, and learning goals.

### `context_model.json`

Define properties of the current situation, such as available time, current
topic, device, location, or immediate goal. Context values are reset when the
page is reloaded; user-model values are retained in the browser.

Each model contains a `fields` array. A field can use these types:

- `text`, `number`, `select`, `tags`, or `boolean`.

The optional `matching` property controls re-ranking:

- `exact`: categorical match;
- `ordinal`: proximity within the listed `options`;
- `token_overlap`: overlap with text/list fields in an item;
- `prefer_smaller`: rewards items that use less of a numeric budget.

`item_fields` names the catalogue attributes to compare, `weight` controls the
field's relative contribution, and `include_in_retrieval` adds the value to the
semantic retrieval query. An optional `constraint` applies a hard `maximum`,
`minimum`, or `equals` filter to one catalogue field.

All `item_fields` and constraint references must name fields defined in
`catalogue_schema.json`. The browser interface, retrieval text, candidate
cards, grounded prompt, and answer validation are generated automatically from
the four files. Students should not need to modify the scripts.

## Run the project

Prerequisites: Python 3.9 or newer and Ollama.

```bash
ollama pull llama3.2:1b
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python server.py
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Open <http://localhost:8080>. The server validates the four JSON files and
automatically creates or refreshes `data/item_embeddings.npy` when the
catalogue changes. The first run may download the embedding model; later runs
use the locally cached copy.

## Inspect what happens

Turn on **Debug mode** in the interface. The debug panel exposes:

- validated user and context models;
- the text embedded for retrieval;
- the raw-query relevance score and local scope decision;
- items removed by hard constraints;
- semantic, user-model, context-model, and final scores;
- the complete grounded request sent to Ollama; and
- Ollama's raw response;
- the server's grounding-validation result; and
- the response ultimately displayed to the user.

This makes it possible to inspect separately what retrieval selected, what
personalisation re-ranked, and what the LLM finally expressed.

Before profile and context values are added, the server embeds the raw student
query and compares it with the catalogue. If its best similarity is below
`MIN_QUERY_SIMILARITY`, the server returns a local out-of-scope message and does
not contact Ollama. This prevents user preferences from making an unrelated
query appear relevant merely because Top-K retrieval always has a nearest item.
The threshold is an inspectable design parameter and should be calibrated when
the catalogue changes substantially.

## Grounding enforcement

The system prompt requires every recommendation to contain the exact ID and
title of a retrieved catalogue item. After generation, the server checks that:

- every cited ID belongs to the current Top-K candidates; and
- at least one citation is accompanied by that item's exact title.

If the LLM fails this check, its ungrounded text is not shown to the user. The
server constructs a deterministic recommendation from the highest-ranked
catalogue item instead. Debug mode retains the raw LLM response for inspection
and labels the displayed response as a `Validated catalogue fallback`. This is
particularly useful for demonstrating why prompt instructions alone cannot
guarantee grounding with a small language model.

## Automatic behavior and parameters

`server.py` uses four candidates by default and combines scores as:

```text
final score = 0.60 × semantic relevance
            + 0.25 × user-model match
            + 0.15 × context-model match
```

These weights, `TOP_K`, the system prompt, history limit, and temperature are
in the constants near the top of `server.py`. They are supplied as instructor
defaults rather than student prerequisites.

## Common errors

- **Configuration error:** read the terminal message and validate the relevant
  JSON file. Property names and commas must be correct.
- **Could not reach Ollama:** start Ollama and run `ollama pull llama3.2:1b`.
- **Port already in use:** stop the earlier Python server before restarting.
- **Catalogue change is not visible:** restart `server.py`; the index refreshes
  during startup, then reload the browser.
