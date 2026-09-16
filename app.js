const state = {
  schemas: null,
  userModel: {},
  contextModel: {},
  messages: [],
  latestDebug: null,
};

const byId = (id) => document.getElementById(id);

function initialValue(field) {
  if (field.type === 'tags') return Array.isArray(field.default) ? field.default : [];
  if (field.type === 'boolean') return field.default ?? false;
  return field.default ?? '';
}

function storageKey() {
  return 'rag-recommender-user-model-v1';
}

function buildModel(schema, container, target, saveLocally) {
  container.replaceChildren();
  for (const field of schema.fields) {
    const wrapper = document.createElement('div');
    wrapper.className = 'field';
    const label = document.createElement('label');
    label.htmlFor = `${saveLocally ? 'user' : 'context'}-${field.id}`;
    label.textContent = field.label;

    let input;
    if (field.type === 'select') {
      input = document.createElement('select');
      for (const option of field.options) {
        const element = document.createElement('option');
        element.value = option;
        element.textContent = option.replaceAll('_', ' ');
        input.append(element);
      }
    } else {
      input = document.createElement('input');
      input.type = field.type === 'number' ? 'number' : field.type === 'boolean' ? 'checkbox' : 'text';
      if (field.minimum !== undefined) input.min = field.minimum;
      if (field.maximum !== undefined) input.max = field.maximum;
      if (field.step !== undefined) input.step = field.step;
      if (field.placeholder) input.placeholder = field.placeholder;
    }
    input.id = label.htmlFor;

    const saved = target[field.id];
    const value = saved === undefined ? initialValue(field) : saved;
    if (field.type === 'boolean') input.checked = Boolean(value);
    else input.value = field.type === 'tags' ? value.join(', ') : value;

    input.addEventListener('change', () => {
      if (field.type === 'boolean') target[field.id] = input.checked;
      else if (field.type === 'number') target[field.id] = Number(input.value);
      else if (field.type === 'tags') {
        target[field.id] = input.value.split(',').map((part) => part.trim()).filter(Boolean);
      } else target[field.id] = input.value;
      if (saveLocally) localStorage.setItem(storageKey(), JSON.stringify(state.userModel));
    });

    wrapper.append(label, input);
    if (field.unit) {
      const hint = document.createElement('small');
      hint.textContent = field.unit;
      wrapper.append(hint);
    }
    container.append(wrapper);
    target[field.id] = value;
  }
}

function resetModel(schema, target, saveLocally) {
  for (const field of schema.fields) target[field.id] = initialValue(field);
  if (saveLocally) localStorage.removeItem(storageKey());
  renderModels();
}

function renderModels() {
  const { user_model: userSchema, context_model: contextSchema } = state.schemas;
  byId('userModelTitle').textContent = userSchema.title;
  byId('userModelDescription').textContent = userSchema.description;
  byId('contextModelTitle').textContent = contextSchema.title;
  byId('contextModelDescription').textContent = contextSchema.description;
  buildModel(userSchema, byId('userModelFields'), state.userModel, true);
  buildModel(contextSchema, byId('contextModelFields'), state.contextModel, false);
}

function optionSelect(className, options, selected = '') {
  const select = document.createElement('select');
  select.className = className;
  for (const [value, label] of options) {
    const option = document.createElement('option');
    option.value = value; option.textContent = label; option.selected = value === selected;
    select.append(option);
  }
  return select;
}

function labelledControl(labelText, control, wide = false) {
  const label = document.createElement('label');
  if (wide) label.className = 'wide';
  const span = document.createElement('span'); span.textContent = labelText;
  label.append(span, control); return label;
}

function csv(value) { return Array.isArray(value) ? value.join(', ') : ''; }

function createCatalogueFieldEditor(field = {}) {
  const row = document.createElement('article'); row.className = 'catalogue-field';
  const id = document.createElement('input'); id.className = 'catalogue-field-id'; id.value = field.id || ''; id.placeholder = 'e.g. genre';
  const label = document.createElement('input'); label.className = 'catalogue-field-label'; label.value = field.label || ''; label.placeholder = 'e.g. Genre';
  const type = optionSelect('catalogue-field-type', [['text','Text'],['number','Number'],['select','Selection list'],['tags','Multiple values'],['boolean','Yes / no']], field.type || 'text');
  const options = document.createElement('input'); options.className = 'catalogue-field-options'; options.value = csv(field.options); options.placeholder = 'Options, if applicable';
  const required = document.createElement('input'); required.type = 'checkbox'; required.className = 'catalogue-field-required'; required.checked = field.required !== false;
  const unique = document.createElement('input'); unique.type = 'checkbox'; unique.className = 'catalogue-field-unique'; unique.checked = Boolean(field.unique);
  const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'danger'; remove.textContent = 'Remove'; remove.addEventListener('click', () => row.remove());
  row.append(labelledControl('Technical ID', id), labelledControl('Label', label), labelledControl('Data type', type), labelledControl('Options', options), labelledControl('Required', required), labelledControl('Unique', unique), remove);
  return row;
}

function catalogueSchemaFromDesigner() {
  const fields = [...byId('catalogueFields').querySelectorAll('.catalogue-field')].map((row) => {
    const field = {
      id: row.querySelector('.catalogue-field-id').value.trim(),
      label: row.querySelector('.catalogue-field-label').value.trim(),
      type: row.querySelector('.catalogue-field-type').value,
      required: row.querySelector('.catalogue-field-required').checked,
    };
    if (row.querySelector('.catalogue-field-unique').checked) field.unique = true;
    const options = row.querySelector('.catalogue-field-options').value.split(',').map((part) => part.trim()).filter(Boolean);
    if (field.type === 'select' && options.length) field.options = options;
    return field;
  });
  return {
    title: byId('catalogueTitle').value.trim(), description: byId('catalogueDescription').value.trim(),
    identity_field: byId('identityField').value, title_field: byId('titleField').value,
    description_field: byId('descriptionField').value,
    embedding_fields: [...byId('embeddingFields').selectedOptions].map((option) => option.value), fields,
  };
}

function populateFieldSelect(select, fields, selected, multiple = false) {
  select.replaceChildren(); select.multiple = multiple;
  for (const field of fields) {
    const option = document.createElement('option'); option.value = field.id; option.textContent = field.label || field.id;
    option.selected = multiple ? (selected || []).includes(field.id) : selected === field.id; select.append(option);
  }
}

function currentCatalogueItems() {
  return [...byId('catalogueItems').querySelectorAll('.catalogue-item')].map((card) => {
    const item = {};
    for (const input of card.querySelectorAll('[data-field-id]')) {
      const field = input.fieldDefinition; let value;
      if (field.type === 'boolean') value = input.checked;
      else if (field.type === 'number') value = Number(input.value || 0);
      else if (field.type === 'tags') value = input.value.split(',').map((part) => part.trim()).filter(Boolean);
      else value = input.value;
      item[field.id] = value;
    }
    return item;
  });
}

function createCatalogueItemEditor(item, fields, position) {
  const card = document.createElement('article'); card.className = 'catalogue-item';
  const heading = document.createElement('div'); heading.className = 'field-editor-heading';
  const title = document.createElement('strong'); title.textContent = `Item ${position}`;
  const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'danger'; remove.textContent = 'Remove item'; remove.addEventListener('click', () => card.remove()); heading.append(title, remove); card.append(heading);
  const grid = document.createElement('div'); grid.className = 'item-grid';
  for (const field of fields) {
    let input;
    if (field.type === 'select') {
      input = optionSelect('', (field.options || []).map((value) => [value, value]), String(item?.[field.id] ?? ''));
    } else {
      input = document.createElement('input'); input.type = field.type === 'number' ? 'number' : field.type === 'boolean' ? 'checkbox' : 'text';
      if (field.type === 'boolean') input.checked = Boolean(item?.[field.id]);
      else input.value = field.type === 'tags' ? csv(item?.[field.id]) : String(item?.[field.id] ?? '');
      if (field.type === 'tags') input.placeholder = 'Separate values with commas';
    }
    input.dataset.fieldId = field.id; input.fieldDefinition = field; grid.append(labelledControl(field.label || field.id, input));
  }
  card.append(grid); return card;
}

function renderCatalogueItems(items = state.schemas.catalogue || []) {
  const container = byId('catalogueItems'); container.replaceChildren();
  const fields = catalogueSchemaFromDesigner().fields;
  items.forEach((item, index) => container.append(createCatalogueItemEditor(item, fields, index + 1)));
}

function applyCatalogueFieldStructure(items = currentCatalogueItems()) {
  const schema = catalogueSchemaFromDesigner();
  const ids = schema.fields.map((field) => field.id).filter(Boolean);
  const previous = {
    identity: byId('identityField').value, title: byId('titleField').value,
    description: byId('descriptionField').value, embeddings: [...byId('embeddingFields').selectedOptions].map((option) => option.value),
  };
  populateFieldSelect(byId('identityField'), schema.fields, ids.includes(previous.identity) ? previous.identity : ids[0]);
  populateFieldSelect(byId('titleField'), schema.fields, ids.includes(previous.title) ? previous.title : ids[1] || ids[0]);
  populateFieldSelect(byId('descriptionField'), schema.fields, ids.includes(previous.description) ? previous.description : ids[2] || ids[0]);
  populateFieldSelect(byId('embeddingFields'), schema.fields, previous.embeddings.filter((id) => ids.includes(id)).length ? previous.embeddings.filter((id) => ids.includes(id)) : ids, true);
  renderCatalogueItems(items.map((item) => Object.fromEntries(ids.map((id) => [id, item[id]]).filter(([, value]) => value !== undefined))));
}

function renderCatalogueDesigner() {
  const schema = state.schemas.catalogue_schema;
  byId('catalogueTitle').value = schema.title || ''; byId('catalogueDescription').value = schema.description || '';
  const fields = byId('catalogueFields'); fields.replaceChildren(); schema.fields.forEach((field) => fields.append(createCatalogueFieldEditor(field)));
  populateFieldSelect(byId('identityField'), schema.fields, schema.identity_field);
  populateFieldSelect(byId('titleField'), schema.fields, schema.title_field);
  populateFieldSelect(byId('descriptionField'), schema.fields, schema.description_field);
  populateFieldSelect(byId('embeddingFields'), schema.fields, schema.embedding_fields, true);
  renderCatalogueItems(state.schemas.catalogue);
}

function inferCatalogueSchema(items) {
  const keys = [...new Set(items.flatMap((item) => Object.keys(item)))];
  const fields = keys.map((id) => {
    const sample = items.map((item) => item[id]).find((value) => value !== undefined && value !== null);
    const type = Array.isArray(sample) ? 'tags' : typeof sample === 'number' ? 'number' : typeof sample === 'boolean' ? 'boolean' : 'text';
    return { id, label: id.replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase()), type, required: items.every((item) => item[id] !== undefined), ...(id === 'id' ? { unique: true } : {}) };
  });
  const pick = (preferred, fallback) => keys.find((key) => preferred.includes(key.toLowerCase())) || keys[fallback] || keys[0];
  return { title: 'Imported catalogue', description: 'Catalogue imported through the visual designer.', identity_field: pick(['id','code','key'], 0), title_field: pick(['title','name','label'], 1), description_field: pick(['description','summary','synopsis'], 2), embedding_fields: keys, fields };
}

async function importCatalogue(event) {
  const status = byId('catalogueStatus'); const file = event.target.files[0]; if (!file) return;
  try {
    const parsed = JSON.parse(await file.text()); const items = Array.isArray(parsed) ? parsed : parsed.catalogue;
    if (!Array.isArray(items) || !items.length || !items.every((item) => item && typeof item === 'object' && !Array.isArray(item))) throw new Error('The file must contain a non-empty JSON array of item objects.');
    const schema = Array.isArray(parsed) ? inferCatalogueSchema(items) : parsed.catalogue_schema || inferCatalogueSchema(items);
    state.schemas.catalogue_schema = schema; state.schemas.catalogue = items; renderCatalogueDesigner();
    status.textContent = `${items.length} items imported. Check the inferred properties, then save.`; status.className = 'designer-success';
  } catch (error) { status.textContent = `Import failed: ${error.message}`; status.className = 'designer-error'; }
  event.target.value = '';
}

async function saveCatalogue() {
  const status = byId('catalogueStatus'); const button = byId('saveCatalogue');
  const schema = catalogueSchemaFromDesigner(); const items = currentCatalogueItems();
  const invalid = schema.fields.filter((field) => !/^[a-z][a-z0-9_]*$/.test(field.id) || !field.label);
  if (invalid.length || !schema.fields.length || !items.length) { status.textContent = 'Add at least one valid property and one item. Property IDs use lowercase letters, numbers, and underscores.'; status.className = 'designer-error'; return; }
  button.disabled = true; button.textContent = 'Validating and rebuilding…';
  try {
    const response = await fetch('/api/catalogue', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ catalogue_schema: schema, catalogue: items }) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Could not save catalogue.');
    Object.assign(state.schemas, result); state.userModel = {}; state.contextModel = {}; localStorage.removeItem(storageKey());
    renderCatalogueDesigner(); renderDesigner(); renderModels(); byId('systemStatus').textContent = `${result.item_count} items · ${state.schemas.generation_model}`;
    status.textContent = result.message; status.className = 'designer-success';
  } catch (error) { status.textContent = error.message; status.className = 'designer-error'; }
  finally { button.disabled = false; button.textContent = 'Save catalogue and rebuild index'; }
}

function createFieldEditor(field = {}) {
  const card = document.createElement('article'); card.className = 'field-editor'; card.sourceField = structuredClone(field);
  const heading = document.createElement('div'); heading.className = 'field-editor-heading';
  const title = document.createElement('strong'); title.textContent = field.label || 'New field';
  const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'danger remove-field'; remove.textContent = 'Remove';
  remove.addEventListener('click', () => card.remove()); heading.append(title, remove);

  const grid = document.createElement('div'); grid.className = 'field-editor-grid';
  const id = document.createElement('input'); id.className = 'field-id'; id.value = field.id || ''; id.placeholder = 'e.g. preferred_genre';
  const label = document.createElement('input'); label.className = 'field-label'; label.value = field.label || ''; label.placeholder = 'e.g. Preferred genre';
  label.addEventListener('input', () => { title.textContent = label.value || 'New field'; });
  const type = optionSelect('field-type', [['text','Text'],['number','Number'],['select','Selection list'],['tags','Multiple values'],['boolean','Yes / no']], field.type || 'text');
  const defaultValue = document.createElement('input'); defaultValue.className = 'field-default'; defaultValue.value = field.type === 'tags' ? csv(field.default) : String(field.default ?? ''); defaultValue.placeholder = 'Starting value';
  const options = document.createElement('input'); options.className = 'field-options'; options.value = csv(field.options); options.placeholder = 'beginner, intermediate, advanced';
  const itemFields = document.createElement('select'); itemFields.className = 'field-item-fields'; itemFields.multiple = true; itemFields.size = Math.min(5, state.schemas.catalogue_schema.fields.length);
  for (const catalogueField of state.schemas.catalogue_schema.fields) {
    const option = document.createElement('option'); option.value = catalogueField.id; option.textContent = catalogueField.label || catalogueField.id;
    option.selected = (field.item_fields || []).includes(catalogueField.id); itemFields.append(option);
  }
  const matching = optionSelect('field-matching', [['','No re-ranking'],['exact','Exact match'],['ordinal','Ordered proximity'],['token_overlap','Text overlap'],['prefer_smaller','Prefer smaller number']], field.matching || '');
  const weight = document.createElement('input'); weight.type = 'number'; weight.min = '0'; weight.step = '0.05'; weight.className = 'field-weight'; weight.value = field.weight ?? 1;
  const retrieval = document.createElement('input'); retrieval.type = 'checkbox'; retrieval.className = 'field-retrieval'; retrieval.checked = Boolean(field.include_in_retrieval);
  const constraintOperator = optionSelect('field-constraint-operator', [['','No hard filter'],['maximum','Item must not exceed value'],['minimum','Item must meet value'],['equals','Item must equal value']], field.constraint?.operator || '');
  const constraintField = optionSelect('field-constraint-field', [['','Choose catalogue field'], ...state.schemas.catalogue_schema.fields.map((f) => [f.id, f.label || f.id])], field.constraint?.item_field || '');

  grid.append(
    labelledControl('Technical ID', id), labelledControl('Label shown to user', label), labelledControl('Input control', type), labelledControl('Default value', defaultValue),
    labelledControl('Options (comma-separated)', options, true), labelledControl('Compare with catalogue fields (⌘/Ctrl-click for several)', itemFields, true),
    labelledControl('Matching method', matching), labelledControl('Importance weight', weight), labelledControl('Add value to semantic retrieval', retrieval),
    labelledControl('Optional hard filter', constraintOperator), labelledControl('Filter catalogue field', constraintField)
  );
  card.append(heading, grid); return card;
}

function renderDesigner() {
  for (const section of document.querySelectorAll('.designer-model')) {
    const schema = state.schemas[section.dataset.model];
    section.querySelector('.model-title').value = schema.title || '';
    section.querySelector('.model-description').value = schema.description || '';
    const fields = section.querySelector('.designer-fields'); fields.replaceChildren();
    for (const field of schema.fields) fields.append(createFieldEditor(field));
  }
}

function parseDefault(type, raw, options) {
  if (type === 'tags') return raw.split(',').map((part) => part.trim()).filter(Boolean);
  if (type === 'number') return raw === '' ? 0 : Number(raw);
  if (type === 'boolean') return raw.trim().toLowerCase() === 'true';
  if (type === 'select') return raw || options[0] || '';
  return raw;
}

function schemaFromDesigner(section) {
  const fields = [...section.querySelectorAll('.field-editor')].map((card) => {
    const type = card.querySelector('.field-type').value;
    const options = card.querySelector('.field-options').value.split(',').map((part) => part.trim()).filter(Boolean);
    const field = {
      ...card.sourceField,
      id: card.querySelector('.field-id').value.trim(),
      label: card.querySelector('.field-label').value.trim(),
      type,
      default: parseDefault(type, card.querySelector('.field-default').value, options),
      include_in_retrieval: card.querySelector('.field-retrieval').checked,
      item_fields: [...card.querySelector('.field-item-fields').selectedOptions].map((option) => option.value),
      weight: Number(card.querySelector('.field-weight').value || 0),
    };
    const matching = card.querySelector('.field-matching').value;
    if (matching) field.matching = matching; else delete field.matching;
    if (type === 'select' || matching === 'ordinal') field.options = options; else delete field.options;
    const operator = card.querySelector('.field-constraint-operator').value;
    const itemField = card.querySelector('.field-constraint-field').value;
    if (operator && itemField) field.constraint = { operator, item_field: itemField }; else delete field.constraint;
    return field;
  });
  return { title: section.querySelector('.model-title').value.trim(), description: section.querySelector('.model-description').value.trim(), fields };
}

async function saveDesignedModels() {
  const status = byId('designerStatus'); const button = byId('saveModels');
  const models = {};
  for (const section of document.querySelectorAll('.designer-model')) models[section.dataset.model] = schemaFromDesigner(section);
  const invalid = Object.entries(models).flatMap(([name, schema]) => schema.fields.filter((field) => !/^[a-z][a-z0-9_]*$/.test(field.id) || !field.label).map((field) => `${name}: “${field.label || field.id || 'new field'}” needs a label and a lowercase ID using letters, numbers, and underscores.`));
  if (invalid.length) { status.textContent = invalid.join(' '); status.className = 'designer-error'; return; }
  button.disabled = true; button.textContent = 'Saving…';
  try {
    const response = await fetch('/api/models', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(models) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Could not save models.');
    state.schemas.user_model = result.user_model; state.schemas.context_model = result.context_model;
    state.userModel = {}; state.contextModel = {}; localStorage.removeItem(storageKey()); renderModels();
    status.textContent = result.message; status.className = 'designer-success';
  } catch (error) { status.textContent = error.message; status.className = 'designer-error'; }
  finally { button.disabled = false; button.textContent = 'Save and generate interface'; }
}

function addMessage(role, content) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const strong = document.createElement('strong');
  strong.textContent = role === 'user' ? 'You' : role === 'error' ? 'Error' : 'Recommender';
  const paragraph = document.createElement('p');
  paragraph.textContent = content;
  article.append(strong, paragraph);
  byId('messages').append(article);
  byId('messages').scrollTop = byId('messages').scrollHeight;
}

function renderCandidates(candidates = []) {
  const container = byId('candidates');
  container.replaceChildren();
  container.classList.toggle('empty', candidates.length === 0);
  if (!candidates.length) {
    container.textContent = 'No candidates satisfied the current request and constraints.';
    return;
  }
  for (const item of candidates) {
    const card = document.createElement('article');
    card.className = 'candidate';
    const reasons = (item.match_reasons || []).join(' · ') || 'No explicit profile/context match.';
    const heading = document.createElement('h3'); heading.textContent = `[${item.id}] ${item.title}`;
    const description = document.createElement('p'); description.textContent = item.description;
    const attributes = document.createElement('dl'); attributes.className = 'candidate-attributes';
    for (const attribute of item.display_fields || []) {
      const term = document.createElement('dt'); term.textContent = attribute.label;
      const detail = document.createElement('dd'); detail.textContent = attribute.value;
      attributes.append(term, detail);
    }
    const reason = document.createElement('p'); reason.textContent = reasons;
    const scores = document.createElement('div'); scores.className = 'score-grid';
    for (const [label, value] of [['semantic', item.semantic_score], ['user', item.user_score], ['context', item.context_score]]) {
      const score = document.createElement('div'); score.className = 'score';
      const strong = document.createElement('strong'); strong.textContent = Number(value).toFixed(3);
      const span = document.createElement('span'); span.textContent = label;
      score.append(strong, span); scores.append(score);
    }
    card.append(heading, description, attributes, reason, scores);
    container.append(card);
  }
}

async function loadConfiguration() {
  const response = await fetch('/api/config');
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Could not load configuration.');
  state.schemas = result;
  try { state.userModel = JSON.parse(localStorage.getItem(storageKey())) || {}; } catch { state.userModel = {}; }
  renderModels();

  const status = byId('systemStatus');
  status.textContent = `${result.item_count} items · ${result.generation_model}`;
  status.className = 'status ready';
}

byId('chatForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = byId('query').value.trim();
  if (!query) return;
  state.messages.push({ role: 'user', content: query });
  addMessage('user', query);
  byId('query').value = '';
  byId('submitButton').disabled = true;
  byId('submitButton').textContent = 'Working…';

  try {
    const response = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_model: state.userModel,
        context_model: state.contextModel,
        messages: state.messages,
        debug: byId('debugToggle').checked,
      }),
    });
    const result = await response.json();
    renderCandidates(result.candidates || []);
    state.latestDebug = result.debug || null;
    byId('debugOutput').textContent = state.latestDebug
      ? JSON.stringify(state.latestDebug, null, 2)
      : 'Debug mode was not enabled for this request.';
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    state.messages.push({ role: 'assistant', content: result.reply });
    addMessage('assistant', result.reply);
  } catch (error) {
    addMessage('error', error instanceof Error ? error.message : String(error));
  } finally {
    byId('submitButton').disabled = false;
    byId('submitButton').textContent = 'Recommend';
  }
});

byId('debugToggle').addEventListener('change', () => {
  byId('debugPanel').hidden = !byId('debugToggle').checked;
});

byId('clearConversation').addEventListener('click', () => {
  state.messages = [];
  state.latestDebug = null;
  byId('messages').replaceChildren();
  addMessage('assistant', 'Conversation cleared. Your saved user model is unchanged.');
  renderCandidates([]);
  byId('debugOutput').textContent = 'Submit a query to inspect the pipeline.';
});

byId('resetUser').addEventListener('click', () => resetModel(state.schemas.user_model, state.userModel, true));
byId('resetContext').addEventListener('click', () => resetModel(state.schemas.context_model, state.contextModel, false));
byId('copyDebug').addEventListener('click', async () => {
  await navigator.clipboard.writeText(byId('debugOutput').textContent);
  byId('copyDebug').textContent = 'Copied';
  setTimeout(() => { byId('copyDebug').textContent = 'Copy JSON'; }, 1200);
});

byId('openDesigner').addEventListener('click', () => { renderCatalogueDesigner(); renderDesigner(); byId('designer').hidden = false; byId('designer').scrollIntoView({ behavior: 'smooth' }); });
byId('closeDesigner').addEventListener('click', () => { byId('designer').hidden = true; });
for (const button of document.querySelectorAll('.add-field')) button.addEventListener('click', () => button.closest('.designer-model').querySelector('.designer-fields').append(createFieldEditor()));
byId('saveModels').addEventListener('click', saveDesignedModels);
byId('addCatalogueField').addEventListener('click', () => byId('catalogueFields').append(createCatalogueFieldEditor()));
byId('applyCatalogueFields').addEventListener('click', () => applyCatalogueFieldStructure());
byId('addCatalogueItem').addEventListener('click', () => { const fields = catalogueSchemaFromDesigner().fields; byId('catalogueItems').append(createCatalogueItemEditor({}, fields, byId('catalogueItems').children.length + 1)); });
byId('catalogueImport').addEventListener('change', importCatalogue);
byId('saveCatalogue').addEventListener('click', saveCatalogue);

loadConfiguration().catch((error) => {
  byId('systemStatus').textContent = error.message;
  byId('systemStatus').className = 'status error';
});
