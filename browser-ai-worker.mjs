import { CreateMLCEngine } from './vendor/webllm.mjs';

const MODEL_ID = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';
const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    results: { type: 'array', items: {
      type: 'object',
      properties: {
        id: { type: 'string' },
        status: { type: 'string', enum: ['matched', 'partial', 'unknown', 'gap'] },
        evidence_ids: { type: 'array', items: { type: 'string' }, maxItems: 2 },
        explanation: { type: 'string' }
      },
      required: ['id', 'status', 'evidence_ids', 'explanation'],
      additionalProperties: false
    } }
  },
  required: ['results'],
  additionalProperties: false
};
function resultSchema(messages) {
  const data = JSON.parse(messages[messages.length - 1].content);
  const ids = data.requirements.map(item => item.id);
  if (ids.length !== 1 || !ids.every(id => typeof id === 'string')) throw new Error('Invalid requirement batch');
  const schema = JSON.parse(JSON.stringify(RESULT_SCHEMA));
  schema.properties.results.minItems = ids.length;
  schema.properties.results.maxItems = ids.length;
  schema.properties.results.items.properties.id.enum = ids;
  const excerpts = data.resume_segment;
  if (!Array.isArray(excerpts) || !excerpts.every(item => typeof item.id === 'string' && typeof item.text === 'string')) throw new Error('Invalid source excerpts');
  if (excerpts.length) schema.properties.results.items.properties.evidence_ids.items.enum = excerpts.map(item => item.id);
  else schema.properties.results.items.properties.evidence_ids.maxItems = 0;
  return JSON.stringify(schema);
}
function restoreCitations(content, messages) {
  const data = JSON.parse(messages[messages.length - 1].content);
  const citations = new Map(data.resume_segment.map(item => [item.id, item.text]));
  const response = JSON.parse(content);
  if (!response || !Array.isArray(response.results)) throw new Error('Model returned invalid result rows');
  for (const row of response.results) {
    if (!Array.isArray(row.evidence_ids) || row.evidence_ids.some(id => !citations.has(id))) throw new Error('Model returned invalid source excerpt IDs');
    row.evidence = row.evidence_ids.map(id => citations.get(id));
    delete row.evidence_ids;
  }
  return JSON.stringify(response);
}
let engine = null;
let busy = false;

function post(type, id, payload = {}) { self.postMessage({ type, id, ...payload }); }
function fail(id, error) {
  post('error', id, { error: { name: String(error && error.name || 'Error'), message: String(error && error.message || error || 'Model runtime failed').slice(0, 2000) } });
}

self.onmessage = async event => {
  const request = event.data;
  if (!request || typeof request !== 'object' || !Number.isInteger(request.id)) return;
  const id = request.id;
  if (busy) { fail(id, new Error('Model runtime is busy')); return; }
  busy = true;
  try {
    if (request.type === 'load') {
      if (request.model !== MODEL_ID) throw new Error('Unsupported model configuration');
      post('progress', id, { progress: 0, message: '正在加载浏览器模型…' });
      if (!engine) {
        engine = await CreateMLCEngine(MODEL_ID, {
          initProgressCallback: report => {
            post('progress', id, { progress: report.progress, message: report.text });
          }
        });
      }
      post('result', id, { result: { loaded: true, model: MODEL_ID } });
    } else if (request.type === 'complete') {
      if (!engine) throw new Error('Model not loaded');
      if (!Array.isArray(request.messages) || !request.messages.length) throw new Error('Invalid messages');
      post('activity', id, { message: '正在理解材料并生成逐项分析…' });
      const stream = await engine.chat.completions.create({
        messages: request.messages,
        max_tokens: request.maxTokens,
        temperature: request.temperature,
        // WebLLM 0.2.89 requires an explicit schema string for json_object.
        response_format: { type: 'json_object', schema: resultSchema(request.messages) },
        stream: true
      });
      let content = '', lastUpdate = 0;
      for await (const chunk of stream) {
        const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta && chunk.choices[0].delta.content;
        if (typeof delta === 'string') content += delta;
        if (Date.now() - lastUpdate > 1000) {
          post('activity', id, { message: '正在生成结果，已生成 ' + content.length + ' 个字符…' });
          lastUpdate = Date.now();
        }
      }
      if (!content) throw new Error('Model returned an invalid response');
      post('result', id, { result: restoreCitations(content, request.messages) });
    } else {
      throw new Error('Unsupported model operation');
    }
  } catch (error) {
    fail(id, error);
  } finally {
    busy = false;
  }
};

post('ready');
