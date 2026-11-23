import { CreateMLCEngine, prebuiltAppConfig } from './vendor/webllm.mjs';

const MODEL_ID = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';
// Official MLC model-kernel artifact, redistributed byte-for-byte.
// Source: https://github.com/mlc-ai/binary-mlc-llm-libs/tree/main/web-llm-models/v0_2_87/base
// MLC LLM is Apache-2.0; license text is included in vendor/WEBLLM-LICENSE.txt.
// model-runtime.wasm SHA-256: 155ef7359c859f7d646f4c5adc548d779475da3e3196701c7a9d4b82827559a8
function localRuntimeURL() { return new URL('./model-runtime.wasm?v=1.3.2', self.location.href).href; }
function modelAppConfig() {
  const preset = prebuiltAppConfig.model_list.find(item => item.model_id === MODEL_ID);
  if (!preset) throw new Error('Selected model preset is unavailable');
  return { ...prebuiltAppConfig, model_list: [{ ...preset, model_lib: localRuntimeURL() }] };
}
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
let activeDownload = null;

function post(type, id, payload = {}) { self.postMessage({ type, id, ...payload }); }
function fail(id, error) {
  post('error', id, { error: { name: String(error && error.name || 'Error'), message: String(error && error.message || error || 'Model runtime failed').slice(0, 2000) } });
}

function assetURL(input) {
  try {
    const url = new URL(typeof input === 'string' ? input : input && input.url || String(input));
    const host = url.hostname;
    if (url.href === localRuntimeURL()) return url;
    return url.protocol === 'https:' && (host === 'huggingface.co' || host.endsWith('.huggingface.co') || host === 'raw.githubusercontent.com' || host.endsWith('.hf.co')) ? url : null;
  } catch (_) { return null; }
}
function assetLabel(url) {
  const path = url.pathname;
  const shard = path.match(/params_shard_(\d+)\.bin$/);
  if (shard) return '参数分片 ' + (Number(shard[1]) + 1);
  if (/\.wasm$/.test(path)) return '运行组件';
  if (/tokenizer/.test(path)) return '分词文件';
  if (/ndarray-cache\.json$/.test(path)) return '参数清单';
  return '模型配置';
}
function downloadFields(tracker) {
  const file = tracker.currentFile;
  return {
    bytesReceived: tracker.bytesReceived,
    currentFile: file ? file.label : '',
    currentFileBytes: file ? file.received : 0,
    currentFileTotal: file ? file.total : 0,
    downloadHost: file ? file.host : ''
  };
}
function postByteProgress(tracker) {
  if (activeDownload !== tracker || !tracker.active || tracker.bytesReceived <= tracker.lastPostedBytes || Date.now() - tracker.lastBytePost < 1000) return;
  const file = tracker.currentFile;
  const mb = value => (value / 1024 / 1024).toFixed(1);
  const fileProgress = file.total ? mb(file.received) + ' / ' + mb(file.total) + ' MB' : mb(file.received) + ' MB';
  tracker.lastBytePost = Date.now(); tracker.lastPostedBytes = tracker.bytesReceived;
  post('progress', tracker.id, {
    progress: tracker.progress,
    message: '正在下载模型资源：累计已收到 ' + mb(tracker.bytesReceived) + ' MB；当前' + file.label + ' ' + fileProgress + '。',
    ...downloadFields(tracker)
  });
}
async function observeResponse(tracker, response, url) {
  let reader;
  try {
    // Observe a tee branch and return the native response unchanged. Rebuilding
    // a Response would lose its URL/type and can alter the SDK's cache handling.
    const copy = response.clone();
    if (!copy.body || typeof copy.body.getReader !== 'function') return;
    reader = copy.body.getReader(); tracker.readers.add(reader);
    const declaredLength = Number(response.headers.get('content-length'));
    const encoded = response.headers.get('content-encoding');
    let host = url.hostname;
    try { const destination = new URL(response.url); if (destination.protocol === 'https:') host = destination.hostname; } catch (_) {}
    const file = {
      label: assetLabel(url), host, received: 0,
      // Stream bytes are decoded bytes, so a compressed Content-Length is not
      // a reliable total. Keep the total unknown rather than show a false %.
      total: !encoded && Number.isFinite(declaredLength) && declaredLength > 0 ? declaredLength : 0
    };
    while (tracker.active && activeDownload === tracker) {
      const chunk = await reader.read();
      if (chunk.done || !tracker.active || activeDownload !== tracker) break;
      const count = chunk.value && chunk.value.byteLength;
      if (!Number.isFinite(count) || count <= 0) continue;
      file.received += count; tracker.bytesReceived += count; tracker.currentFile = file;
      // CORS can hide Content-Encoding. If decoded bytes exceed the declared
      // length, discard that total instead of displaying more than 100%.
      if (file.total && file.received > file.total) file.total = 0;
      postByteProgress(tracker);
    }
  } catch (_) {
    // Progress instrumentation is optional. Its failure must not replace a
    // successful native fetch or claim that a broken model download succeeded.
  } finally {
    if (reader) {
      tracker.readers.delete(reader);
      try { reader.releaseLock(); } catch (_) {}
    }
  }
}
function finishDownload(tracker) {
  tracker.active = false;
  if (activeDownload === tracker) activeDownload = null;
  for (const reader of tracker.readers) {
    try { Promise.resolve(reader.cancel()).catch(() => {}); } catch (_) {}
  }
  tracker.readers.clear();
}
if (typeof self.fetch === 'function') {
  try {
    const nativeFetch = self.fetch.bind(self);
    self.fetch = async (input, init) => {
      const tracker = activeDownload, url = tracker && assetURL(input);
      const response = await nativeFetch(input, init);
      if (tracker && tracker.active && activeDownload === tracker && url && response.ok) {
        void observeResponse(tracker, response, url);
      }
      return response;
    };
  } catch (_) { /* Keep the SDK usable if a host makes fetch read-only. */ }
}

// Cache.add() performs a browser-internal fetch that bypasses the fetch hook.
// Observe WebLLM's official GET assets with the equivalent fetch-and-put path.
if (typeof self.Cache === 'function' && typeof self.Request === 'function' && typeof self.Cache.prototype.add === 'function') {
  try {
    const nativeCacheAdd = self.Cache.prototype.add;
    self.Cache.prototype.add = async function(input) {
      if (!activeDownload || !activeDownload.active || !assetURL(input)) return nativeCacheAdd.call(this, input);
      const request = new self.Request(input);
      if (request.method !== 'GET') return nativeCacheAdd.call(this, input);
      let response;
      try { response = await self.fetch(request); }
      catch (error) { if (request.url === localRuntimeURL()) throw new Error('model-runtime.wasm: ' + (error.message || 'Failed to fetch')); throw error; }
      if (!response.ok) throw new TypeError((request.url === localRuntimeURL() ? 'model-runtime.wasm: ' : '') + 'Model asset download failed with HTTP ' + response.status);
      await this.put(request, response);
    };
  } catch (_) { /* Preserve the SDK if cache methods are read-only. */ }
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
      const tracker = { id, active: true, readers: new Set(), progress: 0, bytesReceived: 0, currentFile: null, lastBytePost: -Infinity, lastPostedBytes: 0 };
      activeDownload = tracker;
      try {
        post('progress', id, { progress: 0, message: '正在加载浏览器模型…', ...downloadFields(tracker) });
        if (!engine) {
          engine = await CreateMLCEngine(MODEL_ID, {
            appConfig: modelAppConfig(),
            initProgressCallback: report => {
              if (!tracker.active || activeDownload !== tracker) return;
              if (Number.isFinite(report.progress)) tracker.progress = Math.max(tracker.progress, Math.min(1, Math.max(0, report.progress)));
              post('progress', id, { progress: tracker.progress, message: report.text, ...downloadFields(tracker) });
            }
          });
        }
      } finally {
        finishDownload(tracker);
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
