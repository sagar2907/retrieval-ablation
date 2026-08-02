"""The single-page UI, inlined.

Kept as a Python string rather than a template file so the service has no
templating dependency and no static-file mount: one import, one route, nothing to
misconfigure in a container. All CSS and JS are inline and no asset is fetched
from a CDN, so the page works with no network and under a strict content policy.

What the page is *for* is citation and score inspection. A demo that shows only an
answer teaches a reader nothing about whether retrieval worked. Here, every
citation in the answer is a button that scrolls to and highlights the passage it
refers to, and every passage shows its rank, raw score, relative score bar,
section path in the filing, and character offsets. A wrong answer should be
diagnosable from this page without opening a terminal.
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>retrieval-ablation</title>
<style>
  :root {
    --bg: #ffffff; --fg: #16181d; --muted: #61656e; --line: #e3e5e9;
    --accent: #1f5fd6; --accent-soft: #e8f0fe; --warn: #a8570d; --warn-soft: #fdf3e6;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --fg: #e6e8ec; --muted: #9aa0ab; --line: #2a2e35;
      --accent: #6aa2ff; --accent-soft: #1b2740; --warn: #e0a86a; --warn-soft: #33281a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 20px 80px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 22px; }
  form { display: flex; gap: 8px; flex-wrap: wrap; }
  input[type=text] {
    flex: 1 1 340px; padding: 11px 13px; font-size: 15px; color: var(--fg);
    background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  }
  input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button {
    padding: 11px 16px; font-size: 14px; font-weight: 600; cursor: pointer;
    border-radius: 8px; border: 1px solid var(--line);
    background: var(--bg); color: var(--fg);
  }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .55; cursor: progress; }
  .status { margin: 14px 0; font-size: 13px; color: var(--muted); }
  .card { border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 18px 0; }
  .answer { font-size: 16px; line-height: 1.7; white-space: pre-wrap; }
  .cite {
    display: inline-block; min-width: 22px; padding: 1px 6px; margin: 0 1px;
    font: 600 12px/1.5 var(--mono); cursor: pointer; vertical-align: baseline;
    color: var(--accent); background: var(--accent-soft);
    border: 1px solid var(--accent); border-radius: 5px;
  }
  .cite.bad { color: var(--warn); background: var(--warn-soft); border-color: var(--warn); }
  .meta { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
  .passage { border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin: 10px 0; }
  .passage.hit { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  .passage.cited { border-left: 4px solid var(--accent); }
  .phead { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .rank {
    font: 700 12px var(--mono); color: #fff; background: var(--muted);
    border-radius: 5px; padding: 2px 7px;
  }
  .passage.cited .rank { background: var(--accent); }
  .doc { font-weight: 600; }
  .section { color: var(--muted); font-size: 12px; }
  .bar { height: 5px; background: var(--line); border-radius: 3px; margin: 9px 0 6px; }
  .bar > i { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
  .ptext {
    font: 12.5px/1.6 var(--mono); white-space: pre-wrap; word-break: break-word;
    max-height: 190px; overflow: auto; margin-top: 8px;
    padding: 10px; background: var(--accent-soft); border-radius: 7px;
  }
  .tag {
    font: 600 11px var(--mono); padding: 1px 6px; border-radius: 4px;
    background: var(--warn-soft); color: var(--warn);
  }
  .err { color: var(--warn); }
  .note { font-size: 12px; color: var(--muted); margin-top: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>retrieval-ablation</h1>
  <div class="sub" id="health">checking index…</div>

  <form id="f">
    <input type="text" id="q" autocomplete="off"
           placeholder="e.g. Apple research and development expense 2025">
    <button type="submit" class="primary" id="btnSearch">Search</button>
    <button type="button" id="btnAnswer">Answer</button>
  </form>
  <div class="status" id="status"></div>

  <div id="answerCard" class="card" style="display:none">
    <div class="answer" id="answer"></div>
    <div class="meta" id="answerMeta"></div>
    <div class="note">Click a citation to jump to the passage it refers to.</div>
  </div>

  <div id="results"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let lastCited = [];

function esc(s) {
  return s.replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

fetch('/health').then(r => r.json()).then(h => {
  $('health').textContent = h.ready
    ? `${h.n_chunks.toLocaleString()} chunks from ${h.n_documents} filings · `
      + `${h.chunker} · ${h.retriever} · built in ${h.index_build_seconds}s`
      + (h.answer_endpoint_available ? '' : ' · no API key: /answer disabled')
    : `not ready — ${h.error || 'building'}`;
}).catch(() => { $('health').textContent = 'service unreachable'; });

function renderPassages(passages, cited) {
  lastCited = cited || [];
  $('results').innerHTML = passages.map(p => {
    const isCited = lastCited.includes(p.chunk_id);
    return `<div class="passage ${isCited ? 'cited' : ''}" id="p${p.rank}">
      <div class="phead">
        <span class="rank">${p.rank}</span>
        <span class="doc">${esc(p.document)}</span>
        ${p.contains_table ? '<span class="tag">table</span>' : ''}
      </div>
      <div class="section">${esc(p.section)}</div>
      <div class="bar"><i style="width:${(p.score_relative * 100).toFixed(1)}%"></i></div>
      <div class="meta">
        <span>score <b>${p.score}</b></span>
        <span>${(p.score_relative * 100).toFixed(0)}% of top</span>
        <span>chars ${p.char_start.toLocaleString()}–${p.char_end.toLocaleString()}</span>
      </div>
      <div class="ptext">${esc(p.text)}</div>
    </div>`;
  }).join('');
}

function renderAnswer(data) {
  const idByNumber = {};
  data.passages.forEach(p => { idByNumber[p.rank] = p.chunk_id; });
  // Citations are rendered as buttons that scroll to their passage. An
  // out-of-range citation is marked rather than dropped, because a model citing
  // a passage it was never given is a distinct and more serious failure.
  const html = esc(data.answer).replace(/\\[(\\d+)\\]/g, (m, n) => {
    const valid = idByNumber[n] !== undefined;
    return `<span class="cite ${valid ? '' : 'bad'}" data-n="${n}"
             title="${valid ? 'go to passage ' + n : 'passage ' + n + ' was never supplied'}">
             [${n}]</span>`;
  });
  $('answer').innerHTML = html || '<i>(empty response)</i>';
  $('answerMeta').innerHTML = [
    data.refused ? '<b>refused: answer not in retrieved passages</b>' : '',
    `${data.prompt_tokens.toLocaleString()} prompt tokens`,
    `${data.output_tokens} output tokens`,
    `${data.took_ms} ms`,
    data.from_cache ? 'served from cache' : 'live call',
    data.invalid_citations.length
      ? `<span class="err">${data.invalid_citations.length} citation(s) out of range</span>` : '',
  ].filter(Boolean).map(s => `<span>${s}</span>`).join('');
  $('answerCard').style.display = 'block';

  document.querySelectorAll('.cite').forEach(el => {
    el.onclick = () => {
      const target = $('p' + el.dataset.n);
      if (!target) return;
      document.querySelectorAll('.passage').forEach(p => p.classList.remove('hit'));
      target.classList.add('hit');
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
  });
}

async function call(path, withAnswer) {
  const query = $('q').value.trim();
  if (!query) return;
  $('btnSearch').disabled = $('btnAnswer').disabled = true;
  $('status').textContent = withAnswer ? 'retrieving and generating…' : 'retrieving…';
  $('answerCard').style.display = withAnswer ? 'block' : 'none';
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 10 }),
    });
    const data = await res.json();
    if (!res.ok) {
      $('status').innerHTML = `<span class="err">${esc(data.detail || res.statusText)}</span>`;
      $('answerCard').style.display = 'none';
      return;
    }
    if (withAnswer) {
      renderPassages(data.passages, data.cited_chunk_ids);
      renderAnswer(data);
      $('status').textContent = '';
    } else {
      renderPassages(data.passages, []);
      $('status').textContent =
        `${data.passages.length} passages from ${data.n_chunks_indexed.toLocaleString()} `
        + `chunks in ${data.took_ms} ms`;
    }
  } catch (e) {
    $('status').innerHTML = `<span class="err">${esc(String(e))}</span>`;
  } finally {
    $('btnSearch').disabled = $('btnAnswer').disabled = false;
  }
}

$('f').onsubmit = (e) => { e.preventDefault(); call('/search', false); };
$('btnAnswer').onclick = () => call('/answer', true);
</script>
</body>
</html>
"""
