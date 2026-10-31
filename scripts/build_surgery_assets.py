"""Build offline research tools from the reviewed public source and model bundles."""

import ast
from html import escape
import hashlib
import json
from pathlib import Path

CSS = '''*{box-sizing:border-box}body{margin:0;background:#f2f6f7;color:#172d40;font:16px/1.6 system-ui,sans-serif}header,main,footer{max-width:1180px;margin:auto;padding:24px}header{background:#102d40;color:white;max-width:none}header>div{max-width:1132px;margin:auto}h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.1;margin:12px 0}h2{line-height:1.2;overflow-wrap:anywhere}nav{display:flex;gap:16px;flex-wrap:wrap}header a{color:#b5eee2}a{color:#006b63;overflow-wrap:anywhere}section,.card{background:white;border:1px solid #d3e0e3;border-radius:12px;padding:24px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}.grid .card{margin:0}label{display:inline-block;margin:8px 18px 8px 0}input,select,button{font:inherit;padding:8px;border:1px solid #9fb8bf;border-radius:6px;max-width:100%}button{cursor:pointer;background:#e6f3ef;color:#00594e}pre{overflow:auto;max-height:540px;background:#102d40;color:#e6f5f4;padding:18px;border-radius:8px;font:13px/1.55 ui-monospace,monospace}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere}.scroll{overflow:auto}table{border-collapse:collapse;table-layout:fixed;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d4e0e4;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf4f4}.small{font-size:13px;color:#526a77}.tag{display:inline-block;font-size:12px;padding:3px 9px;background:#e2f1eb;color:#00594f;border-radius:12px}.notice{border-left:4px solid #007267;background:#edf8f4;padding:14px}details{margin-top:14px}summary{cursor:pointer;font-weight:650}.hash{font-size:11px;word-break:break-all}canvas{display:block;background:white}footer{color:#526a77;font-size:13px}[hidden]{display:none!important}'''


def page(title, body, script=''):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';style-src 'unsafe-inline';script-src 'unsafe-inline';img-src 'self' data:;connect-src 'none';base-uri 'none';form-action 'none'"><title>{escape(title)} | Digital Surgery</title><style>{CSS}</style></head><body><header><div><p>Digital Surgery / research workbench</p><h1>{escape(title)}</h1><nav><a href="SURGERY_TOOLKIT.html">Editing code</a><a href="MODEL_REGISTRY.html">Saved models</a><a href="ACTIVATION_VIEWER.html">Activation viewer</a><a href="SURGERY_PROCESS.html">Process</a><a href="RESEARCH_LOG.html">Research log</a></nav></div></header><main>{body}</main><footer>Offline HTML. No external scripts, uploads or inference API. <a href="../scripts/build_surgery_assets.py">Asset builder source</a> · <a href="../PUBLIC_RELEASE_MANIFEST.json">Release hashes</a></footer>{'<script>'+script+'</script>' if script else ''}</body></html>'''


SOURCES = {
    'Observe': ['interpretability/native_cache_branch.py', 'interpretability/native_capture.py', 'interpretability/native_timeline.py', 'interpretability/probe.py', 'interpretability/response_regions.py', 'interpretability/python_token_regions.py', 'scripts/export_native_trace.py'],
    'Patch activations': ['interpretability/selector_site.py', 'interpretability/mlp_composition.py', 'research/experiments/n34/inference.py'],
    'Edit weights': ['interpretability/stored_neuron_gain.py', 'interpretability/stored_group_gain.py', 'interpretability/gated_mlp_surgery.py', 'interpretability/expansion.py'],
    'Edit connections': ['interpretability/memory_connection.py'],
    'Reconstruct': ['reproduce/model_edits.py'],
    'Evaluate': ['evaluation/paired_generation_metrics.py'],
    'Generate and resume': ['interpretability/native_eos_generation.py'],
}


def toolkit(dest):
    cards, index = [], []
    for category, paths in SOURCES.items():
        for name in paths:
            source = (dest / name).read_text()
            tree = ast.parse(source)
            symbols = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            digest = hashlib.sha256(source.encode()).hexdigest()
            title = ast.get_docstring(tree) or name
            cards.append(f'<article class="card tool" data-category="{escape(category)}"><span class="tag">{escape(category)}</span><h2>{escape(Path(name).stem)}</h2><p>{escape(title)}</p><p><a href="../{escape(name)}">Exact source file</a></p><p class="small">APIs: {escape(", ".join(symbols))}</p><p class="hash">SHA-256 {digest}</p><details><summary>Read the code</summary><pre>{escape(source)}</pre></details></article>')
            index.append({'category': category, 'path': name, 'sha256': digest, 'symbols': symbols})
    body = '<section><h2>Use the tested implementation</h2><p>These are the actual reviewed source files used for observation, scoped activation patches, direct weight edits, structural operations and reconstruction. Each card links the file and embeds its exact contents. A successful software fixture is distinct from a behavioral benefit.</p><label>Find code <input id="query" type="search" placeholder="zero, clone, capture, restore"></label><label>Category <select id="category"><option>All</option>' + ''.join(f'<option>{escape(k)}</option>' for k in SOURCES) + '</select></label><p id="count" class="small"></p></section><div class="grid">' + ''.join(cards) + '</div>'
    script = '''const q=document.getElementById('query'),c=document.getElementById('category');function filter(){let n=0;for(const e of document.querySelectorAll('.tool')){e.hidden=!(e.textContent.toLowerCase().includes(q.value.toLowerCase())&&(c.value==='All'||e.dataset.category===c.value));if(!e.hidden)n++}document.getElementById('count').textContent=n+' source files shown'}q.oninput=c.onchange=filter;filter();'''
    return page('Editing and observation toolkit', body, script), index


def model_registry(dest):
    from safetensors import safe_open
    bundles, tensors = [], []
    for path in sorted((dest / 'models').rglob('*.json')):
        if path.name not in ['bundle.json', 'recipe.json']:
            continue
        data = json.loads(path.read_text())
        base = data.get('base', data.get('parent', 'See recipe'))
        revision = data.get('revision', '')
        if isinstance(base, dict):
            revision = base.get('revision', revision)
            base = next((base[k] for k in ['checkpoint', 'repo', 'repository', 'model_id', 'name'] if isinstance(base.get(k), str)), 'See recipe')
        counts = {k: len(data[k]) for k in ['policies', 'conditions', 'edits'] if isinstance(data.get(k), (list, dict))}
        bundles.append({'path': str(path.relative_to(dest)), 'base': base, 'revision': revision, 'representation': data.get('representation', 'See recipe'), 'kind': data.get('kind', 'Edit recipe'), 'counts': counts, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted((dest / 'models').rglob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as source:
            names = list(source.keys())
            dtypes = sorted({source.get_slice(n).get_dtype() for n in names})
            shapes = sorted({tuple(source.get_slice(n).get_shape()) for n in names})
        tensors.append({'path': str(path.relative_to(dest)), 'bytes': path.stat().st_size, 'tensor_count': len(names), 'dtypes': dtypes, 'shapes': shapes, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    rows = ''.join(f'<tr><td><a href="../{escape(b["path"])}">{escape(b["path"])}</a><p class="hash">{b["sha256"]}</p></td><td>{escape(str(b["base"]))}<p class="hash">{escape(str(b["revision"]))}</p></td><td>{escape(str(b["representation"]))}</td><td>{escape(', '.join(k+': '+str(n) for k,n in b["counts"].items()))}<p class="small">{escape(str(b["kind"]))}</p></td></tr>' for b in bundles)
    weights = ''.join(f'<tr><td><a href="../{escape(t["path"])}" download>{escape(t["path"])}</a><p class="hash">{t["sha256"]}</p></td><td>{t["bytes"]:,}</td><td>{t["tensor_count"]}</td><td>{escape(", ".join(t["dtypes"]))}<p class="small">Shapes: {escape(str(t["shapes"]))}</p></td></tr>' for t in tensors)
    body = f'''<section><h2>The model artifact is reconstructible</h2><p>{len(bundles)} bundle/recipe records and {len(tensors)} exact tensor assets are present in this checkout. Saved tensor payload: {sum(t['bytes'] for t in tensors):,} bytes. Each variant combines an immutable parent, an exact patch or copy recipe and its loader. Full base checkpoints are not duplicated here.</p><p class="notice">N02 has an independently verified public-loader replay in R01. Other recipes state their original-run and package-replay limits. Runtime donor policies need caller-supplied contexts; they are not permanent new weights. Read each validation record before attributing a behavior.</p><p><a href="../models/README.md">Complete reconstruction instructions</a></p></section><section><h2>Reconstruct a physical edit</h2><label>Representation <select id="rep"><option value="bf16_text">BF16 text</option><option value="nf4_text_same_source">Same-source NF4 text</option></select></label><label>Edit <select id="edit">{''.join('<option>'+n+'</option>' for n in ['dense_delete','dense_scale2','dense_clone_unsplit','dense_clone_split','dense_new_zero','dense_recombined'])}</select></label><pre id="command"></pre><p class="small">This command loads the pinned parent only when run. The loader verifies the source and applies the stored edit. No optimization is performed.</p></section><section><h2>Exact saved tensor files</h2><div class="scroll"><table><thead><tr><th>Download and hash</th><th>Bytes</th><th>Tensors</th><th>Stored types</th></tr></thead><tbody>{weights}</tbody></table></div></section><section><h2>All model bundles and runtime policies</h2><div class="scroll"><table><thead><tr><th>Recipe and hash</th><th>Pinned parent</th><th>Representation</th><th>Declared variants</th></tr></thead><tbody>{rows}</tbody></table></div></section>'''
    script = r"""function command(){
const rep=document.getElementById('rep').value,edit=document.getElementById('edit').value;
document.getElementById('command').textContent=`from reproduce.model_edits import reconstruct_n02

model, receipt = reconstruct_n02(
    'models/n02/${rep}',
    '${edit}', device='cuda:0'
)`;
}document.getElementById('rep').onchange=document.getElementById('edit').onchange=command;command();"""
    return page('Saved model registry', body, script), {'bundles': bundles, 'tensor_assets': tensors}


VIEWER_SCRIPT = r'''
const sample={schema:'digital-surgery-trace-v1',provenance:'Synthetic software example, not model evidence',tokens:[{processed:'prompt:end',predicted:'def'},{processed:'def',predicted:'f'},{processed:'f',predicted:'('},{processed:'(',predicted:'x'},{processed:'x',predicted:')'}],units:[{layer:0,channel:3,down_norm:1.2,values:[0.1,0.8,null,-0.4,0]},{layer:0,channel:8,down_norm:0.7,values:[-0.3,0.2,0.6,0.1,-0.5]},{layer:1,channel:2,down_norm:1.5,values:[0,0.1,0.7,-0.9,0.3]}]};
let data=sample,visible=[],start=0;const canvas=document.getElementById('heat'),ctx=canvas.getContext('2d');
function validate(d){if(d.schema!=='digital-surgery-trace-v1'||!Array.isArray(d.tokens)||!d.tokens.length||!Array.isArray(d.units)||!d.units.length)throw Error('Trace schema, tokens and units are required');if(d.tokens.length*d.units.length>2000000)throw Error('Load a compact per-case trace with at most two million recorded cells');let seen=new Set();for(const t of d.tokens)if(typeof t.processed!=='string'||typeof t.predicted!=='string')throw Error('Tokens need processed and predicted labels');for(const u of d.units){let k=u.layer+':'+u.channel;if(!Number.isInteger(u.layer)||!Number.isInteger(u.channel)||u.layer<0||u.channel<0||seen.has(k)||u.values.length!==d.tokens.length)throw Error('Invalid or duplicate unit, or unaligned values');seen.add(k);if(u.down_norm!==undefined&&(!Number.isFinite(u.down_norm)||u.down_norm<0))throw Error('Invalid output-column norm');if(u.values.some(x=>x!==null&&!Number.isFinite(x)))throw Error('Activation values must be finite numbers or null')}}
function load(d){validate(d);data=d;start=0;const s=document.getElementById('layer');s.replaceChildren(new Option('All recorded layers','all'),...[...new Set(d.units.map(u=>u.layer))].sort((a,b)=>a-b).map(n=>new Option('Layer '+n,''+n)));document.getElementById('position').max=Math.max(0,d.tokens.length-32);document.getElementById('position').value=0;document.getElementById('provenance').textContent=d.provenance||'Local trace, not independently validated';draw()}
function value(u,i){const x=u.values[i];return x===null?null:document.getElementById('metric').value==='raw'?x:u.down_norm===undefined?null:Math.abs(x)*u.down_norm}
function draw(){start=+document.getElementById('position').value;visible=data.units.filter(u=>document.getElementById('layer').value==='all'||u.layer===+document.getElementById('layer').value).sort((a,b)=>a.layer-b.layer||a.channel-b.channel).slice(0,64);const n=Math.min(32,data.tokens.length-start),w=30,h=25,left=125,top=110;canvas.width=left+n*w+25;canvas.height=top+visible.length*h+30;ctx.clearRect(0,0,canvas.width,canvas.height);let max=0;for(const u of visible)for(let j=0;j<n;j++){let x=value(u,start+j);if(x!==null)max=Math.max(max,Math.abs(x))}max=max||1;ctx.font='12px system-ui';for(let j=0;j<n;j++){ctx.save();ctx.translate(left+j*w+18,top-8);ctx.rotate(-Math.PI/3);ctx.fillStyle='#173246';ctx.fillText(data.tokens[start+j].predicted.slice(0,16),0,0);ctx.restore()}visible.forEach((u,i)=>{ctx.fillStyle='#173246';ctx.fillText('L'+u.layer+' C'+u.channel,8,top+i*h+17);for(let j=0;j<n;j++){const x=value(u,start+j);ctx.fillStyle=x===null?'#bfc7ca':x===0?'#ffffff':x>0?'rgba(0,107,99,'+(.12+.88*Math.abs(x)/max)+')':'rgba(177,54,54,'+(.12+.88*Math.abs(x)/max)+')';ctx.fillRect(left+j*w,top+i*h,w-1,h-1);ctx.strokeStyle='#d7e2e6';ctx.strokeRect(left+j*w,top+i*h,w-1,h-1)}});document.getElementById('window').textContent='Recorded columns '+start+' to '+(start+n-1)+'; '+visible.length+' units shown. Color is scaled within this window. Gray means unrecorded; white means measured zero.'}
canvas.onmousemove=e=>{const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height,j=Math.floor((x-125)/30),i=Math.floor((y-110)/25);if(i>=0&&i<visible.length&&j>=0&&j<Math.min(32,data.tokens.length-start)){const u=visible[i],t=data.tokens[start+j],v=u.values[start+j];document.getElementById('cell').textContent='L'+u.layer+' C'+u.channel+' | '+(t.processed_position===undefined?'column '+(start+j):'processed position '+t.processed_position+(t.decision===null?' | no decision measured':' | decision '+t.decision))+' | processed '+t.processed+' | predicts '+t.predicted+' | activation '+(v===null?'unrecorded':v)+' | column norm '+(u.down_norm??'not supplied')}};
document.getElementById('file').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;if(f.size>50000000)throw Error('Use a compact per-case trace below 50 MB');load(JSON.parse(await f.text()));document.getElementById('error').textContent=''}catch(err){document.getElementById('error').textContent=err.message}};
for(const id of ['layer','metric','position'])document.getElementById(id).oninput=draw;
document.getElementById('example').onclick=()=>{document.getElementById('file').value='';document.getElementById('error').textContent='';document.getElementById('metric').value='raw';load(sample)};document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='digital-surgery-trace.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),0)};load(sample);
'''


def activation_viewer():
    body = '''<section><h2>Inspect activity at a specific decision</h2><p>This viewer opens compact local traces. The built-in example is synthetic and is not evidence about Gemma. A unit's state belongs to the token being processed and helps predict the next token; both labels are retained.</p><p class="notice">Activity is continuous and can be signed. High activity alone is not a neuron function. Missing observations are not zeros. Raw activation times output-column norm describes a projection term before normalization, not its causal effect on the response.</p><label>Open local JSON <input id="file" type="file" accept="application/json,.json"></label><button id="example">Synthetic example</button> <button id="download">Download current trace</button><p id="error" role="alert"></p><p id="provenance"></p></section><section><label>Layer <select id="layer"></select></label><label>Measure <select id="metric"><option value="raw">Signed activation</option><option value="weighted">Raw projection magnitude</option></select></label><label>Token window <input id="position" type="range" min="0" value="0"></label><p id="window" class="small"></p><div class="scroll"><canvas id="heat" aria-label="Recorded neuron activity heatmap"></canvas></div><p id="cell">Point to a cell for the token and activation.</p></section><section><h2>Trace contract</h2><p>Schema <code>digital-surgery-trace-v1</code>: <code>tokens</code> has processed/predicted text labels; <code>units</code> has layer, channel, optional down_norm and aligned values. Use null for unrecorded values. Export selected traces rather than full activation tensors. The file stays in this browser tab and is never uploaded.</p><p>The observer must first demonstrate unchanged model logits, choices, cache and state. Scientific selection, editing and confirmation are separate from this exploratory display.</p></section>'''
    return page('Token-aligned activation viewer', body, VIEWER_SCRIPT)


def process():
    steps = [
        ('1. Record the unmodified model', 'Pin source, precision, tokenizer and input IDs. Allow adequate completion. Save actual replies, token choices, termination and executable outcomes before comparing an edit.'),
        ('2. Observe a native trajectory', 'Capture actual post-gating MLP values with a read-only hook. Align processed and predicted tokens. Retain compact channel/layer statistics and failures, including incorrect programs.'),
        ('3. State a causal hypothesis', 'Specify the coordinates, positions and proposed effect. Activity is a candidate signal. Freeze the edit, strengths, random matching and all endpoints before new intervention outputs.'),
        ('4. Apply one declared operation', 'Use scoped zeroing, actual donor values, direct weight assignments, coordinated deletion, cloning or copied modules. Keep an identity control and exact restoration. No optimizer or fitted bridge.'),
        ('5. Test behavior and controls', 'Keep every case and condition. Evaluate executable correctness, error classes, retention, necessity, self controls and counterfactual rescue. An injected-error rescue is not an improvement over the original.'),
        ('6. Save a model artifact', 'Store the pinned parent, exact small tensors or deterministic copy recipe, loader and source hashes under the experiment ID. Label current validation and package-replay scope. Keep private raw contexts separate.'),
        ('7. Release and reduce local duplication', 'Review the complete candidate tree and exact diff, publish only authorized assets and verify the remote tree. Remove regenerable caches and verified duplicate exports. Preserve the evidence needed to reconstruct the result.'),
    ]
    body = '<section><h2>A repeatable surgery process</h2><p>Each stage links a measured observation to a concrete intervention and a reviewable artifact. Development selection is not fresh confirmation. Known steering, pruning, copying and cloning are methods, not novelty claims.</p></section>' + ''.join(f'<section><h2>{escape(title)}</h2><p>{escape(text)}</p></section>' for title, text in steps)
    body += '<section><h2>Work from actual code</h2><p><a href="SURGERY_TOOLKIT.html">Observation and edit APIs</a> · <a href="MODEL_REGISTRY.html">Exact model artifacts</a> · <a href="ACTIVATION_VIEWER.html">Local trace inspection</a> · <a href="GOAL_AND_INTENT.html">Current research specification</a></p></section>'
    return page('Digital Surgery process', body)


def build_assets(dest, writer):
    tools, sources = toolkit(dest)
    models, inventory = model_registry(dest)
    writer('docs/SURGERY_TOOLKIT.html', tools)
    writer('docs/MODEL_REGISTRY.html', models)
    writer('docs/ACTIVATION_VIEWER.html', activation_viewer())
    writer('docs/SURGERY_PROCESS.html', process())
    writer('docs/SURGERY_ASSET_INDEX.json', {'project': 'Digital Surgery', 'source_files': sources, **inventory, 'activation_viewer_default_data': 'synthetic_only', 'training_allowed': False})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Reviewed checkout containing models and source files')
    args = parser.parse_args()
    def write_asset(name, value):
        path = args.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')
    build_assets(args.root, write_asset)
    print('Built four offline HTML tools and their source/model index. No model was loaded.')
