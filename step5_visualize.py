"""
Step 5: Interactive Graph Explorer
===================================
Reads the extended coactivation graph (.graphml) and generates
a standalone HTML file with D3.js force-directed visualisation.

Features:
  - Neuron nodes (coloured by layer block)
  - Concept nodes (coloured by COCO supercategory)
  - Coactivation edges (red, ρ-weighted)
  - Concept edges (teal, IoU-weighted)
  - Slider filters for ρ and IoU thresholds
  - Layer block toggles
  - Concept focus dropdown
  - Click-to-highlight neighbourhood
  - Patch metadata on neuron info panel

Reads: results/extended_coactivation_graph.graphml
Writes: results/graph_explorer.html

Usage:
    python step5_visualize.py
"""

import json
import numpy as np
import networkx as nx
from collections import defaultdict

from config import (
    EXTENDED_GRAPH_PATH, EXPLORER_HTML_PATH, CONCEPT_LAYER,
    SUPERCATEGORY_MAP, COCO_ID_TO_NAME,
    ensure_dirs,
)

# Layer → block mapping
def layer_block(layer_name):
    if '.' in layer_name:
        return layer_name.split('.')[0]
    return layer_name

LAYER_ORDER = [
    'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
    'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
    'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
    'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2',
    'fc',
]


def extract_graph_data(graphml_path):
    """Read .graphml and extract nodes + edges for the visualisation."""
    print(f"  Loading: {graphml_path}")
    G = nx.read_graphml(str(graphml_path))
    print(f"    Raw: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    nodes = []
    edges = []
    neuron_stats = defaultdict(lambda: {'coact_neighbors': 0, 'concepts': []})

    # --- Nodes ---
    for nid, d in G.nodes(data=True):
        if d.get('node_type') == 'concept':
            cname = d.get('concept_name', str(nid))
            supercat = SUPERCATEGORY_MAP.get(cname, 'other')
            nodes.append({
                'id': nid, 'type': 'concept', 'name': cname,
                'concept_id': int(d.get('concept_id', 0)),
                'supercategory': supercat, 'n_neurons': 0,
            })
        else:
            layer = d.get('layer', 'unknown')
            block = layer_block(layer)
            nidx = int(d.get('inlayer_id', 0))
            # Parse patch metadata if present
            top_images = json.loads(d.get('top_images', '[]'))
            top_acts = json.loads(d.get('top_activations', '[]'))
            nodes.append({
                'id': nid, 'type': 'neuron', 'layer': layer, 'block': block,
                'neuron_idx': nidx, 'coact_neighbors': 0, 'n_concepts': 0,
                'top_concepts': [], 'top_images': top_images[:3],
                'top_activations': top_acts[:3],
            })

    node_map = {n['id']: n for n in nodes}

    # --- Edges (skip self-loops, filter weak coactivation) ---
    MIN_CORR = 0.60
    n_coact = 0
    n_concept = 0
    n_skipped = 0

    for u, v, d in G.edges(data=True):
        if u == v:
            n_skipped += 1
            continue
        w = float(d.get('weight', 0))
        etype = d.get('edge_type', 'coactivates_with')

        if etype == 'activates_on':
            edges.append({'source': u, 'target': v, 'weight': round(w, 4), 'type': 'concept'})
            n_concept += 1
            # Stats
            if u in node_map:
                cname = node_map.get(v, {}).get('name', '')
                neuron_stats[u]['concepts'].append({'name': cname, 'iou': w})
            if v in node_map and node_map[v]['type'] == 'concept':
                node_map[v]['n_neurons'] = node_map[v].get('n_neurons', 0) + 1
        else:
            if w < MIN_CORR:
                n_skipped += 1
                continue
            u_layer = G.nodes[u].get('layer', '')
            v_layer = G.nodes[v].get('layer', '')
            edges.append({
                'source': u, 'target': v, 'weight': round(w, 4),
                'type': 'coactivation', 'cross_layer': u_layer != v_layer,
            })
            n_coact += 1
            neuron_stats[u]['coact_neighbors'] += 1
            neuron_stats[v]['coact_neighbors'] += 1

    # Attach stats to neuron nodes
    for n in nodes:
        if n['type'] == 'neuron':
            s = neuron_stats[n['id']]
            n['coact_neighbors'] = s['coact_neighbors']
            top_c = sorted(s['concepts'], key=lambda x: x['iou'], reverse=True)[:5]
            n['top_concepts'] = top_c
            n['n_concepts'] = len(s['concepts'])

    print(f"    Export: coact={n_coact}, concept={n_concept}, skipped={n_skipped}")

    layers = sorted(set(n['layer'] for n in nodes if n['type'] == 'neuron'),
                    key=lambda x: LAYER_ORDER.index(x) if x in LAYER_ORDER else 99)

    return {'nodes': nodes, 'edges': edges, 'layers': layers}


def generate_html(graph_data, output_path):
    """Embed data into HTML template and write file."""
    data_json = json.dumps(graph_data, separators=(',', ':'))

    html = HTML_TEMPLATE.replace('__GRAPH_DATA__', data_json)

    with open(str(output_path), 'w') as f:
        f.write(html)
    print(f"  Saved: {output_path} ({len(html)//1024} KB)")


# The full HTML is identical to what we generated before —
# I'll keep it compact here. The key change: neuron info panel
# now shows patch metadata (top activating images).

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SCAG Explorer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:#0a0e17;color:#c8d6e5;overflow:hidden;height:100vh}
.app{display:grid;grid-template-columns:340px 1fr;height:100vh}
.sidebar{background:#0f1420;border-right:1px solid rgba(255,255,255,0.06);padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:16px}
.sidebar::-webkit-scrollbar{width:4px}
.sidebar::-webkit-scrollbar-thumb{background:#2a3444;border-radius:2px}
h1{font-size:17px;font-weight:700;color:#fff;letter-spacing:-0.02em}
.subtitle{font-size:11px;color:#5a6a80;margin-top:3px}
.section-label{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#4a5a6a;margin-bottom:6px}
.stats{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px}
.stat-box{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:8px;text-align:center}
.stat-box .num{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#fff}
.stat-box .lbl{font-size:9px;color:#5a6a80;margin-top:2px}
.slider-group{display:flex;flex-direction:column;gap:12px}
.slider-item label{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;color:#8a9ab0}
.slider-item label .val{font-family:'JetBrains Mono',monospace;color:#fff;font-size:11px;background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:3px}
input[type="range"]{-webkit-appearance:none;width:100%;height:3px;border-radius:2px;background:#1a2030;outline:none}
input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;cursor:pointer;border:2px solid #0f1420}
.corr-slider::-webkit-slider-thumb{background:#ff6b6b;box-shadow:0 0 6px rgba(255,107,107,0.3)}
.iou-slider::-webkit-slider-thumb{background:#4ecdc4;box-shadow:0 0 6px rgba(78,205,196,0.3)}
select{width:100%;padding:7px;background:#1a2030;border:1px solid rgba(255,255,255,0.1);color:#c8d6e5;font-family:'DM Sans',sans-serif;font-size:11px;border-radius:5px;cursor:pointer}
.checkbox-group{display:flex;flex-wrap:wrap;gap:4px}
.chip{font-size:10px;padding:3px 8px;border-radius:12px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03);cursor:pointer;color:#8a9ab0;transition:all 0.15s;user-select:none}
.chip.active{background:rgba(255,255,255,0.1);color:#fff;border-color:rgba(255,255,255,0.2)}
.chip:hover{background:rgba(255,255,255,0.08)}
.btn-row{display:flex;gap:6px}
.btn{flex:1;padding:7px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.04);color:#8a9ab0;font-family:'DM Sans',sans-serif;font-size:11px;border-radius:5px;cursor:pointer;transition:all 0.15s}
.btn:hover{background:rgba(255,255,255,0.08);color:#fff}
.legend{display:flex;flex-wrap:wrap;gap:5px}
.legend-item{display:flex;align-items:center;gap:4px;font-size:10px;color:#6a7a90}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.legend-sq{width:8px;height:8px;border-radius:2px;flex-shrink:0}
.info-panel{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:12px;min-height:60px}
.info-panel h3{font-size:13px;color:#fff;margin-bottom:4px}
.info-panel .detail{font-size:11px;color:#7a8a9a;line-height:1.6}
.info-panel .detail strong{color:#c8d6e5}
.info-panel .concept-tag{display:inline-block;font-size:10px;padding:1px 6px;border-radius:8px;margin:1px 2px;border:1px solid rgba(78,205,196,0.3);color:#4ecdc4}
.info-panel .patch-info{font-size:10px;color:#5a6a80;margin-top:4px;font-family:'JetBrains Mono',monospace}
.graph-area{position:relative;overflow:hidden}
svg{width:100%;height:100%}
.concept-label{font-family:'DM Sans',sans-serif;font-weight:600;fill:#fff;pointer-events:none;text-shadow:0 1px 4px rgba(0,0,0,0.8)}
.tooltip{position:absolute;background:#1a2030;border:1px solid rgba(255,255,255,0.12);border-radius:6px;padding:8px 12px;font-size:11px;color:#c8d6e5;pointer-events:none;opacity:0;transition:opacity 0.12s;z-index:100;max-width:280px;box-shadow:0 6px 24px rgba(0,0,0,0.5)}
.edge-label{position:absolute;bottom:12px;right:12px;display:flex;gap:12px;font-size:10px;color:#5a6a80;pointer-events:none}
.edge-label span{display:flex;align-items:center;gap:4px}
.edge-label .line-sample{width:20px;height:2px;border-radius:1px}
</style>
</head>
<body>
<div class="app">
<div class="sidebar">
    <div><h1>Semantic Coactivation Graph</h1><div class="subtitle">ResNet-18 · SCAG Explorer</div></div>
    <div class="stats">
        <div class="stat-box"><div class="num" id="s-nodes">0</div><div class="lbl">Nodes</div></div>
        <div class="stat-box"><div class="num" id="s-coact">0</div><div class="lbl">Coact.</div></div>
        <div class="stat-box"><div class="num" id="s-concept">0</div><div class="lbl">Concept</div></div>
        <div class="stat-box"><div class="num" id="s-total">0</div><div class="lbl">Total</div></div>
    </div>
    <div>
        <div class="section-label">Thresholds</div>
        <div class="slider-group">
            <div class="slider-item"><label>Coactivation (ρ) <span class="val" id="corr-val">0.75</span></label>
                <input type="range" class="corr-slider" id="corr-slider" min="0.60" max="0.95" step="0.01" value="0.75"></div>
            <div class="slider-item"><label>Concept IoU <span class="val" id="iou-val">0.15</span></label>
                <input type="range" class="iou-slider" id="iou-slider" min="0.10" max="0.35" step="0.01" value="0.15"></div>
        </div>
    </div>
    <div><div class="section-label">Layer Filter</div><div class="checkbox-group" id="layer-chips"></div></div>
    <div><div class="section-label">Focus Concept</div><select id="concept-filter"><option value="all">Show all concepts</option></select></div>
    <div><div class="section-label">Edge Display</div>
        <div class="checkbox-group">
            <div class="chip active" id="toggle-coact" onclick="toggleEdge('coact')">Coactivation</div>
            <div class="chip active" id="toggle-concept" onclick="toggleEdge('concept')">Concepts</div>
            <div class="chip" id="toggle-cross" onclick="toggleEdge('cross')">Cross-layer only</div>
        </div></div>
    <div class="btn-row"><button class="btn" id="btn-reset">Reset</button><button class="btn" id="btn-freeze">Freeze</button></div>
    <div><div class="section-label">Legend</div><div class="legend" id="legend"></div></div>
    <div><div class="section-label">Selected Node</div>
        <div class="info-panel" id="info-panel"><div class="detail" style="color:#5a6a80">Click any node to inspect.</div></div></div>
</div>
<div class="graph-area" id="graph-area">
    <div class="tooltip" id="tooltip"></div>
    <div class="edge-label"><span><div class="line-sample" style="background:#ff6b6b"></div>coactivates_with</span><span><div class="line-sample" style="background:#4ecdc4"></div>activates_on</span></div>
</div>
</div>
<script>
const DATA=__GRAPH_DATA__;
const LC={'layer1':'#6366f1','layer2':'#3b82f6','layer3':'#14b8a6','layer4':'#f59e0b','fc':'#ef4444'};
const SC={person:'#ef4444',vehicle:'#3b82f6',outdoor:'#8b5cf6',animal:'#22c55e',accessory:'#f59e0b',sports:'#ec4899',kitchen:'#6b7280',food:'#eab308',furniture:'#15803d',electronic:'#dc2626',appliance:'#06b6d4',indoor:'#92400e',other:'#64748b'};
let cT=0.75,iT=0.15,fC='all',aL=new Set(['layer4']),sC=true,sCp=true,xO=false,fz=false;
const area=document.getElementById('graph-area'),W=area.clientWidth,H=area.clientHeight;
const svg=d3.select('#graph-area').append('svg').attr('width',W).attr('height',H);
const g=svg.append('g');
const zm=d3.zoom().scaleExtent([0.05,8]).on('zoom',e=>g.attr('transform',e.transform));
svg.call(zm);const tt=d3.select('#tooltip');
const nM={};DATA.nodes.forEach(n=>nM[n.id]=n);
const blocks=[...new Set(DATA.layers.map(l=>l.split('.')[0]))];
const cc=document.getElementById('layer-chips');
blocks.forEach(b=>{const c=document.createElement('div');c.className='chip'+(b==='layer4'?' active':'');c.textContent=b;c.onclick=()=>{c.classList.toggle('active');if(c.classList.contains('active'))aL.add(b);else aL.delete(b);upd()};cc.appendChild(c)});
const cs=document.getElementById('concept-filter');
DATA.nodes.filter(n=>n.type==='concept').sort((a,b)=>(b.n_neurons||0)-(a.n_neurons||0)).forEach(c=>{const o=document.createElement('option');o.value=c.id;o.textContent=`${c.name} (${c.n_neurons||0})`;cs.appendChild(o)});
const lg=document.getElementById('legend');
Object.entries(LC).forEach(([n,c])=>{lg.innerHTML+=`<div class="legend-item"><div class="legend-dot" style="background:${c}"></div>${n}</div>`});
Object.entries(SC).forEach(([n,c])=>{lg.innerHTML+=`<div class="legend-item"><div class="legend-sq" style="background:${c}"></div>${n}</div>`});
const lG=g.append('g'),nG=g.append('g'),laG=g.append('g');
const sim=d3.forceSimulation().force('charge',d3.forceManyBody().strength(-20)).force('center',d3.forceCenter(W/2,H/2)).force('collide',d3.forceCollide().radius(8)).force('x',d3.forceX(W/2).strength(0.02)).force('y',d3.forceY(H/2).strength(0.02)).alphaDecay(0.025);
function toggleEdge(t){if(t==='coact'){sC=!sC;document.getElementById('toggle-coact').classList.toggle('active')}if(t==='concept'){sCp=!sCp;document.getElementById('toggle-concept').classList.toggle('active')}if(t==='cross'){xO=!xO;document.getElementById('toggle-cross').classList.toggle('active')}upd()}
function upd(){
let vE=DATA.edges.filter(e=>{
if(e.type==='coactivation'){if(!sC)return false;if(e.weight<cT)return false;if(xO&&!e.cross_layer)return false;const s=nM[e.source]||nM[typeof e.source==='object'?e.source.id:e.source],t=nM[e.target]||nM[typeof e.target==='object'?e.target.id:e.target];if(!s||!t)return false;return aL.has(s.block)&&aL.has(t.block)}
if(e.type==='concept'){if(!sCp)return false;if(e.weight<iT)return false;const s=nM[e.source]||nM[typeof e.source==='object'?e.source.id:e.source];if(!s)return false;return aL.has(s.block)}return false});
if(fC!=='all'){const ce=vE.filter(e=>{const t=typeof e.target==='object'?e.target.id:e.target;return e.type==='concept'&&t===fC});const cn=new Set();cn.add(fC);ce.forEach(e=>{cn.add(typeof e.source==='object'?e.source.id:e.source)});vE=vE.filter(e=>{const s=typeof e.source==='object'?e.source.id:e.source,t=typeof e.target==='object'?e.target.id:e.target;if(e.type==='concept')return t===fC;return cn.has(s)&&cn.has(t)})}
const vi=new Set();vE.forEach(e=>{vi.add(typeof e.source==='object'?e.source.id:e.source);vi.add(typeof e.target==='object'?e.target.id:e.target)});
const vN=DATA.nodes.filter(n=>vi.has(n.id));
const nc=vE.filter(e=>e.type==='coactivation').length,np=vE.filter(e=>e.type==='concept').length;
document.getElementById('s-nodes').textContent=vN.length;document.getElementById('s-coact').textContent=nc;document.getElementById('s-concept').textContent=np;document.getElementById('s-total').textContent=nc+np;
const lk=lG.selectAll('line').data(vE,e=>{const s=typeof e.source==='object'?e.source.id:e.source,t=typeof e.target==='object'?e.target.id:e.target;return s+'|'+t});lk.exit().remove();
const lkM=lk.enter().append('line').merge(lk).attr('stroke',e=>e.type==='coactivation'?'#ff6b6b':'#4ecdc4').attr('stroke-opacity',e=>e.type==='coactivation'?0.06+0.5*(e.weight-0.6):0.15+0.6*e.weight).attr('stroke-width',e=>e.type==='coactivation'?0.5+3*(e.weight-0.6):0.8+3*e.weight);
const nd=nG.selectAll('.node').data(vN,n=>n.id);nd.exit().remove();
const ne=nd.enter().append('g').attr('class','node').call(d3.drag().on('start',ds).on('drag',dr).on('end',de)).on('mouseover',showT).on('mouseout',hideT).on('click',clickN);
ne.filter(n=>n.type==='neuron').append('circle').attr('r',n=>3+(n.n_concepts||0)*0.3).attr('fill',n=>{const c=LC[n.block]||'#4a5a6a';return c+'33'}).attr('stroke',n=>LC[n.block]||'#5a6a7a').attr('stroke-width',0.8);
ne.filter(n=>n.type==='concept').append('rect').attr('width',16).attr('height',16).attr('x',-8).attr('y',-8).attr('rx',3).attr('fill',n=>SC[n.supercategory]||'#64748b').attr('stroke','#fff').attr('stroke-width',1.5).attr('stroke-opacity',0.7);
const la=laG.selectAll('text').data(vN.filter(n=>n.type==='concept'),n=>n.id);la.exit().remove();la.enter().append('text').attr('class','concept-label').attr('font-size',10).attr('dy',-14).attr('text-anchor','middle').text(n=>n.name);
sim.nodes(vN);sim.force('link',d3.forceLink(vE).id(n=>n.id).distance(e=>e.type==='coactivation'?30:50).strength(e=>e.type==='coactivation'?e.weight*0.15:e.weight*0.6));sim.alpha(0.5).restart();
sim.on('tick',()=>{lkM.attr('x1',e=>e.source.x).attr('y1',e=>e.source.y).attr('x2',e=>e.target.x).attr('y2',e=>e.target.y);nG.selectAll('.node').attr('transform',n=>`translate(${n.x},${n.y})`);laG.selectAll('text').attr('x',n=>n.x).attr('y',n=>n.y)})}
function showT(ev,d){let h;if(d.type==='neuron'){const cs=(d.top_concepts||[]).map(c=>`<span class="concept-tag">${c.name} ${c.iou.toFixed(3)}</span>`).join(' ');h=`<strong>${d.layer}:${d.neuron_idx}</strong><br>Coact. neighbors: ${d.coact_neighbors||0}<br>Concepts: ${d.n_concepts||0}<br>${cs}`}else{h=`<strong>${d.name}</strong> <span style="color:${SC[d.supercategory]}">(${d.supercategory})</span><br>Detecting neurons: ${d.n_neurons||0}`}tt.html(h).style('left',(ev.pageX+14)+'px').style('top',(ev.pageY-8)+'px').style('opacity',1)}
function hideT(){tt.style('opacity',0)}
function clickN(ev,d){
const p=document.getElementById('info-panel');
if(d.type==='neuron'){const cs=(d.top_concepts||[]).map(c=>`<span class="concept-tag">${c.name} (IoU ${c.iou.toFixed(3)})</span>`).join(' ');
const pi=d.top_images&&d.top_images.length?`<div class="patch-info">Top images: ${d.top_images.join(', ')}</div>`:'';
p.innerHTML=`<h3>${d.layer} : neuron ${d.neuron_idx}</h3><div class="detail"><strong>Block:</strong> ${d.block}<br><strong>Coactivation neighbors:</strong> ${d.coact_neighbors||0}<br><strong>Concepts:</strong> ${d.n_concepts||0}<br>${cs||'<em>No concepts</em>'}${pi}</div>`}
else{p.innerHTML=`<h3>${d.name}</h3><div class="detail"><strong>Supercategory:</strong> ${d.supercategory}<br><strong>Detecting neurons:</strong> ${d.n_neurons||0}</div>`}
const cn=new Set();cn.add(d.id);lG.selectAll('line').each(function(e){const s=typeof e.source==='object'?e.source.id:e.source,t=typeof e.target==='object'?e.target.id:e.target;if(s===d.id)cn.add(t);if(t===d.id)cn.add(s)});
nG.selectAll('.node').style('opacity',n=>cn.has(n.id)?1:0.08);lG.selectAll('line').style('opacity',function(e){const s=typeof e.source==='object'?e.source.id:e.source,t=typeof e.target==='object'?e.target.id:e.target;return(s===d.id||t===d.id)?0.7:0.01});laG.selectAll('text').style('opacity',n=>cn.has(n.id)?1:0.08)}
function ds(ev){if(!ev.active)sim.alphaTarget(0.12).restart();ev.subject.fx=ev.subject.x;ev.subject.fy=ev.subject.y}
function dr(ev){ev.subject.fx=ev.x;ev.subject.fy=ev.y}
function de(ev){if(!ev.active)sim.alphaTarget(0);ev.subject.fx=null;ev.subject.fy=null}
document.getElementById('corr-slider').addEventListener('input',function(){cT=parseFloat(this.value);document.getElementById('corr-val').textContent=cT.toFixed(2);upd()});
document.getElementById('iou-slider').addEventListener('input',function(){iT=parseFloat(this.value);document.getElementById('iou-val').textContent=iT.toFixed(2);upd()});
document.getElementById('concept-filter').addEventListener('change',function(){fC=this.value;upd()});
document.getElementById('btn-reset').addEventListener('click',function(){fC='all';document.getElementById('concept-filter').value='all';nG.selectAll('.node').style('opacity',1);lG.selectAll('line').style('opacity',null);laG.selectAll('text').style('opacity',1);svg.transition().duration(400).call(zm.transform,d3.zoomIdentity);upd()});
document.getElementById('btn-freeze').addEventListener('click',function(){fz=!fz;this.textContent=fz?'Unfreeze':'Freeze';if(fz)sim.stop();else sim.alpha(0.3).restart()});
svg.on('click',function(ev){if(ev.target.tagName==='svg'){nG.selectAll('.node').style('opacity',1);lG.selectAll('line').style('opacity',null);laG.selectAll('text').style('opacity',1)}});
upd();
</script>
</body>
</html>"""


def main():
    print("=" * 60)
    print("  Step 5: Interactive Graph Explorer")
    print("=" * 60)

    ensure_dirs()
    graph_data = extract_graph_data(EXTENDED_GRAPH_PATH)
    generate_html(graph_data, EXPLORER_HTML_PATH)

    print("=" * 60)


if __name__ == '__main__':
    main()