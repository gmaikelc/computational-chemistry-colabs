"""Browser-controlled molecular similarity ranking for Google Colab.

Usage:
    from similarity_screening_v5 import launch_similarity_screening
    screening = launch_similarity_screening(df, delay=2.0, max_compounds=10)
    ranked = screening.ranked   # Python DataFrame; computed immediately

Start, Pause, Next Compound, Reset and speed adjustment are implemented in
self-contained JavaScript, avoiding Colab ipywidget callbacks / background threads.
"""
import html
import json
import uuid

import pandas as pd
from IPython.display import display, HTML
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class SimilarityScreening:
    def __init__(self, df, delay=1.5, max_compounds=None,
                 reference_name='Imatinib (reference)', radius=2,
                 fp_size=2048, chart_top=12):
        required = {'name', 'mol', 'group'}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f'Missing DataFrame columns: {sorted(missing)}')
        if df['mol'].isna().any():
            raise ValueError('df contains invalid molecules.')
        refs = df.loc[df['name'].eq(reference_name)]
        if len(refs) != 1:
            raise ValueError(f'Expected exactly one {reference_name!r}; found {len(refs)}.')
        if float(delay) < 0:
            raise ValueError('delay must be >= 0')
        if int(chart_top) < 1:
            raise ValueError('chart_top must be positive')
        self.reference_name = reference_name
        self.delay = max(.1, float(delay))
        self.chart_top = int(chart_top)
        fpgen = GetMorganGenerator(radius=int(radius), fpSize=int(fp_size))
        ref_fp = fpgen.GetFingerprint(refs.iloc[0]['mol'])
        candidates = df.loc[~df['name'].eq(reference_name)].copy()
        if max_compounds is not None:
            candidates = candidates.head(max(0, int(max_compounds)))
        candidates = candidates.reset_index(drop=True)
        self.candidates = candidates
        self.candidates['Tanimoto'] = candidates['mol'].map(
            lambda mol: float(DataStructs.TanimotoSimilarity(ref_fp, fpgen.GetFingerprint(mol)))
        )
        self.ranked = (self.candidates[['name', 'group', 'Tanimoto']]
                       .rename(columns={'name': 'Compound', 'group': 'Group'})
                       .sort_values('Tanimoto', ascending=False, kind='stable')
                       .reset_index(drop=True))
        self._id = 'sim_' + uuid.uuid4().hex

    def show(self):
        items = [{'name': str(r['name']), 'group': str(r['group']),
                  'score': float(r['Tanimoto'])}
                 for _, r in self.candidates.iterrows()]
        payload = json.dumps({'items': items, 'reference': self.reference_name,
                              'delay': self.delay, 'top': self.chart_top}, ensure_ascii=True)
        payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        ident = self._id
        markup = r'''
<div id="__IDENT__" style="font-family:Arial,sans-serif;max-width:1000px;border:1px solid #cbd5e1;border-radius:12px;padding:16px;line-height:1.35;background:white;color:#1e293b">
  <h3 style="margin:0 0 12px">Live Morgan/Tanimoto similarity screening</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <button data-action="play" style="background:#188038;color:white;border:0;border-radius:6px;padding:9px 14px;cursor:pointer">▶ Start / Resume</button>
    <button data-action="pause" style="background:#e9a23b;color:black;border:0;border-radius:6px;padding:9px 14px;cursor:pointer">⏸ Pause</button>
    <button data-action="next" style="background:#2563eb;color:white;border:0;border-radius:6px;padding:9px 14px;cursor:pointer">⏭ Next Compound</button>
    <button data-action="reset" style="background:#e2e8f0;color:#1e293b;border:0;border-radius:6px;padding:9px 14px;cursor:pointer">↺ Reset</button>
  </div>
  <div style="display:flex;align-items:center;gap:12px;margin:12px 0;flex-wrap:wrap">
    <label>Delay: <input data-action="speed" type="range" min="0.1" max="5" step="0.1" style="vertical-align:middle;width:210px"></label>
    <b data-role="speed-value"></b>
    <span data-role="status" aria-live="polite">Ready</span>
  </div>
  <div style="background:#e2e8f0;border-radius:5px;height:12px;overflow:hidden"><div data-role="progress" style="background:#16803c;width:0%;height:100%;transition:width .2s"></div></div>
  <p data-role="summary" style="margin:9px 0">Evaluated 0 compounds.</p>
  <div style="max-height:410px;overflow:auto;border:1px solid #e2e8f0;border-radius:6px">
    <table style="border-collapse:collapse;width:100%;font-size:13px">
      <thead><tr style="background:#f1f5f9"><th style="text-align:left;padding:8px">Rank</th><th style="text-align:left;padding:8px">Compound</th><th style="text-align:left;padding:8px">Group</th><th style="text-align:right;padding:8px">Tanimoto</th></tr></thead>
      <tbody data-role="rows"></tbody>
    </table>
  </div>
  <h4 style="margin:14px 0 8px">Live top-ranked candidates</h4>
  <div data-role="bars" style="font-size:12px"></div>
  <p style="color:#64748b;font-size:12px;margin-bottom:0">Scores use Morgan fingerprints (radius 2 by default). Similarities are calculated once in Python; this panel reveals and ranks them one compound at a time. Controls operate in the browser.</p>
</div>
<script>
(function () {
 const root = document.getElementById('__IDENT__');
 if (!root || root.dataset.ready === '1') return;
 root.dataset.ready = '1';
 const data = __DATA__;
 let index = 0, timer = null, speed = data.delay;
 const find = (s) => root.querySelector(s);
 const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const slider = find('[data-action="speed"]');
 slider.value = Math.max(.1, Math.min(5, speed));
 find('[data-role="speed-value"]').textContent = Number(slider.value).toFixed(1) + ' s';
 function stop() { if(timer !== null) { clearTimeout(timer); timer = null; } }
 function rows() {
   return data.items.slice(0, index).map((x,i)=>({...x, order:i})).sort((a,b)=>b.score-a.score || a.order-b.order);
 }
 function render() {
   const ranked = rows();
   find('[data-role="summary"]').textContent = 'Evaluated ' + index + ' / ' + data.items.length + ' compounds' + (index ? ' — Latest: ' + data.items[index-1].name : '');
   find('[data-role="progress"]').style.width = (data.items.length ? index/data.items.length*100 : 100) + '%';
   const reference = '<tr style="background:#ecfdf5"><td style="padding:7px">REF</td><td style="padding:7px">'+esc(data.reference)+'</td><td style="padding:7px">Reference</td><td style="padding:7px;text-align:right"><b>1.000</b></td></tr>';
   find('[data-role="rows"]').innerHTML = reference + ranked.map((x,i)=>'<tr style="border-top:1px solid #e2e8f0"><td style="padding:7px">'+(i+1)+'</td><td style="padding:7px">'+esc(x.name)+'</td><td style="padding:7px">'+esc(x.group)+'</td><td style="padding:7px;text-align:right">'+x.score.toFixed(3)+'</td></tr>').join('');
   const top = ranked.slice(0, data.top);
   find('[data-role="bars"]').innerHTML = top.length ? top.map(x=>'<div style="display:grid;grid-template-columns:minmax(150px,230px) 1fr 45px;align-items:center;gap:8px;margin:7px 0"><span style="overflow-wrap:anywhere">'+esc(x.name)+'</span><div style="height:17px;background:#e2e8f0;border-radius:3px;overflow:hidden"><div style="height:100%;width:'+(100*x.score).toFixed(2)+'%;background:#4682b4"></div></div><span>'+x.score.toFixed(3)+'</span></div>').join('') : '<span style="color:#64748b">Press Start or Next Compound to begin.</span>';
   if(index === data.items.length) { stop(); find('[data-role="status"]').textContent = 'Completed'; }
 }
 function advance() { if (index < data.items.length) index++; render(); }
 function loop() { if(index >= data.items.length) return; advance(); if(index < data.items.length) timer = setTimeout(loop, speed*1000); }
 function click(action, cb) { find('[data-action="'+action+'"]').addEventListener('click', cb); }
 click('play', () => { if(timer !== null || index >= data.items.length) return; find('[data-role="status"]').textContent = 'Playing'; loop(); });
 click('pause', () => { stop(); find('[data-role="status"]').textContent = 'Paused'; });
 click('next', () => { stop(); advance(); if(index < data.items.length) find('[data-role="status"]').textContent = 'Manual step'; });
 click('reset', () => { stop(); index=0; render(); find('[data-role="status"]').textContent = 'Ready'; });
 slider.addEventListener('input', () => { speed = Number(slider.value); find('[data-role="speed-value"]').textContent = speed.toFixed(1) + ' s'; if(timer !== null) { stop(); timer=setTimeout(loop, speed*1000); } });
 render();
})();
</script>
'''
        markup = markup.replace('__IDENT__', html.escape(ident, quote=True)).replace('__DATA__', payload)
        display(HTML(markup))
        return self


def launch_similarity_screening(df, delay=1.5, max_compounds=None,
                                reference_name='Imatinib (reference)',
                                radius=2, fp_size=2048, chart_top=12):
    """Display an interactive browser-based similarity animation and return results object."""
    return SimilarityScreening(df, delay, max_compounds, reference_name,
                               radius, fp_size, chart_top).show()
