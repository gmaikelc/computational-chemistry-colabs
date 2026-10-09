"""Animated Morgan-Tanimoto ranking for Jupyter / Google Colab.

Usage:
    from similarity_screening_v2 import launch_similarity_screening
    screening = launch_similarity_screening(df)
    # The Next Compound button works without background execution.
    # Auto Play runs in a daemon thread, with Pause and Reset controls.
    ranked = screening.ranked
"""
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class SimilarityScreening:
    def __init__(self, df, reference_name='Imatinib (reference)', delay=1.5,
                 radius=2, fp_size=2048, chart_top=12):
        required = {'name', 'mol', 'group'}
        if not required.issubset(df.columns):
            raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
        if df['mol'].isna().any():
            raise ValueError('Some molecules in df are invalid.')
        refs = df.loc[df.name.eq(reference_name)]
        if len(refs) != 1:
            raise ValueError(f'Expected one {reference_name!r} reference; found {len(refs)}.')
        self.reference_name = reference_name
        fpgen = GetMorganGenerator(radius=radius, fpSize=fp_size)
        ref_fp = fpgen.GetFingerprint(refs.iloc[0]['mol'])
        self.candidates = df.loc[~df.name.eq(reference_name)].reset_index(drop=True).copy()
        self.candidates['_similarity'] = self.candidates['mol'].apply(
            lambda m: float(DataStructs.TanimotoSimilarity(ref_fp, fpgen.GetFingerprint(m)))
        )
        self.chart_top = chart_top
        self.index = 0
        self.results = []
        self._lock = threading.RLock()
        self._generation = 0
        self._running = False
        self.speed = widgets.FloatSlider(value=delay, min=.1, max=5, step=.1,
            description='Delay (s):', continuous_update=False,
            style={'description_width':'initial'}, layout=widgets.Layout(width='400px'))
        self.start_button = widgets.Button(description='Auto Play', button_style='success', icon='play')
        self.next_button = widgets.Button(description='Next Compound', button_style='info', icon='step-forward')
        self.pause_button = widgets.Button(description='Pause', button_style='warning', icon='pause')
        self.reset_button = widgets.Button(description='Reset', icon='refresh')
        self.progress = widgets.IntProgress(value=0, min=0, max=len(self.candidates),
            description='Screening:', layout=widgets.Layout(width='480px'))
        self.status = widgets.HTML('<b>Ready.</b> Use Next Compound or Auto Play.')
        self.output = widgets.Output()
        self.panel = widgets.VBox([
            widgets.HBox([self.start_button,self.next_button,self.pause_button,self.reset_button]),
            self.speed,self.progress,self.status,self.output])
        self.start_button.on_click(self._start)
        self.next_button.on_click(self._next)
        self.pause_button.on_click(self._pause)
        self.reset_button.on_click(self._reset)

    @property
    def ranked(self):
        with self._lock:
            if not self.results:
                return pd.DataFrame(columns=['Compound','Group','Tanimoto'])
            return pd.DataFrame(self.results).sort_values(
                'Tanimoto', ascending=False, kind='stable').reset_index(drop=True)

    def show(self):
        display(self.panel)
        self._render()
        return self

    def _advance(self):
        with self._lock:
            if self.index >= len(self.candidates):
                return False
            row = self.candidates.iloc[self.index]
            self.results.append({'Compound':str(row['name']), 'Group':str(row['group']),
                                 'Tanimoto':float(row['_similarity'])})
            self.index += 1
            self.progress.value = self.index
            complete = self.index == len(self.candidates)
        self._render()
        if complete:
            self.progress.bar_style='success'
            self.status.value='<b>Screening completed.</b>'
        return True

    def _render(self):
        with self.output:
            clear_output(wait=True)
            print(f'Evaluated {self.index} of {len(self.candidates)} compounds')
            ranking = self.ranked
            ref = pd.DataFrame([{'Rank':'REF','Compound':self.reference_name,
                                 'Group':'Reference','Tanimoto':1.}])
            if len(ranking):
                table_rank = ranking.copy()
                table_rank.insert(0,'Rank',range(1,len(table_rank)+1))
                table = pd.concat([ref,table_rank],ignore_index=True)
            else:
                table = ref
            display(table.style.format({'Tanimoto':'{:.3f}'})
                    .background_gradient(subset=['Tanimoto'],cmap='YlGn').hide(axis='index'))
            if len(ranking):
                top = ranking.head(self.chart_top).sort_values('Tanimoto')
                fig,ax = plt.subplots(figsize=(9,max(3.5, .35*len(top)+1)))
                ax.barh(top['Compound'],top['Tanimoto'])
                ax.set(xlim=(0,1.07),xlabel='Morgan Tanimoto similarity',title='Live similarity ranking')
                for i,val in enumerate(top['Tanimoto']):
                    ax.text(float(val)+.012,i,f'{val:.3f}',va='center',fontsize=9)
                fig.tight_layout()
                display(fig)
                plt.close(fig)

    def _next(self, _):
        with self._lock:
            if self._running:
                self.status.value='<b>Pause automatic playback before using Next.</b>'
                return
        self._advance()

    def _worker(self, generation):
        try:
            while True:
                with self._lock:
                    if generation != self._generation or self.index >= len(self.candidates):
                        break
                if not self._advance():
                    break
                remaining = self.speed.value
                while remaining > 0:
                    time.sleep(min(.1,remaining))
                    remaining -= .1
                    with self._lock:
                        if generation != self._generation:
                            return
        except Exception as exc:
            self.progress.bar_style='danger'
            self.status.value=f'<b>Animation stopped:</b> {type(exc).__name__}: {exc}'
        finally:
            with self._lock:
                self._running=False
                self.start_button.disabled=False
                self.next_button.disabled=False

    def _start(self, _):
        with self._lock:
            if self._running or self.index >= len(self.candidates):
                return
            self._generation += 1
            generation = self._generation
            self._running=True
            self.start_button.disabled=True
            self.next_button.disabled=True
            self.status.value='<b>Automatic screening in progress...</b>'
        threading.Thread(target=self._worker,args=(generation,),daemon=True).start()

    def _pause(self, _):
        with self._lock:
            self._generation += 1
            self.status.value='<b>Paused.</b>'

    def _reset(self, _):
        with self._lock:
            self._generation += 1
            self.index=0
            self.results=[]
            self.progress.value=0
            self.progress.bar_style='info'
            self.status.value='<b>Ready.</b> Use Next Compound or Auto Play.'
        self._render()


def launch_similarity_screening(df, reference_name='Imatinib (reference)',
                                delay=1.5, radius=2, fp_size=2048, chart_top=12):
    return SimilarityScreening(df,reference_name,delay,radius,fp_size,chart_top).show()
