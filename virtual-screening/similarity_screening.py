"""Interactive RDKit Morgan-fingerprint similarity ranking for Jupyter/Google Colab.

Usage:
    from similarity_screening import launch_similarity_screening
    screening = launch_similarity_screening(df)
    # After completing the animation:
    ranked = screening.ranked
"""
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class SimilarityScreening:
    """Maintain interactive notebook controls and screening results."""

    def __init__(self, df, reference_name='Imatinib (reference)',
                 delay=1.5, radius=2, fp_size=2048, chart_top=12):
        needed = {'name', 'mol', 'group'}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f'Missing required DataFrame columns: {sorted(missing)}')
        if not (df['mol'].notna().all()):
            raise ValueError('df contains missing/invalid RDKit molecules')
        refs = df.loc[df['name'].eq(reference_name)]
        if len(refs) != 1:
            raise ValueError(f'Expected one reference named {reference_name!r}, found {len(refs)}')
        self.reference_name = reference_name
        self.fpgen = GetMorganGenerator(radius=radius, fpSize=fp_size)
        self.ref_fp = self.fpgen.GetFingerprint(refs.iloc[0]['mol'])
        self.candidates = df.loc[~df['name'].eq(reference_name)].copy().reset_index(drop=True)
        self.candidates['fp'] = self.candidates['mol'].map(self.fpgen.GetFingerprint)
        self.chart_top = chart_top
        self.index = 0
        self.results = []
        self._task = None
        self._run_id = 0
        self._paused = True
        self._running = False

        self.speed_slider = widgets.FloatSlider(
            value=delay, min=0.1, max=5., step=0.1, description='Delay (s):',
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'))
        self.start_button = widgets.Button(description='Start / Resume', button_style='success', icon='play')
        self.pause_button = widgets.Button(description='Pause', button_style='warning', icon='pause')
        self.reset_button = widgets.Button(description='Reset', icon='refresh')
        self.progress = widgets.IntProgress(value=0, min=0, max=len(self.candidates),
            description='Screening:', bar_style='info', layout=widgets.Layout(width='500px'))
        self.status = widgets.HTML('<b>Ready to screen.</b>')
        self.output = widgets.Output()
        self.panel = widgets.VBox([
            widgets.HBox([self.start_button, self.pause_button, self.reset_button]),
            self.speed_slider, self.progress, self.status, self.output])
        self.start_button.on_click(self._on_start)
        self.pause_button.on_click(self._on_pause)
        self.reset_button.on_click(self._on_reset)

    @property
    def ranked(self):
        """All evaluated candidates sorted by descending Tanimoto similarity."""
        if not self.results:
            return pd.DataFrame(columns=['Compound', 'Group', 'Tanimoto'])
        return (pd.DataFrame(self.results)
                .sort_values('Tanimoto', ascending=False, kind='stable')
                .reset_index(drop=True))

    def show(self):
        display(self.panel)
        self._render()
        return self

    def _render(self):
        with self.output:
            clear_output(wait=True)
            print(f'Evaluated {self.index} of {len(self.candidates)} candidate compounds')
            ranking = self.ranked
            ref_row = pd.DataFrame([{
                'Rank': 'REF', 'Compound': self.reference_name,
                'Group': 'Reference', 'Tanimoto': 1.0}])
            if len(ranking):
                ranked_table = ranking.copy()
                ranked_table.insert(0, 'Rank', range(1, len(ranking) + 1))
                table = pd.concat([ref_row, ranked_table], ignore_index=True)
            else:
                table = ref_row
            display(table.style.format({'Tanimoto': '{:.3f}'})
                    .background_gradient(subset=['Tanimoto'], cmap='YlGn')
                    .hide(axis='index'))
            if not ranking.empty:
                top = ranking.head(self.chart_top).sort_values('Tanimoto')
                fig, ax = plt.subplots(figsize=(10, max(3.5, 0.38 * len(top) + 1.0)))
                ax.barh(top['Compound'], top['Tanimoto'], color='steelblue')
                ax.set_xlim(0, 1.08)
                ax.set_xlabel('Morgan fingerprint Tanimoto similarity')
                ax.set_title('Live Virtual Screening Ranking')
                for i, score in enumerate(top['Tanimoto']):
                    ax.text(score + .012, i, f'{score:.3f}', va='center', fontsize=9)
                plt.tight_layout()
                plt.show()
                plt.close(fig)

    async def _animate(self, run_id):
        self._running = True
        try:
            while (self.index < len(self.candidates) and not self._paused
                   and run_id == self._run_id):
                row = self.candidates.iloc[self.index]
                score = DataStructs.TanimotoSimilarity(self.ref_fp, row['fp'])
                self.results.append({'Compound': row['name'], 'Group': row['group'],
                                     'Tanimoto': float(score)})
                self.index += 1
                self.progress.value = self.index
                self._render()
                await asyncio.sleep(self.speed_slider.value)
            if self.index == len(self.candidates) and run_id == self._run_id:
                self.progress.bar_style = 'success'
                self.status.value = '<b>Screening completed.</b>'
            elif run_id == self._run_id:
                self.status.value = '<b>Paused.</b>'
        except Exception as exc:
            self.progress.bar_style = 'danger'
            self.status.value = f'<b>Error:</b> {exc}'
            raise
        finally:
            self._running = False

    def _on_start(self, _):
        if self._running or self.index >= len(self.candidates):
            return
        self._paused = False
        self.status.value = '<b>Screening in progress...</b>'
        self._task = asyncio.create_task(self._animate(self._run_id))

    def _on_pause(self, _):
        self._paused = True
        self.status.value = '<b>Pausing...</b>'

    def _on_reset(self, _):
        self._paused = True
        self._run_id += 1
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._running = False
        self.index = 0
        self.results = []
        self.progress.value = 0
        self.progress.bar_style = 'info'
        self.status.value = '<b>Ready to screen.</b>'
        self._render()


def launch_similarity_screening(df, reference_name='Imatinib (reference)',
                                delay=1.5, radius=2, fp_size=2048, chart_top=12):
    """Display an interactive similarity-screening animation for a molecular DataFrame."""
    return SimilarityScreening(df, reference_name, delay, radius, fp_size, chart_top).show()
