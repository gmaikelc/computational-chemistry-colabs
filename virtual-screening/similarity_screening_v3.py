"""Reliable sequential Morgan/Tanimoto classroom screening for Google Colab.

Use:
    from similarity_screening_v3 import run_similarity_screening
    ranked = run_similarity_screening(df, delay=1.5)

Runs in the active cell: no buttons, background threads, or widget manager needed.
Interrupt the Colab cell to stop. To change speed, rerun with another delay.
"""
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def run_similarity_screening(df, delay=1.5, max_compounds=None,
                             reference_name='Imatinib (reference)',
                             radius=2, fp_size=2048, chart_top=12,
                             show_chart=True):
    """Show a live, sorted ranking and return the final ranked DataFrame."""
    required = {'name','group','mol'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing df columns: {sorted(required - set(df.columns))}')
    if df['mol'].isna().any():
        raise ValueError('df contains invalid molecules (mol=None).')
    refs = df.loc[df['name'].eq(reference_name)]
    if len(refs) != 1:
        raise ValueError(f'Expected exactly one reference {reference_name!r}; found {len(refs)}')
    if delay < 0:
        raise ValueError('delay must be >= 0')

    fpgen = GetMorganGenerator(radius=int(radius), fpSize=int(fp_size))
    ref_fp = fpgen.GetFingerprint(refs.iloc[0]['mol'])
    # Similarities computed up front; reveal one at a time for teaching.
    candidates = df.loc[~df['name'].eq(reference_name)].reset_index(drop=True).copy()
    if max_compounds is not None:
        candidates = candidates.head(int(max_compounds))
    candidates['Tanimoto'] = candidates['mol'].map(
        lambda m: DataStructs.TanimotoSimilarity(ref_fp, fpgen.GetFingerprint(m))
    )
    records = []
    for idx, row in candidates.iterrows():
        records.append({'Compound': row['name'], 'Group': row['group'],
                        'Tanimoto': float(row['Tanimoto'])})
        ranked = (pd.DataFrame(records)
                  .sort_values('Tanimoto', ascending=False, kind='stable')
                  .reset_index(drop=True))
        ref_table = pd.DataFrame([{'Rank': 'REF', 'Compound': reference_name,
                                   'Group': 'Reference', 'Tanimoto': 1.0}])
        candidate_table = ranked.copy()
        candidate_table.insert(0, 'Rank', range(1, len(candidate_table) + 1))
        table = pd.concat([ref_table, candidate_table], ignore_index=True)

        clear_output(wait=True)
        display(HTML(f'<h3>Similarity screening: {idx+1} / {len(candidates)}</h3>'
                     f'<p>Just evaluated: <b>{row["name"]}</b> — '
                     f'Tanimoto = <b>{float(row["Tanimoto"]):.3f}</b></p>'))
        display(table.style.format({'Tanimoto':'{:.3f}'})
                .background_gradient(subset=['Tanimoto'], cmap='YlGn')
                .hide(axis='index'))

        if show_chart:
            top = ranked.head(int(chart_top)).sort_values('Tanimoto')
            fig, ax = plt.subplots(figsize=(9,max(3.5,.38 * len(top) + 1)))
            ax.barh(top['Compound'], top['Tanimoto'])
            ax.set_xlim(0, 1.07)
            ax.set_xlabel('Morgan fingerprint Tanimoto similarity')
            ax.set_title('Live similarity ranking')
            for pos, score in enumerate(top['Tanimoto']):
                ax.text(float(score)+.012, pos, f'{score:.3f}', va='center')
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        if idx < len(candidates)-1:
            time.sleep(delay)

    if not records:
        print('No candidates were provided.')
        return pd.DataFrame(columns=['Compound','Group','Tanimoto'])
    print(f'Completed: {len(records)} candidates screened.')
    return ranked
