"""Colab-friendly sequential Tanimoto screening, with visible per-step snapshots.

Usage:
    from similarity_screening_v4 import run_similarity_screening
    ranked = run_similarity_screening(df, delay=1.5, max_compounds=10)

By default frames accumulate rather than being cleared: this avoids a common
Colab issue where clear_output(wait=True) makes a running cell appear frozen.
Set clear_previous=True to try an in-place animation instead.
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
                             show_chart=True, clear_previous=False):
    """Show successive ranking snapshots and return final ranked candidate DataFrame."""
    required = {'name', 'group', 'mol'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    if df['mol'].isna().any():
        raise ValueError('df contains invalid molecules (mol=None).')
    refs = df.loc[df['name'].eq(reference_name)]
    if len(refs) != 1:
        raise ValueError(f'Expected one reference, found {len(refs)}')
    if delay < 0:
        raise ValueError('delay must be >= 0')

    fpgen = GetMorganGenerator(radius=int(radius), fpSize=int(fp_size))
    ref_fp = fpgen.GetFingerprint(refs.iloc[0]['mol'])
    candidates = df.loc[~df['name'].eq(reference_name)].reset_index(drop=True).copy()
    if max_compounds is not None:
        candidates = candidates.head(int(max_compounds))
    print(f'Starting similarity screening: {len(candidates)} candidates', flush=True)

    records = []
    for idx, row in candidates.iterrows():
        score = DataStructs.TanimotoSimilarity(ref_fp, fpgen.GetFingerprint(row['mol']))
        records.append({'Compound': row['name'], 'Group': row['group'], 'Tanimoto': score})
        ranked = (pd.DataFrame(records)
                  .sort_values('Tanimoto', ascending=False, kind='stable')
                  .reset_index(drop=True))
        ref_table = pd.DataFrame([{'Rank':'REF', 'Compound':reference_name,
                                   'Group':'Reference', 'Tanimoto':1.0}])
        ctab = ranked.copy()
        ctab.insert(0, 'Rank', range(1, len(ctab)+1))
        table = pd.concat([ref_table, ctab], ignore_index=True)

        if clear_previous:
            clear_output(wait=False)
        print(f'Compound {idx+1}/{len(candidates)}: {row["name"]} | Tanimoto={score:.3f}', flush=True)
        display(table.style.format({'Tanimoto':'{:.3f}'})
                .background_gradient(subset=['Tanimoto'], cmap='YlGn')
                .hide(axis='index'))
        if show_chart:
            top = ranked.head(int(chart_top)).sort_values('Tanimoto')
            fig, ax = plt.subplots(figsize=(9, max(3.5, .38*len(top)+1)))
            ax.barh(top['Compound'], top['Tanimoto'])
            ax.set_xlim(0, 1.07)
            ax.set_xlabel('Morgan fingerprint Tanimoto similarity')
            ax.set_title(f'Ranking after {idx+1} of {len(candidates)} candidates')
            for pos, val in enumerate(top['Tanimoto']):
                ax.text(float(val)+.012, pos, f'{val:.3f}', va='center')
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        if idx+1 < len(candidates):
            time.sleep(float(delay))

    if not records:
        print('No candidates were provided.', flush=True)
        return pd.DataFrame(columns=['Compound','Group','Tanimoto'])
    print(f'Completed: {len(records)} candidates.', flush=True)
    return ranked
