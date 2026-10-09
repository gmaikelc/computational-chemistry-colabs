"""Interactive RDKit virtual-screening teaching demo for Jupyter/Google Colab.

Usage:
    from virtual_screening import launch_screening
    launch_screening(df)
"""

import time
import html
import pandas as pd
import ipywidgets as widgets
from IPython.display import HTML, clear_output, display
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

IMATINIB_SMILES = (
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)"
    "NC4=NC=CC(=N4)C5=CN=CC=C5"
)
PATTERNS = {
    "amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])"),
    "aromatic_N": Chem.MolFromSmarts("[n]"),
    "tertiary_amine": Chem.MolFromSmarts("[NX3;H0;!$(N-C=O);!$(N-S(=O)=O)]"),
}
COLORS = {
    "amide": (0.20, 0.45, 0.95),
    "aromatic_N": (0.15, 0.75, 0.30),
    "tertiary_amine": (1.00, 0.55, 0.10),
}
NAMES = {"amide": "Amide", "aromatic_N": "Aromatic nitrogen", "tertiary_amine": "Tertiary amine"}


def prepare_library(df):
    """Return an independent copy of df with corrected reference and calculated descriptors.

    Note: Does not change any analog structures in the caller's dataframe.
    """
    required = {"name", "group", "mol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    prepared = df.copy()
    matches = prepared.index[prepared["name"].eq("Imatinib (reference)")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one 'Imatinib (reference)' row")
    reference = Chem.MolFromSmiles(IMATINIB_SMILES)
    idx = matches[0]
    prepared.at[idx, "mol"] = reference
    if "smiles" in prepared:
        prepared.at[idx, "smiles"] = IMATINIB_SMILES
    if "canonical_smiles" in prepared:
        prepared.at[idx, "canonical_smiles"] = Chem.MolToSmiles(reference)
    if prepared["mol"].isna().any():
        raise ValueError("Molecules include invalid or missing RDKit mol objects")
    fpgen = GetMorganGenerator(radius=2, fpSize=2048)
    fp_ref = fpgen.GetFingerprint(reference)
    prepared["Tanimoto"] = prepared["mol"].map(
        lambda mol: DataStructs.TanimotoSimilarity(fp_ref, fpgen.GetFingerprint(mol)))
    prepared["MW"] = prepared["mol"].map(Descriptors.MolWt)
    prepared["cLogP"] = prepared["mol"].map(Crippen.MolLogP)
    prepared["HBD"] = prepared["mol"].map(Lipinski.NumHDonors)
    prepared["HBA"] = prepared["mol"].map(Lipinski.NumHAcceptors)
    prepared["Ro5_violations"] = (
        (prepared["MW"] > 500).astype(int) + (prepared["cLogP"] > 5).astype(int)
        + (prepared["HBD"] > 5).astype(int) + (prepared["HBA"] > 10).astype(int))
    for label, query in PATTERNS.items():
        prepared[label] = prepared["mol"].map(lambda mol, q=query: mol.HasSubstructMatch(q))
    prepared["demo_fragment_count"] = prepared[list(PATTERNS)].sum(axis=1)
    return prepared


def _highlights(mol, checked):
    ac, bc = {}, {}
    for label in checked:
        color = COLORS[label]
        for matched in mol.GetSubstructMatches(PATTERNS[label]):
            atoms = set(matched)
            for a in atoms:
                ac[a] = color
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms:
                    bc[bond.GetIdx()] = color
    return ac, bc


def _draw(mol, name, similarity, checked=(), size=(650, 355)):
    ac, bc = _highlights(mol, checked)
    display(Draw.MolsToGridImage(
        [mol], legends=[f"{name}\nTanimoto = {similarity:.3f}"], molsPerRow=1,
        subImgSize=size, highlightAtomLists=[list(ac)], highlightBondLists=[list(bc)],
        highlightAtomColors=[ac], highlightBondColors=[bc], useSVG=False))


def _status(results):
    parts = ['<h3>Structural fragment search</h3>']
    for label in PATTERNS:
        outcome = ('FOUND (+1)' if results[label] else 'NOT FOUND (+0)') if label in results else 'Not checked yet'
        rgb = COLORS[label]
        color = '#%02x%02x%02x' % tuple(int(255*v) for v in rgb)
        parts.append(f'<p style="color:{color}"><b>{NAMES[label]}:</b> {outcome}</p>')
    parts.append(f'<b>Detected patterns: {sum(results.values())}/3</b>')
    display(HTML(''.join(parts)))


def _final(selected, total):
    result = pd.DataFrame(selected)
    display(HTML('<h2>Virtual screening completed</h2>'))
    print(f'Screened: {total} | Selected: {len(selected)} | Rejected: {total-len(selected)}')
    if result.empty:
        print('No compounds passed the selected criteria.')
        return result
    result = result.sort_values('Tanimoto', ascending=False).reset_index(drop=True)
    display(result[['name', 'Tanimoto', 'amide', 'aromatic_N', 'tertiary_amine',
                    'demo_fragment_count', 'Ro5_violations', 'MW', 'cLogP']].round(3))
    molecules, legends, als, bls, acs, bcs = [], [], [], [], [], []
    for _, row in result.iterrows():
        ac, bc = _highlights(row['mol'], PATTERNS)
        molecules.append(row['mol'])
        legends.append(f"{row['name']}\nTanimoto = {row['Tanimoto']:.3f}\nFragments = {row['demo_fragment_count']}/3")
        als.append(list(ac)); bls.append(list(bc)); acs.append(ac); bcs.append(bc)
    display(Draw.MolsToGridImage(molecules, legends=legends, molsPerRow=3,
        subImgSize=(450, 320), highlightAtomLists=als, highlightBondLists=bls,
        highlightAtomColors=acs, highlightBondColors=bcs, useSVG=False))
    display(HTML('<p><b>Blue:</b> Amide &nbsp; <b>Green:</b> Aromatic nitrogen '
                 '&nbsp; <b>Orange:</b> Tertiary amine</p>'))
    return result


def launch_screening(df):
    """Display a settings panel and a button to animate the virtual screening.

    Returns a widget panel. The completed shortlist is also saved as
    panel.shortlist (and the prepared data as panel.prepared_df).
    """
    prepared = prepare_library(df)
    candidates = prepared.loc[prepared['group'].ne('reference')].reset_index(drop=True)
    if candidates.empty:
        raise ValueError('No candidate molecules are available')

    def slider(cls, label, value, lo, hi, step):
        w = cls(value=value, min=lo, max=hi, step=step, description=label,
                style={'description_width': '170px'}, layout=widgets.Layout(width='480px'),
                continuous_update=False)
        return w

    sim = slider(widgets.FloatSlider, 'Min Tanimoto:', 0.35, 0, 1, 0.05)
    ro5 = slider(widgets.IntSlider, 'Max Ro5 violations:', 1, 0, 4, 1)
    frag = slider(widgets.IntSlider, 'Min fragments:', 2, 0, 3, 1)
    frag_delay = slider(widgets.FloatSlider, 'Fragment delay (s):', 1.5, 0.1, 8, 0.1)
    result_delay = slider(widgets.FloatSlider, 'Decision delay (s):', 2.5, 0.1, 10, 0.1)
    molecule_delay = slider(widgets.FloatSlider, 'Molecule delay (s):', 1., 0, 5, 0.5)
    num = slider(widgets.IntSlider, 'Compounds:', len(candidates), 1, len(candidates), 1)
    controls = [sim, ro5, frag, frag_delay, result_delay, molecule_delay, num]
    start = widgets.Button(description='Start Virtual Screening', button_style='success', icon='play',
                           layout=widgets.Layout(width='240px', height='42px'))
    prog = widgets.IntProgress(value=0, min=0, max=len(candidates), description='Progress:',
                               layout=widgets.Layout(width='480px'))
    status = widgets.HTML('<b>Ready.</b> Configure settings and press Start.')
    output = widgets.Output()
    panel = widgets.VBox([
        widgets.HTML('<h2>Virtual Screening — Settings</h2><h3>Screening Criteria</h3>'),
        sim, ro5, frag, widgets.HTML('<h3>Animation Settings</h3>'),
        frag_delay, result_delay, molecule_delay, num, start, prog, status, output])
    panel.shortlist = pd.DataFrame()
    panel.prepared_df = prepared

    def run(_):
        conditions = (sim.value, ro5.value, frag.value)
        delays = (frag_delay.value, result_delay.value, molecule_delay.value)
        pool = candidates.head(num.value)
        picked = []
        prog.max = len(pool); prog.value = 0; prog.bar_style = 'info'
        start.disabled = True
        for w in controls: w.disabled = True
        status.value = '<b>Screening in progress...</b>'
        try:
            with output:
                for i, (_, row) in enumerate(pool.iterrows(), start=1):
                    name = row['name']; mol = row['mol']; tanimoto = row['Tanimoto']
                    clear_output(wait=True)
                    display(HTML(f'<h2>Compound {i}/{len(pool)}: {html.escape(str(name))}</h2>'))
                    _draw(mol, name, tanimoto)
                    time.sleep(delays[2])
                    results, checked = {}, []
                    for label, query in PATTERNS.items():
                        results[label] = mol.HasSubstructMatch(query)
                        checked.append(label)
                        clear_output(wait=True)
                        display(HTML(f'<h2>Compound {i}/{len(pool)}: {html.escape(str(name))}</h2>'
                                     f'<h3>Checking: {NAMES[label]}</h3>'))
                        _draw(mol, name, tanimoto, checked)
                        _status(results)
                        time.sleep(delays[0])
                    passed_sim = tanimoto >= conditions[0]
                    passed_ro5 = row['Ro5_violations'] <= conditions[1]
                    passed_frag = sum(results.values()) >= conditions[2]
                    passed = passed_sim and passed_ro5 and passed_frag
                    record = row.to_dict()
                    record.update(results)
                    record['demo_fragment_count'] = sum(results.values())
                    record['passes_demo_rules'] = passed
                    if passed: picked.append(record)
                    clear_output(wait=True)
                    color = '#188038' if passed else '#c5221f'
                    display(HTML(f'<h2 style="color:{color}">{"PASS — SELECTED" if passed else "REJECT"}: '
                                 f'{html.escape(str(name))}</h2>'))
                    _draw(mol, name, tanimoto, PATTERNS)
                    _status(results)
                    display(HTML(f'<p>{"✓" if passed_sim else "✗"} Similarity: {tanimoto:.3f} '
                                 f'(minimum {conditions[0]:.2f})</p>'
                                 f'<p>{"✓" if passed_ro5 else "✗"} Ro5 violations: '
                                 f'{row["Ro5_violations"]} (maximum {conditions[1]})</p>'
                                 f'<p>{"✓" if passed_frag else "✗"} Structural patterns: '
                                 f'{sum(results.values())}/3 (minimum {conditions[2]})</p>'))
                    prog.value = i
                    print(f'Progress: {i}/{len(pool)} | Selected: {len(picked)} | Rejected: {i-len(picked)}')
                    time.sleep(delays[1])
                clear_output(wait=True)
                panel.shortlist = _final(picked, len(pool))
            prog.bar_style = 'success'
            status.value = '<b>Screening completed.</b> Results: panel.shortlist'
        except Exception as exc:
            prog.bar_style = 'danger'
            status.value = f'<b>Screening error:</b> {html.escape(str(exc))}'
            with output:
                print(f'Screening failed: {exc}')
        finally:
            start.disabled = False
            for w in controls: w.disabled = False

    start.on_click(run)
    display(panel)
    return panel
