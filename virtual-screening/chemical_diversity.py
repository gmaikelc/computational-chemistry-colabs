"""Murcko-scaffold diversity analysis for RDKit / Google Colab.

Example
-------
from chemical_diversity import analyze_chemical_diversity
results = analyze_chemical_diversity(df, n_scaffolds=10)
diverse = results.diverse
candidates = results.candidates
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass
class DiversityResults:
    candidates: pd.DataFrame
    diverse: pd.DataFrame
    scaffold_counts: pd.DataFrame


def analyze_chemical_diversity(
    df,
    reference_name="Imatinib (reference)",
    n_scaffolds=10,
    similarity_radius=2,
    fp_size=2048,
    show_structures=True,
    show_property_space=True,
    show_tables=True,
    mols_per_row=3,
    scaffold_color=(0.20, 0.65, 0.95),
):
    """Select the top-Tanimoto candidate per distinct Bemis–Murcko scaffold.

    Highlights scaffold atoms and bonds in each full molecule. Acyclic
    compounds have no Murcko scaffold and are explicitly marked as such;
    they share one '[acyclic]' category for this simple teaching heuristic.
    The input DataFrame is not modified.
    """
    required = {"name", "group", "mol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing DataFrame columns: {sorted(missing)}")
    if df["mol"].isna().any():
        raise ValueError("Some molecules are missing or invalid in df['mol'].")
    ref = df.loc[df["name"].eq(reference_name)]
    if len(ref) != 1:
        raise ValueError(f"Expected exactly one reference '{reference_name}', found {len(ref)}")
    if n_scaffolds < 1:
        raise ValueError("n_scaffolds must be >= 1")

    ref_mol = ref.iloc[0]["mol"]
    fpgen = GetMorganGenerator(radius=int(similarity_radius), fpSize=int(fp_size))
    ref_fp = fpgen.GetFingerprint(ref_mol)
    candidates = df.loc[~df["name"].eq(reference_name)].copy().reset_index(drop=True)
    candidates["Tanimoto"] = candidates["mol"].map(
        lambda m: DataStructs.TanimotoSimilarity(ref_fp, fpgen.GetFingerprint(m))
    )
    candidates["MW"] = candidates["mol"].map(Descriptors.MolWt)
    candidates["cLogP"] = candidates["mol"].map(Crippen.MolLogP)

    def scaffold_smiles(m):
        scaffold = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() else "[acyclic]"

    candidates["scaffold"] = candidates["mol"].map(scaffold_smiles)
    counts = (candidates.groupby("scaffold").size()
              .sort_values(ascending=False).rename("n").to_frame())
    diverse = (candidates.sort_values("Tanimoto", ascending=False, kind="stable")
               .drop_duplicates("scaffold").head(int(n_scaffolds)).copy()
               .reset_index(drop=True))

    if show_tables:
        print("Candidate scaffold counts (top 10):")
        display(counts.head(10))
        print("Selected representatives (highest Tanimoto per scaffold):")
        display(diverse[["name", "Tanimoto", "scaffold"]].round(3))

    if show_property_space:
        fig, ax = plt.subplots(figsize=(9, 6))
        for group, sub in candidates.groupby("group"):
            ax.scatter(sub["MW"], sub["cLogP"], label=str(group), s=65, alpha=0.8)
        ax.scatter([Descriptors.MolWt(ref_mol)], [Crippen.MolLogP(ref_mol)],
                   marker="*", s=250, label=reference_name, zorder=5)
        if not diverse.empty:
            ax.scatter(diverse["MW"], diverse["cLogP"], facecolors="none",
                       edgecolors="black", s=160, linewidths=1.5,
                       label="Distinct scaffold representatives")
        ax.set(xlabel="Molecular weight (g/mol)", ylabel="cLogP",
               title="Chemical property space: virtual screening library")
        ax.legend()
        fig.tight_layout()
        plt.show()

    if show_structures and not diverse.empty:
        print("Blue highlights: Bemis–Murcko scaffold atoms and bonds")
        print("Unhighlighted substituents are side chains; acyclic compounds have no Murcko scaffold.")
        atom_lists, bond_lists, atom_colors, bond_colors, legends = [], [], [], [], []
        for _, row in diverse.iterrows():
            mol = row["mol"]
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            atom_ids = list(mol.GetSubstructMatch(scaffold)) if scaffold.GetNumAtoms() else []
            if scaffold.GetNumAtoms() and not atom_ids:
                raise RuntimeError(f"Could not map scaffold to molecule: {row['name']}")
            atom_set = set(atom_ids)
            bond_ids = [b.GetIdx() for b in mol.GetBonds()
                        if b.GetBeginAtomIdx() in atom_set and b.GetEndAtomIdx() in atom_set]
            atom_lists.append(atom_ids)
            bond_lists.append(bond_ids)
            atom_colors.append({i: scaffold_color for i in atom_ids})
            bond_colors.append({i: scaffold_color for i in bond_ids})
            legends.append(f"{row['name']}\nTanimoto = {row['Tanimoto']:.3f}\n"
                           + ("Acyclic: no scaffold" if not atom_ids else "Murcko scaffold highlighted"))
        #display(Draw.MolsToGridImage(
        #    diverse["mol"].tolist(), legends=legends,
        #    molsPerRow=int(mols_per_row), subImgSize=(420, 300),
        #    highlightAtomLists=atom_lists, highlightBondLists=bond_lists,
        #    highlightAtomColors=atom_colors, highlightBondColors=bond_colors,
        #    useSVG=False,
        #))



       # Drawing options
        draw_options = rdMolDraw2D.MolDrawOptions()

        draw_options.legendFontSize = 54  # Compound names and scores
        draw_options.baseFontSize = 1.1   # Atom labels (N, O, Cl, etc.)
        draw_options.minFontSize = 16     # Minimum atom font size

        display(
            Draw.MolsToGridImage(
                diverse["mol"].tolist(),
                legends=legends,
                molsPerRow=int(mols_per_row),
                subImgSize=(520, 380),
                highlightAtomLists=atom_lists,
                highlightBondLists=bond_lists,
                highlightAtomColors=atom_colors,
                highlightBondColors=bond_colors,
                drawOptions=draw_options,
                useSVG=False
            )
        )

    return DiversityResults(candidates=candidates, diverse=diverse, scaffold_counts=counts)
