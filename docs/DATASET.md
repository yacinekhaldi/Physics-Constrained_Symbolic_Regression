# Dataset

## Source

The experiments use:

Prince Kumar and K. M. Pandey, *CFD-informed machine learning surrogate dataset for thermo-hydraulic prediction in partially porous wavy channels*, Mendeley Data, Version 2, DOI: `10.17632/5b5n3cg32n.2`.

## Files

- `data/raw/ML-CFD-Wavy-Channel-Surrogate.zip` - downloaded source archive.
- `data/raw/mendeley_files.json` - source file metadata captured during download.
- `data/raw/mendeley_snapshot.json` - local source metadata snapshot.
- `data/processed/canonical_wavy_channel_dataset.csv` - canonical CSV used by this workflow.
- `data/processed/column_mapping.json` - source-to-canonical column mapping.

The extracted directory `data/raw/ML-CFD-Wavy-Channel-Surrogate/` is intentionally ignored by Git because it duplicates the archive and contains many large files. Recreate it with `scripts/download_dataset.py`.

## Retained Variables

Inputs:

- `Re` - Reynolds number, dimensionless, 25 to 500.
- `Pr` - Prandtl number, dimensionless, 3 to 50.
- `Da` - Darcy number, dimensionless, 1e-6 to 1e-3.
- `porosity` - porous-zone porosity, dimensionless, 0.70 to 0.85.
- `thickness` - porous-slab thickness, mm, 0.10 to 0.30.
- `amplitude` - wavy-wall amplitude, mm, 0.00 to 0.30.
- `wavelength` - wavy-wall wavelength, mm, 0.00 to 5.00. A value of zero denotes the straight-channel case.

Targets:

- `Nuavg` - average Nusselt number.
- `DelP_Pa` - pressure drop in Pa.

## Checksums

SHA256:

- `data/raw/ML-CFD-Wavy-Channel-Surrogate.zip`: `4B6486704115BE239A3D04344D94AA1AA5736CA685C8B56B6C62D0F69D334945`
- `data/processed/canonical_wavy_channel_dataset.csv`: `D028865707FB42707FD4790D4FCE17963F4E26C3E606BCCFD9D78D2DD79696E8`
