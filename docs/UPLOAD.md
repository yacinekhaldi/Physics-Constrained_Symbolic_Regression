# GitHub Upload Checklist

## Before Upload

1. Confirm Git LFS is installed:

   ```powershell
   git lfs version
   ```

2. Initialize LFS in the repository:

   ```powershell
   git lfs install
   ```

3. Confirm large-file rules:

   ```powershell
   git check-attr filter -- data/raw/ML-CFD-Wavy-Channel-Surrogate.zip
   git check-attr filter -- results/models/random80_20__Nuavg__random_forest.joblib
   ```

4. Add files:

   ```powershell
   git add .gitattributes .gitignore README.md CITATION.cff LICENSE pyproject.toml requirements.txt requirements-optional.txt docs src scripts data results
   ```

5. Verify ignored local content:

   ```powershell
   git status --ignored --short
   ```

   Expected ignored content includes `.venv/`, `__pycache__/`, `literature_text/`, and `data/raw/ML-CFD-Wavy-Channel-Surrogate/`.

6. Verify LFS-tracked files:

   ```powershell
   git lfs ls-files
   ```

7. Commit and push:

   ```powershell
   git commit -m "Initial reproducibility release"
   git branch -M main
   git remote add origin https://github.com/<owner>/<repo>.git
   git push -u origin main
   ```

## Notes

- GitHub rejects normal Git blobs larger than 100 MB. The dataset archive and several model artifacts must remain under Git LFS.
- If repository storage quota is a concern, keep the dataset archive and model artifacts out of Git and attach them to a release or external archive instead.
