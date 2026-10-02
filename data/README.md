# Data handling

The input workbook is kept at the project root for now. The preparation command reads it without editing the source file and writes reconciled customer-level line marts and cutoff snapshots under ignored `data/interim/` and `data/processed/` folders.

The workbook is **Online Retail II** by Daqing Chen, distributed through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) under CC BY 4.0. Preserve source attribution when sharing any part of the project.

The pipeline writes the source SHA-256, workbook worksheet/row lineage for modeled lines, cleaning counts, cutoff labels, and target maturity to a local data-quality manifest. Re-run `python scripts/project.py prepare` after changing the workbook or data policy. Derived Parquet files contain customer IDs and are excluded from Git.

Cleaning counts should be regenerated from the saved manifest, not copied from the earlier exploratory audit. Exact overlapping worksheet rows are reconciled by maximum source-sheet multiplicity; duplicate rows within one sheet remain possible legitimate line items. Credits stay separate because the workbook does not reliably connect them to original sales.
