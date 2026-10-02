"""Generate and execute a portable, plain-Python audit companion notebook.

No IPython-specific features are used. The cells are executed sequentially with
CPython because Jupyter/nbformat are not installed in this research environment.
The output records that execution backend explicitly.
"""

from contextlib import redirect_stdout
from html import escape
from io import StringIO
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "id": uuid4().hex[:8], "metadata": {}, "source": source,
            "execution_count": None, "outputs": []}


cells = [
    markdown("# Supplied workbook: reproducible scope audit\n\n"
             "## tl;dr\n\nSummary is populated from the executed audit below.\n\n"
             "This is a data-quality companion for the project design, not a predictive-model experiment."),
    markdown("## Context & Methods\n\n"
             "The controlling input is online_retail_II.xlsx, with every worksheet read. "
             "The exact profiling implementation is scripts/audit_dataset.py.\n\n"
             "### Key Assumptions\n\n"
             "- Dates remain in their source-local, timezone-naive form.\n"
             "- Forecast intervals are end-exclusive and must be completely observed.\n"
             "- Global exact deduplication and a numeric stock-code pattern define a conservative sensitivity scenario, not final cleaning.\n"
             "- No upstream event log proves recording completeness.\n\n"
             "All code cells were executed sequentially as plain Python by scripts/build_audit_notebook.py. "
             "A Jupyter kernel is not used by this generator; the cells are ordinary Python and can also run in Jupyter."),
    code("from pathlib import Path\nimport json\nimport runpy\nimport pandas as pd\n\n"
         "root = Path.cwd().resolve()\n"
         "if not (root / 'scripts' / 'audit_dataset.py').exists():\n    root = root.parent\n"
         "assert (root / 'online_retail_II.xlsx').exists(), 'Run from the project root or notebooks folder.'\n"
         "print('Input: online_retail_II.xlsx; all worksheets; unchanged source file.')"),
    markdown("## Data\n\n### 1. Recompute the full workbook audit\n\n"
             "This cell reads the workbook and regenerates the aggregate evidence. No model is fitted. "
             "The script performs null, duplicate, key consistency, sign, code, temporal and forecast-maturity checks."),
    code("audit_module = runpy.run_path(str(root / 'scripts' / 'audit_dataset.py'))\n"
         "profile = audit_module['audit']()\n"
         "assert sum(sheet['rows'] for sheet in profile['sheets']) == profile['raw_rows']\n"
         "print('Sheet row counts reconcile with the full raw population.')"),
    markdown("## Results\n\n### 2. Review worksheet coverage and quality issues"),
    code("print(pd.DataFrame(profile['sheets'])[['sheet', 'rows', 'date_min', 'date_max', 'missing_customer_rows']].to_string(index=False))\n"
         "checks = {\n"
         "    'Raw invoice lines': profile['raw_rows'],\n"
         "    'Missing customer lines': profile['nulls']['customer_id']['rows'],\n"
         "    'Missing customer share (%)': round(profile['nulls']['customer_id']['pct'], 2),\n"
         "    'Exact duplicate excess lines': profile['exact_duplicate_excess_rows'],\n"
         "    'Cross-sheet repeated record hashes': profile['cross_sheet_duplicate_hashes'],\n"
         "    'Negative quantity lines': profile['negative_quantity_rows'],\n"
         "    'Nonpositive price lines': profile['nonpositive_price_rows'],\n"
         "}\n"
         "print(pd.Series(checks, name='Measured value').to_string())"),
    markdown("The sheets overlap in early December 2010. Missing IDs affect customer-level coverage. "
             "Credits and special codes require separate policies; the table does not classify every such event as an error."),
    markdown("### 3. Check the conservative purchase sample"),
    code("scenario = profile['provisional_purchase_scenario']\n"
         "keys = ['candidate_product_rows', 'orders', 'customers', 'one_order_customers', 'repeat_order_customers', 'one_purchase_day_customers', 'repeat_purchase_day_customers']\n"
         "print(pd.Series({key: scenario[key] for key in keys}).to_string())\n"
         "assert scenario['one_order_customers'] + scenario['repeat_order_customers'] == scenario['customers']\n"
         "assert scenario['one_purchase_day_customers'] + scenario['repeat_purchase_day_customers'] == scenario['customers']"),
    markdown("These counts describe the audit sensitivity scenario. Final source reconciliation and product classification can change them. "
             "Repeat purchase-day customers, rather than repeat invoice customers, determine daily Gamma-Gamma fitting eligibility."),
    markdown("### 4. Check complete forecast horizons"),
    code("snapshots = pd.DataFrame(profile['snapshots'])\n"
         "columns = ['cutoff', 'horizon_days', 'label_end_exclusive', 'fully_observed', 'historical_customers', 'inactivity_pct_all_history']\n"
         "print(snapshots[columns].round({'inactivity_pct_all_history': 2}).to_string(index=False))\n"
         "assert snapshots.loc[~snapshots['fully_observed'], 'inactivity_pct_all_history'].isna().all()\n"
         "assert snapshots.loc[snapshots['fully_observed'], 'inactivity_pct_all_history'].between(0, 100).all()\n"
         "print('Incomplete labels stay missing; observed inactivity percentages have valid bounds.')"),
    markdown("The 90-day final test can begin in September 2011. The six-calendar-month final test can begin in June 2011 and end "
             "before December 2011; the 180-day diagnostic has slightly different boundaries. Final-period outcome availability does not "
             "prove predictive accuracy, and outcome rates mix cohort composition with seasonality."),
    markdown("## Takeaways\n\nThe workbook supports the planned historical customer intelligence project. "
             "Reconcile the worksheet overlap, preserve signed credits and anonymous sales, and enforce cutoff/target boundaries "
             "before fitting models. Treat forecast revenue, permanent churn and intervention profit as distinct quantities. "
             "See research/DATASET_ASSESSMENT.md and research/PROJECT_DESIGN.md for interpretation and the proposed architecture."),
]

namespace = {"__name__": "__audit_notebook__"}
execution_count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    execution_count += 1
    print(f"Executing audit notebook cell {execution_count}...", flush=True)
    capture = StringIO()
    with redirect_stdout(capture):
        exec(compile(cell["source"], f"audit-notebook-cell-{execution_count}", "exec"), namespace)
    cell["execution_count"] = execution_count
    if capture.getvalue():
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": capture.getvalue()}]

profile = namespace["profile"]
scenario = profile["provisional_purchase_scenario"]
cells[0]["source"] = (
    "# Supplied workbook: reproducible scope audit\n\n## tl;dr\n\n"
    f"The workbook has {profile['raw_rows']:,} invoice lines across two overlapping worksheets. "
    f"{profile['nulls']['customer_id']['pct']:.2f}% lack customer IDs. The conservative purchase scenario contains "
    f"{scenario['customers']:,} customers and {scenario['repeat_purchase_day_customers']:,} repeat purchase-day customers. "
    "It supports the planned project after careful cleaning and temporal validation.\n\n"
    "This is a data-quality companion, not a predictive-model experiment."
)
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "audit_execution": {"backend": "Sequential CPython exec; no IPython features or Jupyter kernel", "code_cells_executed": execution_count},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
assert len({cell["id"] for cell in cells}) == len(cells)
assert all(isinstance(cell["source"], str) and cell["cell_type"] in {"markdown", "code"} for cell in cells)
assert all(cell["execution_count"] is not None for cell in cells if cell["cell_type"] == "code")
output = ROOT / "notebooks" / "00_dataset_scope_audit.ipynb"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
assert json.loads(output.read_text(encoding="utf-8"))["nbformat"] == 4

parts = ["<!doctype html><meta charset='utf-8'><title>Workbook audit companion</title>",
         "<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.5 system-ui;color:#17212b}"
         "pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 Consolas,monospace}"
         "section{padding:14px 0;border-bottom:1px solid #ddd}.code{background:#f5f7fa;padding:16px}</style>"]
for cell in cells:
    if cell["cell_type"] == "markdown":
        parts.append("<section><pre>" + escape(cell["source"]) + "</pre></section>")
    else:
        parts.append("<details><summary>Python cell " + str(cell["execution_count"]) + "</summary><pre class='code'>" + escape(cell["source"]) + "</pre></details>")
        for result in cell["outputs"]:
            parts.append("<section><pre>" + escape(result["text"]) + "</pre></section>")
(ROOT / "research" / "evidence" / "audit_notebook_preview.html").write_text("\n".join(parts), encoding="utf-8")
print(f"Executed {execution_count} plain-Python cells; saved {output.relative_to(ROOT)}", flush=True)
