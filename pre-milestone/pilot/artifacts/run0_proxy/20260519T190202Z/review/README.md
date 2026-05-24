# Run 0 Review Dashboard

Static, local HTML dashboard for reviewing all 500 prompts and 8 rollouts each from Run 0. No server required.

## Quick start

From this directory (`review/`):

```bash
python build_review_dashboard.py
./serve.sh
```

Then open **http://localhost:8765** (recommended for LaTeX / KaTeX).

You can also `open index.html` directly, but `file://` may block the KaTeX CDN; a yellow banner will suggest using `serve.sh`.

## Regenerate after jsonl changes

The build script reads parent artifact files (default: **both** raw and cleaned labels):

- `../raw_predictions.jsonl`
- `../cleaned/predictions.jsonl`
- `../prompt_inputs.jsonl`

and writes `data.js`. Re-run whenever those files change:

```bash
python build_review_dashboard.py
```

Single-source builds (backward compatible):

```bash
python build_review_dashboard.py --source raw
python build_review_dashboard.py --source cleaned
```

Then refresh the browser tab.

## Requirements

- Python 3.9+ (stdlib only)
- A modern browser
- Internet connection for KaTeX CDN (math rendering) and Google Fonts (optional; UI works without fonts)

## Usage

| Action | Control |
|--------|---------|
| Select prompt | Click in left list |
| Next / previous (filtered list) | `j` / `↓` or **Next**; `k` / `↑` or **Prev** |
| Switch label source (raw ↔ cleaned) | `;` (header pill shows **RAW** / **CLEAN**) |
| Toggle “What is Run 0?” panel | `i` or click summary in sidebar |
| Toggle quick stats panel | `s` or click summary in sidebar |
| Expand / collapse all completions (current prompt) | `l` or checkbox in header |
| Filter | All · Has correct · No correct · Partial (1–7) — uses **active** source counts |
| Search | `prompt_id` substring |

Each prompt shows the problem, gold answer, parsed answer, and completions rendered with **KaTeX** (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`). Math in long completions is typeset when you expand a rollout (or press `l` for all eight).

**Delta highlighting:** Amber highlight when **parsed** or **correct** differs between raw and clean (not cluster id; not run-on fallback rejections). Prompts with any such delta show **Δ** in the sidebar.

## Files

| File | Role |
|------|------|
| `build_review_dashboard.py` | Builds `data.js` from jsonl (default `--source both`) |
| `index.html` | Main UI shell |
| `app.js` | Client logic |
| `styles.css` | Layout and theme |
| `data.js` | Generated bundle (gitignored-sized; regenerate locally) |

`data.js` is generated and may be ~8–10 MB. Commit it only if you want a frozen snapshot; otherwise regenerate after pulling artifacts.
