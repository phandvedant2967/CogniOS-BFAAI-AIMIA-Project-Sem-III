# CogniOS design system

Single source of truth: `app/static/css/cognios.css`. New pages should reuse these tokens and classes and not add new colours, radii or shadows.

**Tokens** (CSS variables on `:root`)
- Surfaces: `--bg` page, `--surface` cards, `--surface-sunken` inputs, `--line` dividers
- Text: `--ink` (navy, headings), `--ink-2` (body), `--ink-3` (labels, secondary)
- Accent: `--accent`, `--accent-strong`, `--accent-soft` (muted blue-purple). One accent only.
- Slate panel: `--slate` for the occasional dark card (`.card.slate`)
- Soft status tints (background + text pairs): `rose`, `blue`, `lav`, `sand`, `ok`. Used for chips and notices only.
- Radius: `--r-sm` 10, `--r-md` 16, `--r-lg` 24, `--r-pill`. Shadow: `--shadow` (one level).
- Type: system/SF/Inter stack; title 28, h2 18, body 14, small 12.5. Space scale `--s1`..`--s6` (4 to 32px).

**Components**: `.shell` / `.sidebar` / `.main` / `.topbar`, `.card` (+ `.card-head`), `.btn` (+ `.secondary`), `.field`, `.select`, `.textarea`, `.segmented`, `.chip` (+ tone), `.notice` (`info`, `warn`, `error`), `.kv`, `table`, `.timeline`, `.score-row`, `.radio`.

**Rules**
- Colour carries meaning only (class tone, notice level). No decorative gradients.
- Missing data uses `.missing` (muted italic "not recorded"). Never a zero or a blank.
- Model output always carries the non-causal wording and the "not calibrated" label (`render.js`: `PROB_NOTE`, `NON_CAUSAL_NOTE`).
- Markup lives in `render.js` (pure, escaped), logic in `logic.js` (pure), DOM and fetch only in `app.js`.

**Stage 9 additions (same tokens, no new colours):** `.filter-bar`, `.kpis` / `.kpi` (`.ai` lavender edge, `.doctor` green edge), `.bar-row` / `.bar-line`, `.legend`, `.supp` (suppressed "<5" cell), `.review-list`. AI predictions use the lavender chip, doctor assessments the green chip, everywhere.
