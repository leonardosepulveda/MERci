---
description: Write a prompt_rationales/ HTML doc explaining how a task's conclusions were reached, for later review/learning
---

Write a rationale document for a recently-completed task in this project, in
`prompt_rationales/{same-basename}.html` (create the folder if it doesn't
exist), matching the basename of the corresponding `prompt_history/` entry
(so `prompt_history/2026_07_13_1259_foo.md` pairs with
`prompt_rationales/2026_07_13_1259_foo.html`).

**Which task**: if the user gave an argument, use it to find the matching
`prompt_history/` entry (by basename substring or most similar). Otherwise,
default to the most recent `prompt_history/` entry from *this conversation*.

**Purpose**: unlike the `prompt_history/` Summary (compressed prose, no code),
this is a *narrative walkthrough* of the reasoning process -- written so the
user can later understand not just what was done, but how you figured it out:
what you read, what you tried that didn't work, the moment something clicked,
and why you trusted a conclusion enough to act on it. Write it from the live
conversation transcript/context while it's still fresh -- if this task has
already been compacted out of context, say so explicitly rather than
fabricating detail that isn't actually remembered; fall back to what the
`prompt_history/` Summary and any code changes reveal, and note the reduced
fidelity in the doc itself.

**What makes a good rationale doc** (see an existing one in
`prompt_rationales/` for the template/style if any exist yet):
- Organize as a handful of named sections, each answering one real question
  ("how did I know the file format?", "why didn't the first approach work?"),
  not a chronological blow-by-blow of every tool call.
- Include real code snippets you actually read or wrote, with a file-path
  caption, plus a sentence of *why* that snippet mattered -- not just what it
  says.
- Call out dead ends and wrong first guesses, not just the final answer --
  that's most of the actual learning value.
- Flag the specific moment a hypothesis got confirmed or falsified (an "aha"
  or "this didn't work" callout), not just the eventual conclusion.
- Self-contained HTML (inline `<style>`, no external assets) so it opens
  directly in a browser from disk. Check color contrast on a white page
  background specifically: a color picked to read well on a dark `<pre>`
  code-block background (e.g. a light blue/gray) will look faint and be
  hard to read if the same CSS class gets reused inline in normal prose --
  give light-on-dark colors their own selector scoped to `pre`/`code`
  (e.g. `pre .file { color: ... }`) rather than one shared class, and keep
  regular-prose text (including captions/muted text) dark enough for
  reasonable contrast against white (roughly `#4a4a4a` or darker, not
  lighter grays like `#888`+).
- Verify the HTML tags actually balance before calling it done -- an
  unclosed `<pre>`/`<code>` (easy to introduce with a "file-path caption
  line, then a code block" pattern: `<pre><code class="file">path</code>`
  followed by a second `<pre><code>...</code></pre>` looks fine to read but
  is missing the first `</pre>`) makes the browser nest everything after it
  inside the still-open dark `<pre>` background, so later callout boxes and
  text visually inherit a black background even though their own CSS is
  correct. This was the actual root cause the one time this looked like a
  "text hard to read" color bug and wasn't -- don't assume it's a color
  issue just because that's what it looks like; check tag balance first
  (e.g. feed the file through `html.parser.HTMLParser` and confirm the open
  tag stack is empty at EOF) whenever a rendering complaint doesn't fully
  match what the CSS says it should do.
- Also include `<meta name="color-scheme" content="light">` in `<head>`
  and `html { color-scheme: light; }` + an explicit `background: #ffffff`
  in the body's CSS, as cheap insurance against browser/OS forced-dark-mode
  features -- but don't stop at this if a contrast complaint persists after
  adding it; check tag balance next rather than tweaking more hex values.
- Colored callout boxes (e.g. green/blue/yellow for "confirmed step" /
  "aha" / "dead end") are NOT a universal or self-evident convention --
  always add a short, explicit "Color key" legend near the top of the doc
  (right after the table of contents) spelling out what each color means,
  using the actual CSS classes so the legend doubles as a live example.
- Embed real images wherever the narrative describes a plot/visual result
  (a before/after comparison, a histogram, a segmentation overlay, a
  progression across attempts) rather than only describing it in prose --
  save the actual matplotlib/notebook output (or regenerate it from the
  same data/script if it wasn't saved) into `prompt_rationales/images/`
  (create it if needed) as `{same-basename}_<short-name>.png`, reference it
  with a relative `<img src="images/...">`, and add a one-line `.caption`
  underneath saying what to look for -- don't just assert "the plot showed
  X," show the plot. If regenerating isn't possible (data no longer
  available) or would take disproportionate effort for a minor point, say
  so explicitly rather than skip the image silently.
- Keep it proportional: a task that was mostly mechanical (renaming files,
  applying an agreed pattern) doesn't need one at all -- this is for tasks
  with genuine investigation/non-obvious discovery.

**When to use this proactively** (not just when explicitly invoked): at the
end of a task that involved real investigation -- reverse-engineering an
undocumented format, debugging by reading unfamiliar source, iterating on an
approach that didn't work the first time -- offer the user a rationale doc
for it rather than waiting to be asked, since the reasoning is cheap to
capture immediately and gets harder to reconstruct once the conversation is
compacted.

After writing, tell the user the file path and a one-line description of what
it covers -- don't paste the HTML into the chat.
