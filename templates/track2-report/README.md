# Galileo Hackathon 2026 — Research report template

## Start here

Prepare a separate report for each task; fill in Team ID and Task slug in the
title area, in addition to the team name and member names.
This is a standard LaTeX source package. Extract `galileo-report-template.zip`
and keep `main.tex` and `neurips_2026.sty` in the same directory. Replace every
bracketed prompt with your own account. Compile from that directory:

```sh
latexmk -pdf main.tex
```

Alternatively run twice:

```sh
pdflatex main.tex
pdflatex main.tex
```

The included [preview.pdf](preview.pdf) is an unfilled example, not a model scientific report.
It has two body pages plus one references page. You need not fill four pages.

## Length and format

The report body is at most FOUR pages, including title, authors, abstract,
figures, tables, all six required sections and the optional reward-hacking
section. References may take additional pages. Do not add narrative appendices.
The AI-use statement also counts toward the four-page body. There is no minimum
page count. There is no 800-word or 1,500-word requirement. Do not alter the style's font
sizes, margins or spacing to fit more text; do not use negative spacing to
squeeze content onto a page. Keep plots and tables legible.
The source puts references on a new page to make the body boundary easy to
check; the template does not automatically enforce the final page count.
Keep all substantive argument in the body. External evidence may contain the
complete original logs, code, model artifacts and reproducibility materials;
it must not function as an extra narrative appendix.

## Required content

### 1. Scientific goal & initial hypothesis

Explain the scientific objective beyond the leaderboard metric. State the
initial hypothesis and the observation that would support or contradict it.

### 2. Research trajectory

Connect decisions to evidence: Run ID / why / change / result / next decision.
Include real failures or negative results where they occurred, not invented
failures or a compulsory failure count. Six or fewer major decision points is
a suggestion for clarity, not a cap on experiments or a requirement to reach
six. Provide complete original logs separately and link them from the report.

### 3. Final model & comparisons

Describe the final model and compare the official baseline, final model and a
meaningful alternative. Identify split, metric and direction, relevant gates
and their outcomes. Report compute/time/cost if available and uncertainty where
measured. Mark unavailable measurements explicitly. Bind the report to the
exact submission ID, artifact/version or checksum and code revision. Avoid
comparing numbers from different splits or protocols without explaining why.

### 4. Key insights

Give at most three insights, each with claim / evidence / limitation. Tie
claims to specific runs or observations. Label causal guesses as hypotheses.
A well-supported negative result can earn the Research & Insight award;
a positive score change is not required for a valuable scientific insight.

### 5. Limitations & next experiment

State the important remaining uncertainty or failure mode and a concrete next
experiment that could distinguish competing explanations.

### 6. Reproducibility, contributions & AI use

Include commands, versions/environment, seeds where relevant, data access and
split usage, evidence links, member roles, and AI uses plus verification.
Describe actual data use accurately; do not imply held-out generalization from
public-validation measurements. Follow the event's current train-only rules.
Do not publish credentials, private data or unauthorized participant details.

## Optional: Reward hacking discovery

Keep this section before references and within the four-page body. Explain:

- scientific goal versus metric incentive;
- triggering condition;
- minimal reproducible example;
- evidence of high score alongside scientific failure;
- proposed fix and potential false positives.

Delete the optional section if it does not apply. This is not a requirement to
exploit a system; a carefully evidenced discovery and explanation is the aim.

## AI policy

Team members must write their own research judgments, insights, limitations,
and loophole explanations. AI may assist with organizing logs, translating and
proofreading; it must not generate the core scientific arguments. AI assistance
for code is allowed. Disclose the tools and purposes and explain how generated
outputs and code were verified. Authors remain responsible for every claim,
measurement and citation. AI detectors are not used to determine authorship.

## Deadline and submission

Reports are due November 21, 2026, at 4:30 pm PST (UTC−8). Model uploads close
at 4:00 pm. Use the final 30 minutes to organize existing evidence and finish
the report for your submitted model; this does not extend model submissions
or permit new competition scores. Use the separate report PDF form
in the same Track 2 portal: choose the task and link your own submission ID for
that task. It permits report replacement before the deadline without using the
ten-upload model quota. A report about a failed submission does
not confer model validity.

## Awards and review context

The existing awards are Solution x2, Research & Insight x1, and Reward Hacking
x1. The model and report are considered together. This template introduces no
scoring weights. Use the report submission procedure above. Do not infer NeurIPS or ML4PS
submission, acceptance, sponsorship or endorsement from this formatting.

## Attribution and adaptation

The formatting is the official NeurIPS 2026 template, linked by the ML4PS 2026
guidelines. We use its single-blind workshop mode to show team/member names;
this does not establish a new anonymity policy for another venue. The style
file is unchanged. Only the main document's venue notice is overridden to
identify Galileo Hackathon 2026. Our sections and policies are Galileo-specific,
not a reproduction of the ML4PS submission rules. See [LICENSE-ATTRIBUTION.txt](LICENSE-ATTRIBUTION.txt).
