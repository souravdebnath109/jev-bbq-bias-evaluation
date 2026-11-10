# Jev Ã— BBQ â€” Reproducible experiment using OpenCode 1.13 Free

**Status:** Source code and offline tests provided. **This repository contains no API-generated BBQ predictions until YOU run it with your own key.** The one simple doctor/nurse example executed in PowerShell is **not** a BBQ benchmark result.

## Purpose

Test Jev's structured choice responses on the official BBQ dataset. Measure overall accuracy, separate ambiguous/disambiguated accuracy, unnecessary abstentions, category errors, and the calibration of selected-choice probabilities. This is a baseline experiment; gold-label agreement alone cannot reveal the model's internal justification.

Jev is a hosted API, accessed here through OpenCode **`jev-1.13-free`**, with one BBQ question per request. OpenCode describes this free model as **limited-time**, and usage limits may apply. If a model is no longer free, stop instead of substituting paid `jev-1.13` silently.

## Required software

- Windows PowerShell
- Python 3.10+ (`py -3 --version`)
- Git for Windows (`git --version`)
- An OpenCode API key that successfully calls `https://opencode.ai/inference/systemone/v1/systemone`

No `pip install` is required; all Python packages are from the standard library.

## Exact Windows PowerShell commands

Extract the ZIP. Navigate into the extracted folder (example when inside Downloads):

```powershell
cd "$HOME\Downloads\Jev_BBQ_OpenCode_Ready"
```

If you extracted it elsewhere, open its folder in File Explorer, click the address bar, type `powershell`, and press Enter.

**1. Clone the original dataset** (the `BBQ` folder is intentionally ignored by Git):

```powershell
git clone https://github.com/nyu-mll/BBQ.git BBQ
```

**2. Verify Python and run tests** (never calls API):

```powershell
py -3 --version
py -3 -m unittest discover -s tests -v
```

**3. Configure your OpenCode key** in this *same* PowerShell session if the prior session was closed:

```powershell
$secret = Read-Host 'Enter NEW OpenCode API key' -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try { $env:OPENCODE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
```

Do not paste any key into a command, README, chat, GitHub, or screenshot. Keys previously exposed in a screenshot/chat should be revoked.

**4. First real experiment â€” 10 BBQ items** (five ambiguous/five disambiguated, roughly category balanced):

```powershell
py -3 jev_bbq.py run --bbq-dir BBQ/data --limit 10 --output results/pilot10.jsonl
```

The program prints each selected answer and the gold label; it records the full request body and full API response (without your key) in the prediction file.

**5. Analyze the 10 real answers:**

```powershell
py -3 jev_bbq.py score --bbq-dir BBQ/data --predictions results/pilot10.jsonl --output results/pilot10_report.json
```

Read `results/pilot10_report.json` for summary, and `results/pilot10_report.audit.csv` for question-by-question comparison. Scores from a 10-item sample are not reliable full-benchmark estimates.

**6. Extend to 200 items:**

```powershell
py -3 jev_bbq.py run --bbq-dir BBQ/data --limit 200 --output results/pilot200.jsonl
py -3 jev_bbq.py score --bbq-dir BBQ/data --predictions results/pilot200.jsonl --output results/pilot200_report.json
```

**7. Study choice order** (200 pairs, one original and one reversed):

```powershell
py -3 jev_bbq.py run --bbq-dir BBQ/data --limit 200 --output results/order_normal.jsonl
py -3 jev_bbq.py run --bbq-dir BBQ/data --limit 200 --reverse --output results/order_reversed.jsonl
py -3 jev_bbq.py compare-order --original results/order_normal.jsonl --reversed results/order_reversed.jsonl
```

**8. Only after checking limits, run the full original dataset:**

```powershell
py -3 jev_bbq.py run --bbq-dir BBQ/data --all --output results/full_predictions.jsonl
py -3 jev_bbq.py score --bbq-dir BBQ/data --predictions results/full_predictions.jsonl --output results/full_report.json
```

The full run creates **one API request per question**, which may take substantial time and could encounter rate or free-tier limits. Already completed predictions are preserved; rerunning the same command resumes.

## What is saved

- `results/pilot10.jsonl`: your real Jev API predictions, full API request and response, prompt version, server-returned model name, time and cost. No token/key headers are saved.
- `results/pilot10_report.json`: offline-computed accuracy, category and condition breakdown, ECE and abstention metrics.
- `results/pilot10_report.audit.csv`: each actual question, passage, expected answer and Jev answer.

**The script does NOT compute official BBQ stereotype bias scores**; these require the official `additional_metadata.csv` and corresponding polarity-aware scoring. Its 'unsupported ambiguous guess' metric identifies non-unknown answers when evidence is absent, but **does not alone establish stereotype direction**.

## GitHub sharing

Create an empty GitHub repository named `jev-bbq-bias-evaluation` at https://github.com/new, then from the project folder:

```powershell
git init
git add .
git status
git commit -m "Reproducible Jev BBQ evaluation with OpenCode API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/jev-bbq-bias-evaluation.git
git push -u origin main
```

Replace `YOUR_USERNAME`. **Before `git add .`, verify there is no API key or private credentials in any file.** Git ignores the `BBQ/` checkout and secrets. For a small pilot, your actual JSONL prediction and audit output can be committed. Cite the dataset repository rather than redistributing the full benchmark itself.

For stronger paper-worthy research, extend with official BBQ stereotype-bias scoring, template-clustered uncertainty intervals, genuine evidence counterfactuals, and matched model baselines. Accuracy alone has already been measured in other works.

## Independent sources

- Official dataset: https://github.com/nyu-mll/BBQ
- OpenCode Jev API: https://opencode.ai/docs/console/models/
- JEVal benchmark: https://arxiv.org/abs/2610.03935
- Earlier public Jev-BBQ evaluation: https://github.com/simonmesmith/jev-bbq-experiment

## Source and provenance

This evaluation code is an independent implementation, not copied from the published Jev-BBQ experiment. It is under the MIT license; BBQ dataset has its own licensing and attribution. `jev_confidence` returned by the API is distinct from `max(probabilities)`, which is used for ECE.

## Experimental Results

Model: jev-1.13-free
API: OpenCode
Dataset: BBQ Bias Benchmark
Dataset source: https://github.com/nyu-mll/BBQ

### Original Evaluation

| Metric | Result |
|---|---|
| Evaluated questions | 200 |
| Overall accuracy | 91.5% (183/200) |
| Ambiguous accuracy | 100% (100/100) |
| Disambiguated accuracy | 83% (83/100) |
| Incorrect predictions | 17 |
| Unnecessary abstentions | 15 |
| Expected calibration error (10 bins) | 0.0375 |

### Answer-Order Sensitivity

| Metric | Result |
|---|---|
| Paired questions | 49 |
| Changed answers | 1 |
| Observed answer-flip rate | 2.04% |
| Status | Incomplete due to API usage limit |

### Important Notes

- All predictions were obtained from actual Jev API calls.
- The 200-question experiment is a sample, not the entire BBQ dataset.
- Answer-order results are preliminary and may include stochastic variation.
- Official BBQ stereotype bias scores have not yet been computed.
- Dataset annotation inconsistencies are being examined separately.
- API credentials are not included in this repository.
