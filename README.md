# CO₂ Emissions Explorer

**What actually drives US per-capita CO₂ emissions?** A state-level regression study over a 1,300-observation panel (50 states across 26 years, 1998 to 2023), shipped as an interactive explorer.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)

**[Open the live explorer](https://co2-emissions-explorer.vercel.app)** · [Project video](https://youtu.be/_4QkfWFUnI4) · [Full case study](https://zhifeng-portfolio.vercel.app/projects/co2-modeling)

## The question

Per-capita CO₂ emissions vary almost twelve-fold across US states. In 2023 Maryland emitted 7.8 tonnes per person and Wyoming emitted 92.9, an 11.9-fold gap, against a national median of 14.0. A uniform emissions cap would punish states whose energy systems differ structurally rather than by choice.

So: can a state's per-capita emissions be predicted from five structural variables, and can the drivers justify targeted rather than uniform climate policy? The five are renewable energy consumption, coal consumed for electricity, natural gas consumed for electricity, personal consumption expenditure, and urban population.

## What we found

A single pooled model across all 50 states was statistical noise. State-level structure swamps any shared pattern, so we stopped forcing it and fitted the three highest-emitting states separately.

| State | Adjusted R² | Significant drivers at the 5% level |
|---|---|---|
| Wyoming | 0.98 | Coal, urban population |
| North Dakota | 0.84 | Coal, natural gas, consumer spending |
| Alaska | 0.81 | Natural gas, consumer spending, urban population |

**There is no one-size-fits-all model.** Coal dominates Wyoming and North Dakota. Coal is not significant in Alaska at all, where spending and urbanisation carry the model instead. Natural gas lowers predicted emissions in one state and raises them in another.

## What the model gets wrong

Each model was refit on 1998 to 2019 alone, then asked to predict four years it had never seen. Reporting only the fit statistic would hide this:

| State | Holdout RMSE | Share of that state's observed range | Direction of error |
|---|---|---|---|
| Wyoming | 5.34 t | 14% | Over-predicts every unseen year |
| North Dakota | 1.38 t | 11% | Tracks closely |
| Alaska | 7.39 t | 28% | Under-predicts every unseen year |

Three things worth saying plainly:

- **A high adjusted R² is not a forecast.** Wyoming explains 98% of in-sample variance and still runs about 5 t per person high on every held-out year.
- **Alaska's error is one-signed.** Every held-out year is under-predicted and the gap widens as emissions recover after 2020. A consistently signed error is structural, not noise.
- **Five predictors cannot see flaring.** Gas flared at the wellhead is material in the Bakken and on the North Slope, and it never enters the model's inputs.

### One numerical detail that changed a conclusion

The five predictors span about nine orders of magnitude, which pushes the condition number of the design matrix to roughly 1.2e9 and of XᵀX to around 1e18. Computing coefficient standard errors by inverting XᵀX returns noise at that scale: it reported Wyoming's intercept standard error as 5e-10, turning a t statistic of 8.6 into 2.5e11, and it promoted natural gas to significant in Wyoming on a spurious t of -2.58. Deriving the same errors from a QR decomposition puts that t back at -1.41, which is not significant. The significance columns above come from the QR path. See `standard_errors` in [scripts/export_web_data.py](scripts/export_web_data.py).

## Two front ends, one model

The repository ships the study twice, and both read the same fitted coefficients so they cannot disagree.

| | Stack | Purpose |
|---|---|---|
| `Home.py`, `pages/` | Streamlit, Python | The working app. Fits the models live with `numpy.linalg.lstsq`. |
| `web/` | Static HTML, CSS and JavaScript, no build step | The hosted explorer. Runs the same arithmetic in the browser against exported coefficients. |

`scripts/export_web_data.py` is the bridge. It reads the panel, pulls the coefficients straight out of `data_service.fit_state_models()`, computes the fit diagnostics and the holdout test, and writes `web/data/co2.json`. A test asserts that the exported coefficients reproduce `predict_co2` exactly, so the browser and Python cannot drift apart.

## Quick start

```bash
git clone https://github.com/Kopi-O-Kosong-Beng/CO2-Emissions-Explorer.git
cd CO2-Emissions-Explorer

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt
```

Run the Streamlit app:

```bash
streamlit run Home.py
```

Run the static explorer (any static server works, it has no build step):

```bash
python scripts/export_web_data.py   # refresh web/data/co2.json
python -m http.server 8000 -d web
```

Then open <http://localhost:8000>.

## Tests

```bash
python -m pytest
python scripts/check_web_bundle.py
```

28 tests cover data loading, column contracts, series filtering and sorting, model structure, the prediction identity (zero inputs must return the intercept), the export payload, the holdout split, the t-distribution implementation against known critical values, and the guard against ill-conditioned standard errors.

`check_web_bundle.py` verifies the static bundle before it ships: every local reference in `index.html` resolves, the data payload parses and covers 50 states, and no long dashes or leftover placeholders reach a reader.

Both run on every push through [GitHub Actions](.github/workflows/tests.yml), against Python 3.10, 3.11 and 3.12.

## Project structure

```
├── Home.py                     # Streamlit entry point
├── case_service.py             # Load and filter the merged panel for case studies
├── data_service.py             # Fit state models, feature bounds, prediction API
├── pages/
│   ├── 02_Case_Studies.py      # Historical trends and driver narratives
│   ├── 03_Prediction.py        # Interactive forecasting UI
│   └── 04_HASS_Reflection.py   # Environmental justice reflection
├── scripts/
│   ├── export_web_data.py      # Panel and coefficients to web/data/co2.json
│   └── check_web_bundle.py     # Pre-deploy checks for the static bundle
├── web/                        # Static explorer deployed to Vercel
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── data/co2.json           # Generated, committed so the site needs no build
├── tests/                      # pytest suite
├── assets/
│   └── All main data (1998 to 2023).xlsx
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Adds pytest for local work and CI
└── vercel.json                 # Static deploy config for web/
```

## Deploying

**The static explorer, on Vercel.** Live at [co2-emissions-explorer.vercel.app](https://co2-emissions-explorer.vercel.app), redeployed automatically on every push to `main`. `vercel.json` sets the framework to none, the build command to none and the output directory to `web/`, so there is nothing to configure in the dashboard.

The page reads `?theme=light` or `?theme=dark` and accepts a `{type: "set-theme", theme}` postMessage, so it can be embedded in an iframe that follows the host page's theme. Adding `?embed=1` hides its own header and footer.

**The Streamlit app, on Streamlit Community Cloud.** Streamlit needs a long-running server holding a WebSocket per visitor, which Vercel's serverless model cannot provide, so it is hosted separately. At [share.streamlit.io](https://share.streamlit.io), point a new app at this repository with `Home.py` as the entry point. It is free for public repositories.

## Data

- US Energy Information Administration, State Energy Data System
- US Bureau of Economic Analysis, personal consumption expenditure
- US Census Bureau, urban population estimates

## Team and attribution

Built for SUTD's Design Thinking Project III by a five-person team.

I was one of two core modellers and the developer of this app. Together with my teammate I gathered the dataset and built the regression models in Excel. He wrote the core maths-algorithm code; I did the final checking and integrated the models into the app. I also originated the evaluation methodology (adjusted R² and the RMSE-variant scoring), wrote the report, and led the model-strengths analysis in the final presentation.

The static explorer, the export pipeline, the test suite and the numerical work described above are mine. Full role breakdown on the [case study page](https://zhifeng-portfolio.vercel.app/projects/co2-modeling).
