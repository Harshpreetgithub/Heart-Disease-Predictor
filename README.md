# Cardiac Risk Estimator — deploy to Vercel

A small full-stack app around the Random Forest model from the report:
a static HTML/CSS/JS form (`/public`) calling a FastAPI prediction
endpoint (`main.py`), served together as one Vercel project.

## What's in this folder

```
main.py               FastAPI app — loads the model and exposes POST /api/predict
model.pkl              trained RandomForestClassifier (Chapter 8 pipeline)
scaler.pkl              StandardScaler fitted on the training split
label_encoders.pkl      LabelEncoders for Sex / ExerciseAngina
feature_columns.json    exact column order the model expects
train_model.py          the training script that produced the .pkl files
heart.csv               the 918-row dataset used to train them
requirements.txt        Python dependencies
vercel.json             sets the function timeout
public/
  index.html            the form + results UI
  style.css
  app.js                calls /api/predict and renders the gauge
```

## 1. Retrain locally first (optional but recommended)

If you want to confirm everything reproduces the report's numbers before
deploying:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt scikit-learn pandas
python3 train_model.py
```

This regenerates `model.pkl`, `scaler.pkl`, `label_encoders.pkl` and
`feature_columns.json` from `heart.csv`. You should see Random Forest at
~88% test accuracy / ~87.6% 10-fold CV, matching Chapter 6 of the report.

## 2. Test locally with Vercel's dev server

```bash
npm i -g vercel        # if you don't already have the CLI
pip install -r requirements.txt
vercel dev
```

Open the printed local URL — the form should load and `/api/predict`
should return a JSON prediction.

## 3. Deploy

**Easiest — via GitHub:**
1. Push this folder to a new GitHub repository.
2. Go to vercel.com → **Add New… → Project** → import that repo.
3. Vercel auto-detects the FastAPI app (`main.py` exports `app`) and the
   `public/` static folder — no build settings to change.
4. Click **Deploy**. You'll get a `https://your-project.vercel.app` URL.

**Or via CLI, from inside this folder:**
```bash
vercel login
vercel        # preview deployment
vercel --prod # promote to production
```

## Notes

- The model is a `RandomForestClassifier` (scikit-learn defaults), trained
  with the same encoding/scaling pipeline as Chapter 8 of the report:
  label encoding for `Sex`/`ExerciseAngina`, one-hot (drop-first) for
  `ChestPainType`/`RestingECG`/`ST_Slope`, `StandardScaler` on all 15
  resulting columns, 80/20 stratified split, `random_state=0`.
- `main.py` rebuilds that exact same 15-column row for every request
  (see `feature_columns.json`) before scaling and calling
  `model.predict_proba`.
- This is a training-report demo, not a medical device — the UI says so,
  and it's worth keeping that disclaimer if you share the link.
