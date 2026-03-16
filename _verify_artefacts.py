"""Verify all required artefacts exist."""
import os, glob, json

errors = []

# featured.csv
f = "data/processed/featured.csv"
if os.path.exists(f) and os.path.getsize(f) > 0:
    import pandas as pd
    df = pd.read_csv(f)
    print(f"featured.csv: OK ({len(df)} rows)")
else:
    errors.append("featured.csv: MISSING or empty")

# model_results.csv
r = "data/processed/model_results.csv"
if os.path.exists(r) and os.path.getsize(r) > 0:
    import pandas as pd
    df2 = pd.read_csv(r)
    status = "OK" if len(df2) == 20 else f"WRONG ({len(df2)} rows, expected 20)"
    print(f"model_results.csv: {status}")
    if len(df2) != 20:
        errors.append(f"model_results.csv has {len(df2)} rows, expected 20")
else:
    errors.append("model_results.csv: MISSING or empty")

# feature_columns.json
j = "models/feature_columns.json"
if os.path.exists(j):
    cols = json.load(open(j))
    status = "OK" if len(cols) == 13 else f"WRONG ({len(cols)} cols, expected 13)"
    print(f"feature_columns.json: {len(cols)} cols — {status}")
    if len(cols) != 13:
        errors.append(f"feature_columns.json has {len(cols)} cols, expected 13")
else:
    errors.append("feature_columns.json: MISSING")

# .joblib files
joblibfiles = sorted(glob.glob("models/*.joblib"))
status = "OK" if len(joblibfiles) == 20 else f"WRONG (expected 20)"
print(f".joblib files: {len(joblibfiles)} — {status}")
for j in joblibfiles:
    print(f"  {os.path.basename(j)}")
if len(joblibfiles) != 20:
    errors.append(f"{len(joblibfiles)} .joblib files, expected 20")

# PNG files
pngs = sorted(glob.glob("charts/*.png"))
print(f"PNG files: {len(pngs)}")
for p in pngs:
    print(f"  {os.path.basename(p)}")

# Summary
if errors:
    print("\nFAILED:")
    for e in errors:
        print(f"  {e}")
else:
    print("\nAll artefacts verified OK")
