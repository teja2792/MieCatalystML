"""
build_validation_summary.py

Pulls together every real-vs-predicted validation check performed
across this repo's phases into one plain-language table, written as
both a CSV (results/validation_summary.csv) and a Markdown table
printed to the terminal for direct paste into the README. Intended for
a non-specialist reader -- the point is "does the physics engine
reproduce real published measurements," not the underlying math.

All numbers here were computed in earlier phases (not recomputed here)
-- this script just assembles and formats them, with citations.
"""

import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ROWS = [
    {
        "category": "Mie theory sanity check",
        "test": "Where electric and magnetic scattering are equal (lossless test sphere, not real Cu2O)",
        "real": "1548 nm",
        "predicted": "1540 nm",
        "diff": "-0.5%",
        "status": "Validated",
        "source": "SI Figure S7 (Mohammadparast et al., 2020)",
    },
    {
        "category": "Near-field brightness",
        "test": "Magnetic field hotspot strength at resonance, 175nm Cu2O sphere",
        "real": "~25x",
        "predicted": "29.2x",
        "diff": "+17%",
        "status": "Validated (right shape and order of magnitude)",
        "source": "SI Figure S5a (Mohammadparast et al., 2020)",
    },
    {
        "category": "Near-field brightness",
        "test": "Electric field hotspot strength at resonance, 175nm Cu2O sphere",
        "real": "~8x",
        "predicted": "13.0x",
        "diff": "+62%",
        "status": "Approximate (correct shape, magnitude off)",
        "source": "SI Figure S5b (Mohammadparast et al., 2020)",
    },
    {
        "category": "Size-dependent scattering",
        "test": "Scattering brightness peak, 300nm diameter sphere",
        "real": "~3.7 (x1e-13 m2)",
        "predicted": "3.75 (x1e-13 m2)",
        "diff": "+1%",
        "status": "Validated",
        "source": "SI Figure S3b (Mohammadparast et al., 2020)",
    },
    {
        "category": "Size-dependent scattering",
        "test": "Scattering brightness peak, 350nm diameter sphere",
        "real": "~5.3 (x1e-13 m2)",
        "predicted": "5.25 (x1e-13 m2)",
        "diff": "-1%",
        "status": "Validated",
        "source": "SI Figure S3b (Mohammadparast et al., 2020)",
    },
    {
        "category": "Size-dependent scattering",
        "test": "Scattering brightness peak, 400nm diameter sphere",
        "real": "~7.2 (x1e-13 m2)",
        "predicted": "7.09 (x1e-13 m2)",
        "diff": "-2%",
        "status": "Validated",
        "source": "SI Figure S3b (Mohammadparast et al., 2020)",
    },
    {
        "category": "Catalytic rate (particle size, green light)",
        "test": "How much faster the reaction runs on 145nm vs. 42nm spheres",
        "real": "9.6x (+/- 1.6x)",
        "predicted": "8.2x",
        "diff": "-15% (within real measurement uncertainty)",
        "status": "Validated",
        "source": "Addanki Tirumala et al., ACS Catalysis 2022",
    },
    {
        "category": "Catalytic rate (particle size, red light)",
        "test": "How much faster a different reaction runs on 145nm vs. 37nm spheres",
        "real": "2.6x (+/- 0.4x)",
        "predicted": "12.2x",
        "diff": "Far outside real range",
        "status": "Documented limitation (different reaction mechanism; not tuned away)",
        "source": "Addanki Tirumala et al., ACS Appl. Nano Mater. 2022",
    },
    {
        "category": "Catalytic rate (light intensity)",
        "test": "How reaction rate scales with light brightness, fixed 1300nm particles",
        "real": "5 measured points",
        "predicted": "Fit R2 = 0.999",
        "diff": "N/A",
        "status": "Validated",
        "source": "Addanki Tirumala et al., ACS Sustainable Chem. Eng. 2023",
    },
]

csv_path = os.path.join(RESULTS_DIR, "validation_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["category", "test", "real", "predicted", "diff", "status", "source"])
    writer.writeheader()
    writer.writerows(ROWS)
print(f"Saved {csv_path}")
print()

print("Markdown table (paste directly into README):")
print()
print("| What we checked | Real measurement | Our prediction | Difference | Result |")
print("|---|---|---|---|---|")
for r in ROWS:
    print(f"| {r['test']} | {r['real']} | {r['predicted']} | {r['diff']} | {r['status']} |")
print()
print("Sources, in order: SI of Mohammadparast et al., ACS Appl. Nano Mater. 2020; ")
print("Addanki Tirumala et al., ACS Catalysis 2022; Addanki Tirumala et al., ACS Appl. Nano Mater. 2022; ")
print("Addanki Tirumala et al., ACS Sustainable Chem. Eng. 2023.")