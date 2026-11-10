# Dataset Card – Kerala PCOS dataset (Kottarathil, 2020)

Verified on 2026-10-17 unless marked *inferred* or *unverified*.

## Source and access
| Item | Value |
|---|---|
| Title | Polycystic ovary syndrome (PCOS) |
| Uploader / curator | Prasoon Kottarathil (Kaggle user `prasoonkottarathil`), 2020 |
| URL | https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos |
| Download | `https://www.kaggle.com/api/v1/datasets/download/prasoonkottarathil/polycystic-ovary-syndrome-pcos` (anonymous HTTP 200 on 2026-10-17; no login needed) |
| Files | `PCOS_data_without_infertility.xlsx` (sheets `Instructions`, `Full_new`; 541 x 45), `PCOS_infertility.csv` (541 x 6, IDs 10001-10541; **not used**) |
| Original archive SHA-256 | `380ce7ff504cb8aa89349a251c64841a09aadbfe410627df94e6a62e67a23891` (checked by `experiments/01_ingest_and_quality.py`) |
| Stated collection | "data is collect from 10 different hospital across Kerala, India" (Kaggle description). Some papers say one hospital; unresolved. |
| Peer-reviewed data descriptor / ethics approval / consent statement | **None found.** Secondary papers (e.g. PMC8965679, PMC13508585) state only that the data are public and anonymised. |

## License and permitted use
- Kaggle metadata (API, retrieved 2026-10-17): **CC BY-NC-SA 4.0**. The license is asserted by the uploader; whether the uploader owns the rights to the underlying hospital data is **unverified**.
- **Attribution required**: cite Kottarathil (2020) as above.
- **Non-commercial only**: academic research and a non-commercial demo are within scope; commercial deployment is not.
- **Share-alike**: any redistributed copy of the data or derived data (`data/processed/`) must carry CC BY-NC-SA 4.0. Whether a trained model counts as "adapted material" is a legal question I cannot settle; the conservative course is to release the model and processed data under the same license and not offer a commercial service.
- **Public deployment**: a free, non-commercial, clearly labelled research demo does not appear to violate the license, but the absent ethics/consent documentation is a reason to keep any deployment private or limited to the user's own academic use unless a supervisor/IRB confirms otherwise.
- The repository must not be presented as covered by a more permissive license than the data.

## Target
`PCOS (Y/N)`: 1 = PCOS, 0 = not. 177 positive / 364 negative (32.7%). **No label definition is given.** The label agrees with "max follicle count >= 10" in 85.4% of rows and with a crude Rotterdam-style proxy (>= 2 of: irregular cycle, hirsutism/acne/hair-loss, >= 12 follicles) in 83.9% of rows (`results/metrics/label_circularity.json`); this suggests clinical, criteria-based labelling but it is **not documented**. The study therefore treats the label only as "the dataset's PCOS label".

## Variables used in the deployed model (non-invasive)
| Model feature | Raw column | Definition status |
|---|---|---|
| `age_years` | `Age (yrs)` | years, range 20-48 |
| `bmi_derived` | from `Weight (Kg)`, `Height(Cm)` | derived = kg/m2. Reported `BMI` disagrees with it by > 0.5 in 4 rows; reported BMI is not used |
| `cycle_irregular` | `Cycle(R/I)` | Coded 2 (n=390), 4 (n=150), 5 (n=1). **Meaning inferred**: 2 = regular, 4 and 5 = irregular, because secondary papers report 390 regular / 151 irregular. No original codebook found. |
| `weight_gain`, `hair_growth`, `skin_darkening`, `hair_loss`, `pimples` | `... (Y/N)` | 1 = Yes (per the workbook's `Instructions` sheet). Definitions, severity scale and who assessed them (self-report vs clinician) are **not documented**. Papers read them as weight gain, hirsutism, acanthosis nigricans, androgenic alopecia and acne respectively (*unverified*). |
| `fast_food`, `regular_exercise` | `Fast food (Y/N)`, `Reg.Exercise(Y/N)` | Y/N; frequency/threshold undefined |

## Variables deliberately not used in the primary model
`pregnant`, `marriage_years`, `abortions` (reproductive-history proxies), `blood_group`, `hb_g_dl`, `pulse_bpm`, `resp_rate`, `bp_sys`, `bp_dia` (clinic measurements, error-prone), `cycle_length_days` (values 2-12, median 5: more likely bleeding duration than cycle length; meaning **unverified**, tested only as a labelled sensitivity analysis), `waist_inch`/`hip_inch`/`whr` (no meaningful gain, see `results/metrics/feature_redundancy_*.csv`), all hormone/laboratory and ultrasound variables (comparison experiments only).

## Preprocessing (`src/pcos/data.py`)
No rows removed. Logged actions (`results/metrics/cleaning_log.csv`): string typos `1.99.` (beta-HCG, trailing dot stripped) and `a` (AMH -> missing); cycle recoding; derived BMI; empty column `Unnamed: 44` dropped. Values judged implausible are **flagged, not altered** (`results/metrics/quality_flags.csv`: 19 flags, mostly in variables outside the primary model; 6 weights of 31-34 kg and 4 derived BMIs of 12.4-13.9 touch primary inputs: extreme but not provably wrong, so kept; 4 further rows have a reported BMI that differs from weight/height by > 0.5, which is why BMI is re-derived). Remaining missing in primary features: 1 `fast_food` value, imputed with the training-split mode inside the model pipeline.

## Known data-quality issues
Undocumented label and codes; several entry errors (pulse 13, BP 12/8, FSH 5052, LH 2018, Vit D3 6014); one malformed AMH and beta-HCG entry; possible duplication of patients across hospitals cannot be checked (no hospital or time field).

## Population limitations
Single-region Indian cohort, ages 20-48; no ethnicity, income, hospital, date or measurement-protocol fields; many married and pregnant women, so likely clinic attendees (selection bias); PCOS prevalence 32.7% is far above population prevalence, so predictive values do not transfer. No second labelled open dataset exists (`docs/DATASET_SELECTION_REPORT.md`), so **no external validation is possible or claimed**.
