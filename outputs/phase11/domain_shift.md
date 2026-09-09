# Domain-shift characterisation: APTOS 2019 vs Messidor-2 vs IDRiD

_Compiled from the dataset publications and `DATA_PROVENANCE.md`. "Not published"
means the axis is not reported in the public releases used here._

| Axis | APTOS 2019 (development) | Messidor-2 (external) | IDRiD (external) |
|---|---|---|---|
| Country / population | India (Aravind Eye Hospital); telescreening | France (Messidor programme partners) | India (Kasturba Medical College, Manipal) |
| Camera / device | Multiple, not individually published (multi-site) | Topcon TRC NW6 non-mydriatic | Kowa VX-10 alpha / Nidek AFC-210 |
| Native resolution | Heterogeneous; ~1.5-3 MP after vendor processing | 1440x960 to 2304x1536 | 4288x2848 |
| Field of view | ~45 deg, macula-centred | ~45 deg, macula-centred | 50 deg, macula-centred |
| Illumination / colour | Variable across sites; partly pre-processed by the Kaggle release | more uniform (single programme) | high-resolution, well-exposed; least colour variation |
| Mydriasis | mixed / not published | non-mydriatic | mydriatic |
| Grading scale | ICDR 0-4 | ICDR 0-4 | ICDR 0-4 |
| Reference standard | single grader (Kaggle labels) | **adjudicated consensus** (Krause et al. 2018) | expert consensus (release) |
| Grade prevalence 0/1/2/3/4 (%) | 49.3 / 10.1 / 27.3 / 5.3 / 8.1 | 58.3 / 15.5 / 19.9 / 4.3 / 2.0 | 28.4 / 4.8 / 34.3 / 18.5 / 14.1 |
| Referable DR (grade >= 2), % | 40.7 | 26.2 | 67.6 |
| Image quality | variable; some low-quality frames retained | curated; adjudicated-gradable subset only | high and uniform |
| Preprocessing here | single Phase 3 pipeline (crop / pad / 224 / ImageNet norm) | same, all 1744 | same, all 455 |

**Interpretation.** Messidor-2 differs from APTOS most on the reference standard
(adjudicated consensus vs single grader) and acquisition programme, and least on
FOV/scale; its referable-DR prevalence is *lower* (26% vs 41%). IDRiD differs most
on resolution, mydriasis, and prevalence (referable DR 68%), and least on colour
uniformity. This matches the observed failure modes: on Messidor-2 (stricter
labels, similar low-severity skew) QWK collapses by an ordinal-structure
mechanism; on IDRiD (very different base rate) accuracy falls more than QWK.
