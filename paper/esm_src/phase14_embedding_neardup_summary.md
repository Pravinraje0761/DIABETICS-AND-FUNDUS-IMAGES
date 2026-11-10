# Phase 14 -- embedding near-duplicate check

Feature extractor: ImageNet-pretrained EfficientNet-B3, 1536-d pooled, L2-normalised. Cosine similarity, all cross-dataset pairs.

| Dataset pair | pairs cos>=0.90 | >=0.95 | >=0.98 |
|---|--:|--:|--:|
| APTOS <-> Messidor-2 | 47791 | 99 | 0 |
| APTOS <-> IDRiD | 303 | 0 | 0 |
| Messidor-2 <-> IDRiD | 507 | 0 | 0 |

## APTOS internal-test images with a near-duplicate elsewhere

| target pool | test images cos>=0.95 | cos>=0.90 |
|---|--:|--:|
| APTOS train/val | 89 / 733 | 367 / 733 |
| Messidor-2 (external) | 7 / 733 | 214 / 733 |
| IDRiD (external) | 0 / 733 | 38 / 733 |

Cross-dataset near-duplicates into an external set: **252** APTOS-test images with any cos>=0.90 match (detail in `aptos_test_vs_external_matches.csv`; expected ~0).

Top cross-dataset pairs are in `cross_dataset_top_pairs.csv`; inspect the highest-cosine rows manually -- fundus frames are all centred discs on black, so even unrelated images sit around cos 0.6-0.85.
