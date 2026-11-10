# Phase 12 - duplicate / leakage check

5861 preprocessed images. Two tests: (a) exact byte-identical duplicates (MD5 of the processed PNG); (b) near-duplicates by a 1024-bit fine difference-hash, Hamming $\le$ 24.

## (a) Exact duplicates (MD5)

- Exact-duplicate groups: **132**
- Pairs crossing a split or dataset boundary: **58**
- Boundary-crossing exact duplicates are listed in `exact_duplicates.csv` (boundary_crossed == True); the affected split or external comparison should be re-run with them removed.

## (b) Near-duplicates (fine dhash)

| Comparison | near-duplicate pairs (dhash Hamming <= 24) |
|---|--:|
| APTOS:train vs APTOS:test | 40 |
| APTOS:train vs APTOS:val | 16 |
| APTOS:val vs APTOS:test | 6 |

Near-duplicate pairs (if any) are in `leakage_pairs.csv`. Perceptual near-duplicate detection on tightly cropped, resized fundus frames is intrinsically noisy (every image is a centred disc on black); the exact-MD5 test above is the reliable one, and an embedding-based near-duplicate check is listed as a follow-up.
