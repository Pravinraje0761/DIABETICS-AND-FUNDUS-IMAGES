# Depth x activation ablation -- Cross-dataset external validation of a retinal foundation model

Model(s): efficientnet_b3@aptos_internal, efficientnet_b3@idrid, efficientnet_b3@messidor2, retfound_vit_large_patch16@aptos_internal, retfound_vit_large_patch16@idrid, retfound_vit_large_patch16@messidor2
Grid: depths [0, 3, 4, 5, 6, 7] x activations ['gelu', 'identity', 'relu', 'tanh'] x seeds [0, 1, 2, 3, 4]
Width policy: constant
Total runs: 630

## Seed noise floor

Standard deviation across seed replicates, within each fixed
(model, depth, activation) cell. A depth effect smaller than this is
not a depth effect.

- median within-cell SD of MCC: 0.0169
- max within-cell SD of MCC:    0.0603

## Mean MCC by depth x activation (averaged over seeds)

|   depth |     gelu |   identity |     relu |     tanh |
|--------:|---------:|-----------:|---------:|---------:|
|       0 | nan      |     0.4135 | nan      | nan      |
|       3 |   0.414  |     0.4236 |   0.4193 |   0.4261 |
|       4 |   0.422  |     0.4174 |   0.4226 |   0.423  |
|       5 |   0.4266 |     0.4085 |   0.4279 |   0.4154 |
|       6 |   0.4164 |     0.4028 |   0.4132 |   0.4185 |
|       7 |   0.4196 |     0.4142 |   0.4097 |   0.4247 |

## Mean balanced accuracy by depth x activation

|   depth |     gelu |   identity |     relu |     tanh |
|--------:|---------:|-----------:|---------:|---------:|
|       0 | nan      |     0.4947 | nan      | nan      |
|       3 |   0.4839 |     0.4943 |   0.4839 |   0.4961 |
|       4 |   0.4868 |     0.4932 |   0.4854 |   0.4924 |
|       5 |   0.4824 |     0.4847 |   0.4864 |   0.4901 |
|       6 |   0.4686 |     0.486  |   0.4744 |   0.4912 |
|       7 |   0.4751 |     0.487  |   0.4779 |   0.4933 |

## Best cell

- model `efficientnet_b3@aptos_internal`, depth 4, activation `gelu` -> mean MCC 0.7009

Treat 'best' as provisional until you have checked it against the noise
floor above. On a grid this size the top cell is frequently the luckiest
cell rather than the best architecture.

## Depth effect against the noise floor

- best-minus-worst depth, averaged over activations and seeds: 0.0085
- median within-cell seed SD: 0.0169
- ratio: 0.51x

A ratio below about 2 means depth did not move this task beyond what
reseeding moves it on its own.

## How to read this

Compare the spread across the depth column against the seed noise floor
above. If the best and worst depths differ by less than roughly two
within-cell standard deviations, the honest conclusion is that depth did
not matter on this task at this width -- which is a result worth
reporting, not a failed experiment.

Confusion matrices for all 630 cells are in
confusion_matrices.json, keyed `model|depth|activation|seed`, as raw
integer counts over classes ['0 none', '1 mild', '2 moderate', '3 severe', '4 proliferative'].