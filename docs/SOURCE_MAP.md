# Documentation source map

This README documents the supplied 30-day expansion v2 files, not a live repository checkout. No training code, weights, or notebook outputs were changed while preparing the documentation.

## Source snapshots

| File | SHA-256 |
|---|---|
| [`Cell_1_30day_expansion.py`](../Cell_1_30day_expansion.py) | `48b23918f26be35dec56e6f9b6796c94a5342f647d970f7251f3a3ecca4c650b` |
| [`Cell_2_30day_expansion.py`](../Cell_2_30day_expansion.py) | `f1f766023e72ab55c99748d1191ddd168d06e2cd5febb7d8faf727d80062ca1b` |
| [`Kaggriculture_30day_expansion.ipynb`](../Kaggriculture_30day_expansion.ipynb) | `69e1472a035c9fbbcb70ca94e76de49e36e8c93f02736dc24a8aba3c6fada6fa` |

The two main code cells in `Kaggriculture_30day_expansion.ipynb` were compared with the replacement Python files and match byte-for-byte. Notebook output cells were not used as evidence of measured performance.

## Implementation locations

| README topic | Source |
|---|---|
| Game horizon, crops, regions, roles | [Cell 1, lines 33-75](../Cell_1_30day_expansion.py#L33-L75) |
| Heuristic economic model and strategic value | [Cell 1, lines 425-1408](../Cell_1_30day_expansion.py#L425-L1408) |
| Board encoding, spatial CNN and auxiliary loss | [Cell 1, lines 1419-1915](../Cell_1_30day_expansion.py#L1419-L1915) |
| Planner features, GRU experts and grand planner | [Cell 1, lines 2019-2568](../Cell_1_30day_expansion.py#L2019-L2568) |
| Spatial targets, warm-up and forced-action bookkeeping | [Cell 1, lines 2575-3449](../Cell_1_30day_expansion.py#L2575-L3449) |
| Worker features, transformer, masks and action conversion | [Cell 1, lines 3470-4324](../Cell_1_30day_expansion.py#L3470-L4324) |
| Expansion constants, costs and market execution | [Cell 1, lines 4331-4874](../Cell_1_30day_expansion.py#L4331-L4874) |
| Worker rewards, replay and DQN | [Cell 1, lines 4881-5570](../Cell_1_30day_expansion.py#L4881-L5570) |
| Phase-planner PPO, teacher loss and numerical guards | [Cell 1, lines 5577-5916](../Cell_1_30day_expansion.py#L5577-L5916) |
| Policy snapshots and resumable checkpoints | [Cell 1, lines 5922-6123](../Cell_1_30day_expansion.py#L5922-L6123) |
| ArenaAdapter and frozen policy | [Cell 1, lines 6129-6809](../Cell_1_30day_expansion.py#L6129-L6809) |
| Head-to-head challenger evaluation | [Cell 1, lines 6816-7080](../Cell_1_30day_expansion.py#L6816-L7080) |
| Run settings, Drive paths, network creation and restoration | [Cell 2, lines 1-520](../Cell_2_30day_expansion.py#L1-L520) |
| Opponent construction and starter health checks | [Cell 2, lines 528-754](../Cell_2_30day_expansion.py#L528-L754) |
| Curriculum, rollout, grand PPO, reporting and promotion | [Cell 2, lines 865-2948](../Cell_2_30day_expansion.py#L865-L2948) |

## Documentation checks

Documented tensor dimensions were checked by constructing the models from Cell 1 and running shape-only forward passes on a constructed observation: board `17 x 10 x 10`, embedding `128`, crop logits `5 x 10 x 10`, planner state `182`, grand state `29`, worker context `289`, tile features `18`, and worker Q-values `14`. The environment import was excluded for this isolated shape check. This was not a Kaggriculture episode, a training run, or a benchmark.

The technical diagrams were rendered from their editable Graphviz sources. All documentation image paths, diagram-source paths, and in-page navigation anchors were checked. Source-file links intentionally point to the existing training files in the repository; those files are not duplicated in the documentation ZIP.

## Version distinctions retained in the README

- `main_fixed.py` is an earlier submission artifact. It does not define `apply_expansion_floor` and retains the older market executor.
- The phase-planner update has the revised numerical guards. The separate grand-planner update still uses an unclamped exponential ratio and MSE value loss.
- Training and frozen inference use different grand-phase selection schedules.
- The market counters record emitted actions, not confirmed changes in farm state.
- The expansion target is subject to executable cash and order constraints.

These are observations about the supplied source version, not changes made by this documentation package.

## External references

External references are used only for runtime installation, submission packaging, and checkpoint-loading security. Architecture, hyperparameters, expansion behavior, and training control flow are documented from the local files above.

- [Kaggle Environments](https://github.com/Kaggle/kaggle-environments)
- [Kaggriculture agent guide](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/AGENTS.md)
- [PyTorch torch.load documentation](https://docs.pytorch.org/docs/stable/generated/torch.load.html)

## Graphics

The banner uses illustrative generated artwork, not a game screenshot. Only its decorative title/header is used. All four technical diagrams were separately constructed from the implementation; no generated performance curves or benchmark figures are included.

SVGs are the images embedded in the README. Matching PNGs are included for tools that do not display SVGs. Each diagram can be edited through its `.dot` source and regenerated with Graphviz.
