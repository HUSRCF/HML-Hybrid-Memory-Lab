# Complete normal-update load diagnostic

All 3,456 readouts, 36 cells and 576 episode cells are retained; eight independent streams. Full and privileged Oracle are separate. No new training.

| Model | Reader | Condition | Correct | Mean answer NLL |
|---|---|---|---:|---:|
| gdn_all_tokens | full | before | 89/96 | 0.081741 |
| gdn_all_tokens | full | related-512-load16 | 92/96 | 0.046453 |
| gdn_all_tokens | full | related-512-load2 | 80/96 | 0.245891 |
| gdn_all_tokens | full | related-64-load16 | 96/96 | 0.005381 |
| gdn_all_tokens | full | related-64-load2 | 95/96 | 0.008768 |
| gdn_all_tokens | full | unrelated-512-load16 | 89/96 | 0.045716 |
| gdn_all_tokens | full | unrelated-512-load2 | 93/96 | 0.038662 |
| gdn_all_tokens | full | unrelated-64-load16 | 95/96 | 0.017996 |
| gdn_all_tokens | full | unrelated-64-load2 | 92/96 | 0.034110 |
| gdn_all_tokens | oracle | before | 96/96 | 0.001000 |
| gdn_all_tokens | oracle | related-512-load16 | 96/96 | 0.062487 |
| gdn_all_tokens | oracle | related-512-load2 | 96/96 | 0.069157 |
| gdn_all_tokens | oracle | related-64-load16 | 96/96 | 0.031094 |
| gdn_all_tokens | oracle | related-64-load2 | 96/96 | 0.030091 |
| gdn_all_tokens | oracle | unrelated-512-load16 | 96/96 | 0.077834 |
| gdn_all_tokens | oracle | unrelated-512-load2 | 96/96 | 0.077843 |
| gdn_all_tokens | oracle | unrelated-64-load16 | 96/96 | 0.034715 |
| gdn_all_tokens | oracle | unrelated-64-load2 | 96/96 | 0.026337 |
| native | full | before | 70/96 | 0.289366 |
| native | full | related-512-load16 | 36/96 | 0.632718 |
| native | full | related-512-load2 | 44/96 | 0.599259 |
| native | full | related-64-load16 | 71/96 | 0.417622 |
| native | full | related-64-load2 | 63/96 | 0.447800 |
| native | full | unrelated-512-load16 | 47/96 | 0.577264 |
| native | full | unrelated-512-load2 | 58/96 | 0.534234 |
| native | full | unrelated-64-load16 | 65/96 | 0.452372 |
| native | full | unrelated-64-load2 | 61/96 | 0.385274 |
| native | oracle | before | 96/96 | 0.088239 |
| native | oracle | related-512-load16 | 83/96 | 0.386511 |
| native | oracle | related-512-load2 | 92/96 | 0.311289 |
| native | oracle | related-64-load16 | 96/96 | 0.261614 |
| native | oracle | related-64-load2 | 96/96 | 0.218315 |
| native | oracle | unrelated-512-load16 | 94/96 | 0.322997 |
| native | oracle | unrelated-512-load2 | 91/96 | 0.314771 |
| native | oracle | unrelated-64-load16 | 96/96 | 0.219306 |
| native | oracle | unrelated-64-load2 | 95/96 | 0.202533 |

Length and record positions match across load and naming arms; entity recurrence and text still change. Accuracy alone does not identify GDN interference. Oracle is not a deployable method. All earlier failed scientific criteria remain false.

See all role/style/name strata and paired per-stream gains/losses in the adjacent CSV files. These are descriptive pilot results; there is no significance, architecture, long-context scaleup or novelty claim.
