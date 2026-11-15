# Complete single-token version-question control

All 1,728 new Full readouts and 1,728 explicitly archived references are retained. New matched comparison: state_term -> event_term.

| Origin | Model | Question style | Condition | Correct | Mean answer NLL |
|---|---|---|---|---:|---:|
| archived | gdn_all_tokens | entity_first | before | 44/48 | 0.098736 |
| archived | gdn_all_tokens | entity_first | related-512-load16 | 46/48 | 0.058362 |
| archived | gdn_all_tokens | entity_first | related-512-load2 | 40/48 | 0.178307 |
| archived | gdn_all_tokens | entity_first | related-64-load16 | 48/48 | 0.008836 |
| archived | gdn_all_tokens | entity_first | related-64-load2 | 47/48 | 0.016630 |
| archived | gdn_all_tokens | entity_first | unrelated-512-load16 | 46/48 | 0.028152 |
| archived | gdn_all_tokens | entity_first | unrelated-512-load2 | 47/48 | 0.026685 |
| archived | gdn_all_tokens | entity_first | unrelated-64-load16 | 48/48 | 0.010845 |
| archived | gdn_all_tokens | entity_first | unrelated-64-load2 | 47/48 | 0.016848 |
| archived | gdn_all_tokens | original | before | 45/48 | 0.064746 |
| archived | gdn_all_tokens | original | related-512-load16 | 46/48 | 0.034544 |
| archived | gdn_all_tokens | original | related-512-load2 | 40/48 | 0.313476 |
| archived | gdn_all_tokens | original | related-64-load16 | 48/48 | 0.001926 |
| archived | gdn_all_tokens | original | related-64-load2 | 48/48 | 0.000906 |
| archived | gdn_all_tokens | original | unrelated-512-load16 | 43/48 | 0.063280 |
| archived | gdn_all_tokens | original | unrelated-512-load2 | 46/48 | 0.050640 |
| archived | gdn_all_tokens | original | unrelated-64-load16 | 47/48 | 0.025147 |
| archived | gdn_all_tokens | original | unrelated-64-load2 | 45/48 | 0.051373 |
| archived | native | entity_first | before | 33/48 | 0.305312 |
| archived | native | entity_first | related-512-load16 | 16/48 | 0.646522 |
| archived | native | entity_first | related-512-load2 | 20/48 | 0.628778 |
| archived | native | entity_first | related-64-load16 | 30/48 | 0.473784 |
| archived | native | entity_first | related-64-load2 | 25/48 | 0.534401 |
| archived | native | entity_first | unrelated-512-load16 | 18/48 | 0.606657 |
| archived | native | entity_first | unrelated-512-load2 | 25/48 | 0.596783 |
| archived | native | entity_first | unrelated-64-load16 | 28/48 | 0.508125 |
| archived | native | entity_first | unrelated-64-load2 | 25/48 | 0.471552 |
| archived | native | original | before | 37/48 | 0.273419 |
| archived | native | original | related-512-load16 | 20/48 | 0.618913 |
| archived | native | original | related-512-load2 | 24/48 | 0.569740 |
| archived | native | original | related-64-load16 | 41/48 | 0.361460 |
| archived | native | original | related-64-load2 | 38/48 | 0.361199 |
| archived | native | original | unrelated-512-load16 | 29/48 | 0.547871 |
| archived | native | original | unrelated-512-load2 | 33/48 | 0.471685 |
| archived | native | original | unrelated-64-load16 | 37/48 | 0.396619 |
| archived | native | original | unrelated-64-load2 | 36/48 | 0.298996 |
| new | gdn_all_tokens | event_term | before | 45/48 | 0.025966 |
| new | gdn_all_tokens | event_term | related-512-load16 | 47/48 | 0.015638 |
| new | gdn_all_tokens | event_term | related-512-load2 | 41/48 | 0.272202 |
| new | gdn_all_tokens | event_term | related-64-load16 | 48/48 | 0.003747 |
| new | gdn_all_tokens | event_term | related-64-load2 | 48/48 | 0.001276 |
| new | gdn_all_tokens | event_term | unrelated-512-load16 | 40/48 | 0.060885 |
| new | gdn_all_tokens | event_term | unrelated-512-load2 | 41/48 | 0.108960 |
| new | gdn_all_tokens | event_term | unrelated-64-load16 | 48/48 | 0.015572 |
| new | gdn_all_tokens | event_term | unrelated-64-load2 | 44/48 | 0.061299 |
| new | gdn_all_tokens | state_term | before | 45/48 | 0.064335 |
| new | gdn_all_tokens | state_term | related-512-load16 | 47/48 | 0.028801 |
| new | gdn_all_tokens | state_term | related-512-load2 | 39/48 | 0.334197 |
| new | gdn_all_tokens | state_term | related-64-load16 | 48/48 | 0.002417 |
| new | gdn_all_tokens | state_term | related-64-load2 | 48/48 | 0.001237 |
| new | gdn_all_tokens | state_term | unrelated-512-load16 | 45/48 | 0.036112 |
| new | gdn_all_tokens | state_term | unrelated-512-load2 | 46/48 | 0.052861 |
| new | gdn_all_tokens | state_term | unrelated-64-load16 | 48/48 | 0.015789 |
| new | gdn_all_tokens | state_term | unrelated-64-load2 | 46/48 | 0.034700 |
| new | native | event_term | before | 46/48 | 0.193160 |
| new | native | event_term | related-512-load16 | 28/48 | 0.527837 |
| new | native | event_term | related-512-load2 | 19/48 | 0.614480 |
| new | native | event_term | related-64-load16 | 37/48 | 0.351950 |
| new | native | event_term | related-64-load2 | 39/48 | 0.356844 |
| new | native | event_term | unrelated-512-load16 | 27/48 | 0.499767 |
| new | native | event_term | unrelated-512-load2 | 28/48 | 0.481206 |
| new | native | event_term | unrelated-64-load16 | 34/48 | 0.414497 |
| new | native | event_term | unrelated-64-load2 | 34/48 | 0.371674 |
| new | native | state_term | before | 40/48 | 0.237672 |
| new | native | state_term | related-512-load16 | 28/48 | 0.564435 |
| new | native | state_term | related-512-load2 | 25/48 | 0.548841 |
| new | native | state_term | related-64-load16 | 42/48 | 0.356034 |
| new | native | state_term | related-64-load2 | 39/48 | 0.342508 |
| new | native | state_term | unrelated-512-load16 | 32/48 | 0.489163 |
| new | native | state_term | unrelated-512-load2 | 35/48 | 0.423025 |
| new | native | state_term | unrelated-64-load16 | 37/48 | 0.393489 |
| new | native | state_term | unrelated-64-load2 | 38/48 | 0.290158 |

Each new style/role/episode/model has16case/name repetitions, with only8underlying streams. Every known question has24tokens and each pair changes only its sixth token. Historical source caches match the predecessor.

Old versus new sentence syntax is not token matched; old rows are archived evidence, not fresh inference. Token-matched paraphrase effects do not identify a unique memory mechanism, prove an update method, or establish architecture effectiveness. All prior failed criteria remain false.

Plots use U/R/L for unrelated/related/active accounts;64/512are tail record counts. Full role tables, all864paired gains/losses, per-stream counts and every generated answer value attribution are adjacent.
