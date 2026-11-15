# 尾部后递归状态、卷积与KV交叉：完整结果

复用8条已测事件流、32个尾部上下文、192个问题，保留所有基线失败。固定Qwen3.5-2B IT与已有GDN适配器，全部6层Attention仍在；不是Base实测或新训练。
每个尾部结束后构造全部8种缓存，再供全部问题读取。来源顺序为18个递归矩阵、18个卷积状态、6层Attention KV；N为正常尾部，H为尾部双门关闭。12份前缀KV切片固定。
正式新测4608项；1536条历史NNN/HHH端点参考组成6144项完整因子表。另1536项新端点重放全部与历史五字段答案精确一致，作为运行控制单列，不计为新主结果。
正式运行manifest SHA256：`ba01f38b49f23af7af758331694ba4e9024ebffbaa12437862356517475b69de`。登记包SHA256：`209c3dc8185f66e60bbcb05abec9086082c64b17a825a2aca3753a82f8af9dfd`。
214份测量源码不变，全部128份N/H源缓存和512份因子缓存通过审计。混合缓存没有冒用源logits，问题后缀重新计算实际输出；源缓存与所有组合独立存储。

## 全部32条件

每行192个读出。Joint3为64个流×尾部×文体，Joint6为32个流×尾部；独立事件流仅8条。Oracle使用真实支持标注，只是能力对照。

| Model | R/C/KV | Reader | Origin | Current /64 | Initial /64 | Protected /64 | Exact /192 | Joint3 /64 | Joint6 /32 | All /8 | NLL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | NNN | full | archived_endpoint_reference | 52 | 56 | 64 | 172 | 46 | 19 | 2 | 0.312241 |
| native | NNN | oracle | archived_endpoint_reference | 61 | 64 | 64 | 189 | 61 | 30 | 6 | 0.232877 |
| native | NNH | full | new_primary | 11 | 24 | 35 | 70 | 5 | 2 | 0 | 1.016190 |
| native | NNH | oracle | new_primary | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.643140 |
| native | NHN | full | new_primary | 53 | 57 | 64 | 174 | 49 | 20 | 3 | 0.311303 |
| native | NHN | oracle | new_primary | 62 | 64 | 64 | 190 | 62 | 30 | 6 | 0.236987 |
| native | NHH | full | new_primary | 12 | 23 | 34 | 69 | 5 | 2 | 0 | 1.036659 |
| native | NHH | oracle | new_primary | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.648636 |
| native | HNN | full | new_primary | 48 | 47 | 64 | 159 | 36 | 16 | 1 | 0.281911 |
| native | HNN | oracle | new_primary | 64 | 64 | 64 | 192 | 64 | 32 | 8 | 0.099778 |
| native | HNH | full | new_primary | 14 | 20 | 38 | 72 | 6 | 3 | 0 | 1.005799 |
| native | HNH | oracle | new_primary | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.537157 |
| native | HHN | full | new_primary | 46 | 43 | 64 | 153 | 33 | 15 | 1 | 0.299479 |
| native | HHN | oracle | new_primary | 64 | 64 | 64 | 192 | 64 | 32 | 8 | 0.114732 |
| native | HHH | full | archived_endpoint_reference | 12 | 20 | 38 | 70 | 6 | 3 | 0 | 1.026273 |
| native | HHH | oracle | archived_endpoint_reference | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.550146 |
| gdn_all_tokens | NNN | full | archived_endpoint_reference | 64 | 63 | 64 | 191 | 63 | 31 | 7 | 0.006624 |
| gdn_all_tokens | NNN | oracle | archived_endpoint_reference | 64 | 64 | 64 | 192 | 64 | 32 | 8 | 0.043133 |
| gdn_all_tokens | NNH | full | new_primary | 9 | 44 | 62 | 115 | 8 | 3 | 0 | 0.691272 |
| gdn_all_tokens | NNH | oracle | new_primary | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.471116 |
| gdn_all_tokens | NHN | full | new_primary | 64 | 62 | 64 | 190 | 62 | 30 | 7 | 0.007558 |
| gdn_all_tokens | NHN | oracle | new_primary | 64 | 64 | 64 | 192 | 64 | 32 | 8 | 0.054140 |
| gdn_all_tokens | NHH | full | new_primary | 7 | 45 | 61 | 113 | 6 | 2 | 0 | 0.684464 |
| gdn_all_tokens | NHH | oracle | new_primary | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.486058 |
| gdn_all_tokens | HNN | full | new_primary | 34 | 62 | 64 | 160 | 33 | 16 | 0 | 0.298314 |
| gdn_all_tokens | HNN | oracle | new_primary | 64 | 64 | 64 | 192 | 64 | 32 | 8 | 0.000683 |
| gdn_all_tokens | HNH | full | new_primary | 12 | 48 | 64 | 124 | 11 | 3 | 0 | 0.703416 |
| gdn_all_tokens | HNH | oracle | new_primary | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.504131 |
| gdn_all_tokens | HHN | full | new_primary | 34 | 62 | 64 | 160 | 33 | 16 | 0 | 0.317237 |
| gdn_all_tokens | HHN | oracle | new_primary | 64 | 64 | 64 | 192 | 64 | 32 | 8 | 0.001706 |
| gdn_all_tokens | HHH | full | archived_endpoint_reference | 8 | 47 | 64 | 119 | 6 | 2 | 0 | 0.693763 |
| gdn_all_tokens | HHH | oracle | archived_endpoint_reference | 32 | 64 | 64 | 160 | 32 | 16 | 0 | 0.501331 |

## 全部128个尾部条件

| Model | Condition.Reader | Episode | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint6 /8 | NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| native | NNN.full | unrelated-8 | 13 | 15 | 16 | 44 | 5 | 0.285379 |
| native | NNN.full | related-8 | 16 | 13 | 16 | 45 | 6 | 0.267190 |
| native | NNN.full | unrelated-64 | 8 | 15 | 16 | 39 | 2 | 0.357843 |
| native | NNN.full | related-64 | 15 | 13 | 16 | 44 | 6 | 0.338554 |
| native | NNN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.187921 |
| native | NNN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.193789 |
| native | NNN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.278382 |
| native | NNN.oracle | related-64 | 13 | 16 | 16 | 45 | 6 | 0.271415 |
| native | NNH.full | unrelated-8 | 10 | 9 | 14 | 33 | 2 | 0.421460 |
| native | NNH.full | related-8 | 0 | 9 | 14 | 23 | 0 | 1.228437 |
| native | NNH.full | unrelated-64 | 1 | 3 | 3 | 7 | 0 | 0.896213 |
| native | NNH.full | related-64 | 0 | 3 | 4 | 7 | 0 | 1.518649 |
| native | NNH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.187921 |
| native | NNH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 1.090155 |
| native | NNH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.278382 |
| native | NNH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 1.016104 |
| native | NHN.full | unrelated-8 | 14 | 15 | 16 | 45 | 5 | 0.296444 |
| native | NHN.full | related-8 | 16 | 14 | 16 | 46 | 6 | 0.276791 |
| native | NHN.full | unrelated-64 | 9 | 15 | 16 | 40 | 3 | 0.348609 |
| native | NHN.full | related-64 | 14 | 13 | 16 | 43 | 6 | 0.323367 |
| native | NHN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.196916 |
| native | NHN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.206396 |
| native | NHN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.273691 |
| native | NHN.oracle | related-64 | 14 | 16 | 16 | 46 | 6 | 0.270943 |
| native | NHH.full | unrelated-8 | 10 | 9 | 14 | 33 | 2 | 0.444935 |
| native | NHH.full | related-8 | 0 | 9 | 14 | 23 | 0 | 1.242688 |
| native | NHH.full | unrelated-64 | 2 | 2 | 3 | 7 | 0 | 0.932071 |
| native | NHH.full | related-64 | 0 | 3 | 3 | 6 | 0 | 1.526942 |
| native | NHH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.196916 |
| native | NHH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 1.107378 |
| native | NHH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.273691 |
| native | NHH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 1.016559 |
| native | HNN.full | unrelated-8 | 16 | 13 | 16 | 45 | 6 | 0.206508 |
| native | HNN.full | related-8 | 12 | 12 | 16 | 40 | 4 | 0.248939 |
| native | HNN.full | unrelated-64 | 16 | 12 | 16 | 44 | 5 | 0.255843 |
| native | HNN.full | related-64 | 4 | 10 | 16 | 30 | 1 | 0.416354 |
| native | HNN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.065219 |
| native | HNN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.100179 |
| native | HNN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.083518 |
| native | HNN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.150196 |
| native | HNH.full | unrelated-8 | 14 | 8 | 16 | 38 | 3 | 0.340664 |
| native | HNH.full | related-8 | 0 | 8 | 16 | 24 | 0 | 1.263643 |
| native | HNH.full | unrelated-64 | 0 | 2 | 3 | 5 | 0 | 0.850259 |
| native | HNH.full | related-64 | 0 | 2 | 3 | 5 | 0 | 1.568631 |
| native | HNH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.065219 |
| native | HNH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 1.028527 |
| native | HNH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.083518 |
| native | HNH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 0.971363 |
| native | HHN.full | unrelated-8 | 16 | 12 | 16 | 44 | 5 | 0.228338 |
| native | HHN.full | related-8 | 12 | 11 | 16 | 39 | 4 | 0.264425 |
| native | HHN.full | unrelated-64 | 16 | 10 | 16 | 42 | 5 | 0.291882 |
| native | HHN.full | related-64 | 2 | 10 | 16 | 28 | 1 | 0.413271 |
| native | HHN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.077387 |
| native | HHN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.123369 |
| native | HHN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.093538 |
| native | HHN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.164632 |
| native | HHH.full | unrelated-8 | 12 | 8 | 16 | 36 | 3 | 0.365341 |
| native | HHH.full | related-8 | 0 | 8 | 16 | 24 | 0 | 1.270605 |
| native | HHH.full | unrelated-64 | 0 | 2 | 3 | 5 | 0 | 0.884144 |
| native | HHH.full | related-64 | 0 | 2 | 3 | 5 | 0 | 1.585004 |
| native | HHH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.077387 |
| native | HHH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 1.048189 |
| native | HHH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.093538 |
| native | HHH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 0.981470 |
| gdn_all_tokens | NNN.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.003735 |
| gdn_all_tokens | NNN.full | related-8 | 16 | 16 | 16 | 48 | 8 | 0.005241 |
| gdn_all_tokens | NNN.full | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.009447 |
| gdn_all_tokens | NNN.full | related-64 | 16 | 15 | 16 | 47 | 7 | 0.008072 |
| gdn_all_tokens | NNN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.024612 |
| gdn_all_tokens | NNN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.017480 |
| gdn_all_tokens | NNN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.085847 |
| gdn_all_tokens | NNN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.044593 |
| gdn_all_tokens | NNH.full | unrelated-8 | 9 | 15 | 16 | 40 | 3 | 0.055278 |
| gdn_all_tokens | NNH.full | related-8 | 0 | 15 | 16 | 31 | 0 | 1.262466 |
| gdn_all_tokens | NNH.full | unrelated-64 | 0 | 6 | 15 | 21 | 0 | 0.347988 |
| gdn_all_tokens | NNH.full | related-64 | 0 | 8 | 15 | 23 | 0 | 1.099354 |
| gdn_all_tokens | NNH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.024612 |
| gdn_all_tokens | NNH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 0.980318 |
| gdn_all_tokens | NNH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.085847 |
| gdn_all_tokens | NNH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 0.793686 |
| gdn_all_tokens | NHN.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.003344 |
| gdn_all_tokens | NHN.full | related-8 | 16 | 16 | 16 | 48 | 8 | 0.007578 |
| gdn_all_tokens | NHN.full | unrelated-64 | 16 | 15 | 16 | 47 | 7 | 0.011004 |
| gdn_all_tokens | NHN.full | related-64 | 16 | 15 | 16 | 47 | 7 | 0.008309 |
| gdn_all_tokens | NHN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.034182 |
| gdn_all_tokens | NHN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.026235 |
| gdn_all_tokens | NHN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.095657 |
| gdn_all_tokens | NHN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.060487 |
| gdn_all_tokens | NHH.full | unrelated-8 | 7 | 15 | 16 | 38 | 2 | 0.054144 |
| gdn_all_tokens | NHH.full | related-8 | 0 | 14 | 16 | 30 | 0 | 1.261581 |
| gdn_all_tokens | NHH.full | unrelated-64 | 0 | 7 | 15 | 22 | 0 | 0.340273 |
| gdn_all_tokens | NHH.full | related-64 | 0 | 9 | 14 | 23 | 0 | 1.081858 |
| gdn_all_tokens | NHH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.034182 |
| gdn_all_tokens | NHH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 0.991770 |
| gdn_all_tokens | NHH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.095657 |
| gdn_all_tokens | NHH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 0.822621 |
| gdn_all_tokens | HNN.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.003128 |
| gdn_all_tokens | HNN.full | related-8 | 2 | 16 | 16 | 34 | 1 | 0.231654 |
| gdn_all_tokens | HNN.full | unrelated-64 | 16 | 15 | 16 | 47 | 7 | 0.029078 |
| gdn_all_tokens | HNN.full | related-64 | 0 | 15 | 16 | 31 | 0 | 0.929397 |
| gdn_all_tokens | HNN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.000485 |
| gdn_all_tokens | HNN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.000711 |
| gdn_all_tokens | HNN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.000836 |
| gdn_all_tokens | HNN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000701 |
| gdn_all_tokens | HNH.full | unrelated-8 | 11 | 15 | 16 | 42 | 3 | 0.038231 |
| gdn_all_tokens | HNH.full | related-8 | 0 | 16 | 16 | 32 | 0 | 1.413137 |
| gdn_all_tokens | HNH.full | unrelated-64 | 1 | 10 | 16 | 27 | 0 | 0.240555 |
| gdn_all_tokens | HNH.full | related-64 | 0 | 7 | 16 | 23 | 0 | 1.121743 |
| gdn_all_tokens | HNH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.000485 |
| gdn_all_tokens | HNH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 1.040933 |
| gdn_all_tokens | HNH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.000836 |
| gdn_all_tokens | HNH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 0.974269 |
| gdn_all_tokens | HHN.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.003372 |
| gdn_all_tokens | HHN.full | related-8 | 2 | 16 | 16 | 34 | 1 | 0.284598 |
| gdn_all_tokens | HHN.full | unrelated-64 | 16 | 15 | 16 | 47 | 7 | 0.030602 |
| gdn_all_tokens | HHN.full | related-64 | 0 | 15 | 16 | 31 | 0 | 0.950376 |
| gdn_all_tokens | HHN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.001345 |
| gdn_all_tokens | HHN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.002305 |
| gdn_all_tokens | HHN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.001138 |
| gdn_all_tokens | HHN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.002037 |
| gdn_all_tokens | HHH.full | unrelated-8 | 8 | 14 | 16 | 38 | 2 | 0.039148 |
| gdn_all_tokens | HHH.full | related-8 | 0 | 15 | 16 | 31 | 0 | 1.397982 |
| gdn_all_tokens | HHH.full | unrelated-64 | 0 | 10 | 16 | 26 | 0 | 0.209469 |
| gdn_all_tokens | HHH.full | related-64 | 0 | 8 | 16 | 24 | 0 | 1.128455 |
| gdn_all_tokens | HHH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.001345 |
| gdn_all_tokens | HHH.oracle | related-8 | 0 | 16 | 16 | 32 | 0 | 1.043377 |
| gdn_all_tokens | HHH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.001138 |
| gdn_all_tokens | HHH.oracle | related-64 | 0 | 16 | 16 | 32 | 0 | 0.959462 |

## 解读边界

`readouts.csv`明确区分新主测量和历史参考；`endpoint-replays.csv`单列精确端点重放。全部角色、文体、流和证据来源见`strata.csv`，全部单分量替换/修复、Oracle及相关/无关对照见`contrasts.csv`。
`nll-interactions.csv`给出每个固定问题的7项有限差分，含二阶与三阶项；它们是冻结缓存干预的NLL交互，不是学到的集合效用或显著性证据。
`cache-components.csv`逐层报告全部3072项N/H分量比较，`cache-identities.csv`保留512个组合及等价组合清单；重复条件不重采样、不删除。
相关尾部的当前答案来自新写入，和固定前缀证据分开报告。相关/无关匹配时该答案改变，不能把这个比较叫同一事实损伤。
交叉缓存可能偏离自然分布。分量替换会改变后续Query和答案计算；回答恢复不证明事实存储或删除，也不证明共享控制的新架构必要。
保留完整逐张量指纹、布局、实际属性与运行检查，没有保存所有缓存张量的全量副本。精确重跑需要相同本地权重及后端。
本轮复用已测队列，是机制定位，不是新数据确认；没有训练、显著性声明、架构成功或扩模门槛，旧失败门槛保持原判。
