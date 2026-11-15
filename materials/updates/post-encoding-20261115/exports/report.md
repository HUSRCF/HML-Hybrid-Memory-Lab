# 正常编码后再更新：完整诊断结果

本轮使用固定Qwen3.5-2B IT及四种已有适配器部署，不是Base实测或新架构训练。8条新事件流、40个上下文、240个唯一问题，正式读出8160项。
前缀正常编码一次；同一源缓存服务全部问题。干预只覆盖后续更新，Oracle只在问题阶段限制历史读取，仍使用后见支持标签。

数据manifest SHA256：`d3e65bc72abd1a901c26c4c69f768bb83d9095c6bb2d9e0987ffa2a8545a0051`。正式运行manifest SHA256：`3bdd8f7e7ee657d270911d77ab15bf8b6b292aaca11fcf68b5afdc6714829b8a`。
259项测试、两轮相反模型顺序的各510项smoke、206份源码、全部680份源缓存、640种尾部条件、160项no-op缓存/输出检查与320项no-op读出通过审计。
所有前缀KV切片逐位保留；问题阶段部署与原生模型在问题到来前的缓存完全一致。两门均关闭只保证18个递归矩阵不变，卷积仍运行。

## 全部170条件

每行48个问题读出（8流×3角色×2文体）。Joint3以16个流×文体为分母，Joint6以8个流为分母；这些读出不是48个独立事件流。

| Model | Episode | Gates | Reader | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint3 /16 | Joint6 /8 | NLL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| native | before | none | full | 16 | 10 | 16 | 42 | 10 | 4 | 0.226558 |
| native | before | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.063338 |
| native | unrelated-8 | none | full | 13 | 15 | 16 | 44 | 12 | 5 | 0.285379 |
| native | unrelated-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.187921 |
| native | unrelated-8 | decay | full | 13 | 12 | 16 | 41 | 9 | 4 | 0.311319 |
| native | unrelated-8 | decay | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.185443 |
| native | unrelated-8 | delta | full | 13 | 10 | 16 | 39 | 8 | 4 | 0.384245 |
| native | unrelated-8 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.055888 |
| native | unrelated-8 | both | full | 12 | 8 | 16 | 36 | 6 | 3 | 0.365341 |
| native | unrelated-8 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.077387 |
| native | related-8 | none | full | 16 | 13 | 16 | 45 | 13 | 6 | 0.267190 |
| native | related-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.193789 |
| native | related-8 | decay | full | 0 | 13 | 16 | 29 | 0 | 0 | 1.071877 |
| native | related-8 | decay | oracle | 2 | 16 | 16 | 34 | 2 | 1 | 0.590056 |
| native | related-8 | delta | full | 0 | 10 | 16 | 26 | 0 | 0 | 1.177186 |
| native | related-8 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.898171 |
| native | related-8 | both | full | 0 | 8 | 16 | 24 | 0 | 0 | 1.270605 |
| native | related-8 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 1.048189 |
| native | unrelated-64 | none | full | 8 | 15 | 16 | 39 | 8 | 2 | 0.357843 |
| native | unrelated-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.278382 |
| native | unrelated-64 | decay | full | 7 | 5 | 10 | 22 | 2 | 0 | 0.616727 |
| native | unrelated-64 | decay | oracle | 13 | 16 | 15 | 44 | 13 | 6 | 0.241885 |
| native | unrelated-64 | delta | full | 11 | 11 | 14 | 36 | 6 | 2 | 0.418708 |
| native | unrelated-64 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.038597 |
| native | unrelated-64 | both | full | 0 | 2 | 3 | 5 | 0 | 0 | 0.884144 |
| native | unrelated-64 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.093538 |
| native | related-64 | none | full | 15 | 13 | 16 | 44 | 13 | 6 | 0.338554 |
| native | related-64 | none | oracle | 13 | 16 | 16 | 45 | 13 | 6 | 0.271415 |
| native | related-64 | decay | full | 0 | 4 | 9 | 13 | 0 | 0 | 1.390687 |
| native | related-64 | decay | oracle | 0 | 15 | 15 | 30 | 0 | 0 | 1.071646 |
| native | related-64 | delta | full | 0 | 10 | 14 | 24 | 0 | 0 | 1.257058 |
| native | related-64 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 1.004841 |
| native | related-64 | both | full | 0 | 2 | 3 | 5 | 0 | 0 | 1.585004 |
| native | related-64 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.981470 |
| shared | before | none | full | 16 | 16 | 16 | 48 | 16 | 8 | 0.007689 |
| shared | before | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001271 |
| shared | unrelated-8 | none | full | 15 | 16 | 16 | 47 | 15 | 7 | 0.012738 |
| shared | unrelated-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.025480 |
| shared | unrelated-8 | decay | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.041886 |
| shared | unrelated-8 | decay | oracle | 14 | 16 | 16 | 46 | 14 | 7 | 0.027402 |
| shared | unrelated-8 | delta | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.033401 |
| shared | unrelated-8 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.000788 |
| shared | unrelated-8 | both | full | 3 | 14 | 16 | 33 | 2 | 0 | 0.059901 |
| shared | unrelated-8 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.004681 |
| shared | related-8 | none | full | 16 | 16 | 16 | 48 | 16 | 8 | 0.011073 |
| shared | related-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.012772 |
| shared | related-8 | decay | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.483125 |
| shared | related-8 | decay | oracle | 6 | 16 | 16 | 38 | 6 | 2 | 0.240149 |
| shared | related-8 | delta | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.338300 |
| shared | related-8 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.655201 |
| shared | related-8 | both | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.480075 |
| shared | related-8 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.956574 |
| shared | unrelated-64 | none | full | 15 | 15 | 16 | 46 | 14 | 6 | 0.034402 |
| shared | unrelated-64 | none | oracle | 15 | 16 | 16 | 47 | 15 | 7 | 0.090691 |
| shared | unrelated-64 | decay | full | 10 | 11 | 13 | 34 | 5 | 1 | 0.330081 |
| shared | unrelated-64 | decay | oracle | 14 | 16 | 14 | 44 | 13 | 6 | 0.071124 |
| shared | unrelated-64 | delta | full | 16 | 9 | 16 | 41 | 9 | 4 | 0.140804 |
| shared | unrelated-64 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001614 |
| shared | unrelated-64 | both | full | 3 | 8 | 16 | 27 | 2 | 0 | 0.335832 |
| shared | unrelated-64 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.006396 |
| shared | related-64 | none | full | 16 | 15 | 16 | 47 | 15 | 7 | 0.013500 |
| shared | related-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.036486 |
| shared | related-64 | decay | full | 0 | 11 | 12 | 23 | 0 | 0 | 1.580270 |
| shared | related-64 | decay | oracle | 0 | 16 | 14 | 30 | 0 | 0 | 1.001074 |
| shared | related-64 | delta | full | 0 | 9 | 16 | 25 | 0 | 0 | 1.385535 |
| shared | related-64 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.950828 |
| shared | related-64 | both | full | 0 | 10 | 16 | 26 | 0 | 0 | 1.188225 |
| shared | related-64 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.846445 |
| shared_question_trained_all | before | none | full | 16 | 14 | 15 | 45 | 13 | 6 | 0.062600 |
| shared_question_trained_all | before | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.000549 |
| shared_question_trained_all | unrelated-8 | none | full | 15 | 14 | 16 | 45 | 13 | 6 | 0.044342 |
| shared_question_trained_all | unrelated-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.008696 |
| shared_question_trained_all | unrelated-8 | decay | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.067204 |
| shared_question_trained_all | unrelated-8 | decay | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.010235 |
| shared_question_trained_all | unrelated-8 | delta | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.046936 |
| shared_question_trained_all | unrelated-8 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.000487 |
| shared_question_trained_all | unrelated-8 | both | full | 15 | 14 | 16 | 45 | 13 | 6 | 0.086524 |
| shared_question_trained_all | unrelated-8 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.000916 |
| shared_question_trained_all | related-8 | none | full | 13 | 15 | 16 | 44 | 13 | 6 | 0.082910 |
| shared_question_trained_all | related-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.007803 |
| shared_question_trained_all | related-8 | decay | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.567328 |
| shared_question_trained_all | related-8 | decay | oracle | 7 | 16 | 16 | 39 | 7 | 3 | 0.306140 |
| shared_question_trained_all | related-8 | delta | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.443161 |
| shared_question_trained_all | related-8 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.692675 |
| shared_question_trained_all | related-8 | both | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.626752 |
| shared_question_trained_all | related-8 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.873047 |
| shared_question_trained_all | unrelated-64 | none | full | 15 | 15 | 16 | 46 | 14 | 6 | 0.050919 |
| shared_question_trained_all | unrelated-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.085744 |
| shared_question_trained_all | unrelated-64 | decay | full | 11 | 9 | 13 | 33 | 4 | 1 | 0.328393 |
| shared_question_trained_all | unrelated-64 | decay | oracle | 15 | 16 | 16 | 47 | 15 | 7 | 0.080465 |
| shared_question_trained_all | unrelated-64 | delta | full | 16 | 6 | 16 | 38 | 6 | 3 | 0.166826 |
| shared_question_trained_all | unrelated-64 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001605 |
| shared_question_trained_all | unrelated-64 | both | full | 6 | 8 | 16 | 30 | 3 | 1 | 0.314369 |
| shared_question_trained_all | unrelated-64 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001831 |
| shared_question_trained_all | related-64 | none | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.045285 |
| shared_question_trained_all | related-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.055655 |
| shared_question_trained_all | related-64 | decay | full | 0 | 10 | 11 | 21 | 0 | 0 | 1.504789 |
| shared_question_trained_all | related-64 | decay | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.983844 |
| shared_question_trained_all | related-64 | delta | full | 0 | 6 | 16 | 22 | 0 | 0 | 1.499280 |
| shared_question_trained_all | related-64 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.952125 |
| shared_question_trained_all | related-64 | both | full | 0 | 9 | 16 | 25 | 0 | 0 | 1.226132 |
| shared_question_trained_all | related-64 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.853866 |
| shared_question_trained_question | before | none | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.051182 |
| shared_question_trained_question | before | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.000711 |
| shared_question_trained_question | unrelated-8 | none | full | 15 | 14 | 16 | 45 | 13 | 6 | 0.026398 |
| shared_question_trained_question | unrelated-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.020993 |
| shared_question_trained_question | unrelated-8 | decay | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.064199 |
| shared_question_trained_question | unrelated-8 | decay | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.011696 |
| shared_question_trained_question | unrelated-8 | delta | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.044080 |
| shared_question_trained_question | unrelated-8 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.000362 |
| shared_question_trained_question | unrelated-8 | both | full | 14 | 14 | 16 | 44 | 12 | 5 | 0.081080 |
| shared_question_trained_question | unrelated-8 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001330 |
| shared_question_trained_question | related-8 | none | full | 14 | 15 | 16 | 45 | 14 | 7 | 0.043676 |
| shared_question_trained_question | related-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.017962 |
| shared_question_trained_question | related-8 | decay | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.549247 |
| shared_question_trained_question | related-8 | decay | oracle | 6 | 16 | 16 | 38 | 6 | 3 | 0.321581 |
| shared_question_trained_question | related-8 | delta | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.435075 |
| shared_question_trained_question | related-8 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.700806 |
| shared_question_trained_question | related-8 | both | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.581965 |
| shared_question_trained_question | related-8 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.888477 |
| shared_question_trained_question | unrelated-64 | none | full | 15 | 15 | 16 | 46 | 14 | 6 | 0.053645 |
| shared_question_trained_question | unrelated-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.113111 |
| shared_question_trained_question | unrelated-64 | decay | full | 13 | 10 | 13 | 36 | 6 | 3 | 0.284761 |
| shared_question_trained_question | unrelated-64 | decay | oracle | 14 | 16 | 15 | 45 | 14 | 7 | 0.087156 |
| shared_question_trained_question | unrelated-64 | delta | full | 16 | 4 | 16 | 36 | 4 | 1 | 0.168485 |
| shared_question_trained_question | unrelated-64 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001438 |
| shared_question_trained_question | unrelated-64 | both | full | 6 | 8 | 16 | 30 | 3 | 1 | 0.375060 |
| shared_question_trained_question | unrelated-64 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.002509 |
| shared_question_trained_question | related-64 | none | full | 16 | 14 | 16 | 46 | 14 | 7 | 0.051383 |
| shared_question_trained_question | related-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.082680 |
| shared_question_trained_question | related-64 | decay | full | 0 | 12 | 12 | 24 | 0 | 0 | 1.471878 |
| shared_question_trained_question | related-64 | decay | oracle | 0 | 16 | 15 | 31 | 0 | 0 | 0.990315 |
| shared_question_trained_question | related-64 | delta | full | 0 | 5 | 16 | 21 | 0 | 0 | 1.501585 |
| shared_question_trained_question | related-64 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.953873 |
| shared_question_trained_question | related-64 | both | full | 0 | 9 | 16 | 25 | 0 | 0 | 1.262208 |
| shared_question_trained_question | related-64 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.871873 |
| gdn_all_tokens | before | none | full | 16 | 16 | 16 | 48 | 16 | 8 | 0.006375 |
| gdn_all_tokens | before | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001111 |
| gdn_all_tokens | unrelated-8 | none | full | 16 | 16 | 16 | 48 | 16 | 8 | 0.003735 |
| gdn_all_tokens | unrelated-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.024612 |
| gdn_all_tokens | unrelated-8 | decay | full | 16 | 16 | 16 | 48 | 16 | 8 | 0.019680 |
| gdn_all_tokens | unrelated-8 | decay | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.009277 |
| gdn_all_tokens | unrelated-8 | delta | full | 16 | 15 | 16 | 47 | 15 | 7 | 0.038382 |
| gdn_all_tokens | unrelated-8 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.000706 |
| gdn_all_tokens | unrelated-8 | both | full | 8 | 14 | 16 | 38 | 6 | 2 | 0.039148 |
| gdn_all_tokens | unrelated-8 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001345 |
| gdn_all_tokens | related-8 | none | full | 16 | 16 | 16 | 48 | 16 | 8 | 0.005241 |
| gdn_all_tokens | related-8 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.017480 |
| gdn_all_tokens | related-8 | decay | full | 0 | 16 | 16 | 32 | 0 | 0 | 1.540408 |
| gdn_all_tokens | related-8 | decay | oracle | 4 | 16 | 16 | 36 | 4 | 2 | 0.334745 |
| gdn_all_tokens | related-8 | delta | full | 0 | 15 | 16 | 31 | 0 | 0 | 1.285261 |
| gdn_all_tokens | related-8 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.764074 |
| gdn_all_tokens | related-8 | both | full | 0 | 15 | 16 | 31 | 0 | 0 | 1.397982 |
| gdn_all_tokens | related-8 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 1.043377 |
| gdn_all_tokens | unrelated-64 | none | full | 16 | 16 | 16 | 48 | 16 | 8 | 0.009447 |
| gdn_all_tokens | unrelated-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.085847 |
| gdn_all_tokens | unrelated-64 | decay | full | 9 | 9 | 11 | 29 | 2 | 0 | 0.387082 |
| gdn_all_tokens | unrelated-64 | decay | oracle | 13 | 15 | 14 | 42 | 13 | 6 | 0.156653 |
| gdn_all_tokens | unrelated-64 | delta | full | 14 | 14 | 16 | 44 | 12 | 5 | 0.132195 |
| gdn_all_tokens | unrelated-64 | delta | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.003665 |
| gdn_all_tokens | unrelated-64 | both | full | 0 | 10 | 16 | 26 | 0 | 0 | 0.209469 |
| gdn_all_tokens | unrelated-64 | both | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.001138 |
| gdn_all_tokens | related-64 | none | full | 16 | 15 | 16 | 47 | 15 | 7 | 0.008072 |
| gdn_all_tokens | related-64 | none | oracle | 16 | 16 | 16 | 48 | 16 | 8 | 0.044593 |
| gdn_all_tokens | related-64 | decay | full | 0 | 8 | 11 | 19 | 0 | 0 | 1.575396 |
| gdn_all_tokens | related-64 | decay | oracle | 0 | 15 | 14 | 29 | 0 | 0 | 1.043530 |
| gdn_all_tokens | related-64 | delta | full | 0 | 14 | 16 | 30 | 0 | 0 | 1.306974 |
| gdn_all_tokens | related-64 | delta | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.901575 |
| gdn_all_tokens | related-64 | both | full | 0 | 8 | 16 | 24 | 0 | 0 | 1.128455 |
| gdn_all_tokens | related-64 | both | oracle | 0 | 16 | 16 | 32 | 0 | 0 | 0.959462 |

## 更新与保持的边界

全部编码后基线失败均保留。`before-after.csv`逐条件报告both_correct、gain、loss与both_wrong，并单列固定前缀证据和三个角色；`strata.csv`保留角色、文体及逐流统计。
相关尾部的当前值需要编码新的最后一条SET，每条件16次，单独作为正确更新对照；这些答案的证据不在固定前缀。其余问题对应已编码的前缀证据。
`contrasts.csv`保留同一尾部/问题下各门控对none、Oracle对full，以及同长度相关对无关尾部的比较。相关与无关尾部的当前答案不同，不能把其比较解释为同一目标答案的损伤。

## 解释限制

门控仍会改变后续Query、隐藏表示、尾部KV与卷积状态。旧KV相同不唯一定位到递归矩阵存储损坏，也不分离Delta擦除与插入。接尾部前后还存在距离/位置变化；主要干预对照固定实际尾部Token和位置。
缓存记录保留逐张量指纹、布局、实际元数据和检查回执，未保存680份完整缓存张量。逐张量相等检查由封存的运行代码执行；外部复现这些计算仍需在相同模型和后端上重跑。
本轮是冻结模型的机制诊断，没有新架构成功门槛、显著性声明、训练或扩模触发条件。此前失败的科学门槛全部保持原判。
