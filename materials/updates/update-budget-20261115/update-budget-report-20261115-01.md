# 完整记录写入预算控制：全部审计结果

本报告由完整审计和逐字段核验的导出数据生成。共13440次新物理读出，加6720项历史参考；统一表20160项，复用8条事件流、48个相关案例。

全部六个Attention层保留。控制只改变一条SET的身份；COPY路径已由Oracle类型预算决定。预算仍有后见信息，不是一个对所有问题共用的在线状态；旧科学失败判定不变。

六项联合为两种顺序×当前、历史、无关三个问题全部正确，分母48；三个角色分别为96项相关绑定，物理读出分母336。联合答案可能来自不同的后见前缀状态。

## native

| 条件 | 来源 | 联合/48 | 当前/96 | 历史/96 | 无关/96 | 物理/336 | 深度1/2/3各/16 |
|---|---|---|---|---|---|---|---|
| full | 历史参考 | 1 | 13 | 96 | 92 | 249 | 1/0/0 |
| delta_full | 历史参考 | 1 | 54 | 29 | 54 | 164 | 1/0/0 |
| random0_full | 新测量 | 0 | 2 | 2 | 0 | 4 | 0/0/0 |
| random1_full | 新测量 | 0 | 0 | 0 | 1 | 1 | 0/0/0 |
| early_full | 新测量 | 0 | 0 | 29 | 0 | 45 | 0/0/0 |
| recent_full | 新测量 | 0 | 42 | 0 | 0 | 42 | 0/0/0 |
| oracle | 历史参考 | 13 | 44 | 96 | 89 | 277 | 6/4/3 |
| delta_oracle | 历史参考 | 48 | 96 | 96 | 96 | 336 | 16/16/16 |
| random0_oracle | 新测量 | 0 | 2 | 4 | 0 | 6 | 0/0/0 |
| random1_oracle | 新测量 | 0 | 4 | 0 | 2 | 8 | 0/0/0 |
| early_oracle | 新测量 | 0 | 0 | 96 | 0 | 128 | 0/0/0 |
| recent_oracle | 新测量 | 0 | 48 | 0 | 0 | 48 | 0/0/0 |

## shared

| 条件 | 来源 | 联合/48 | 当前/96 | 历史/96 | 无关/96 | 物理/336 | 深度1/2/3各/16 |
|---|---|---|---|---|---|---|---|
| full | 历史参考 | 14 | 45 | 92 | 96 | 281 | 12/2/0 |
| delta_full | 历史参考 | 9 | 45 | 60 | 78 | 219 | 7/2/0 |
| random0_full | 新测量 | 0 | 2 | 3 | 0 | 5 | 0/0/0 |
| random1_full | 新测量 | 0 | 0 | 0 | 2 | 3 | 0/0/0 |
| early_full | 新测量 | 0 | 0 | 60 | 0 | 83 | 0/0/0 |
| recent_full | 新测量 | 0 | 33 | 0 | 0 | 33 | 0/0/0 |
| oracle | 历史参考 | 48 | 96 | 96 | 96 | 336 | 16/16/16 |
| delta_oracle | 历史参考 | 48 | 96 | 96 | 96 | 336 | 16/16/16 |
| random0_oracle | 新测量 | 0 | 2 | 4 | 0 | 6 | 0/0/0 |
| random1_oracle | 新测量 | 0 | 4 | 0 | 2 | 8 | 0/0/0 |
| early_oracle | 新测量 | 0 | 0 | 96 | 0 | 128 | 0/0/0 |
| recent_oracle | 新测量 | 0 | 48 | 0 | 0 | 48 | 0/0/0 |

## shared_question_trained_all

| 条件 | 来源 | 联合/48 | 当前/96 | 历史/96 | 无关/96 | 物理/336 | 深度1/2/3各/16 |
|---|---|---|---|---|---|---|---|
| full | 历史参考 | 15 | 43 | 93 | 96 | 280 | 13/1/1 |
| delta_full | 历史参考 | 4 | 48 | 33 | 83 | 195 | 3/1/0 |
| random0_full | 新测量 | 0 | 2 | 2 | 0 | 4 | 0/0/0 |
| random1_full | 新测量 | 0 | 0 | 0 | 2 | 4 | 0/0/0 |
| early_full | 新测量 | 0 | 0 | 33 | 0 | 50 | 0/0/0 |
| recent_full | 新测量 | 0 | 34 | 0 | 0 | 34 | 0/0/0 |
| oracle | 历史参考 | 47 | 96 | 96 | 95 | 334 | 16/15/16 |
| delta_oracle | 历史参考 | 48 | 96 | 96 | 96 | 336 | 16/16/16 |
| random0_oracle | 新测量 | 0 | 2 | 4 | 0 | 6 | 0/0/0 |
| random1_oracle | 新测量 | 0 | 4 | 0 | 2 | 8 | 0/0/0 |
| early_oracle | 新测量 | 0 | 0 | 96 | 0 | 128 | 0/0/0 |
| recent_oracle | 新测量 | 0 | 48 | 0 | 0 | 48 | 0/0/0 |

## shared_question_trained_question

| 条件 | 来源 | 联合/48 | 当前/96 | 历史/96 | 无关/96 | 物理/336 | 深度1/2/3各/16 |
|---|---|---|---|---|---|---|---|
| full | 历史参考 | 15 | 39 | 96 | 96 | 279 | 15/0/0 |
| delta_full | 历史参考 | 5 | 49 | 32 | 84 | 196 | 4/1/0 |
| random0_full | 新测量 | 0 | 2 | 2 | 0 | 4 | 0/0/0 |
| random1_full | 新测量 | 0 | 0 | 0 | 2 | 4 | 0/0/0 |
| early_full | 新测量 | 0 | 0 | 32 | 0 | 49 | 0/0/0 |
| recent_full | 新测量 | 0 | 33 | 0 | 0 | 33 | 0/0/0 |
| oracle | 历史参考 | 46 | 96 | 96 | 94 | 334 | 16/15/15 |
| delta_oracle | 历史参考 | 48 | 96 | 96 | 96 | 336 | 16/16/16 |
| random0_oracle | 新测量 | 0 | 2 | 4 | 0 | 6 | 0/0/0 |
| random1_oracle | 新测量 | 0 | 4 | 0 | 2 | 8 | 0/0/0 |
| early_oracle | 新测量 | 0 | 0 | 96 | 0 | 128 | 0/0/0 |
| recent_oracle | 新测量 | 0 | 48 | 0 | 0 | 48 | 0/0/0 |

## gdn_all_tokens

| 条件 | 来源 | 联合/48 | 当前/96 | 历史/96 | 无关/96 | 物理/336 | 深度1/2/3各/16 |
|---|---|---|---|---|---|---|---|
| full | 历史参考 | 16 | 38 | 96 | 96 | 278 | 15/0/1 |
| delta_full | 历史参考 | 9 | 65 | 47 | 80 | 229 | 7/2/0 |
| random0_full | 新测量 | 0 | 2 | 2 | 0 | 4 | 0/0/0 |
| random1_full | 新测量 | 0 | 2 | 0 | 2 | 6 | 0/0/0 |
| early_full | 新测量 | 0 | 0 | 47 | 0 | 70 | 0/0/0 |
| recent_full | 新测量 | 0 | 45 | 0 | 0 | 45 | 0/0/0 |
| oracle | 历史参考 | 46 | 95 | 96 | 95 | 334 | 16/15/15 |
| delta_oracle | 历史参考 | 48 | 96 | 96 | 96 | 336 | 16/16/16 |
| random0_oracle | 新测量 | 0 | 2 | 4 | 0 | 6 | 0/0/0 |
| random1_oracle | 新测量 | 0 | 4 | 0 | 2 | 8 | 0/0/0 |
| early_oracle | 新测量 | 0 | 0 | 96 | 0 | 128 | 0/0/0 |
| recent_oracle | 新测量 | 0 | 48 | 0 | 0 | 48 | 0/0/0 |

## 全部预定配对比较

增益／退化为同一题或同一相关案例上的正确性变化，不能将相关绑定或分层当成独立样本。当前列分母96，联合48，物理336；损失差为候选减参考。

### native

| 候选 vs 参考 | 当前增益/退化 | 联合增益/退化 | 物理增益/退化 | 当前平均NLL差 |
|---|---|---|---|---|
| random0_full vs delta_full | 0/52 | 0/1 | 0/160 | 2.417071 |
| random0_oracle vs delta_oracle | 0/94 | 0/48 | 0/330 | 2.65423325 |
| random1_full vs delta_full | 0/54 | 0/1 | 0/163 | 2.33771517 |
| random1_oracle vs delta_oracle | 0/92 | 0/48 | 0/328 | 2.58850115 |
| early_full vs delta_full | 0/54 | 0/1 | 0/119 | 2.29068549 |
| early_oracle vs delta_oracle | 0/96 | 0/48 | 0/208 | 2.55017081 |
| recent_full vs delta_full | 0/12 | 0/1 | 0/122 | 1.08010267 |
| recent_oracle vs delta_oracle | 0/48 | 0/48 | 0/288 | 1.39608492 |
| random0_oracle vs random0_full | 0/0 | 0/0 | 2/0 | -0.266576431 |
| random1_oracle vs random1_full | 4/0 | 0/0 | 7/0 | -0.252952701 |
| early_oracle vs early_full | 0/0 | 0/0 | 83/0 | -0.244253352 |
| recent_oracle vs recent_full | 6/0 | 0/0 | 6/0 | -0.18775643 |
| random0_full vs full | 2/13 | 0/1 | 2/247 | 2.02480176 |
| random0_oracle vs oracle | 1/43 | 0/13 | 1/272 | 2.16171123 |
| random1_full vs full | 0/13 | 0/1 | 0/248 | 1.94544593 |
| random1_oracle vs oracle | 4/44 | 0/13 | 6/275 | 2.09597913 |
| early_full vs full | 0/13 | 0/1 | 0/204 | 1.89841624 |
| early_oracle vs oracle | 0/44 | 0/13 | 0/149 | 2.0576488 |
| recent_full vs full | 33/4 | 0/1 | 33/240 | 0.687833425 |
| recent_oracle vs oracle | 17/13 | 0/13 | 17/246 | 0.903562903 |

### shared

| 候选 vs 参考 | 当前增益/退化 | 联合增益/退化 | 物理增益/退化 | 当前平均NLL差 |
|---|---|---|---|---|
| random0_full vs delta_full | 0/43 | 0/9 | 0/214 | 2.22667027 |
| random0_oracle vs delta_oracle | 0/94 | 0/48 | 0/330 | 2.4232991 |
| random1_full vs delta_full | 0/45 | 0/9 | 0/216 | 2.18712223 |
| random1_oracle vs delta_oracle | 0/92 | 0/48 | 0/328 | 2.37246625 |
| early_full vs delta_full | 0/45 | 0/9 | 0/136 | 2.25689997 |
| early_oracle vs delta_oracle | 0/96 | 0/48 | 0/208 | 2.28058308 |
| recent_full vs delta_full | 0/12 | 0/9 | 0/186 | 0.938478753 |
| recent_oracle vs delta_oracle | 0/48 | 0/48 | 0/288 | 1.27207113 |
| random0_oracle vs random0_full | 0/0 | 0/0 | 1/0 | -0.375060601 |
| random1_oracle vs random1_full | 4/0 | 0/0 | 5/0 | -0.386345409 |
| early_oracle vs early_full | 0/0 | 0/0 | 45/0 | -0.548006323 |
| recent_oracle vs recent_full | 15/0 | 0/0 | 15/0 | -0.238097053 |
| random0_full vs full | 2/45 | 0/14 | 2/278 | 2.01175732 |
| random0_oracle vs oracle | 0/94 | 0/48 | 0/330 | 2.39877499 |
| random1_full vs full | 0/45 | 0/14 | 0/278 | 1.97220928 |
| random1_oracle vs oracle | 0/92 | 0/48 | 0/328 | 2.34794214 |
| early_full vs full | 0/45 | 0/14 | 1/199 | 2.04198702 |
| early_oracle vs oracle | 0/96 | 0/48 | 0/208 | 2.25605897 |
| recent_full vs full | 14/26 | 0/14 | 14/262 | 0.723565798 |
| recent_oracle vs oracle | 0/48 | 0/48 | 0/288 | 1.24754702 |

### shared_question_trained_all

| 候选 vs 参考 | 当前增益/退化 | 联合增益/退化 | 物理增益/退化 | 当前平均NLL差 |
|---|---|---|---|---|
| random0_full vs delta_full | 0/46 | 0/4 | 0/191 | 2.33483104 |
| random0_oracle vs delta_oracle | 0/94 | 0/48 | 0/330 | 2.4193501 |
| random1_full vs delta_full | 0/48 | 0/4 | 0/191 | 2.27815043 |
| random1_oracle vs delta_oracle | 0/92 | 0/48 | 0/328 | 2.36244213 |
| early_full vs delta_full | 0/48 | 0/4 | 0/145 | 2.45703431 |
| early_oracle vs delta_oracle | 0/96 | 0/48 | 0/208 | 2.3395866 |
| recent_full vs delta_full | 0/14 | 0/4 | 0/161 | 1.06604256 |
| recent_oracle vs delta_oracle | 0/48 | 0/48 | 0/288 | 1.25777561 |
| random0_oracle vs random0_full | 0/0 | 0/0 | 2/0 | -0.400138846 |
| random1_oracle vs random1_full | 4/0 | 0/0 | 4/0 | -0.400366222 |
| early_oracle vs early_full | 0/0 | 0/0 | 78/0 | -0.602105618 |
| recent_oracle vs recent_full | 14/0 | 0/0 | 14/0 | -0.29292486 |
| random0_full vs full | 2/43 | 0/15 | 2/278 | 2.19468534 |
| random0_oracle vs oracle | 0/94 | 0/47 | 0/328 | 2.37657893 |
| random1_full vs full | 0/43 | 0/15 | 0/276 | 2.13800474 |
| random1_oracle vs oracle | 0/92 | 0/47 | 0/326 | 2.31967095 |
| early_full vs full | 0/43 | 0/15 | 0/230 | 2.31688861 |
| early_oracle vs oracle | 0/96 | 0/47 | 1/207 | 2.29681542 |
| recent_full vs full | 17/26 | 0/15 | 17/263 | 0.925896861 |
| recent_oracle vs oracle | 0/48 | 0/47 | 0/286 | 1.21500443 |

### shared_question_trained_question

| 候选 vs 参考 | 当前增益/退化 | 联合增益/退化 | 物理增益/退化 | 当前平均NLL差 |
|---|---|---|---|---|
| random0_full vs delta_full | 0/47 | 0/5 | 0/192 | 2.34100053 |
| random0_oracle vs delta_oracle | 0/94 | 0/48 | 0/330 | 2.43061864 |
| random1_full vs delta_full | 0/49 | 0/5 | 0/192 | 2.28408658 |
| random1_oracle vs delta_oracle | 0/92 | 0/48 | 0/328 | 2.38355729 |
| early_full vs delta_full | 0/49 | 0/5 | 0/147 | 2.43017174 |
| early_oracle vs delta_oracle | 0/96 | 0/48 | 0/208 | 2.32303143 |
| recent_full vs delta_full | 0/16 | 0/5 | 0/163 | 1.07933776 |
| recent_oracle vs delta_oracle | 0/48 | 0/48 | 0/288 | 1.27126808 |
| random0_oracle vs random0_full | 0/0 | 0/0 | 2/0 | -0.417269784 |
| random1_oracle vs random1_full | 4/0 | 0/0 | 4/0 | -0.407417182 |
| early_oracle vs early_full | 0/0 | 0/0 | 79/0 | -0.614028202 |
| recent_oracle vs recent_full | 15/0 | 0/0 | 15/0 | -0.314957573 |
| random0_full vs full | 2/39 | 0/15 | 2/277 | 2.11483199 |
| random0_oracle vs oracle | 0/94 | 0/46 | 0/328 | 2.3457189 |
| random1_full vs full | 0/39 | 0/15 | 0/275 | 2.05791804 |
| random1_oracle vs oracle | 0/92 | 0/46 | 0/326 | 2.29865755 |
| early_full vs full | 0/39 | 0/15 | 0/230 | 2.2040032 |
| early_oracle vs oracle | 0/96 | 0/46 | 0/206 | 2.23813169 |
| recent_full vs full | 19/25 | 0/15 | 19/265 | 0.853169221 |
| recent_oracle vs oracle | 0/48 | 0/46 | 0/286 | 1.18636834 |

### gdn_all_tokens

| 候选 vs 参考 | 当前增益/退化 | 联合增益/退化 | 物理增益/退化 | 当前平均NLL差 |
|---|---|---|---|---|
| random0_full vs delta_full | 0/63 | 0/9 | 0/225 | 2.20328322 |
| random0_oracle vs delta_oracle | 0/94 | 0/48 | 0/330 | 2.34655401 |
| random1_full vs delta_full | 0/63 | 0/9 | 0/223 | 2.15646174 |
| random1_oracle vs delta_oracle | 0/92 | 0/48 | 0/328 | 2.29132453 |
| early_full vs delta_full | 0/65 | 0/9 | 0/159 | 2.39767005 |
| early_oracle vs delta_oracle | 0/96 | 0/48 | 0/208 | 2.29706011 |
| recent_full vs delta_full | 0/20 | 0/9 | 0/184 | 0.943081099 |
| recent_oracle vs delta_oracle | 0/48 | 0/48 | 0/288 | 1.25493649 |
| random0_oracle vs random0_full | 0/0 | 0/0 | 2/0 | -0.286488646 |
| random1_oracle vs random1_full | 2/0 | 0/0 | 2/0 | -0.294896642 |
| early_oracle vs early_full | 0/0 | 0/0 | 58/0 | -0.530369371 |
| recent_oracle vs recent_full | 3/0 | 0/0 | 3/0 | -0.117904046 |
| random0_full vs full | 2/38 | 0/16 | 2/276 | 1.77980813 |
| random0_oracle vs oracle | 0/93 | 0/46 | 0/328 | 2.27359823 |
| random1_full vs full | 2/38 | 0/16 | 2/274 | 1.73298665 |
| random1_oracle vs oracle | 0/91 | 0/46 | 0/326 | 2.21836876 |
| early_full vs full | 0/38 | 0/16 | 0/208 | 1.97419496 |
| early_oracle vs oracle | 0/95 | 0/46 | 0/206 | 2.22410434 |
| recent_full vs full | 28/21 | 0/16 | 28/261 | 0.519606004 |
| recent_oracle vs oracle | 0/47 | 0/46 | 0/286 | 1.18198071 |

## 全部描述性损失交互

`L(f_oracle)-L(f_full)-L(delta_oracle)+L(delta_full)`；平均值仅作描述，没有新增显著性检验、科学成功阈值或架构有效声明。全部1680个物理问题向量及原始NLL另存于factorial.csv。

| 配置 | 写入族 | 336物理问题均值 | 96当前绑定均值 |
|---|---|---|---|
| native | random0 | 0.37909125 | 0.237162246 |
| native | random1 | 0.391041592 | 0.250785975 |
| native | early | 0.156288316 | 0.259485324 |
| native | recent | 0.335908469 | 0.315982247 |
| shared | random0 | 0.234903233 | 0.196628827 |
| shared | random1 | 0.262424829 | 0.185344018 |
| shared | early | -0.111600625 | 0.0236831049 |
| shared | recent | 0.291321361 | 0.333592375 |
| shared_question_trained_all | random0 | 0.0995708148 | 0.0845190674 |
| shared_question_trained_all | random1 | 0.12967442 | 0.0842916914 |
| shared_question_trained_all | early | -0.229461881 | -0.117447704 |
| shared_question_trained_all | recent | 0.167406178 | 0.191733053 |
| shared_question_trained_question | random0 | 0.0901705508 | 0.0896181062 |
| shared_question_trained_question | random1 | 0.128596427 | 0.0994707082 |
| shared_question_trained_question | early | -0.23388514 | -0.107140312 |
| shared_question_trained_question | recent | 0.169730765 | 0.191930317 |
| gdn_all_tokens | random0 | 0.173631359 | 0.143270789 |
| gdn_all_tokens | random1 | 0.180222803 | 0.134862793 |
| gdn_all_tokens | early | -0.0909075845 | -0.100609936 |
| gdn_all_tokens | recent | 0.205416838 | 0.311855389 |

## 完整性与边界

241项测试、两轮相反模型顺序的各720次smoke、240项完整前缀缓存检查通过。197份正式运行源码、实际18层GDN门控、六层Attention观测、原生权重和固定step432适配检查点均由原始审计核验。原理想FP32内核校验失败及其nativeBF16参考修正继续保留，未更改容差或原生内核。

随机策略与原支持写入偶然完全重合的题数为6和8，最早128、最近48，均保留；交叉两种读取和五配置时，必须精确重放1900项旧五字段读出。

类型和Token数相同不代表写入能量、全部位置或状态轨迹相同。Oracle读取固定为原支持记录；不同问题可使用不同预算，COPY选择已由预算揭示。本轮不能证明语义控制器学会依赖链、GDN自然损坏，或新混合架构有效。

跨写入策略固定的是输入Token、位置和Reader保留的记录身份，不是KV向量、Query或后续隐层表示。beta=0同时停止Delta擦除与插入，原生衰减继续执行；结果不能唯一定位到循环矩阵损坏或单独擦除项。

全部角色/顺序/深度/文体的逐项计数见完整审计；事件流、深度、文体和干扰负载的联合分层见strata.csv。所有100组比较、对应逐题退化和分层均保留，没有从结果选择模型或控制族。

