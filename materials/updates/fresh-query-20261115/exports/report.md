# 新事件流上的递归状态与问题Query来源：完整结果

Qwen3.5-2B IT及已有GDN适配器，8条新事件流、40个上下文、240个问题。全部3264项主读出：1536项尾部后未替换基线、1536项交叉Query干预和192项接尾部前基线。保留所有失败。
另1536项同来源真实替换重放与原五字段结果及读取概率完全一致；1536次问题专用捕获记录9216个实际Query张量。运行控制与主结果分开。
登记包SHA256：`78c42d760f1d0108e8a79adea427f8638bca5be88be3b8933574632ba1d82f60`。正式运行manifest SHA256：`a9059e6309feae40f69c72ccacbb544875bd73cd65d366027de6ff0c9e44ce17`。
284项测试、两轮相反模型顺序smoke、222份测量源码、16个前缀与128个尾部后缓存通过审计。状态N为正常尾部；H只恢复旧递归矩阵，卷积和全部KV保持正常尾部结果。
标签第一位是接收方状态，第二位是问题Query供体：NN/HH未替换，NH/HN交叉替换。只在已知问题后缀的一次forward替换6层Attention实际post-normalization/post-RoPE Query；后续gold评分与自由生成不再替换。
供体只使用已知问题文本；捕获产生的预测logits被丢弃，不做gold评分或生成。Full/Oracle供体严格分开，Oracle使用真实支持标注，仅作能力控制。

## 全部20条件

Before每行48次读出，接尾部后每行192次；分母显式列出。Joint6对应流×尾部，独立事件流仅8条。

| Model | Condition | Reader | Exact | Current | Initial | Protected | Joint3 | Joint6 | All /8 | NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | before | full | 38/48 | 14/16 | 8/16 | 16/16 | 6/16 | 3/8 | 3 | 0.252064 |
| native | before | oracle | 48/48 | 16/16 | 16/16 | 16/16 | 16/16 | 8/8 | 8 | 0.067179 |
| native | NN | full | 159/192 | 46/64 | 49/64 | 64/64 | 37/64 | 16/32 | 2 | 0.343151 |
| native | NN | oracle | 191/192 | 64/64 | 64/64 | 63/64 | 63/64 | 31/32 | 7 | 0.204843 |
| native | NH | full | 152/192 | 41/64 | 47/64 | 64/64 | 30/64 | 14/32 | 0 | 0.345646 |
| native | NH | oracle | 191/192 | 63/64 | 64/64 | 64/64 | 63/64 | 31/32 | 7 | 0.176783 |
| native | HN | full | 161/192 | 50/64 | 47/64 | 64/64 | 39/64 | 19/32 | 2 | 0.300040 |
| native | HN | oracle | 192/192 | 64/64 | 64/64 | 64/64 | 64/64 | 32/32 | 8 | 0.112724 |
| native | HH | full | 151/192 | 40/64 | 47/64 | 64/64 | 29/64 | 14/32 | 0 | 0.323792 |
| native | HH | oracle | 192/192 | 64/64 | 64/64 | 64/64 | 64/64 | 32/32 | 8 | 0.097294 |
| gdn_all_tokens | before | full | 48/48 | 16/16 | 16/16 | 16/16 | 16/16 | 8/8 | 8 | 0.017717 |
| gdn_all_tokens | before | oracle | 48/48 | 16/16 | 16/16 | 16/16 | 16/16 | 8/8 | 8 | 0.002502 |
| gdn_all_tokens | NN | full | 191/192 | 63/64 | 64/64 | 64/64 | 63/64 | 31/32 | 7 | 0.007206 |
| gdn_all_tokens | NN | oracle | 192/192 | 64/64 | 64/64 | 64/64 | 64/64 | 32/32 | 8 | 0.030103 |
| gdn_all_tokens | NH | full | 163/192 | 35/64 | 64/64 | 64/64 | 35/64 | 17/32 | 0 | 0.230215 |
| gdn_all_tokens | NH | oracle | 192/192 | 64/64 | 64/64 | 64/64 | 64/64 | 32/32 | 8 | 0.015371 |
| gdn_all_tokens | HN | full | 191/192 | 63/64 | 64/64 | 64/64 | 63/64 | 31/32 | 7 | 0.008109 |
| gdn_all_tokens | HN | oracle | 192/192 | 64/64 | 64/64 | 64/64 | 64/64 | 32/32 | 8 | 0.001675 |
| gdn_all_tokens | HH | full | 162/192 | 34/64 | 64/64 | 64/64 | 34/64 | 17/32 | 0 | 0.255647 |
| gdn_all_tokens | HH | oracle | 192/192 | 64/64 | 64/64 | 64/64 | 64/64 | 32/32 | 8 | 0.000955 |

## 全部68个阶段与尾部条件

| Model | Cell | Episode | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint6 /8 | NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| native | before.full | before | 14 | 8 | 16 | 38 | 3 | 0.252064 |
| native | before.oracle | before | 16 | 16 | 16 | 48 | 8 | 0.067179 |
| native | NN.full | unrelated-8 | 12 | 11 | 16 | 39 | 3 | 0.324639 |
| native | NN.full | related-8 | 14 | 12 | 16 | 42 | 6 | 0.295707 |
| native | NN.full | unrelated-64 | 7 | 14 | 16 | 37 | 2 | 0.388959 |
| native | NN.full | related-64 | 13 | 12 | 16 | 41 | 5 | 0.363300 |
| native | NN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.163656 |
| native | NN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.169913 |
| native | NN.oracle | unrelated-64 | 16 | 16 | 15 | 47 | 7 | 0.233809 |
| native | NN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.251994 |
| native | NH.full | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.282136 |
| native | NH.full | related-8 | 12 | 12 | 16 | 40 | 5 | 0.305030 |
| native | NH.full | unrelated-64 | 13 | 11 | 16 | 40 | 3 | 0.330847 |
| native | NH.full | related-64 | 2 | 12 | 16 | 30 | 1 | 0.464571 |
| native | NH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.120046 |
| native | NH.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.149520 |
| native | NH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.193975 |
| native | NH.oracle | related-64 | 15 | 16 | 16 | 47 | 7 | 0.243589 |
| native | HN.full | unrelated-8 | 14 | 11 | 16 | 41 | 4 | 0.271255 |
| native | HN.full | related-8 | 14 | 12 | 16 | 42 | 6 | 0.266055 |
| native | HN.full | unrelated-64 | 10 | 12 | 16 | 38 | 4 | 0.323642 |
| native | HN.full | related-64 | 12 | 12 | 16 | 40 | 5 | 0.339209 |
| native | HN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.096559 |
| native | HN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.112879 |
| native | HN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.103433 |
| native | HN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.138023 |
| native | HH.full | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.245990 |
| native | HH.full | related-8 | 12 | 12 | 16 | 40 | 5 | 0.289890 |
| native | HH.full | unrelated-64 | 14 | 11 | 16 | 41 | 4 | 0.287155 |
| native | HH.full | related-64 | 0 | 12 | 16 | 28 | 0 | 0.472133 |
| native | HH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.067045 |
| native | HH.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.095253 |
| native | HH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.085670 |
| native | HH.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.141207 |
| gdn_all_tokens | before.full | before | 16 | 16 | 16 | 48 | 8 | 0.017717 |
| gdn_all_tokens | before.oracle | before | 16 | 16 | 16 | 48 | 8 | 0.002502 |
| gdn_all_tokens | NN.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004112 |
| gdn_all_tokens | NN.full | related-8 | 16 | 16 | 16 | 48 | 8 | 0.004048 |
| gdn_all_tokens | NN.full | unrelated-64 | 15 | 16 | 16 | 47 | 7 | 0.019994 |
| gdn_all_tokens | NN.full | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000671 |
| gdn_all_tokens | NN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.014396 |
| gdn_all_tokens | NN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.013051 |
| gdn_all_tokens | NN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.057187 |
| gdn_all_tokens | NN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.035778 |
| gdn_all_tokens | NH.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004220 |
| gdn_all_tokens | NH.full | related-8 | 3 | 16 | 16 | 35 | 1 | 0.200113 |
| gdn_all_tokens | NH.full | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.006703 |
| gdn_all_tokens | NH.full | related-64 | 0 | 16 | 16 | 32 | 0 | 0.709824 |
| gdn_all_tokens | NH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.008204 |
| gdn_all_tokens | NH.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.008199 |
| gdn_all_tokens | NH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.026228 |
| gdn_all_tokens | NH.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.018853 |
| gdn_all_tokens | HN.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.006125 |
| gdn_all_tokens | HN.full | related-8 | 16 | 16 | 16 | 48 | 8 | 0.006902 |
| gdn_all_tokens | HN.full | unrelated-64 | 15 | 16 | 16 | 47 | 7 | 0.018412 |
| gdn_all_tokens | HN.full | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000996 |
| gdn_all_tokens | HN.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.001614 |
| gdn_all_tokens | HN.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.001302 |
| gdn_all_tokens | HN.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.002522 |
| gdn_all_tokens | HN.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.001263 |
| gdn_all_tokens | HH.full | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.005643 |
| gdn_all_tokens | HH.full | related-8 | 2 | 16 | 16 | 34 | 1 | 0.249895 |
| gdn_all_tokens | HH.full | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.004433 |
| gdn_all_tokens | HH.full | related-64 | 0 | 16 | 16 | 32 | 0 | 0.762616 |
| gdn_all_tokens | HH.oracle | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.000620 |
| gdn_all_tokens | HH.oracle | related-8 | 16 | 16 | 16 | 48 | 8 | 0.000978 |
| gdn_all_tokens | HH.oracle | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.001128 |
| gdn_all_tokens | HH.oracle | related-64 | 16 | 16 | 16 | 48 | 8 | 0.001095 |

## 解释与恢复限制

全部角色、文体、流和证据来源见strata.csv。contrasts.csv保留Query修复、反向破坏、状态变化和Oracle读取的全部配对增益/退化；before-after.csv单列固定前缀证据，相关更新的当前答案变化不能当成同一事实遗忘。
donor-queries.csv列出9216个供体张量的实际字节指纹、布局与来源，replacement-calls.csv列出18432个实际替换层调用及解码直通计数。cache-fields.csv记录6912个源缓存张量。完整张量不另存副本，复现需相同本地权重与后端重跑。
Query包含上游递归与Attention计算，交叉激活可能偏离自然分布。恢复不唯一证明语义地址构造，失败也不证明Query无关；没有中介百分比、显著性、创新性或新架构收益声明。
所有旧失败门槛不变；本轮不训练、不扩模、不下载新权重、不创建环境。
