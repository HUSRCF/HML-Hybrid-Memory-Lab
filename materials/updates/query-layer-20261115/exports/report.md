# 单层问题 Query 定位：完整结果

复用八条事件流、32 个尾部上下文、192 个问题。4,608 项新读出与 1,536 项已标明来源的历史 Full 参考共 6,144 项；运行重放不重复计入新主结果。保留所有基线失败。
原生及已有 GDN 适配器，六层分别单独替换，HN/NH 两方向，只使用 Full 供体。其余层使用当次真实原生 Query 对象，后续变化自然传播；答案及解码直通。
1,536 项历史参考精确重放、384 项单层同来源真实替换、768 次问题专用捕获及 4,608 个实际供体张量独立审计通过。299 项测试、相反模型顺序检查、230 份原封存源码及四份审计修正源码、固定参数通过核验。原审计器局部变量覆盖的失败及单独登记修正保留；模型计算与门槛未变。
登记 manifest SHA256 `ddc38548a1ac566aac0d44d35b25c108f21a0edc1e58ffd98e79a8fde7e22737`；正式 manifest SHA256 `942bb9f514e7a78ad4206ebaebd368776266218f11356beda33ba0e19ae05d2b`。

## 全部 32 条件

all6 是历史参考；NN/HH 为其未替换基线，HN/NH 为其六层同时交叉替换。Lxx 是本轮只替换 xx 层的读出。每行分母均为 192，三个角色各 64；新尾部证据分母 32，固定前缀证据分母 160。

| Model | Cell | Origin | Exact /192 | Current /64 | Initial /64 | Protected /64 | New /32 | Prefix /160 | Joint6 /32 | NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | NN.all6 | archived_reference | 159 | 46 | 49 | 64 | 27 | 132 | 16 | 0.343151 |
| native | NH.all6 | archived_reference | 152 | 41 | 47 | 64 | 14 | 138 | 14 | 0.345646 |
| native | HN.all6 | archived_reference | 161 | 50 | 47 | 64 | 26 | 135 | 19 | 0.300040 |
| native | HH.all6 | archived_reference | 151 | 40 | 47 | 64 | 12 | 139 | 14 | 0.323792 |
| native | HN.L03 | new_single_layer_query | 151 | 39 | 48 | 64 | 11 | 140 | 14 | 0.309429 |
| native | HN.L07 | new_single_layer_query | 150 | 39 | 47 | 64 | 11 | 139 | 13 | 0.327442 |
| native | HN.L11 | new_single_layer_query | 158 | 48 | 46 | 64 | 20 | 138 | 17 | 0.288393 |
| native | HN.L15 | new_single_layer_query | 156 | 44 | 48 | 64 | 19 | 137 | 15 | 0.307533 |
| native | HN.L19 | new_single_layer_query | 159 | 48 | 47 | 64 | 23 | 136 | 16 | 0.303620 |
| native | HN.L23 | new_single_layer_query | 150 | 39 | 47 | 64 | 11 | 139 | 13 | 0.326258 |
| native | NH.L03 | new_single_layer_query | 157 | 43 | 50 | 64 | 24 | 133 | 16 | 0.355010 |
| native | NH.L07 | new_single_layer_query | 159 | 45 | 50 | 64 | 25 | 134 | 17 | 0.331374 |
| native | NH.L11 | new_single_layer_query | 158 | 44 | 50 | 64 | 24 | 134 | 16 | 0.342637 |
| native | NH.L15 | new_single_layer_query | 156 | 45 | 47 | 64 | 21 | 135 | 15 | 0.343573 |
| native | NH.L19 | new_single_layer_query | 154 | 42 | 48 | 64 | 18 | 136 | 15 | 0.348518 |
| native | NH.L23 | new_single_layer_query | 157 | 43 | 50 | 64 | 26 | 131 | 16 | 0.343897 |
| gdn_all_tokens | NN.all6 | archived_reference | 191 | 63 | 64 | 64 | 32 | 159 | 31 | 0.007206 |
| gdn_all_tokens | NH.all6 | archived_reference | 163 | 35 | 64 | 64 | 3 | 160 | 17 | 0.230215 |
| gdn_all_tokens | HN.all6 | archived_reference | 191 | 63 | 64 | 64 | 32 | 159 | 31 | 0.008109 |
| gdn_all_tokens | HH.all6 | archived_reference | 162 | 34 | 64 | 64 | 2 | 160 | 17 | 0.255647 |
| gdn_all_tokens | HN.L03 | new_single_layer_query | 162 | 34 | 64 | 64 | 2 | 160 | 17 | 0.263122 |
| gdn_all_tokens | HN.L07 | new_single_layer_query | 163 | 35 | 64 | 64 | 3 | 160 | 17 | 0.257574 |
| gdn_all_tokens | HN.L11 | new_single_layer_query | 172 | 44 | 64 | 64 | 12 | 160 | 22 | 0.096730 |
| gdn_all_tokens | HN.L15 | new_single_layer_query | 191 | 63 | 64 | 64 | 32 | 159 | 31 | 0.015615 |
| gdn_all_tokens | HN.L19 | new_single_layer_query | 188 | 60 | 64 | 64 | 28 | 160 | 30 | 0.021753 |
| gdn_all_tokens | HN.L23 | new_single_layer_query | 162 | 34 | 64 | 64 | 2 | 160 | 17 | 0.309211 |
| gdn_all_tokens | NH.L03 | new_single_layer_query | 191 | 63 | 64 | 64 | 32 | 159 | 31 | 0.008528 |
| gdn_all_tokens | NH.L07 | new_single_layer_query | 190 | 62 | 64 | 64 | 32 | 158 | 30 | 0.009514 |
| gdn_all_tokens | NH.L11 | new_single_layer_query | 192 | 64 | 64 | 64 | 32 | 160 | 32 | 0.006385 |
| gdn_all_tokens | NH.L15 | new_single_layer_query | 186 | 58 | 64 | 64 | 26 | 160 | 28 | 0.033883 |
| gdn_all_tokens | NH.L19 | new_single_layer_query | 181 | 53 | 64 | 64 | 21 | 160 | 26 | 0.038838 |
| gdn_all_tokens | NH.L23 | new_single_layer_query | 191 | 63 | 64 | 64 | 32 | 159 | 31 | 0.008074 |

## 全部 128 个尾部条件

| Model | Cell | Episode | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint6 /8 | NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| native | NN.all6 | unrelated-8 | 12 | 11 | 16 | 39 | 3 | 0.324639 |
| native | NN.all6 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.295707 |
| native | NN.all6 | unrelated-64 | 7 | 14 | 16 | 37 | 2 | 0.388959 |
| native | NN.all6 | related-64 | 13 | 12 | 16 | 41 | 5 | 0.363300 |
| native | NH.all6 | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.282136 |
| native | NH.all6 | related-8 | 12 | 12 | 16 | 40 | 5 | 0.305030 |
| native | NH.all6 | unrelated-64 | 13 | 11 | 16 | 40 | 3 | 0.330847 |
| native | NH.all6 | related-64 | 2 | 12 | 16 | 30 | 1 | 0.464571 |
| native | HN.all6 | unrelated-8 | 14 | 11 | 16 | 41 | 4 | 0.271255 |
| native | HN.all6 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.266055 |
| native | HN.all6 | unrelated-64 | 10 | 12 | 16 | 38 | 4 | 0.323642 |
| native | HN.all6 | related-64 | 12 | 12 | 16 | 40 | 5 | 0.339209 |
| native | HH.all6 | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.245990 |
| native | HH.all6 | related-8 | 12 | 12 | 16 | 40 | 5 | 0.289890 |
| native | HH.all6 | unrelated-64 | 14 | 11 | 16 | 41 | 4 | 0.287155 |
| native | HH.all6 | related-64 | 0 | 12 | 16 | 28 | 0 | 0.472133 |
| native | HN.L03 | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.241744 |
| native | HN.L03 | related-8 | 11 | 12 | 16 | 39 | 4 | 0.285656 |
| native | HN.L03 | unrelated-64 | 14 | 12 | 16 | 42 | 5 | 0.260137 |
| native | HN.L03 | related-64 | 0 | 12 | 16 | 28 | 0 | 0.450179 |
| native | HN.L07 | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.249918 |
| native | HN.L07 | related-8 | 11 | 12 | 16 | 39 | 4 | 0.290505 |
| native | HN.L07 | unrelated-64 | 14 | 11 | 16 | 41 | 4 | 0.297966 |
| native | HN.L07 | related-64 | 0 | 12 | 16 | 28 | 0 | 0.471379 |
| native | HN.L11 | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.253698 |
| native | HN.L11 | related-8 | 12 | 12 | 16 | 40 | 5 | 0.282482 |
| native | HN.L11 | unrelated-64 | 14 | 11 | 16 | 41 | 4 | 0.272110 |
| native | HN.L11 | related-64 | 8 | 11 | 16 | 35 | 3 | 0.345284 |
| native | HN.L15 | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.255768 |
| native | HN.L15 | related-8 | 13 | 12 | 16 | 41 | 5 | 0.279077 |
| native | HN.L15 | unrelated-64 | 11 | 12 | 16 | 39 | 4 | 0.309192 |
| native | HN.L15 | related-64 | 6 | 12 | 16 | 34 | 1 | 0.386094 |
| native | HN.L19 | unrelated-8 | 14 | 11 | 16 | 41 | 4 | 0.254855 |
| native | HN.L19 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.267703 |
| native | HN.L19 | unrelated-64 | 11 | 12 | 16 | 39 | 4 | 0.308597 |
| native | HN.L19 | related-64 | 9 | 12 | 16 | 37 | 2 | 0.383326 |
| native | HN.L23 | unrelated-8 | 14 | 12 | 16 | 42 | 5 | 0.244185 |
| native | HN.L23 | related-8 | 11 | 12 | 16 | 39 | 4 | 0.291572 |
| native | HN.L23 | unrelated-64 | 14 | 11 | 16 | 41 | 4 | 0.288000 |
| native | HN.L23 | related-64 | 0 | 12 | 16 | 28 | 0 | 0.481276 |
| native | NH.L03 | unrelated-8 | 12 | 12 | 16 | 40 | 4 | 0.329475 |
| native | NH.L03 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.295956 |
| native | NH.L03 | unrelated-64 | 7 | 14 | 16 | 37 | 2 | 0.404294 |
| native | NH.L03 | related-64 | 10 | 12 | 16 | 38 | 4 | 0.390314 |
| native | NH.L07 | unrelated-8 | 13 | 12 | 16 | 41 | 4 | 0.310433 |
| native | NH.L07 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.287558 |
| native | NH.L07 | unrelated-64 | 7 | 14 | 16 | 37 | 2 | 0.362050 |
| native | NH.L07 | related-64 | 11 | 12 | 16 | 39 | 5 | 0.365456 |
| native | NH.L11 | unrelated-8 | 12 | 12 | 16 | 40 | 4 | 0.308942 |
| native | NH.L11 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.287393 |
| native | NH.L11 | unrelated-64 | 8 | 14 | 16 | 38 | 2 | 0.387027 |
| native | NH.L11 | related-64 | 10 | 12 | 16 | 38 | 4 | 0.387183 |
| native | NH.L15 | unrelated-8 | 14 | 11 | 16 | 41 | 4 | 0.310227 |
| native | NH.L15 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.292185 |
| native | NH.L15 | unrelated-64 | 10 | 12 | 16 | 38 | 2 | 0.368625 |
| native | NH.L15 | related-64 | 7 | 12 | 16 | 35 | 3 | 0.403257 |
| native | NH.L19 | unrelated-8 | 13 | 12 | 16 | 41 | 5 | 0.313825 |
| native | NH.L19 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.304646 |
| native | NH.L19 | unrelated-64 | 11 | 12 | 16 | 39 | 2 | 0.361991 |
| native | NH.L19 | related-64 | 4 | 12 | 16 | 32 | 2 | 0.413608 |
| native | NH.L23 | unrelated-8 | 11 | 12 | 16 | 39 | 4 | 0.330036 |
| native | NH.L23 | related-8 | 14 | 12 | 16 | 42 | 6 | 0.291457 |
| native | NH.L23 | unrelated-64 | 6 | 14 | 16 | 36 | 1 | 0.393548 |
| native | NH.L23 | related-64 | 12 | 12 | 16 | 40 | 5 | 0.360548 |
| gdn_all_tokens | NN.all6 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004112 |
| gdn_all_tokens | NN.all6 | related-8 | 16 | 16 | 16 | 48 | 8 | 0.004048 |
| gdn_all_tokens | NN.all6 | unrelated-64 | 15 | 16 | 16 | 47 | 7 | 0.019994 |
| gdn_all_tokens | NN.all6 | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000671 |
| gdn_all_tokens | NH.all6 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004220 |
| gdn_all_tokens | NH.all6 | related-8 | 3 | 16 | 16 | 35 | 1 | 0.200113 |
| gdn_all_tokens | NH.all6 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.006703 |
| gdn_all_tokens | NH.all6 | related-64 | 0 | 16 | 16 | 32 | 0 | 0.709824 |
| gdn_all_tokens | HN.all6 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.006125 |
| gdn_all_tokens | HN.all6 | related-8 | 16 | 16 | 16 | 48 | 8 | 0.006902 |
| gdn_all_tokens | HN.all6 | unrelated-64 | 15 | 16 | 16 | 47 | 7 | 0.018412 |
| gdn_all_tokens | HN.all6 | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000996 |
| gdn_all_tokens | HH.all6 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.005643 |
| gdn_all_tokens | HH.all6 | related-8 | 2 | 16 | 16 | 34 | 1 | 0.249895 |
| gdn_all_tokens | HH.all6 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.004433 |
| gdn_all_tokens | HH.all6 | related-64 | 0 | 16 | 16 | 32 | 0 | 0.762616 |
| gdn_all_tokens | HN.L03 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.006043 |
| gdn_all_tokens | HN.L03 | related-8 | 2 | 16 | 16 | 34 | 1 | 0.246425 |
| gdn_all_tokens | HN.L03 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.004003 |
| gdn_all_tokens | HN.L03 | related-64 | 0 | 16 | 16 | 32 | 0 | 0.796016 |
| gdn_all_tokens | HN.L07 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004702 |
| gdn_all_tokens | HN.L07 | related-8 | 3 | 16 | 16 | 35 | 1 | 0.251396 |
| gdn_all_tokens | HN.L07 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.003978 |
| gdn_all_tokens | HN.L07 | related-64 | 0 | 16 | 16 | 32 | 0 | 0.770220 |
| gdn_all_tokens | HN.L11 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.006963 |
| gdn_all_tokens | HN.L11 | related-8 | 4 | 16 | 16 | 36 | 2 | 0.170899 |
| gdn_all_tokens | HN.L11 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.009620 |
| gdn_all_tokens | HN.L11 | related-64 | 8 | 16 | 16 | 40 | 4 | 0.199439 |
| gdn_all_tokens | HN.L15 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004674 |
| gdn_all_tokens | HN.L15 | related-8 | 16 | 16 | 16 | 48 | 8 | 0.018345 |
| gdn_all_tokens | HN.L15 | unrelated-64 | 15 | 16 | 16 | 47 | 7 | 0.014553 |
| gdn_all_tokens | HN.L15 | related-64 | 16 | 16 | 16 | 48 | 8 | 0.024890 |
| gdn_all_tokens | HN.L19 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.005561 |
| gdn_all_tokens | HN.L19 | related-8 | 14 | 16 | 16 | 46 | 7 | 0.038425 |
| gdn_all_tokens | HN.L19 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.009537 |
| gdn_all_tokens | HN.L19 | related-64 | 14 | 16 | 16 | 46 | 7 | 0.033490 |
| gdn_all_tokens | HN.L23 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.005816 |
| gdn_all_tokens | HN.L23 | related-8 | 2 | 16 | 16 | 34 | 1 | 0.292256 |
| gdn_all_tokens | HN.L23 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.004434 |
| gdn_all_tokens | HN.L23 | related-64 | 0 | 16 | 16 | 32 | 0 | 0.934338 |
| gdn_all_tokens | NH.L03 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004730 |
| gdn_all_tokens | NH.L03 | related-8 | 16 | 16 | 16 | 48 | 8 | 0.004346 |
| gdn_all_tokens | NH.L03 | unrelated-64 | 15 | 16 | 16 | 47 | 7 | 0.024184 |
| gdn_all_tokens | NH.L03 | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000852 |
| gdn_all_tokens | NH.L07 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004697 |
| gdn_all_tokens | NH.L07 | related-8 | 16 | 16 | 16 | 48 | 8 | 0.003991 |
| gdn_all_tokens | NH.L07 | unrelated-64 | 14 | 16 | 16 | 46 | 6 | 0.028734 |
| gdn_all_tokens | NH.L07 | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000633 |
| gdn_all_tokens | NH.L11 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.003707 |
| gdn_all_tokens | NH.L11 | related-8 | 16 | 16 | 16 | 48 | 8 | 0.004002 |
| gdn_all_tokens | NH.L11 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.013604 |
| gdn_all_tokens | NH.L11 | related-64 | 16 | 16 | 16 | 48 | 8 | 0.004226 |
| gdn_all_tokens | NH.L15 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004635 |
| gdn_all_tokens | NH.L15 | related-8 | 15 | 16 | 16 | 47 | 7 | 0.041237 |
| gdn_all_tokens | NH.L15 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.006655 |
| gdn_all_tokens | NH.L15 | related-64 | 11 | 16 | 16 | 43 | 5 | 0.083007 |
| gdn_all_tokens | NH.L19 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004343 |
| gdn_all_tokens | NH.L19 | related-8 | 14 | 16 | 16 | 46 | 7 | 0.030246 |
| gdn_all_tokens | NH.L19 | unrelated-64 | 16 | 16 | 16 | 48 | 8 | 0.009819 |
| gdn_all_tokens | NH.L19 | related-64 | 7 | 16 | 16 | 39 | 3 | 0.110942 |
| gdn_all_tokens | NH.L23 | unrelated-8 | 16 | 16 | 16 | 48 | 8 | 0.004373 |
| gdn_all_tokens | NH.L23 | related-8 | 16 | 16 | 16 | 48 | 8 | 0.003848 |
| gdn_all_tokens | NH.L23 | unrelated-64 | 15 | 16 | 16 | 47 | 7 | 0.023417 |
| gdn_all_tokens | NH.L23 | related-64 | 16 | 16 | 16 | 48 | 8 | 0.000659 |

## 解释限制

全部配对增益、退化及 NLL 差见 contrasts.csv；全部问题、文体、角色和事件流见 primary.csv 与 strata.csv。历史参考及其新重放分开列出；它们不增加独立事件流数量。
question-calls.csv 记录 29,952 个问题阶段层调用，其中仅 4,992 个实际复制 Query，其余调用传入真实原生 Query 对象。供体张量只记录字节指纹与布局；重建实际激活需要记录的同环境和本地权重。
这是复用队列的定位，不是另一轮未见数据确认。单层成功不证明必要性或可加性，失败不证明该层无关；交叉激活可能分布外，供体包含上游 Attention 和递归计算。
没有中介百分比、显著性、创新性、架构有效性或扩模声明。所有旧失败判据保持原判，本轮没有新训练或环境。
