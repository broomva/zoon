# Candidate persons on training groups

This compares the six coded persons and their twins on 210 training groups before
registration.
The 143 held-out groups are absent from the file it reads.
The prompted arms are not run here.
The result concerns prediction and the five summaries, not what causes human play.
The settings used are the labelled lines at the top of the output;
`docs/interaction/design.md` defines the candidates, the selection rule and the measures.

## Selection

Among candidates, A, B, C, D, E tie at step 2; D has the highest score;
the rule chooses B because it has the fewest fitted numbers among those tied.
Among twins, B twin has the highest score, is alone at step 2, and is chosen by the rule.

## Scores and counts

| Form | Fitted numbers | Open-loop nats per decision | Closed-loop composite |
|---|---|---|---|
| A | 38 | -0.9330 [-0.9930, -0.8699] | +0.1191 [+0.1126, +0.1261] |
| B | 20 | -0.9300 [-0.9905, -0.8667] | +0.1187 [+0.1120, +0.1255] |
| C | 40 | -0.9328 [-0.9928, -0.8697] | +0.1191 [+0.1126, +0.1261] |
| D | 38 | -0.9292 [-0.9923, -0.8658] | +0.1195 [+0.1126, +0.1266] |
| E | 39 | -0.9329 [-0.9930, -0.8697] | +0.1185 [+0.1120, +0.1255] |
| F | 36 | -1.1904 [-1.2582, -1.1236] | +0.1590 [+0.1524, +0.1661] |
| A twin | 36 | -0.9597 [-1.0204, -0.8962] | +0.1322 [+0.1269, +0.1378] |
| B twin | 18 | -0.9553 [-1.0161, -0.8930] | +0.1311 [+0.1260, +0.1365] |
| C twin | 36 | -0.9597 [-1.0204, -0.8962] | +0.1322 [+0.1269, +0.1378] |
| D twin | 36 | -0.9597 [-1.0204, -0.8962] | +0.1322 [+0.1269, +0.1378] |
| E twin | 37 | -0.9599 [-1.0208, -0.8966] | +0.1322 [+0.1269, +0.1378] |
| F twin | 34 | -1.4687 [-1.5227, -1.4162] | +0.1891 [+0.1824, +0.1965] |

## Coded-arm gains

| Measure | Against selected twin | Against own twin |
|---|---|---|
| H2 | +0.0253 [+0.0179, +0.0328] | +0.0253 [+0.0179, +0.0328] |
| composite | +0.0124 [+0.0085, +0.0162] | +0.0124 [+0.0085, +0.0162] |
| level | +0.0051 [+0.0005, +0.0098] | +0.0051 [+0.0005, +0.0098] |
| members apart | +0.0466 [+0.0355, +0.0571] | +0.0466 [+0.0355, +0.0571] |
| trend | -0.0023 [-0.0045, -0.0001] | -0.0023 [-0.0045, -0.0001] |
| movement | +0.0012 [+0.0004, +0.0020] | +0.0012 [+0.0004, +0.0020] |
| type gap | +0.0114 [+0.0040, +0.0184] | +0.0114 [+0.0040, +0.0184] |

## Coverage

| Summary | Selected candidate | Selected twin | Own twin |
|---|---|---|---|
| level | 0.7333 | 0.6143 | 0.6143 |
| members apart | 0.6714 | 0.6667 | 0.6667 |
| trend | 0.8952 | 0.8619 | 0.8619 |
| movement | 0.5476 | 0.4429 | 0.4429 |
| type gap | 0.8075 | 0.9938 | 0.9938 |

## Power

| Measure | Selected twin, training effect | Selected twin, half effect | Own twin, training effect | Own twin, half effect |
|---|---|---|---|---|
| H2 | 1.000 | 0.782 | 1.000 | 0.782 |
| composite | 0.999 | 0.737 | 0.999 | 0.737 |
| level | 0.434 | 0.146 | 0.434 | 0.146 |
| members apart | 1.000 | 0.938 | 1.000 | 0.938 |
| trend | 0.395 | 0.135 | 0.395 | 0.135 |
| movement | 0.639 | 0.212 | 0.639 | 0.212 |
| type gap | 0.731 | 0.251 | 0.731 | 0.251 |

## A game left out

The game-left-out calculation is defined only for forms with an intercept.

| Arm | Nats per decision | Loss from leaving the game out |
|---|---|---|
| candidate | -0.9357 [-0.9981, -0.8714] | +0.0058 [+0.0018, +0.0098] |
| twin | -0.9586 [-1.0218, -0.8944] | +0.0033 [-0.0003, +0.0072] |

| Game | Candidate | Twin |
|---|---|---|
| 2-AI | -0.8812 | -0.8915 |
| 2-EI | -0.9194 | -0.9203 |
| 2-FE | -0.7243 | -0.7800 |
| 2-MI | -0.9696 | -0.9748 |
| 2-PI | -0.8604 | -0.9047 |
| 4-AI | -0.9518 | -0.9626 |
| 4-FE | -1.1331 | -1.1582 |
| 4-MI | -1.0728 | -1.1012 |

## Parameterisation

The first parameterisation fitted a cut origin beside each rule's constants.
A check of reproducibility showed that the two were redundant.
The first cut is now fixed at 0.
As a matter of procedure, this change was not made in response to a comparison result.

The largest absolute fitted-number changes over reversed groups and a nudged start
are 0.006276 for B and 0.004199 for B twin.

## What this means for the pre-registration

This settles the choice of the coded person and coded comparator for this comparison.
The labelled settings and amount table are the proposed registered choices.
The smallest differences, prompted arms, rendered instructions, recalibration, recall
check, spending bound, stopping rules and treatment of finite simulation error remain open.
No result here decides them.

## Reproduce

```sh
uv sync --extra groups
uv run --extra groups python -m zoon.interaction.data
uv run --extra groups python -m zoon.interaction.compare
```

```
zoon interaction: candidates on training groups
[data] training groups = 210
[data] decisions = 10240
[data] held-out groups = 143
[simulation] groups per game and fold = 100
[fit] ridge = 0.01
[fit] probability floor = 0.0000000010
[fit] log gap limit = 8.0
[fit] iteration limit = 2000
[fit] gradient tolerance = 0.00000010
[fit] function tolerance = 0.0000000001
[fit] reproducibility start nudge = 0.000001
[fit] optimiser = L-BFGS-B
[fit] gradient = analytic
[fit] B mean endowment = 24
[fit] line-search limit = 50
[fit] first cut = 0
[fit] constant, weight and log-gap start = 0
[fit] trace fraction start = 0.5
[fit] trace fraction bounds = 0, 1
[fit] other bounds = none
[fit] starts per fit = 1
[fit] stopping = either successful condition; failure raises
[simulation] seed = 20261120
[interval] resamples = 2000
[amounts, endowment 12, band 1] points:count = 1:73, 2:96, 3:77, 4:73
[amounts, endowment 12, band 2] points:count = 5:84, 6:171, 7:62, 8:153
[amounts, endowment 12, band 3] points:count = 9:99, 10:307, 11:86
[amounts, endowment 24, band 1] points:count = 1:71, 2:62, 3:69, 4:64, 5:107, 6:96, 7:37, 8:97
[amounts, endowment 24, band 2] points:count = 9:31, 10:233, 11:39, 12:272, 13:27, 14:38, 15:117, 16:47
[amounts, endowment 24, band 3] points:count = 17:20, 18:44, 19:20, 20:144, 21:18, 22:43, 23:40
[amounts, endowment 36, band 1] points:count = 1:76, 2:77, 3:56, 4:61, 5:66, 6:78, 7:37, 8:64, 9:42, 10:201, 11:19, 12:143
[amounts, endowment 36, band 2] points:count = 13:29, 14:27, 15:44, 16:38, 17:8, 18:70, 19:12, 20:157, 21:20, 22:21, 23:19, 24:71
[amounts, endowment 36, band 3] points:count = 25:39, 26:37, 27:26, 28:35, 29:8, 30:200, 31:9, 32:46, 33:12, 34:68, 35:39
[count] A = 38
[numbers, A] first constant 2-AI type 0 = +2.225187
[numbers, A] first constant 2-AI type 1 = +2.600094
[numbers, A] first constant 2-EI type 0 = +2.278260
[numbers, A] first constant 2-EI type 1 = +3.271519
[numbers, A] first constant 2-FE type 0 = +2.940388
[numbers, A] first constant 2-MI type 0 = +1.903734
[numbers, A] first constant 2-MI type 1 = +3.117496
[numbers, A] first constant 2-PI type 0 = +2.945633
[numbers, A] first constant 2-PI type 1 = +2.505683
[numbers, A] first constant 4-AI type 0 = +2.917562
[numbers, A] first constant 4-AI type 1 = +3.026033
[numbers, A] first constant 4-FE type 0 = +3.207432
[numbers, A] first constant 4-MI type 0 = +2.808929
[numbers, A] first constant 4-MI type 1 = +3.525203
[numbers, A] first log gap 1 = +0.758398
[numbers, A] first log gap 2 = +0.135634
[numbers, A] first log gap 3 = -1.115819
[numbers, A] later constant 2-AI type 0 = +3.467716
[numbers, A] later constant 2-AI type 1 = +4.002843
[numbers, A] later constant 2-EI type 0 = +3.074553
[numbers, A] later constant 2-EI type 1 = +4.060168
[numbers, A] later constant 2-FE type 0 = +3.869889
[numbers, A] later constant 2-MI type 0 = +2.997609
[numbers, A] later constant 2-MI type 1 = +3.948466
[numbers, A] later constant 2-PI type 0 = +3.604125
[numbers, A] later constant 2-PI type 1 = +3.755371
[numbers, A] later constant 4-AI type 0 = +3.500935
[numbers, A] later constant 4-AI type 1 = +3.645997
[numbers, A] later constant 4-FE type 0 = +3.741336
[numbers, A] later constant 4-MI type 0 = +3.189701
[numbers, A] later constant 4-MI type 1 = +3.652699
[numbers, A] later log gap 1 = +0.941484
[numbers, A] later log gap 2 = +0.805294
[numbers, A] later log gap 3 = +0.407276
[numbers, A] weight own = +7.019937
[numbers, A] weight others = +2.046825
[numbers, A] trace fraction own = +0.658659
[numbers, A] trace fraction others = +1.000000
[open loop] A = -0.9330 [-0.9930, -0.8699]
[closed loop, composite] A = +0.1191 [+0.1126, +0.1261]
[closed loop, level] A = +0.1723 [+0.1574, +0.1878]
[closed loop, members apart] A = +0.0984 [+0.0895, +0.1081]
[closed loop, trend] A = +0.1511 [+0.1355, +0.1684]
[closed loop, movement] A = +0.0552 [+0.0509, +0.0599]
[closed loop, type gap] A = +0.1187 [+0.1032, +0.1356]
[coverage, level] A = 0.7476
[coverage, members apart] A = 0.6905
[coverage, trend] A = 0.8571
[coverage, movement] A = 0.5476
[coverage, type gap] A = 0.7764
[count] B = 20
[numbers, B] first constant = +2.683038
[numbers, B] first endowment = -0.690416
[numbers, B] first return = +0.987392
[numbers, B] first product = -0.759102
[numbers, B] first four seats = +0.432596
[numbers, B] first log gap 1 = +0.758890
[numbers, B] first log gap 2 = +0.129728
[numbers, B] first log gap 3 = -1.117268
[numbers, B] later constant = +3.646176
[numbers, B] later endowment = -0.622400
[numbers, B] later return = +0.286668
[numbers, B] later product = +1.168089
[numbers, B] later four seats = -0.062715
[numbers, B] later log gap 1 = +0.936284
[numbers, B] later log gap 2 = +0.805321
[numbers, B] later log gap 3 = +0.405616
[numbers, B] weight own = +7.079080
[numbers, B] weight others = +1.940825
[numbers, B] trace fraction own = +0.658825
[numbers, B] trace fraction others = +1.000000
[open loop] B = -0.9300 [-0.9905, -0.8667]
[closed loop, composite] B = +0.1187 [+0.1120, +0.1255]
[closed loop, level] B = +0.1679 [+0.1551, +0.1816]
[closed loop, members apart] B = +0.1017 [+0.0926, +0.1115]
[closed loop, trend] B = +0.1488 [+0.1331, +0.1660]
[closed loop, movement] B = +0.0547 [+0.0504, +0.0594]
[closed loop, type gap] B = +0.1205 [+0.1045, +0.1384]
[coverage, level] B = 0.7333
[coverage, members apart] B = 0.6714
[coverage, trend] B = 0.8952
[coverage, movement] B = 0.5476
[coverage, type gap] B = 0.8075
[count] C = 40
[numbers, C] first constant 2-AI type 0 = +2.227235
[numbers, C] first constant 2-AI type 1 = +2.601268
[numbers, C] first constant 2-EI type 0 = +2.279831
[numbers, C] first constant 2-EI type 1 = +3.272960
[numbers, C] first constant 2-FE type 0 = +2.942232
[numbers, C] first constant 2-MI type 0 = +1.905826
[numbers, C] first constant 2-MI type 1 = +3.119060
[numbers, C] first constant 2-PI type 0 = +2.947352
[numbers, C] first constant 2-PI type 1 = +2.507082
[numbers, C] first constant 4-AI type 0 = +2.918474
[numbers, C] first constant 4-AI type 1 = +3.027571
[numbers, C] first constant 4-FE type 0 = +3.208787
[numbers, C] first constant 4-MI type 0 = +2.810365
[numbers, C] first constant 4-MI type 1 = +3.526466
[numbers, C] first log gap 1 = +0.759039
[numbers, C] first log gap 2 = +0.135719
[numbers, C] first log gap 3 = -1.115578
[numbers, C] later constant 2-AI type 0 = +3.468535
[numbers, C] later constant 2-AI type 1 = +4.003971
[numbers, C] later constant 2-EI type 0 = +3.074991
[numbers, C] later constant 2-EI type 1 = +4.061653
[numbers, C] later constant 2-FE type 0 = +3.870775
[numbers, C] later constant 2-MI type 0 = +2.998078
[numbers, C] later constant 2-MI type 1 = +3.949904
[numbers, C] later constant 2-PI type 0 = +3.605078
[numbers, C] later constant 2-PI type 1 = +3.756276
[numbers, C] later constant 4-AI type 0 = +3.525071
[numbers, C] later constant 4-AI type 1 = +3.619287
[numbers, C] later constant 4-FE type 0 = +3.742201
[numbers, C] later constant 4-MI type 0 = +3.218507
[numbers, C] later constant 4-MI type 1 = +3.629028
[numbers, C] later log gap 1 = +0.941364
[numbers, C] later log gap 2 = +0.806192
[numbers, C] later log gap 3 = +0.407564
[numbers, C] weight own = +7.020209
[numbers, C] weight same = +0.914232
[numbers, C] weight other = +1.133759
[numbers, C] trace fraction own = +0.658733
[numbers, C] trace fraction same = +1.000000
[numbers, C] trace fraction other = +1.000000
[open loop] C = -0.9328 [-0.9928, -0.8697]
[closed loop, composite] C = +0.1191 [+0.1126, +0.1261]
[closed loop, level] C = +0.1722 [+0.1575, +0.1878]
[closed loop, members apart] C = +0.0983 [+0.0895, +0.1081]
[closed loop, trend] C = +0.1507 [+0.1351, +0.1680]
[closed loop, movement] C = +0.0551 [+0.0509, +0.0599]
[closed loop, type gap] C = +0.1191 [+0.1036, +0.1360]
[coverage, level] C = 0.7429
[coverage, members apart] C = 0.6905
[coverage, trend] C = 0.8571
[coverage, movement] C = 0.5476
[coverage, type gap] C = 0.7764
[count] D = 38
[numbers, D] first constant 2-AI type 0 = +2.225523
[numbers, D] first constant 2-AI type 1 = +2.600536
[numbers, D] first constant 2-EI type 0 = +2.278907
[numbers, D] first constant 2-EI type 1 = +3.272164
[numbers, D] first constant 2-FE type 0 = +2.940747
[numbers, D] first constant 2-MI type 0 = +1.903873
[numbers, D] first constant 2-MI type 1 = +3.117887
[numbers, D] first constant 2-PI type 0 = +2.945998
[numbers, D] first constant 2-PI type 1 = +2.506486
[numbers, D] first constant 4-AI type 0 = +2.917516
[numbers, D] first constant 4-AI type 1 = +3.026504
[numbers, D] first constant 4-FE type 0 = +3.207170
[numbers, D] first constant 4-MI type 0 = +2.809619
[numbers, D] first constant 4-MI type 1 = +3.525844
[numbers, D] first log gap 1 = +0.758562
[numbers, D] first log gap 2 = +0.135682
[numbers, D] first log gap 3 = -1.115867
[numbers, D] later constant 2-AI type 0 = +4.579754
[numbers, D] later constant 2-AI type 1 = +3.368741
[numbers, D] later constant 2-EI type 0 = +4.245200
[numbers, D] later constant 2-EI type 1 = +3.222087
[numbers, D] later constant 2-FE type 0 = +3.853439
[numbers, D] later constant 2-MI type 0 = +4.124270
[numbers, D] later constant 2-MI type 1 = +3.343046
[numbers, D] later constant 2-PI type 0 = +3.620862
[numbers, D] later constant 2-PI type 1 = +3.772964
[numbers, D] later constant 4-AI type 0 = +4.259717
[numbers, D] later constant 4-AI type 1 = +3.101468
[numbers, D] later constant 4-FE type 0 = +3.732976
[numbers, D] later constant 4-MI type 0 = +3.973116
[numbers, D] later constant 4-MI type 1 = +3.101189
[numbers, D] later log gap 1 = +0.946756
[numbers, D] later log gap 2 = +0.818870
[numbers, D] later log gap 3 = +0.407957
[numbers, D] weight own = +7.044364
[numbers, D] weight others = +2.553511
[numbers, D] trace fraction own = +0.661770
[numbers, D] trace fraction others = +1.000000
[open loop] D = -0.9292 [-0.9923, -0.8658]
[closed loop, composite] D = +0.1195 [+0.1126, +0.1266]
[closed loop, level] D = +0.1720 [+0.1572, +0.1880]
[closed loop, members apart] D = +0.0977 [+0.0888, +0.1078]
[closed loop, trend] D = +0.1514 [+0.1356, +0.1692]
[closed loop, movement] D = +0.0581 [+0.0536, +0.0630]
[closed loop, type gap] D = +0.1180 [+0.1031, +0.1347]
[coverage, level] D = 0.7333
[coverage, members apart] D = 0.6667
[coverage, trend] D = 0.8762
[coverage, movement] D = 0.5048
[coverage, type gap] D = 0.8012
[count] E = 39
[numbers, E] first constant 2-AI type 0 = +2.225414
[numbers, E] first constant 2-AI type 1 = +2.599330
[numbers, E] first constant 2-EI type 0 = +2.277669
[numbers, E] first constant 2-EI type 1 = +3.271336
[numbers, E] first constant 2-FE type 0 = +2.940454
[numbers, E] first constant 2-MI type 0 = +1.904158
[numbers, E] first constant 2-MI type 1 = +3.117532
[numbers, E] first constant 2-PI type 0 = +2.945799
[numbers, E] first constant 2-PI type 1 = +2.504932
[numbers, E] first constant 4-AI type 0 = +2.916772
[numbers, E] first constant 4-AI type 1 = +3.025944
[numbers, E] first constant 4-FE type 0 = +3.206528
[numbers, E] first constant 4-MI type 0 = +2.808447
[numbers, E] first constant 4-MI type 1 = +3.525014
[numbers, E] first log gap 1 = +0.758240
[numbers, E] first log gap 2 = +0.135633
[numbers, E] first log gap 3 = -1.115927
[numbers, E] later constant 2-AI type 0 = +3.501390
[numbers, E] later constant 2-AI type 1 = +4.059313
[numbers, E] later constant 2-EI type 0 = +3.077970
[numbers, E] later constant 2-EI type 1 = +4.085979
[numbers, E] later constant 2-FE type 0 = +3.873251
[numbers, E] later constant 2-MI type 0 = +3.007805
[numbers, E] later constant 2-MI type 1 = +3.960596
[numbers, E] later constant 2-PI type 0 = +3.592930
[numbers, E] later constant 2-PI type 1 = +3.776547
[numbers, E] later constant 4-AI type 0 = +3.503799
[numbers, E] later constant 4-AI type 1 = +3.650406
[numbers, E] later constant 4-FE type 0 = +3.726768
[numbers, E] later constant 4-MI type 0 = +3.194589
[numbers, E] later constant 4-MI type 1 = +3.641200
[numbers, E] later log gap 1 = +0.941350
[numbers, E] later log gap 2 = +0.805569
[numbers, E] later log gap 3 = +0.409983
[numbers, E] weight own = +6.801897
[numbers, E] weight others = +2.108211
[numbers, E] weight anchor = +0.325060
[numbers, E] trace fraction own = +0.685259
[numbers, E] trace fraction others = +1.000000
[open loop] E = -0.9329 [-0.9930, -0.8697]
[closed loop, composite] E = +0.1185 [+0.1120, +0.1255]
[closed loop, level] E = +0.1713 [+0.1569, +0.1867]
[closed loop, members apart] E = +0.0980 [+0.0892, +0.1075]
[closed loop, trend] E = +0.1504 [+0.1345, +0.1681]
[closed loop, movement] E = +0.0546 [+0.0504, +0.0594]
[closed loop, type gap] E = +0.1182 [+0.1030, +0.1348]
[coverage, level] E = 0.7524
[coverage, members apart] E = 0.6905
[coverage, trend] E = 0.8476
[coverage, movement] E = 0.5667
[coverage, type gap] E = 0.7950
[count] F = 36
[numbers, F] first constant 2-AI type 0 = +2.224544
[numbers, F] first constant 2-AI type 1 = +2.600362
[numbers, F] first constant 2-EI type 0 = +2.277941
[numbers, F] first constant 2-EI type 1 = +3.271718
[numbers, F] first constant 2-FE type 0 = +2.940518
[numbers, F] first constant 2-MI type 0 = +1.902942
[numbers, F] first constant 2-MI type 1 = +3.117561
[numbers, F] first constant 2-PI type 0 = +2.945555
[numbers, F] first constant 2-PI type 1 = +2.505572
[numbers, F] first constant 4-AI type 0 = +2.918073
[numbers, F] first constant 4-AI type 1 = +3.026091
[numbers, F] first constant 4-FE type 0 = +3.207129
[numbers, F] first constant 4-MI type 0 = +2.808947
[numbers, F] first constant 4-MI type 1 = +3.525645
[numbers, F] first log gap 1 = +0.758388
[numbers, F] first log gap 2 = +0.135664
[numbers, F] first log gap 3 = -1.115567
[numbers, F] later constant 2-AI type 0 = +2.059834
[numbers, F] later constant 2-AI type 1 = +3.473768
[numbers, F] later constant 2-EI type 0 = +1.180781
[numbers, F] later constant 2-EI type 1 = +4.073324
[numbers, F] later constant 2-FE type 0 = +2.833009
[numbers, F] later constant 2-MI type 0 = +1.024030
[numbers, F] later constant 2-MI type 1 = +3.799604
[numbers, F] later constant 2-PI type 0 = +2.564992
[numbers, F] later constant 2-PI type 1 = +2.529743
[numbers, F] later constant 4-AI type 0 = +2.292901
[numbers, F] later constant 4-AI type 1 = +2.970781
[numbers, F] later constant 4-FE type 0 = +2.581727
[numbers, F] later constant 4-MI type 0 = +2.056403
[numbers, F] later constant 4-MI type 1 = +3.052077
[numbers, F] later log gap 1 = +0.628128
[numbers, F] later log gap 2 = +0.246197
[numbers, F] later log gap 3 = -0.115523
[numbers, F] weight others = +5.186466
[numbers, F] trace fraction others = +0.752608
[open loop] F = -1.1904 [-1.2582, -1.1236]
[closed loop, composite] F = +0.1590 [+0.1524, +0.1661]
[closed loop, level] F = +0.1875 [+0.1704, +0.2055]
[closed loop, members apart] F = +0.1412 [+0.1296, +0.1539]
[closed loop, trend] F = +0.1446 [+0.1300, +0.1606]
[closed loop, movement] F = +0.1825 [+0.1729, +0.1922]
[closed loop, type gap] F = +0.1391 [+0.1212, +0.1582]
[coverage, level] F = 0.4524
[coverage, members apart] F = 0.4238
[coverage, trend] F = 0.9000
[coverage, movement] F = 0.1905
[coverage, type gap] F = 0.3292
[count] A twin = 36
[numbers, A twin] first constant 2-AI type 0 = +2.224255
[numbers, A twin] first constant 2-AI type 1 = +2.600039
[numbers, A twin] first constant 2-EI type 0 = +2.277548
[numbers, A twin] first constant 2-EI type 1 = +3.271196
[numbers, A twin] first constant 2-FE type 0 = +2.940210
[numbers, A twin] first constant 2-MI type 0 = +1.902703
[numbers, A twin] first constant 2-MI type 1 = +3.117051
[numbers, A twin] first constant 2-PI type 0 = +2.945090
[numbers, A twin] first constant 2-PI type 1 = +2.505243
[numbers, A twin] first constant 4-AI type 0 = +2.917505
[numbers, A twin] first constant 4-AI type 1 = +3.025625
[numbers, A twin] first constant 4-FE type 0 = +3.207130
[numbers, A twin] first constant 4-MI type 0 = +2.808551
[numbers, A twin] first constant 4-MI type 1 = +3.525027
[numbers, A twin] first log gap 1 = +0.758201
[numbers, A twin] first log gap 2 = +0.135683
[numbers, A twin] first log gap 3 = -1.115922
[numbers, A twin] later constant 2-AI type 0 = +3.646241
[numbers, A twin] later constant 2-AI type 1 = +3.762148
[numbers, A twin] later constant 2-EI type 0 = +3.469705
[numbers, A twin] later constant 2-EI type 1 = +3.543430
[numbers, A twin] later constant 2-FE type 0 = +3.815666
[numbers, A twin] later constant 2-MI type 0 = +3.376446
[numbers, A twin] later constant 2-MI type 1 = +3.398804
[numbers, A twin] later constant 2-PI type 0 = +3.481586
[numbers, A twin] later constant 2-PI type 1 = +3.670722
[numbers, A twin] later constant 4-AI type 0 = +3.642795
[numbers, A twin] later constant 4-AI type 1 = +3.570969
[numbers, A twin] later constant 4-FE type 0 = +3.701355
[numbers, A twin] later constant 4-MI type 0 = +3.324210
[numbers, A twin] later constant 4-MI type 1 = +3.555576
[numbers, A twin] later log gap 1 = +0.885865
[numbers, A twin] later log gap 2 = +0.778342
[numbers, A twin] later log gap 3 = +0.380303
[numbers, A twin] weight own = +8.135944
[numbers, A twin] trace fraction own = +0.631019
[open loop] A twin = -0.9597 [-1.0204, -0.8962]
[closed loop, composite] A twin = +0.1322 [+0.1269, +0.1378]
[closed loop, level] A twin = +0.1751 [+0.1596, +0.1912]
[closed loop, members apart] A twin = +0.1438 [+0.1337, +0.1543]
[closed loop, trend] A twin = +0.1494 [+0.1330, +0.1677]
[closed loop, movement] A twin = +0.0564 [+0.0519, +0.0614]
[closed loop, type gap] A twin = +0.1362 [+0.1263, +0.1464]
[coverage, level] A twin = 0.6048
[coverage, members apart] A twin = 0.6952
[coverage, trend] A twin = 0.8143
[coverage, movement] A twin = 0.4524
[coverage, type gap] A twin = 1.0000
[count] B twin = 18
[numbers, B twin] first constant = +2.683730
[numbers, B twin] first endowment = -0.690312
[numbers, B twin] first return = +0.986683
[numbers, B twin] first product = -0.758062
[numbers, B twin] first four seats = +0.432390
[numbers, B twin] first log gap 1 = +0.759150
[numbers, B twin] first log gap 2 = +0.129725
[numbers, B twin] first log gap 3 = -1.117532
[numbers, B twin] later constant = +3.589777
[numbers, B twin] later endowment = -0.087788
[numbers, B twin] later return = -0.005112
[numbers, B twin] later product = +1.360966
[numbers, B twin] later four seats = -0.013597
[numbers, B twin] later log gap 1 = +0.881900
[numbers, B twin] later log gap 2 = +0.780900
[numbers, B twin] later log gap 3 = +0.380904
[numbers, B twin] weight own = +8.134181
[numbers, B twin] trace fraction own = +0.631896
[open loop] B twin = -0.9553 [-1.0161, -0.8930]
[closed loop, composite] B twin = +0.1311 [+0.1260, +0.1365]
[closed loop, level] B twin = +0.1730 [+0.1585, +0.1880]
[closed loop, members apart] B twin = +0.1484 [+0.1386, +0.1583]
[closed loop, trend] B twin = +0.1465 [+0.1300, +0.1640]
[closed loop, movement] B twin = +0.0559 [+0.0513, +0.0608]
[closed loop, type gap] B twin = +0.1319 [+0.1219, +0.1429]
[coverage, level] B twin = 0.6143
[coverage, members apart] B twin = 0.6667
[coverage, trend] B twin = 0.8619
[coverage, movement] B twin = 0.4429
[coverage, type gap] B twin = 0.9938
[count] C twin = 36
[numbers, C twin] first constant 2-AI type 0 = +2.224255
[numbers, C twin] first constant 2-AI type 1 = +2.600039
[numbers, C twin] first constant 2-EI type 0 = +2.277548
[numbers, C twin] first constant 2-EI type 1 = +3.271196
[numbers, C twin] first constant 2-FE type 0 = +2.940210
[numbers, C twin] first constant 2-MI type 0 = +1.902703
[numbers, C twin] first constant 2-MI type 1 = +3.117051
[numbers, C twin] first constant 2-PI type 0 = +2.945090
[numbers, C twin] first constant 2-PI type 1 = +2.505243
[numbers, C twin] first constant 4-AI type 0 = +2.917505
[numbers, C twin] first constant 4-AI type 1 = +3.025625
[numbers, C twin] first constant 4-FE type 0 = +3.207130
[numbers, C twin] first constant 4-MI type 0 = +2.808551
[numbers, C twin] first constant 4-MI type 1 = +3.525027
[numbers, C twin] first log gap 1 = +0.758201
[numbers, C twin] first log gap 2 = +0.135683
[numbers, C twin] first log gap 3 = -1.115922
[numbers, C twin] later constant 2-AI type 0 = +3.646241
[numbers, C twin] later constant 2-AI type 1 = +3.762148
[numbers, C twin] later constant 2-EI type 0 = +3.469705
[numbers, C twin] later constant 2-EI type 1 = +3.543430
[numbers, C twin] later constant 2-FE type 0 = +3.815666
[numbers, C twin] later constant 2-MI type 0 = +3.376446
[numbers, C twin] later constant 2-MI type 1 = +3.398804
[numbers, C twin] later constant 2-PI type 0 = +3.481586
[numbers, C twin] later constant 2-PI type 1 = +3.670722
[numbers, C twin] later constant 4-AI type 0 = +3.642795
[numbers, C twin] later constant 4-AI type 1 = +3.570969
[numbers, C twin] later constant 4-FE type 0 = +3.701355
[numbers, C twin] later constant 4-MI type 0 = +3.324210
[numbers, C twin] later constant 4-MI type 1 = +3.555576
[numbers, C twin] later log gap 1 = +0.885865
[numbers, C twin] later log gap 2 = +0.778342
[numbers, C twin] later log gap 3 = +0.380303
[numbers, C twin] weight own = +8.135944
[numbers, C twin] trace fraction own = +0.631019
[open loop] C twin = -0.9597 [-1.0204, -0.8962]
[closed loop, composite] C twin = +0.1322 [+0.1269, +0.1378]
[closed loop, level] C twin = +0.1751 [+0.1596, +0.1912]
[closed loop, members apart] C twin = +0.1438 [+0.1337, +0.1543]
[closed loop, trend] C twin = +0.1494 [+0.1330, +0.1677]
[closed loop, movement] C twin = +0.0564 [+0.0519, +0.0614]
[closed loop, type gap] C twin = +0.1362 [+0.1263, +0.1464]
[coverage, level] C twin = 0.6048
[coverage, members apart] C twin = 0.6952
[coverage, trend] C twin = 0.8143
[coverage, movement] C twin = 0.4524
[coverage, type gap] C twin = 1.0000
[count] D twin = 36
[numbers, D twin] first constant 2-AI type 0 = +2.224255
[numbers, D twin] first constant 2-AI type 1 = +2.600039
[numbers, D twin] first constant 2-EI type 0 = +2.277548
[numbers, D twin] first constant 2-EI type 1 = +3.271196
[numbers, D twin] first constant 2-FE type 0 = +2.940210
[numbers, D twin] first constant 2-MI type 0 = +1.902703
[numbers, D twin] first constant 2-MI type 1 = +3.117051
[numbers, D twin] first constant 2-PI type 0 = +2.945090
[numbers, D twin] first constant 2-PI type 1 = +2.505243
[numbers, D twin] first constant 4-AI type 0 = +2.917505
[numbers, D twin] first constant 4-AI type 1 = +3.025625
[numbers, D twin] first constant 4-FE type 0 = +3.207130
[numbers, D twin] first constant 4-MI type 0 = +2.808551
[numbers, D twin] first constant 4-MI type 1 = +3.525027
[numbers, D twin] first log gap 1 = +0.758201
[numbers, D twin] first log gap 2 = +0.135683
[numbers, D twin] first log gap 3 = -1.115922
[numbers, D twin] later constant 2-AI type 0 = +3.646241
[numbers, D twin] later constant 2-AI type 1 = +3.762148
[numbers, D twin] later constant 2-EI type 0 = +3.469705
[numbers, D twin] later constant 2-EI type 1 = +3.543430
[numbers, D twin] later constant 2-FE type 0 = +3.815666
[numbers, D twin] later constant 2-MI type 0 = +3.376446
[numbers, D twin] later constant 2-MI type 1 = +3.398804
[numbers, D twin] later constant 2-PI type 0 = +3.481586
[numbers, D twin] later constant 2-PI type 1 = +3.670722
[numbers, D twin] later constant 4-AI type 0 = +3.642795
[numbers, D twin] later constant 4-AI type 1 = +3.570969
[numbers, D twin] later constant 4-FE type 0 = +3.701355
[numbers, D twin] later constant 4-MI type 0 = +3.324210
[numbers, D twin] later constant 4-MI type 1 = +3.555576
[numbers, D twin] later log gap 1 = +0.885865
[numbers, D twin] later log gap 2 = +0.778342
[numbers, D twin] later log gap 3 = +0.380303
[numbers, D twin] weight own = +8.135944
[numbers, D twin] trace fraction own = +0.631019
[open loop] D twin = -0.9597 [-1.0204, -0.8962]
[closed loop, composite] D twin = +0.1322 [+0.1269, +0.1378]
[closed loop, level] D twin = +0.1751 [+0.1596, +0.1912]
[closed loop, members apart] D twin = +0.1438 [+0.1337, +0.1543]
[closed loop, trend] D twin = +0.1494 [+0.1330, +0.1677]
[closed loop, movement] D twin = +0.0564 [+0.0519, +0.0614]
[closed loop, type gap] D twin = +0.1362 [+0.1263, +0.1464]
[coverage, level] D twin = 0.6048
[coverage, members apart] D twin = 0.6952
[coverage, trend] D twin = 0.8143
[coverage, movement] D twin = 0.4524
[coverage, type gap] D twin = 1.0000
[count] E twin = 37
[numbers, E twin] first constant 2-AI type 0 = +2.226397
[numbers, E twin] first constant 2-AI type 1 = +2.600610
[numbers, E twin] first constant 2-EI type 0 = +2.279391
[numbers, E twin] first constant 2-EI type 1 = +3.274041
[numbers, E twin] first constant 2-FE type 0 = +2.941342
[numbers, E twin] first constant 2-MI type 0 = +1.904623
[numbers, E twin] first constant 2-MI type 1 = +3.119642
[numbers, E twin] first constant 2-PI type 0 = +2.947568
[numbers, E twin] first constant 2-PI type 1 = +2.506879
[numbers, E twin] first constant 4-AI type 0 = +2.917961
[numbers, E twin] first constant 4-AI type 1 = +3.027989
[numbers, E twin] first constant 4-FE type 0 = +3.208908
[numbers, E twin] first constant 4-MI type 0 = +2.810517
[numbers, E twin] first constant 4-MI type 1 = +3.528144
[numbers, E twin] first log gap 1 = +0.758979
[numbers, E twin] first log gap 2 = +0.135737
[numbers, E twin] first log gap 3 = -1.115540
[numbers, E twin] later constant 2-AI type 0 = +3.649628
[numbers, E twin] later constant 2-AI type 1 = +3.765750
[numbers, E twin] later constant 2-EI type 0 = +3.470976
[numbers, E twin] later constant 2-EI type 1 = +3.543817
[numbers, E twin] later constant 2-FE type 0 = +3.815798
[numbers, E twin] later constant 2-MI type 0 = +3.378426
[numbers, E twin] later constant 2-MI type 1 = +3.398515
[numbers, E twin] later constant 2-PI type 0 = +3.480319
[numbers, E twin] later constant 2-PI type 1 = +3.672276
[numbers, E twin] later constant 4-AI type 0 = +3.643382
[numbers, E twin] later constant 4-AI type 1 = +3.571351
[numbers, E twin] later constant 4-FE type 0 = +3.700116
[numbers, E twin] later constant 4-MI type 0 = +3.324956
[numbers, E twin] later constant 4-MI type 1 = +3.554132
[numbers, E twin] later log gap 1 = +0.885665
[numbers, E twin] later log gap 2 = +0.778290
[numbers, E twin] later log gap 3 = +0.380515
[numbers, E twin] weight own = +8.119241
[numbers, E twin] weight anchor = +0.028731
[numbers, E twin] trace fraction own = +0.632898
[open loop] E twin = -0.9599 [-1.0208, -0.8966]
[closed loop, composite] E twin = +0.1322 [+0.1269, +0.1378]
[closed loop, level] E twin = +0.1750 [+0.1596, +0.1911]
[closed loop, members apart] E twin = +0.1439 [+0.1337, +0.1544]
[closed loop, trend] E twin = +0.1495 [+0.1331, +0.1680]
[closed loop, movement] E twin = +0.0563 [+0.0518, +0.0614]
[closed loop, type gap] E twin = +0.1361 [+0.1263, +0.1463]
[coverage, level] E twin = 0.6095
[coverage, members apart] E twin = 0.7048
[coverage, trend] E twin = 0.8143
[coverage, movement] E twin = 0.4571
[coverage, type gap] E twin = 1.0000
[count] F twin = 34
[numbers, F twin] first constant 2-AI type 0 = +2.226028
[numbers, F twin] first constant 2-AI type 1 = +2.599738
[numbers, F twin] first constant 2-EI type 0 = +2.278428
[numbers, F twin] first constant 2-EI type 1 = +3.272497
[numbers, F twin] first constant 2-FE type 0 = +2.941383
[numbers, F twin] first constant 2-MI type 0 = +1.904622
[numbers, F twin] first constant 2-MI type 1 = +3.118487
[numbers, F twin] first constant 2-PI type 0 = +2.946588
[numbers, F twin] first constant 2-PI type 1 = +2.505660
[numbers, F twin] first constant 4-AI type 0 = +2.917101
[numbers, F twin] first constant 4-AI type 1 = +3.026772
[numbers, F twin] first constant 4-FE type 0 = +3.207140
[numbers, F twin] first constant 4-MI type 0 = +2.809274
[numbers, F twin] first constant 4-MI type 1 = +3.526341
[numbers, F twin] first log gap 1 = +0.758593
[numbers, F twin] first log gap 2 = +0.135698
[numbers, F twin] first log gap 3 = -1.115731
[numbers, F twin] later constant 2-AI type 0 = +2.187663
[numbers, F twin] later constant 2-AI type 1 = +2.729046
[numbers, F twin] later constant 2-EI type 0 = +1.648300
[numbers, F twin] later constant 2-EI type 1 = +2.751623
[numbers, F twin] later constant 2-FE type 0 = +2.665942
[numbers, F twin] later constant 2-MI type 0 = +1.304462
[numbers, F twin] later constant 2-MI type 1 = +2.326298
[numbers, F twin] later constant 2-PI type 0 = +2.037248
[numbers, F twin] later constant 2-PI type 1 = +2.057385
[numbers, F twin] later constant 4-AI type 0 = +2.441211
[numbers, F twin] later constant 4-AI type 1 = +2.850352
[numbers, F twin] later constant 4-FE type 0 = +2.243835
[numbers, F twin] later constant 4-MI type 0 = +2.288788
[numbers, F twin] later constant 4-MI type 1 = +2.851798
[numbers, F twin] later log gap 1 = +0.252638
[numbers, F twin] later log gap 2 = -0.233475
[numbers, F twin] later log gap 3 = -0.572120
[open loop] F twin = -1.4687 [-1.5227, -1.4162]
[closed loop, composite] F twin = +0.1891 [+0.1824, +0.1965]
[closed loop, level] F twin = +0.2226 [+0.2045, +0.2421]
[closed loop, members apart] F twin = +0.1882 [+0.1734, +0.2036]
[closed loop, trend] F twin = +0.1411 [+0.1226, +0.1603]
[closed loop, movement] F twin = +0.2681 [+0.2575, +0.2788]
[closed loop, type gap] F twin = +0.1252 [+0.1089, +0.1432]
[coverage, level] F twin = 0.1381
[coverage, members apart] F twin = 0.2190
[coverage, trend] F twin = 0.6905
[coverage, movement] F twin = 0.0143
[coverage, type gap] F twin = 0.5963
[selection, candidates] highest = D
[selection, candidates] tied at step 2 = A, B, C, D, E
[selection, candidates] winner = B
[selection, twins] highest = B twin
[selection, twins] tied at step 2 = B twin
[selection, twins] winner = B twin
[gain, selected twin, open loop] candidate minus twin = +0.0253 [+0.0179, +0.0328]
[gain, selected twin, closed loop, composite] twin minus candidate = +0.0124 [+0.0085, +0.0162]
[gain, selected twin, closed loop, level] twin minus candidate = +0.0051 [+0.0005, +0.0098]
[gain, selected twin, closed loop, members apart] twin minus candidate = +0.0466 [+0.0355, +0.0571]
[gain, selected twin, closed loop, trend] twin minus candidate = -0.0023 [-0.0045, -0.0001]
[gain, selected twin, closed loop, movement] twin minus candidate = +0.0012 [+0.0004, +0.0020]
[gain, selected twin, closed loop, type gap] twin minus candidate = +0.0114 [+0.0040, +0.0184]
[power, selected twin, H2] training effect = 1.000
[power, selected twin, H2] half effect = 0.782
[power, selected twin, composite] training effect = 0.999
[power, selected twin, composite] half effect = 0.737
[power, selected twin, level] training effect = 0.434
[power, selected twin, level] half effect = 0.146
[power, selected twin, members apart] training effect = 1.000
[power, selected twin, members apart] half effect = 0.938
[power, selected twin, trend] training effect = 0.395
[power, selected twin, trend] half effect = 0.135
[power, selected twin, movement] training effect = 0.639
[power, selected twin, movement] half effect = 0.212
[power, selected twin, type gap] training effect = 0.731
[power, selected twin, type gap] half effect = 0.251
[gain, own twin, open loop] candidate minus twin = +0.0253 [+0.0179, +0.0328]
[gain, own twin, closed loop, composite] twin minus candidate = +0.0124 [+0.0085, +0.0162]
[gain, own twin, closed loop, level] twin minus candidate = +0.0051 [+0.0005, +0.0098]
[gain, own twin, closed loop, members apart] twin minus candidate = +0.0466 [+0.0355, +0.0571]
[gain, own twin, closed loop, trend] twin minus candidate = -0.0023 [-0.0045, -0.0001]
[gain, own twin, closed loop, movement] twin minus candidate = +0.0012 [+0.0004, +0.0020]
[gain, own twin, closed loop, type gap] twin minus candidate = +0.0114 [+0.0040, +0.0184]
[power, own twin, H2] training effect = 1.000
[power, own twin, H2] half effect = 0.782
[power, own twin, composite] training effect = 0.999
[power, own twin, composite] half effect = 0.737
[power, own twin, level] training effect = 0.434
[power, own twin, level] half effect = 0.146
[power, own twin, members apart] training effect = 1.000
[power, own twin, members apart] half effect = 0.938
[power, own twin, trend] training effect = 0.395
[power, own twin, trend] half effect = 0.135
[power, own twin, movement] training effect = 0.639
[power, own twin, movement] half effect = 0.212
[power, own twin, type gap] training effect = 0.731
[power, own twin, type gap] half effect = 0.251
[reproducibility, B] largest absolute change = 0.006276
[game left out, candidate] score = -0.9357 [-0.9981, -0.8714]
[game left out, candidate] loss = +0.0058 [+0.0018, +0.0098]
[game left out, candidate, 2-AI] score = -0.8812
[game left out, candidate, 2-EI] score = -0.9194
[game left out, candidate, 2-FE] score = -0.7243
[game left out, candidate, 2-MI] score = -0.9696
[game left out, candidate, 2-PI] score = -0.8604
[game left out, candidate, 4-AI] score = -0.9518
[game left out, candidate, 4-FE] score = -1.1331
[game left out, candidate, 4-MI] score = -1.0728
[reproducibility, B twin] largest absolute change = 0.004199
[game left out, twin] score = -0.9586 [-1.0218, -0.8944]
[game left out, twin] loss = +0.0033 [-0.0003, +0.0072]
[game left out, twin, 2-AI] score = -0.8915
[game left out, twin, 2-EI] score = -0.9203
[game left out, twin, 2-FE] score = -0.7800
[game left out, twin, 2-MI] score = -0.9748
[game left out, twin, 2-PI] score = -0.9047
[game left out, twin, 4-AI] score = -0.9626
[game left out, twin, 4-FE] score = -1.1582
[game left out, twin, 4-MI] score = -1.1012
```
