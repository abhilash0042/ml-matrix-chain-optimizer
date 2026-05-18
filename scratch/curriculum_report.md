# In-Depth Matrix Chain Multiplication Model Comparison

## Stage 1: Very Small (n=5)

### Test Case #1 (Uniform)

**Input Dimensions ($n=5$):**
`[230, 493, 121, 336, 97, 68]`

**Dynamic Programming Optimal Cost:** `16,508,672`

#### Predictions & MAPE:
- **🟢 GNN:** `16,747,788`
  - Error (MAPE): **1.4484%** 

- **🟡 Pointer Network:** `16,508,672`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `16,752,708`
  - Error (MAPE): **1.4782%** 

- **🌳 Random Forest:** `17,367,894`
  - Error (MAPE): **5.2047%** 

🏆 **Final Verdict:** The best performing model is **Pointer Network** (Error: 0.0000%)
---

### Test Case #2 (Spiky)

**Input Dimensions ($n=5$):**
`[21, 701, 45, 978, 20, 921]`

**Dynamic Programming Optimal Cost:** `1,948,365`

#### Predictions & MAPE:
- **🟢 GNN:** `1,948,365`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `1,948,365`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `1,528,044`
  - Error (MAPE): **21.5730%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `1,456,672`
  - Error (MAPE): **25.2362%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #3 (Bottleneck)

**Input Dimensions ($n=5$):**
`[794, 513, 3, 985, 750, 635]`

**Dynamic Programming Optimal Cost:** `6,379,536`

#### Predictions & MAPE:
- **🟢 GNN:** `6,379,536`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `6,379,536`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `6,134,432`
  - Error (MAPE): **3.8420%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `5,044,508`
  - Error (MAPE): **20.9267%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #4 (Monotone)

**Input Dimensions ($n=5$):**
`[29, 46, 63, 80, 97, 114]`

**Dynamic Programming Optimal Cost:** `775,924`

#### Predictions & MAPE:
- **🟢 GNN:** `775,924`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `775,924`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `711,311`
  - Error (MAPE): **8.3272%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `607,245`
  - Error (MAPE): **21.7391%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

## Stage 2: Small (n=10)

### Test Case #1 (Uniform)

**Input Dimensions ($n=10$):**
`[290, 357, 335, 61, 276, 496, 469, 438, 69, 93, 76]`

**Dynamic Programming Optimal Cost:** `52,677,645`

#### Predictions & MAPE:
- **🟢 GNN:** `52,677,645`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `52,692,654`
  - Error (MAPE): **0.0285%** 

- **🌲 XGBoost:** `44,748,508`
  - Error (MAPE): **15.0522%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `37,013,829`
  - Error (MAPE): **29.7352%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #2 (Spiky)

**Input Dimensions ($n=10$):**
`[25, 926, 31, 988, 24, 509, 20, 610, 28, 662, 27]`

**Dynamic Programming Optimal Cost:** `2,598,114`

#### Predictions & MAPE:
- **🟢 GNN:** `2,598,114`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `2,598,114`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `5,500,545`
  - Error (MAPE): **111.7130%** 

- **🌳 Random Forest:** `5,021,328`
  - Error (MAPE): **93.2682%** 

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #3 (Bottleneck)

**Input Dimensions ($n=10$):**
`[993, 932, 882, 899, 713, 655, 602, 1, 824, 928, 539]`

**Dynamic Programming Optimal Cost:** `5,842,821`

#### Predictions & MAPE:
- **🟢 GNN:** `5,842,821`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `5,842,821`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `4,790,725`
  - Error (MAPE): **18.0066%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `5,250,248`
  - Error (MAPE): **10.1419%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #4 (Monotone)

**Input Dimensions ($n=10$):**
`[87, 100, 113, 126, 139, 152, 165, 178, 191, 204, 217]`

**Dynamic Programming Optimal Cost:** `20,519,820`

#### Predictions & MAPE:
- **🟢 GNN:** `20,519,820`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `20,519,820`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `18,406,061`
  - Error (MAPE): **10.3011%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `15,606,864`
  - Error (MAPE): **23.9425%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

## Stage 3: Intermediate (n=20)

### Test Case #1 (Uniform)

**Input Dimensions ($n=20$):**
`[285, 92, 206, 407, 337, 223, 374, 135, 267, 145, 246, 417, 169, 481, 208, 141, 334, 42, 171, 89, 416]`

**Dynamic Programming Optimal Cost:** `49,865,046`

#### Predictions & MAPE:
- **🟢 GNN:** `49,865,046`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `49,865,046`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `44,568,124`
  - Error (MAPE): **10.6225%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `46,577,776`
  - Error (MAPE): **6.5923%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #2 (Spiky)

**Input Dimensions ($n=20$):**
`[26, 915, 7, 808, 48, 638, 34, 624, 31, 845, 41, 652, 33, 504, 46, 633, 35, 796, 41, 905, 10]`

**Dynamic Programming Optimal Cost:** `3,237,766`

#### Predictions & MAPE:
- **🟢 GNN:** `3,237,766`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `3,237,766`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `9,133,258`
  - Error (MAPE): **182.0852%** 

- **🌳 Random Forest:** `12,790,106`
  - Error (MAPE): **295.0287%** 

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #3 (Bottleneck)

**Input Dimensions ($n=20$):**
`[647, 906, 818, 580, 860, 651, 538, 908, 934, 973, 751, 863, 1, 659, 630, 560, 902, 878, 848, 842, 562]`

**Dynamic Programming Optimal Cost:** `11,195,246`

#### Predictions & MAPE:
- **🟢 GNN:** `11,195,246`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `11,195,246`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `11,169,258`
  - Error (MAPE): **0.2321%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `9,623,520`
  - Error (MAPE): **14.0392%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #4 (Monotone)

**Input Dimensions ($n=20$):**
`[93, 139, 185, 231, 277, 323, 369, 415, 461, 507, 553, 599, 645, 691, 737, 783, 829, 875, 921, 967, 1013]`

**Dynamic Programming Optimal Cost:** `697,482,609`

#### Predictions & MAPE:
- **🟢 GNN:** `697,482,609`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `697,482,609`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `666,230,030`
  - Error (MAPE): **4.4808%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `689,777,994`
  - Error (MAPE): **1.1046%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

## Stage 4: Advanced (n=35)

### Test Case #1 (Uniform)

**Input Dimensions ($n=35$):**
`[481, 192, 255, 280, 235, 383, 189, 280, 223, 311, 37, 214, 173, 84, 397, 405, 71, 474, 165, 378, 7, 158, 83, 112, 492, 398, 297, 214, 426, 307, 183, 121, 468, 231, 177, 133]`

**Dynamic Programming Optimal Cost:** `14,995,533`

#### Predictions & MAPE:
- **🟢 GNN:** `14,995,533`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `14,995,533`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `13,625,223`
  - Error (MAPE): **9.1381%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `15,412,750`
  - Error (MAPE): **2.7823%** 

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #2 (Spiky)

**Input Dimensions ($n=35$):**
`[20, 788, 48, 559, 30, 953, 18, 711, 23, 685, 21, 940, 6, 840, 17, 761, 22, 899, 7, 960, 9, 545, 48, 769, 48, 801, 25, 910, 13, 516, 15, 793, 39, 577, 45, 939]`

**Dynamic Programming Optimal Cost:** `3,984,038`

#### Predictions & MAPE:
- **🟢 GNN:** `3,984,038`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `3,984,038`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `15,079,036`
  - Error (MAPE): **278.4862%** 

- **🌳 Random Forest:** `12,346,669`
  - Error (MAPE): **209.9034%** 

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #3 (Bottleneck)

**Input Dimensions ($n=35$):**
`[968, 543, 968, 971, 914, 654, 796, 846, 946, 702, 848, 983, 828, 778, 668, 662, 803, 611, 553, 856, 823, 584, 572, 2, 621, 752, 804, 984, 936, 577, 806, 909, 885, 887, 671, 734]`

**Dynamic Programming Optimal Cost:** `42,491,182`

#### Predictions & MAPE:
- **🟢 GNN:** `42,491,182`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `42,491,182`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `37,461,929`
  - Error (MAPE): **11.8360%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `37,353,239`
  - Error (MAPE): **12.0918%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #4 (Monotone)

**Input Dimensions ($n=35$):**
`[812, 791, 770, 749, 728, 707, 686, 665, 644, 623, 602, 581, 560, 539, 518, 497, 476, 455, 434, 413, 392, 371, 350, 329, 308, 287, 266, 245, 224, 203, 182, 161, 140, 119, 98, 77]`

**Dynamic Programming Optimal Cost:** `652,827,098`

#### Predictions & MAPE:
- **🟢 GNN:** `652,827,098`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `652,827,098`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `660,803,416`
  - Error (MAPE): **1.2218%** 

- **🌳 Random Forest:** `647,165,263`
  - Error (MAPE): **0.8673%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

## Stage 5: Expert / OOD (n=50)

### Test Case #1 (Uniform)

**Input Dimensions ($n=50$):**
`[400, 422, 50, 78, 102, 113, 132, 230, 399, 455, 395, 375, 385, 310, 259, 173, 450, 145, 378, 45, 162, 444, 497, 383, 276, 107, 91, 425, 179, 292, 239, 335, 416, 225, 69, 68, 344, 133, 141, 248, 92, 138, 49, 345, 245, 288, 122, 33, 252, 86, 189]`

**Dynamic Programming Optimal Cost:** `104,099,094`

#### Predictions & MAPE:
- **🟢 GNN:** `104,099,094`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `104,099,094`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `83,976,989`
  - Error (MAPE): **19.3298%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `76,321,898`
  - Error (MAPE): **26.6834%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #2 (Spiky)

**Input Dimensions ($n=50$):**
`[5, 928, 35, 663, 47, 828, 17, 896, 42, 763, 46, 875, 12, 713, 48, 788, 7, 746, 41, 580, 17, 691, 17, 672, 25, 788, 23, 879, 30, 830, 13, 557, 42, 657, 11, 689, 25, 509, 37, 896, 39, 517, 27, 618, 24, 832, 42, 945, 22, 646, 21]`

**Dynamic Programming Optimal Cost:** `5,183,535`

#### Predictions & MAPE:
- **🟢 GNN:** `5,183,535`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `5,183,535`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `12,450,398`
  - Error (MAPE): **140.1913%** 

- **🌳 Random Forest:** `22,709,798`
  - Error (MAPE): **338.1141%** 

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #3 (Bottleneck)

**Input Dimensions ($n=50$):**
`[830, 512, 708, 885, 874, 610, 518, 827, 987, 820, 750, 3, 821, 512, 780, 532, 850, 821, 980, 812, 843, 688, 912, 972, 674, 740, 652, 815, 553, 694, 914, 543, 724, 597, 804, 777, 636, 744, 558, 568, 718, 792, 932, 996, 727, 604, 583, 915, 878, 545, 639]`

**Dynamic Programming Optimal Cost:** `81,250,896`

#### Predictions & MAPE:
- **🟢 GNN:** `81,250,896`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `81,250,896`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `71,738,681`
  - Error (MAPE): **11.7072%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `71,489,052`
  - Error (MAPE): **12.0144%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

### Test Case #4 (Monotone)

**Input Dimensions ($n=50$):**
`[2536, 2487, 2438, 2389, 2340, 2291, 2242, 2193, 2144, 2095, 2046, 1997, 1948, 1899, 1850, 1801, 1752, 1703, 1654, 1605, 1556, 1507, 1458, 1409, 1360, 1311, 1262, 1213, 1164, 1115, 1066, 1017, 968, 919, 870, 821, 772, 723, 674, 625, 576, 527, 478, 429, 380, 331, 282, 233, 184, 135, 86]`

**Dynamic Programming Optimal Cost:** `9,536,956,240`

#### Predictions & MAPE:
- **🟢 GNN:** `9,536,956,240`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🟡 Pointer Network:** `9,536,956,240`
  - Error (MAPE): **0.0000%** 🎯 PERFECT

- **🌲 XGBoost:** `6,683,352,177`
  - Error (MAPE): **29.9215%** ❌ INVALID (lower than DP cost)

- **🌳 Random Forest:** `5,684,954,905`
  - Error (MAPE): **40.3903%** ❌ INVALID (lower than DP cost)

🏆 **Final Verdict:** The best performing model is **GNN** (Error: 0.0000%)
---

