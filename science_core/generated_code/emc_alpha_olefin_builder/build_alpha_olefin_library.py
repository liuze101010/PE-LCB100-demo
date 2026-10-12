#!/usr/bin/env python
"""Generate literature-grounded EMC setup files for isotactic alpha-olefins.

This builder writes EMC `.esh` inputs plus a manifest for three use cases:
1. A literature-grounded first-round screening preset.
2. A broader side-chain library for later expansion.
3. A 30-case LCB screening library on an iPP backbone.

The current project constraint is "150 chains/box" with a target of roughly
"100 total carbons/chain". Because repeat-unit carbon counts vary with side
chain length, exact equality is not possible for every alpha-olefin. This
script therefore records both the target and the actually realized total
carbons for each generated case.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ALKENE_NAMES = {
    3: "propylene",
    4: "1-butene",
    5: "1-pentene",
    6: "1-hexene",
    7: "1-heptene",
    8: "1-octene",
    9: "1-nonene",
    10: "1-decene",
    11: "1-undecene",
    12: "1-dodecene",
    13: "1-tridecene",
    14: "1-tetradecene",
    15: "1-pentadecene",
    16: "1-hexadecene",
    17: "1-heptadecene",
    18: "1-octadecene",
    19: "1-nonadecene",
    20: "1-eicosene",
    21: "1-heneicosene",
    22: "1-docosene",
}


# 30-case LCB library on propylene (C3) backbone.
# Columns: seq, branch_points (b), branch_carbons (L), topology, position_pattern, regime_tag.
# Macro SMILES encodes the whole chain as a single EMC monomer with n=1.
# Orthogonal coverage: b in {1,2,3}; L in {5,8,12,16,20}; 7 topology/position variants.
LCB_LEGACY30_SPEC: list[tuple[int, int, int, str, str, str]] = [
    (1, 1, 5, "three-arm-star", "mid", "sub-entangled-LCB"),
    (2, 1, 8, "three-arm-star", "mid", "sub-entangled-LCB"),
    (3, 1, 12, "three-arm-star", "mid", "sub-entangled-LCB"),
    (4, 1, 16, "three-arm-star", "mid", "sub-entangled-LCB"),
    (5, 1, 20, "three-arm-star", "mid", "sub-entangled-LCB"),
    (6, 1, 8, "center-branched", "near-end", "sub-entangled-LCB"),
    (7, 1, 12, "center-branched", "near-end", "sub-entangled-LCB"),
    (8, 1, 16, "center-branched", "near-end", "sub-entangled-LCB"),
    (9, 2, 5, "random-comb", "uniform", "sub-entangled-LCB"),
    (10, 2, 8, "random-comb", "uniform", "sub-entangled-LCB"),
    (11, 2, 12, "random-comb", "uniform", "sub-entangled-LCB"),
    (12, 2, 16, "random-comb", "uniform", "sub-entangled-LCB"),
    (13, 2, 20, "random-comb", "uniform", "sub-entangled-LCB"),
    (14, 2, 8, "H-type", "two-ends", "sub-entangled-LCB"),
    (15, 2, 12, "H-type", "two-ends", "sub-entangled-LCB"),
    (16, 2, 16, "H-type", "two-ends", "sub-entangled-LCB"),
    (17, 2, 20, "H-type", "two-ends", "sub-entangled-LCB"),
    (18, 2, 8, "dumbbell-mid", "mid-pair", "sub-entangled-LCB"),
    (19, 2, 12, "dumbbell-mid", "mid-pair", "sub-entangled-LCB"),
    (20, 3, 5, "random-comb", "uniform", "sub-entangled-LCB"),
    (21, 3, 8, "random-comb", "uniform", "sub-entangled-LCB"),
    (22, 3, 12, "random-comb", "uniform", "sub-entangled-LCB"),
    (23, 3, 16, "random-comb", "uniform", "sub-entangled-LCB"),
    (24, 3, 8, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (25, 3, 12, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (26, 3, 16, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (27, 3, 12, "asymmetric-comb", "end-cluster", "sub-entangled-LCB"),
    (28, 1, 20, "center-branched", "near-end", "sub-entangled-LCB"),
    (29, 2, 20, "dumbbell-mid", "mid-pair", "sub-entangled-LCB"),
    (30, 3, 20, "pom-pom", "two-ends+mid", "oligomer-LCB"),
]

LCB70_EXTRA_SPEC: list[tuple[int, int, int, str, str, str]] = [
    # Group A: b=1 补全 (6 cases, lcb31-lcb36)
    (31, 1, 3, "three-arm-star", "mid", "sub-entangled-LCB"),
    (32, 1, 4, "three-arm-star", "mid", "sub-entangled-LCB"),
    (33, 1, 5, "three-arm-star", "far-end", "sub-entangled-LCB"),
    (34, 1, 5, "center-branched", "near-start", "sub-entangled-LCB"),
    (35, 1, 12, "center-branched", "near-start", "sub-entangled-LCB"),
    (36, 1, 3, "center-branched", "near-end", "sub-entangled-LCB"),

    # Group B: b=2 补全+新分布 (16 cases, lcb37-lcb52)
    (37, 2, 3, "random-comb", "uniform", "sub-entangled-LCB"),
    (38, 2, 4, "random-comb", "uniform", "sub-entangled-LCB"),
    (39, 2, 5, "H-type", "H-asymmetric", "sub-entangled-LCB"),
    (40, 2, 8, "H-type", "H-asymmetric", "sub-entangled-LCB"),
    (41, 2, 12, "H-type", "H-asymmetric", "sub-entangled-LCB"),
    (42, 2, 16, "H-type", "H-asymmetric", "sub-entangled-LCB"),
    (43, 2, 5, "random-comb", "clustered", "sub-entangled-LCB"),
    (44, 2, 8, "random-comb", "clustered", "sub-entangled-LCB"),
    (45, 2, 12, "random-comb", "clustered", "sub-entangled-LCB"),
    (46, 2, 16, "random-comb", "clustered", "sub-entangled-LCB"),
    (47, 2, 5, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (48, 2, 8, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (49, 2, 12, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (50, 2, 16, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (51, 2, 3, "H-type", "two-ends", "sub-entangled-LCB"),
    (52, 2, 4, "H-type", "two-ends", "sub-entangled-LCB"),

    # Group C: b=3 补全 (12 cases, lcb53-lcb64)
    (53, 3, 3, "random-comb", "uniform", "sub-entangled-LCB"),
    (54, 3, 4, "random-comb", "uniform", "sub-entangled-LCB"),
    (55, 3, 5, "pom-pom", "pompom-dense", "sub-entangled-LCB"),
    (56, 3, 3, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (57, 3, 4, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (58, 3, 5, "random-comb", "clustered", "sub-entangled-LCB"),
    (59, 3, 8, "random-comb", "clustered", "sub-entangled-LCB"),
    (60, 3, 12, "random-comb", "clustered", "sub-entangled-LCB"),
    (61, 3, 5, "three-point", "three-point", "sub-entangled-LCB"),
    (62, 3, 8, "three-point", "three-point", "sub-entangled-LCB"),
    (63, 3, 12, "three-point", "three-point", "sub-entangled-LCB"),
    (64, 3, 3, "asymmetric-comb", "end-cluster", "sub-entangled-LCB"),

    # Group D: b=4 新系列 (16 cases, lcb65-lcb80)
    (65, 4, 3, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (66, 4, 5, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (67, 4, 8, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (68, 4, 12, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (69, 4, 3, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (70, 4, 5, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (71, 4, 8, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (72, 4, 12, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (73, 4, 3, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (74, 4, 5, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (75, 4, 8, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (76, 4, 12, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (77, 4, 5, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (78, 4, 8, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (79, 4, 12, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (80, 4, 3, "quad-gradient", "gradient", "sub-entangled-LCB"),

    # Group E: b=5 超支化系列 (10 cases, lcb81-lcb90)
    (81, 5, 3, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),
    (82, 5, 5, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),
    (83, 5, 8, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),
    (84, 5, 3, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (85, 5, 5, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (86, 5, 8, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (87, 5, 3, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (88, 5, 5, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (89, 5, 8, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (90, 5, 3, "penta-gradient", "gradient-5", "sub-entangled-LCB"),

    # Group F: 压力测试+极端点 (10 cases, lcb91-lcb100)
    (91, 4, 16, "quad-comb", "comb-uniform-4", "oligomer-LCB"),
    (92, 4, 20, "quad-comb", "comb-uniform-4", "oligomer-LCB"),
    (93, 5, 12, "penta-comb", "comb-uniform-5", "oligomer-LCB"),
    (94, 5, 5, "penta-gradient", "gradient-5", "sub-entangled-LCB"),
    (95, 1, 20, "three-arm-star", "far-end", "sub-entangled-LCB"),
    (96, 2, 20, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (97, 3, 20, "three-point", "three-point", "oligomer-LCB"),
    (98, 3, 16, "three-point", "three-point", "sub-entangled-LCB"),
    (99, 4, 16, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (100, 5, 8, "penta-gradient", "gradient-5", "sub-entangled-LCB"),
]

LCB100_SPEC = LCB_LEGACY30_SPEC + LCB70_EXTRA_SPEC

LCB200_EXTRA_SPEC: list[tuple[int, int, int, str, str, str]] = [
    # Group G: b=1 interpolation and single-branch distribution variants (8 cases, lcb101-lcb108)
    (101, 1, 6, "three-arm-star", "mid", "sub-entangled-LCB"),
    (102, 1, 10, "three-arm-star", "mid", "sub-entangled-LCB"),
    (103, 1, 14, "three-arm-star", "mid", "sub-entangled-LCB"),
    (104, 1, 18, "three-arm-star", "mid", "sub-entangled-LCB"),
    (105, 1, 6, "center-branched", "near-start", "sub-entangled-LCB"),
    (106, 1, 10, "center-branched", "near-start", "sub-entangled-LCB"),
    (107, 1, 14, "center-branched", "near-end", "sub-entangled-LCB"),
    (108, 1, 18, "center-branched", "far-end", "sub-entangled-LCB"),

    # Group H: b=2 interpolation across density and distribution axes (16 cases, lcb109-lcb124)
    (109, 2, 6, "random-comb", "uniform", "sub-entangled-LCB"),
    (110, 2, 10, "random-comb", "uniform", "sub-entangled-LCB"),
    (111, 2, 14, "random-comb", "uniform", "sub-entangled-LCB"),
    (112, 2, 18, "random-comb", "uniform", "sub-entangled-LCB"),
    (113, 2, 6, "H-type", "two-ends", "sub-entangled-LCB"),
    (114, 2, 10, "H-type", "two-ends", "sub-entangled-LCB"),
    (115, 2, 14, "H-type", "two-ends", "sub-entangled-LCB"),
    (116, 2, 18, "H-type", "two-ends", "sub-entangled-LCB"),
    (117, 2, 6, "random-comb", "clustered", "sub-entangled-LCB"),
    (118, 2, 10, "random-comb", "clustered", "sub-entangled-LCB"),
    (119, 2, 14, "random-comb", "clustered", "sub-entangled-LCB"),
    (120, 2, 18, "random-comb", "clustered", "sub-entangled-LCB"),
    (121, 2, 6, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (122, 2, 10, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (123, 2, 14, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),
    (124, 2, 18, "dumbbell-far", "dumbbell-far", "sub-entangled-LCB"),

    # Group I: b=3 interpolation and topology isomers (16 cases, lcb125-lcb140)
    (125, 3, 6, "random-comb", "uniform", "sub-entangled-LCB"),
    (126, 3, 10, "random-comb", "uniform", "sub-entangled-LCB"),
    (127, 3, 14, "random-comb", "uniform", "sub-entangled-LCB"),
    (128, 3, 18, "random-comb", "uniform", "sub-entangled-LCB"),
    (129, 3, 6, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (130, 3, 10, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (131, 3, 14, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (132, 3, 18, "pom-pom", "two-ends+mid", "sub-entangled-LCB"),
    (133, 3, 6, "three-point", "three-point", "sub-entangled-LCB"),
    (134, 3, 10, "three-point", "three-point", "sub-entangled-LCB"),
    (135, 3, 14, "three-point", "three-point", "sub-entangled-LCB"),
    (136, 3, 18, "three-point", "three-point", "sub-entangled-LCB"),
    (137, 3, 6, "random-comb", "clustered", "sub-entangled-LCB"),
    (138, 3, 10, "random-comb", "clustered", "sub-entangled-LCB"),
    (139, 3, 14, "random-comb", "clustered", "sub-entangled-LCB"),
    (140, 3, 18, "random-comb", "clustered", "sub-entangled-LCB"),

    # Group J: b=4 missing/intermediate length grid (20 cases, lcb141-lcb160)
    (141, 4, 4, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (142, 4, 6, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (143, 4, 10, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (144, 4, 14, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (145, 4, 18, "quad-comb", "comb-uniform-4", "sub-entangled-LCB"),
    (146, 4, 4, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (147, 4, 6, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (148, 4, 10, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (149, 4, 14, "quad-pompom", "pompom-4", "sub-entangled-LCB"),
    (150, 4, 4, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (151, 4, 6, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (152, 4, 10, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (153, 4, 14, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (154, 4, 18, "quad-cluster", "cluster-4", "sub-entangled-LCB"),
    (155, 4, 4, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (156, 4, 6, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (157, 4, 10, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (158, 4, 14, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (159, 4, 18, "quad-gradient", "gradient", "sub-entangled-LCB"),
    (160, 4, 20, "quad-gradient", "gradient", "sub-entangled-LCB"),

    # Group K: b=5 missing/intermediate length grid (20 cases, lcb161-lcb180)
    (161, 5, 4, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),
    (162, 5, 6, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),
    (163, 5, 10, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),
    (164, 5, 14, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),
    (165, 5, 4, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (166, 5, 6, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (167, 5, 10, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (168, 5, 14, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (169, 5, 4, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (170, 5, 6, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (171, 5, 10, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (172, 5, 14, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (173, 5, 4, "penta-gradient", "gradient-5", "sub-entangled-LCB"),
    (174, 5, 6, "penta-gradient", "gradient-5", "sub-entangled-LCB"),
    (175, 5, 10, "penta-gradient", "gradient-5", "sub-entangled-LCB"),
    (176, 5, 14, "penta-gradient", "gradient-5", "sub-entangled-LCB"),
    (177, 5, 12, "penta-pompom", "pompom-5", "sub-entangled-LCB"),
    (178, 5, 12, "penta-cluster", "cluster-5", "sub-entangled-LCB"),
    (179, 5, 12, "penta-gradient", "gradient-5", "sub-entangled-LCB"),
    (180, 5, 16, "penta-comb", "comb-uniform-5", "sub-entangled-LCB"),

    # Group L: b=6 high-density LCB axis (20 cases, lcb181-lcb200)
    (181, 6, 3, "hexa-comb", "comb-uniform-6", "sub-entangled-LCB"),
    (182, 6, 4, "hexa-comb", "comb-uniform-6", "sub-entangled-LCB"),
    (183, 6, 5, "hexa-comb", "comb-uniform-6", "sub-entangled-LCB"),
    (184, 6, 6, "hexa-comb", "comb-uniform-6", "sub-entangled-LCB"),
    (185, 6, 8, "hexa-comb", "comb-uniform-6", "sub-entangled-LCB"),
    (186, 6, 10, "hexa-comb", "comb-uniform-6", "sub-entangled-LCB"),
    (187, 6, 12, "hexa-comb", "comb-uniform-6", "sub-entangled-LCB"),
    (188, 6, 3, "hexa-pompom", "pompom-6", "sub-entangled-LCB"),
    (189, 6, 4, "hexa-pompom", "pompom-6", "sub-entangled-LCB"),
    (190, 6, 5, "hexa-pompom", "pompom-6", "sub-entangled-LCB"),
    (191, 6, 6, "hexa-pompom", "pompom-6", "sub-entangled-LCB"),
    (192, 6, 8, "hexa-pompom", "pompom-6", "sub-entangled-LCB"),
    (193, 6, 10, "hexa-pompom", "pompom-6", "sub-entangled-LCB"),
    (194, 6, 12, "hexa-pompom", "pompom-6", "sub-entangled-LCB"),
    (195, 6, 3, "hexa-cluster", "cluster-6", "sub-entangled-LCB"),
    (196, 6, 6, "hexa-cluster", "cluster-6", "sub-entangled-LCB"),
    (197, 6, 10, "hexa-cluster", "cluster-6", "sub-entangled-LCB"),
    (198, 6, 4, "hexa-gradient", "gradient-6", "sub-entangled-LCB"),
    (199, 6, 8, "hexa-gradient", "gradient-6", "sub-entangled-LCB"),
    (200, 6, 12, "hexa-gradient", "gradient-6", "sub-entangled-LCB"),
]

LCB200_SPEC = LCB100_SPEC + LCB200_EXTRA_SPEC

LCB300_EXTRA_SPEC: list[tuple[int, int, int, str, str, str]] = [
    # Group M: b=6 distribution isomers over the feasible L axis (33 cases, lcb201-lcb233)
    (201, 6, 3, "hexa-center", "center-packed", "sub-entangled-LCB"),
    (202, 6, 3, "hexa-start", "start-packed", "sub-entangled-LCB"),
    (203, 6, 3, "hexa-end", "end-packed", "sub-entangled-LCB"),
    (204, 6, 3, "hexa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (205, 6, 3, "hexa-wide-center", "wide-center", "sub-entangled-LCB"),
    (206, 6, 4, "hexa-center", "center-packed", "sub-entangled-LCB"),
    (207, 6, 4, "hexa-start", "start-packed", "sub-entangled-LCB"),
    (208, 6, 4, "hexa-end", "end-packed", "sub-entangled-LCB"),
    (209, 6, 4, "hexa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (210, 6, 4, "hexa-wide-center", "wide-center", "sub-entangled-LCB"),
    (211, 6, 5, "hexa-center", "center-packed", "sub-entangled-LCB"),
    (212, 6, 5, "hexa-start", "start-packed", "sub-entangled-LCB"),
    (213, 6, 5, "hexa-end", "end-packed", "sub-entangled-LCB"),
    (214, 6, 5, "hexa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (215, 6, 5, "hexa-wide-center", "wide-center", "sub-entangled-LCB"),
    (216, 6, 6, "hexa-center", "center-packed", "sub-entangled-LCB"),
    (217, 6, 6, "hexa-start", "start-packed", "sub-entangled-LCB"),
    (218, 6, 6, "hexa-end", "end-packed", "sub-entangled-LCB"),
    (219, 6, 6, "hexa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (220, 6, 6, "hexa-wide-center", "wide-center", "sub-entangled-LCB"),
    (221, 6, 8, "hexa-center", "center-packed", "sub-entangled-LCB"),
    (222, 6, 8, "hexa-start", "start-packed", "sub-entangled-LCB"),
    (223, 6, 8, "hexa-end", "end-packed", "sub-entangled-LCB"),
    (224, 6, 8, "hexa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (225, 6, 8, "hexa-wide-center", "wide-center", "sub-entangled-LCB"),
    (226, 6, 10, "hexa-center", "center-packed", "sub-entangled-LCB"),
    (227, 6, 10, "hexa-start", "start-packed", "sub-entangled-LCB"),
    (228, 6, 10, "hexa-end", "end-packed", "sub-entangled-LCB"),
    (229, 6, 10, "hexa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (230, 6, 10, "hexa-wide-center", "wide-center", "sub-entangled-LCB"),
    (231, 6, 12, "hexa-start", "start-packed", "oligomer-LCB"),
    (232, 6, 12, "hexa-end", "end-packed", "oligomer-LCB"),
    (233, 6, 12, "hexa-left-gradient", "left-gradient", "oligomer-LCB"),

    # Group N: b=7 high-density LCB axis (36 cases, lcb234-lcb269)
    (234, 7, 3, "hepta-comb", "inner-uniform", "sub-entangled-LCB"),
    (235, 7, 3, "hepta-pompom", "terminal-split", "sub-entangled-LCB"),
    (236, 7, 3, "hepta-center", "center-packed", "sub-entangled-LCB"),
    (237, 7, 3, "hepta-end", "end-packed", "sub-entangled-LCB"),
    (238, 7, 3, "hepta-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (239, 7, 3, "hepta-right-gradient", "right-gradient", "sub-entangled-LCB"),
    (240, 7, 4, "hepta-comb", "inner-uniform", "sub-entangled-LCB"),
    (241, 7, 4, "hepta-pompom", "terminal-split", "sub-entangled-LCB"),
    (242, 7, 4, "hepta-center", "center-packed", "sub-entangled-LCB"),
    (243, 7, 4, "hepta-end", "end-packed", "sub-entangled-LCB"),
    (244, 7, 4, "hepta-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (245, 7, 4, "hepta-right-gradient", "right-gradient", "sub-entangled-LCB"),
    (246, 7, 5, "hepta-comb", "inner-uniform", "sub-entangled-LCB"),
    (247, 7, 5, "hepta-pompom", "terminal-split", "sub-entangled-LCB"),
    (248, 7, 5, "hepta-center", "center-packed", "sub-entangled-LCB"),
    (249, 7, 5, "hepta-end", "end-packed", "sub-entangled-LCB"),
    (250, 7, 5, "hepta-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (251, 7, 5, "hepta-right-gradient", "right-gradient", "sub-entangled-LCB"),
    (252, 7, 6, "hepta-comb", "inner-uniform", "sub-entangled-LCB"),
    (253, 7, 6, "hepta-pompom", "terminal-split", "sub-entangled-LCB"),
    (254, 7, 6, "hepta-center", "center-packed", "sub-entangled-LCB"),
    (255, 7, 6, "hepta-end", "end-packed", "sub-entangled-LCB"),
    (256, 7, 6, "hepta-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (257, 7, 6, "hepta-right-gradient", "right-gradient", "sub-entangled-LCB"),
    (258, 7, 8, "hepta-comb", "inner-uniform", "sub-entangled-LCB"),
    (259, 7, 8, "hepta-pompom", "terminal-split", "sub-entangled-LCB"),
    (260, 7, 8, "hepta-center", "center-packed", "sub-entangled-LCB"),
    (261, 7, 8, "hepta-end", "end-packed", "sub-entangled-LCB"),
    (262, 7, 8, "hepta-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (263, 7, 8, "hepta-right-gradient", "right-gradient", "sub-entangled-LCB"),
    (264, 7, 10, "hepta-comb", "inner-uniform", "oligomer-LCB"),
    (265, 7, 10, "hepta-pompom", "terminal-split", "oligomer-LCB"),
    (266, 7, 10, "hepta-center", "center-packed", "oligomer-LCB"),
    (267, 7, 10, "hepta-end", "end-packed", "oligomer-LCB"),
    (268, 7, 10, "hepta-left-gradient", "left-gradient", "oligomer-LCB"),
    (269, 7, 10, "hepta-right-gradient", "right-gradient", "oligomer-LCB"),

    # Group O: b=8 high-density LCB axis (25 cases, lcb270-lcb294)
    (270, 8, 3, "octa-comb", "inner-uniform", "sub-entangled-LCB"),
    (271, 8, 3, "octa-pompom", "terminal-split", "sub-entangled-LCB"),
    (272, 8, 3, "octa-center", "center-packed", "sub-entangled-LCB"),
    (273, 8, 3, "octa-start", "start-packed", "sub-entangled-LCB"),
    (274, 8, 3, "octa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (275, 8, 4, "octa-comb", "inner-uniform", "sub-entangled-LCB"),
    (276, 8, 4, "octa-pompom", "terminal-split", "sub-entangled-LCB"),
    (277, 8, 4, "octa-center", "center-packed", "sub-entangled-LCB"),
    (278, 8, 4, "octa-start", "start-packed", "sub-entangled-LCB"),
    (279, 8, 4, "octa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (280, 8, 5, "octa-comb", "inner-uniform", "sub-entangled-LCB"),
    (281, 8, 5, "octa-pompom", "terminal-split", "sub-entangled-LCB"),
    (282, 8, 5, "octa-center", "center-packed", "sub-entangled-LCB"),
    (283, 8, 5, "octa-start", "start-packed", "sub-entangled-LCB"),
    (284, 8, 5, "octa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (285, 8, 6, "octa-comb", "inner-uniform", "sub-entangled-LCB"),
    (286, 8, 6, "octa-pompom", "terminal-split", "sub-entangled-LCB"),
    (287, 8, 6, "octa-center", "center-packed", "sub-entangled-LCB"),
    (288, 8, 6, "octa-start", "start-packed", "sub-entangled-LCB"),
    (289, 8, 6, "octa-left-gradient", "left-gradient", "sub-entangled-LCB"),
    (290, 8, 8, "octa-comb", "inner-uniform", "sub-entangled-LCB"),
    (291, 8, 8, "octa-pompom", "terminal-split", "sub-entangled-LCB"),
    (292, 8, 8, "octa-center", "center-packed", "sub-entangled-LCB"),
    (293, 8, 8, "octa-start", "start-packed", "sub-entangled-LCB"),
    (294, 8, 8, "octa-left-gradient", "left-gradient", "sub-entangled-LCB"),

    # Group P: b=5 L14 distribution controls (6 cases, lcb295-lcb300)
    (295, 5, 14, "penta-start", "start-packed", "oligomer-LCB"),
    (296, 5, 14, "penta-end", "end-packed", "oligomer-LCB"),
    (297, 5, 14, "penta-center", "center-packed", "oligomer-LCB"),
    (298, 5, 14, "penta-split", "terminal-split", "oligomer-LCB"),
    (299, 5, 14, "penta-inner-split", "inner-split", "oligomer-LCB"),
    (300, 5, 14, "penta-right-gradient", "right-gradient", "oligomer-LCB"),
]

LCB300_SPEC = LCB200_SPEC + LCB300_EXTRA_SPEC


LCB_TOPOLOGY_ANCHOR = {
    "three-arm-star": "Gloger 2022 (star/T1 branch) + Yazdanbakhsh 2025 (length scan)",
    "center-branched": "Yazdanbakhsh 2025 (distribution: asymmetric single branch)",
    "random-comb": "Gloger 2022 (comb series) + Yazdanbakhsh 2025 (density scan)",
    "H-type": "Gloger 2022 (H-architecture) + McLeish tube-theory classic",
    "dumbbell-mid": "Yazdanbakhsh 2025 (distribution: clustered branches)",
    "dumbbell-far": "Yazdanbakhsh 2025 (distribution: far-spaced branches)",
    "pom-pom": "Gloger 2022 (pom-pom) + McLeish 1998 pom-pom model + Jie 2026 (long-arm LCB)",
    "asymmetric-comb": "Yazdanbakhsh 2025 (distribution: end-clustered branches)",
    "three-point": "Three-point symmetric branching (1/4, 1/2, 3/4)",
    "quad-comb": "Four-branch uniform comb (new)",
    "quad-pompom": "Four-branch pom-pom topology (new)",
    "quad-cluster": "Four-branch clustered (new)",
    "quad-gradient": "Four-branch gradient spacing (new)",
    "penta-comb": "Five-branch uniform comb (new)",
    "penta-pompom": "Five-branch pom-pom topology (new)",
    "penta-cluster": "Five-branch clustered (new)",
    "penta-gradient": "Five-branch gradient spacing (new)",
    "hexa-comb": "Six-branch uniform comb extension of the LCB100 density axis",
    "hexa-pompom": "Six-branch pom-pom extension for end-loaded LCB density",
    "hexa-cluster": "Six-branch clustered extension for local branch-density stress tests",
    "hexa-gradient": "Six-branch gradient spacing extension for ML distribution features",
    "hexa-center": "Six-branch center-packed distribution control inspired by Yazdanbakhsh 2025",
    "hexa-start": "Six-branch start-packed distribution control inspired by Yazdanbakhsh 2025",
    "hexa-end": "Six-branch end-packed distribution control inspired by Yazdanbakhsh 2025",
    "hexa-left-gradient": "Six-branch one-sided gradient distribution for ML topology resolution",
    "hexa-wide-center": "Six-branch broad central distribution for density-distribution decoupling",
    "hepta-comb": "Seven-branch uniform comb high-density LCB extension",
    "hepta-pompom": "Seven-branch terminal split high-density pom-pom extension",
    "hepta-center": "Seven-branch center-packed high-density LCB extension",
    "hepta-end": "Seven-branch end-packed distribution control",
    "hepta-left-gradient": "Seven-branch left-gradient distribution control",
    "hepta-right-gradient": "Seven-branch right-gradient distribution control",
    "octa-comb": "Eight-branch uniform comb high-density LCB extension",
    "octa-pompom": "Eight-branch terminal split high-density pom-pom extension",
    "octa-center": "Eight-branch center-packed high-density LCB extension",
    "octa-start": "Eight-branch start-packed distribution control",
    "octa-left-gradient": "Eight-branch left-gradient distribution control",
    "penta-start": "Five-branch start-packed L14 distribution control",
    "penta-end": "Five-branch end-packed L14 distribution control",
    "penta-center": "Five-branch center-packed L14 distribution control",
    "penta-split": "Five-branch terminal split L14 distribution control",
    "penta-inner-split": "Five-branch inner split L14 distribution control",
    "penta-right-gradient": "Five-branch right-gradient L14 distribution control",
}


# PE-LCB100 uses the literature to define design axes, while the exact 100-case
# discretization is a project choice constrained by the C100 budget and builder
# grammar. Keep these labels separate from the inherited PP-LCB anchors so the
# generated provenance does not overstate paper-to-case correspondence.
PELCB_TOPOLOGY_ANCHOR = {
    "three-arm-star": "Horton 1989 star-branched polyethylene topology context",
    "center-branched": "Zhang and Larson 2018 central branch-placement context",
    "random-comb": "Yazdanbakhsh 2025 branch-content and distribution context",
    "H-type": "project-defined two-end placement extension",
    "dumbbell-mid": "Zhang and Larson 2018 grouped branch-placement context",
    "dumbbell-far": "Zhang and Larson 2018 separated branch-placement context",
    "pom-pom": "project-defined terminal-and-middle placement extension",
    "asymmetric-comb": "Yazdanbakhsh 2025 asymmetric distribution context",
    "three-point": "project-defined symmetric three-point placement extension",
    "quad-comb": "project-defined four-branch uniform-comb extension",
    "quad-pompom": "project-defined four-branch terminal extension",
    "quad-cluster": "project-defined four-branch clustered extension",
    "quad-gradient": "project-defined four-branch gradient extension",
    "penta-comb": "project-defined five-branch uniform-comb extension",
    "penta-pompom": "project-defined five-branch terminal extension",
    "penta-cluster": "project-defined five-branch clustered extension",
    "penta-gradient": "project-defined five-branch gradient extension",
}

LCB_TOPOLOGY_SLUG = {
    "three-arm-star": "star3",
    "center-branched": "center",
    "random-comb": "comb",
    "H-type": "htype",
    "dumbbell-mid": "dbmid",
    "dumbbell-far": "dbfar",
    "pom-pom": "pompom",
    "asymmetric-comb": "acomb",
    "three-point": "tri",
    "quad-comb": "q4comb",
    "quad-pompom": "q4pompom",
    "quad-cluster": "q4clust",
    "quad-gradient": "q4grad",
    "penta-comb": "q5comb",
    "penta-pompom": "q5pompom",
    "penta-cluster": "q5clust",
    "penta-gradient": "q5grad",
    "hexa-comb": "q6comb",
    "hexa-pompom": "q6pompom",
    "hexa-cluster": "q6clust",
    "hexa-gradient": "q6grad",
    "hexa-center": "q6center",
    "hexa-start": "q6start",
    "hexa-end": "q6end",
    "hexa-left-gradient": "q6lgrad",
    "hexa-wide-center": "q6wide",
    "hepta-comb": "q7comb",
    "hepta-pompom": "q7pompom",
    "hepta-center": "q7center",
    "hepta-end": "q7end",
    "hepta-left-gradient": "q7lgrad",
    "hepta-right-gradient": "q7rgrad",
    "octa-comb": "q8comb",
    "octa-pompom": "q8pompom",
    "octa-center": "q8center",
    "octa-start": "q8start",
    "octa-left-gradient": "q8lgrad",
    "penta-start": "q5start",
    "penta-end": "q5end",
    "penta-center": "q5center",
    "penta-split": "q5split",
    "penta-inner-split": "q5isplit",
    "penta-right-gradient": "q5rgrad",
}

PRESETS = {
    "first_round_lopez": {
        "sidechain_carbons": [4, 8, 16],
        "default_chains": 150,
        "description": (
            "Short/mid/long literature-grounded screening set anchored to "
            "Lopez-Barron 2021. These correspond to poly(1-hexene), "
            "poly(1-decene), and poly(1-octadecene), spanning the reported "
            "poly(alpha-olefin) range from linear-like to bottlebrush-like."
        ),
        "literature_anchor": {
            "primary": "Lopez-Barron 2021, Macromolecules, 10.1021/acs.macromol.1c00725",
            "reason": (
                "Use three representative side-chain lengths within the "
                "poly(1-hexene) to poly(1-octadecene) range to align the first "
                "round MD structures with the experimentally discussed "
                "conformational crossover window."
            ),
        },
    },
    "first_round_lopez_smoke": {
        "sidechain_carbons": [4, 8, 16],
        "default_chains": 10,
        "description": (
            "Small-box smoke preset that keeps the same literature-grounded "
            "C4/C8/C16 chemistry as first_round_lopez, but reduces chain count "
            "to validate the remote EMC build pipeline before production-scale runs."
        ),
        "literature_anchor": {
            "primary": "Lopez-Barron 2021, Macromolecules, 10.1021/acs.macromol.1c00725",
            "reason": (
                "Reuse the same short/mid/long side-chain chemistry as the "
                "production preset, but at 10 chains/box so the first remote "
                "validation checks pipeline correctness rather than production runtime."
            ),
        },
    },
    "exact_100": {
        "sidechain_carbons": [2, 3, 8, 18],
        "default_chains": 150,
        "description": (
            "Subset whose repeat-unit carbon counts divide 100 exactly. "
            "Useful when strict equality of total carbons/chain is more "
            "important than literature coverage."
        ),
        "literature_anchor": {
            "primary": "Project control variable",
            "reason": "Exact divisors of the current target total carbons per chain.",
        },
    },
    "screen_lcb100": {
        "architecture": "lcb",
        "default_chains": 150,
        "description": (
            "Canonical 100-case LCB library on propylene (C3) backbone. Covers "
            "b=1/2/3/4/5 series, C3/C4 short-branch completions, "
            "and new distribution patterns (near-start, far-end, clustered, "
            "H-asymmetric, dumbbell-far, three-point, pompom-dense, gradient). "
            "Provides dense coverage for ML training: all 100 cases hold "
            "actual total carbons within [85, 105] to allow backbone flexibility."
        ),
        "literature_anchor": {
            "primary": (
                "Gloger 2022 + Yazdanbakhsh 2025 + Jie 2026"
            ),
            "reason": (
                "LCB100 is the canonical systematic 100-point library covering "
                "b∈{1,2,3,4,5}, L∈{3,4,5,8,12,16,20}, "
                "and 13 branch position patterns. New b=4/5 and short L=3/4 cases "
                "fill gaps for extreme topology testing. New patterns (clustered, "
                "gradient, three-point, dumbbell-far) enable finer ML descriptor "
                "training without systematic literature gaps."
            ),
        },
    },
    "screen_pelcb100": {
        "architecture": "pe_lcb",
        "polymer_family": "PE",
        "default_chains": 150,
        "description": (
            "Canonical 100-case LCB library on polyethylene (C2) backbone. "
            "It reuses the LCB100 branch-count, branch-arm length, topology, "
            "and branch-position design matrix, but replaces the isotactic "
            "propylene backbone with an achiral polyethylene backbone. Case "
            "names use generic polymer-family labels for automated comparison "
            "with PP-LCB releases."
        ),
        "literature_anchor": {
            "primary": (
                "Horton 1989 (10.1021/ma00192a029) + "
                "Zhang and Larson 2018 (10.1021/acs.macromol.8b00958) + "
                "Yazdanbakhsh 2025 (10.1007/s10965-025-04467-0)"
            ),
            "reason": (
                "The three sources define polyethylene topology, branch-length/count, "
                "and branch-placement/distribution as design axes. The discrete "
                "bp/bc/pd grid is project-defined under the approximately C100 "
                "carbon budget and registered builder constraints."
            ),
        },
    },
    "screen_lcb200": {
        "architecture": "lcb",
        "default_chains": 150,
        "description": (
            "Canonical 200-case LCB library on propylene (C3) backbone. The "
            "first 100 cases exactly preserve screen_lcb100. Cases 101-200 "
            "add literature- and ML-balanced coverage across intermediate "
            "branch lengths, topology isomers, branch distributions, and a "
            "new b=6 high-density LCB axis. All cases hold actual total "
            "carbons within [99, 101] for direct comparison with LCB100."
        ),
        "literature_anchor": {
            "primary": (
                "Gloger 2022 + Yazdanbakhsh 2025 + Jie 2026 + Munstedt 2026"
            ),
            "reason": (
                "LCB200 preserves the current LCB100 design and adds 100 "
                "new cases for systematic ML-oriented coverage of branch "
                "count, branch length, distribution, and topology. The "
                "extension fills L=6/10/14/18 interpolation gaps for b=1-5, "
                "adds b=4/5 missing topology-length isomers, and introduces "
                "b=6 hexa-comb, hexa-pompom, hexa-cluster, and hexa-gradient "
                "families without adding SCB, copolymer, or replicate axes."
            ),
        },
    },
    "screen_lcb300": {
        "architecture": "lcb",
        "default_chains": 150,
        "description": (
            "Canonical 300-case LCB library on propylene (C3) backbone. The "
            "first 200 cases preserve screen_lcb200. Cases 201-300 add "
            "high-density b=7/8 LCB axes and additional distribution isomers "
            "for b=5/6, while keeping actual total carbons within [99, 101]."
        ),
        "literature_anchor": {
            "primary": (
                "Gloger 2022 + Yazdanbakhsh 2025 + Jie 2026 + Munstedt 2026"
            ),
            "reason": (
                "LCB300 preserves the current LCB200 design and adds 100 "
                "new literature- and ML-balanced cases. The extension focuses "
                "on branch density and distribution decoupling: b=6 isomer "
                "controls, new b=7/8 high-density LCB families, and b=5 L14 "
                "distribution controls. The design keeps the project on the "
                "LCB/iPP route without SCB, copolymer, or replicate axes."
            ),
        },
    },
    "screen_c1_c20": {
        "sidechain_carbons": list(range(1, 21)),
        "default_chains": 150,
        "description": "Full side-chain screening library for the current project scope.",
        "literature_anchor": {
            "primary": "Project scope",
            "reason": "C1-C20 side-chain library for later large-scale screening.",
        },
    },
    "screen_lcb30": {
        "architecture": "lcb",
        "default_chains": 150,
        "description": (
            "Legacy 30-case LCB prototype on propylene (C3) backbone. Superseded "
            "by screen_lcb100 for current MD/ML planning. Orthogonal "
            "coverage of branch-points (1/2/3), branch-length (C5/8/12/16/20) "
            "and 7 topologies (three-arm-star, center-branched, random-comb, "
            "H-type, dumbbell-mid, pom-pom, asymmetric-comb). Every chain "
            "holds actual total carbons within [99, 101] against the project "
            "target of 100 C/chain."
        ),
        "literature_anchor": {
            "primary": (
                "Gloger 2022 (10.1021/acs.macromol.1c02113) + "
                "Yazdanbakhsh 2025 (10.1007/s10965-025-04467-0) + "
                "Jie 2026 (10.1021/acs.macromol.5c03206)"
            ),
            "reason": (
                "Gloger 2022 anchors LCB-PP architecture variants "
                "(star/comb/H/pom-pom) on an iPP backbone. Yazdanbakhsh 2025 "
                "provides the (length x density x distribution) scan grid. "
                "Jie 2026 supports the long-arm (L=20) design points. The "
                "30-case matrix fits these three references into the project "
                "constraints of ~100 total carbons per chain and all-"
                "isotactic ([C@H]) stereocenters."
            ),
        },
    },
}


@dataclass
class CaseMetadata:
    case_id: str
    preset: str
    sidechain_carbons: int
    repeat_unit_total_carbons: int
    # For current homopolymer presets, this matches repeat_unit_total_carbons.
    # Keep it separate so future copolymer cases can assign monomer-level values.
    monomer_total_carbons: int
    monomer_name: str
    chain_count: int
    target_total_carbons_per_chain: int
    repeat_units_per_chain: int
    actual_total_carbons_per_chain: int
    carbon_delta: int
    temperature_k: float
    initial_density_g_per_cc: float
    pressure_atm: float
    force_field: str
    monomer_smiles: str
    tacticity: str
    literature_anchor: dict
    # Optional LCB extension fields. Linear presets leave these at defaults.
    # For LCB cases monomer_smiles holds a macro SMILES encoding the full chain
    # (backbone + all branches) and repeat_units_per_chain=1; backbone_repeat_units
    # stores the propylene unit count of the main chain.
    architecture: str = "linear"
    backbone_monomer: str | None = None
    backbone_repeat_units: int | None = None
    branch_points: int | None = None
    branch_carbons: int | None = None
    branch_topology: str | None = None
    branch_position_pattern: str | None = None
    branch_positions: list[int] | None = None
    branch_smiles: str | None = None
    regime_tag: str | None = None
    ml_feature_coord: str | None = None
    polymer_family: str = "PP"
    source_case_id: str | None = None
    generic_case_name: str | None = None
    generic_plot_label: str | None = None
    branch_distribution_code: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EMC inputs for isotactic alpha-olefin homopolymers."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="first_round_lopez",
        help="Named case set to generate.",
    )
    parser.add_argument(
        "--sidechain-carbons",
        type=int,
        nargs="*",
        help="Override preset with explicit side-chain carbon counts.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help="Number of polymer chains in the build box. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--target-total-carbons",
        type=int,
        default=100,
        help="Target total carbons per chain before integer repeat rounding.",
    )
    parser.add_argument(
        "--repeat-mode",
        choices=["nearest", "floor", "ceil", "exact"],
        default="nearest",
        help="How to convert target total carbons into an integer repeat count.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=400.0,
        help="Initial EMC/MD temperature in K.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.85,
        help="Initial build density in g/cc.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Target pressure in atm.",
    )
    parser.add_argument(
        "--field",
        default="opls/2024/opls-aa",
        help="EMC force-field identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Root directory that will receive generated cases.",
    )
    return parser.parse_args()


def choose_repeat_units(
    target_total_carbons: int,
    repeat_unit_total_carbons: int,
    mode: str,
) -> int:
    raw = target_total_carbons / repeat_unit_total_carbons
    if mode == "nearest":
        return max(1, int(raw + 0.5))
    if mode == "floor":
        return max(1, int(raw))
    if mode == "ceil":
        return max(1, int(-(-target_total_carbons // repeat_unit_total_carbons)))
    if target_total_carbons % repeat_unit_total_carbons != 0:
        raise ValueError(
            f"target_total_carbons={target_total_carbons} is not divisible by "
            f"repeat_unit_total_carbons={repeat_unit_total_carbons}"
        )
    return target_total_carbons // repeat_unit_total_carbons


def alkyl_smiles(sidechain_carbons: int) -> str:
    if sidechain_carbons < 1:
        raise ValueError("sidechain_carbons must be >= 1 for alpha-olefins")
    return "C" * sidechain_carbons


def monomer_smiles(sidechain_carbons: int) -> str:
    # Use a fixed stereocenter marker to encode isotactic repeat units.
    return f"*[C@H]({alkyl_smiles(sidechain_carbons)})C*"


def monomer_name(sidechain_carbons: int) -> str:
    return ALKENE_NAMES.get(sidechain_carbons + 2, f"1-alkene-C{sidechain_carbons + 2}")


def case_id(sidechain_carbons: int, repeat_units: int, actual_total: int) -> str:
    return f"sc{sidechain_carbons:02d}_r{repeat_units:02d}_c{actual_total:03d}"


def lcb_case_id(
    sequence: int,
    branch_points: int,
    branch_carbons: int,
    topology: str,
    actual_total: int,
) -> str:
    topology_slug = LCB_TOPOLOGY_SLUG[topology]
    return (
        f"lcb{sequence:02d}_{topology_slug}_b{branch_points}"
        f"_l{branch_carbons:02d}_c{actual_total:03d}"
    )


GENERIC_DISTRIBUTION_CODES = {
    "mid": "mid",
    "near-end": "end",
    "near-start": "end",
    "far-end": "end",
    "uniform": "uni",
    "inner-uniform": "uni",
    "two-ends": "term",
    "two-ends+mid": "term",
    "terminal-split": "term",
    "clustered": "clus",
    "cluster-4": "clus",
    "cluster-5": "clus",
    "cluster-6": "clus",
    "center-packed": "clus",
    "end-cluster": "clus",
    "start-packed": "end",
    "end-packed": "end",
    "gradient": "grad",
    "gradient-5": "grad",
    "gradient-6": "grad",
    "left-gradient": "grad",
    "right-gradient": "grad",
    "three-point": "tri",
    "dumbbell-far": "db",
    "mid-pair": "db",
    "H-asymmetric": "asym",
    "inner-split": "split",
    "wide-center": "wide",
}


def generic_distribution_code(pattern: str) -> str:
    key = str(pattern).strip()
    if key in GENERIC_DISTRIBUTION_CODES:
        return GENERIC_DISTRIBUTION_CODES[key]
    if key.startswith("comb-uniform"):
        return "uni"
    if key.startswith("pompom"):
        return "term"
    if key.startswith("cluster"):
        return "clus"
    if key.startswith("gradient"):
        return "grad"
    return re.sub(r"[^a-z0-9]+", "", key.lower()) or "na"


def generic_lcb_case_id(
    polymer_family: str,
    sequence: int,
    branch_points: int,
    branch_carbons: int,
    pattern: str,
    actual_total: int,
) -> str:
    family = polymer_family.lower()
    dist = generic_distribution_code(pattern)
    return (
        f"lcb_{family}_{sequence:03d}_bp{branch_points:02d}"
        f"_bc{branch_carbons:03d}_pd{dist}_c{actual_total:03d}"
    )


def generic_lcb_plot_label(
    polymer_family: str,
    sequence: int,
    branch_points: int,
    branch_carbons: int,
    pattern: str,
) -> str:
    family = polymer_family.upper()
    dist = generic_distribution_code(pattern).title()
    return f"{family}{sequence:03d}/BP{branch_points}-BC{branch_carbons}-{dist}"


def lcb_monomer_name(topology: str, branch_points: int, branch_carbons: int) -> str:
    return f"LCB-iPP {topology} (b={branch_points}, L=C{branch_carbons})"


def clamp_branch_position(backbone_repeat_units: int, position: int) -> int:
    if backbone_repeat_units <= 2:
        return max(1, min(backbone_repeat_units, position))
    return max(2, min(backbone_repeat_units - 1, position))


def normalize_branch_positions(backbone_repeat_units: int, positions: Iterable[int]) -> list[int]:
    normalized: list[int] = []
    for position in sorted({clamp_branch_position(backbone_repeat_units, pos) for pos in positions}):
        if position not in normalized:
            normalized.append(position)
    if not normalized:
        raise ValueError("LCB branch positions collapsed to an empty set")
    return normalized


def evenly_spaced_positions(
    backbone_repeat_units: int,
    count: int,
    left: int | None = None,
    right: int | None = None,
) -> list[int]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if left is None:
        left = 2 if backbone_repeat_units > 3 else 1
    if right is None:
        right = backbone_repeat_units - 1 if backbone_repeat_units > 3 else backbone_repeat_units
    if count == 1:
        return [clamp_branch_position(backbone_repeat_units, int((left + right) / 2))]
    values: list[int] = []
    span = right - left
    for index in range(count):
        raw = left + (span * index) / (count - 1)
        values.append(int(round(raw)))
    return normalize_branch_positions(backbone_repeat_units, values)


def branch_site_candidates(backbone_repeat_units: int) -> list[int]:
    if backbone_repeat_units <= 3:
        return list(range(1, backbone_repeat_units + 1))
    return list(range(2, backbone_repeat_units))


def unique_positions_from_scores(
    backbone_repeat_units: int,
    count: int,
    scores: Iterable[float],
) -> list[int]:
    candidates = branch_site_candidates(backbone_repeat_units)
    if len(candidates) < count:
        raise ValueError(
            f"Not enough unique branch positions for count={count}, "
            f"backbone_repeat_units={backbone_repeat_units}"
        )
    selected: list[int] = []
    left = min(candidates)
    right = max(candidates)
    for score in scores:
        target = left + (score * (right - left))
        for candidate in sorted(candidates, key=lambda value: (abs(value - target), value)):
            if candidate not in selected:
                selected.append(candidate)
                break
    for candidate in candidates:
        if len(selected) >= count:
            break
        if candidate not in selected:
            selected.append(candidate)
    return normalize_branch_positions(backbone_repeat_units, selected[:count])


def center_packed_positions(backbone_repeat_units: int, count: int) -> list[int]:
    candidates = branch_site_candidates(backbone_repeat_units)
    center = (min(candidates) + max(candidates)) / 2.0
    selected = sorted(candidates, key=lambda value: (abs(value - center), value))[:count]
    return normalize_branch_positions(backbone_repeat_units, selected)


def terminal_split_positions(backbone_repeat_units: int, count: int) -> list[int]:
    candidates = branch_site_candidates(backbone_repeat_units)
    left_count = count // 2
    right_count = count - left_count
    return normalize_branch_positions(
        backbone_repeat_units,
        candidates[:left_count] + candidates[-right_count:],
    )


def inner_split_positions(backbone_repeat_units: int, count: int) -> list[int]:
    candidates = branch_site_candidates(backbone_repeat_units)
    left_count = count // 2
    right_count = count - left_count
    if len(candidates) >= count + 2:
        positions = candidates[1 : 1 + left_count] + candidates[-right_count - 1 : -1]
    else:
        positions = candidates[:left_count] + candidates[-right_count:]
    return normalize_branch_positions(backbone_repeat_units, positions)


def branch_positions_for_pattern(
    backbone_repeat_units: int,
    branch_points: int,
    pattern: str,
) -> list[int]:
    near_end = clamp_branch_position(backbone_repeat_units, max(2, int(round(backbone_repeat_units * 0.2))))
    far_end = clamp_branch_position(backbone_repeat_units, backbone_repeat_units - near_end + 1)
    center = clamp_branch_position(backbone_repeat_units, int(round(backbone_repeat_units / 2)))
    if pattern == "mid":
        return [center]
    if pattern == "near-end":
        return [near_end]
    if pattern == "uniform":
        return evenly_spaced_positions(backbone_repeat_units, branch_points)
    if pattern == "two-ends":
        return normalize_branch_positions(backbone_repeat_units, [near_end, far_end])
    if pattern == "mid-pair":
        return normalize_branch_positions(
            backbone_repeat_units,
            [center - 2, center + 2],
        )
    if pattern == "two-ends+mid":
        return normalize_branch_positions(backbone_repeat_units, [near_end, center, far_end])
    if pattern == "end-cluster":
        return normalize_branch_positions(backbone_repeat_units, [near_end, near_end + 2, near_end + 4])
    if pattern == "near-start":
        return [clamp_branch_position(backbone_repeat_units, 2)]
    if pattern == "far-end":
        return [clamp_branch_position(backbone_repeat_units, backbone_repeat_units - 1)]
    if pattern == "clustered":
        zone = max(branch_points + 2, backbone_repeat_units // 3)
        return normalize_branch_positions(backbone_repeat_units, evenly_spaced_positions(zone, branch_points))
    if pattern == "H-asymmetric":
        return normalize_branch_positions(backbone_repeat_units, [near_end, center])
    if pattern == "dumbbell-far":
        q1 = clamp_branch_position(backbone_repeat_units, backbone_repeat_units // 4)
        q3 = clamp_branch_position(backbone_repeat_units, 3 * backbone_repeat_units // 4)
        return normalize_branch_positions(backbone_repeat_units, [q1, q3])
    if pattern == "three-point":
        q1 = clamp_branch_position(backbone_repeat_units, backbone_repeat_units // 4)
        q2 = center
        q3 = clamp_branch_position(backbone_repeat_units, 3 * backbone_repeat_units // 4)
        return normalize_branch_positions(backbone_repeat_units, [q1, q2, q3])
    if pattern == "pompom-dense":
        return normalize_branch_positions(backbone_repeat_units, [near_end, max(2, center - 2), far_end])
    if pattern == "gradient":
        if branch_points == 4:
            return normalize_branch_positions(
                backbone_repeat_units,
                [
                    2,
                    max(4, int(round(backbone_repeat_units * 0.25))),
                    max(5, int(round(backbone_repeat_units * 0.55))),
                    backbone_repeat_units - 1,
                ],
            )
        return evenly_spaced_positions(backbone_repeat_units, branch_points)
    if pattern in ("comb-uniform-4", "comb-uniform-5", "comb-uniform-6"):
        return evenly_spaced_positions(backbone_repeat_units, branch_points)
    if pattern == "pompom-4":
        return normalize_branch_positions(
            backbone_repeat_units,
            [2, 4, backbone_repeat_units - 3, backbone_repeat_units - 1],
        )
    if pattern == "cluster-4":
        return normalize_branch_positions(
            backbone_repeat_units,
            [center - 3, center - 1, center + 1, center + 3],
        )
    if pattern == "pompom-5":
        return normalize_branch_positions(
            backbone_repeat_units,
            [2, 4, center, backbone_repeat_units - 3, backbone_repeat_units - 1],
        )
    if pattern == "cluster-5":
        return normalize_branch_positions(
            backbone_repeat_units,
            [center - 4, center - 2, center, center + 2, center + 4],
        )
    if pattern == "gradient-5":
        return normalize_branch_positions(
            backbone_repeat_units,
            [2, 4, max(5, center - 2), min(backbone_repeat_units - 2, center + 4), backbone_repeat_units - 1],
        )
    if pattern == "pompom-6":
        return normalize_branch_positions(
            backbone_repeat_units,
            [2, 3, 4, backbone_repeat_units - 4, backbone_repeat_units - 2, backbone_repeat_units - 1],
        )
    if pattern == "cluster-6":
        return normalize_branch_positions(
            backbone_repeat_units,
            [center - 3, center - 2, center - 1, center + 1, center + 2, center + 3],
        )
    if pattern == "gradient-6":
        return normalize_branch_positions(
            backbone_repeat_units,
            [
                2,
                4,
                max(5, int(round(backbone_repeat_units * 0.35))),
                max(6, int(round(backbone_repeat_units * 0.55))),
                max(7, int(round(backbone_repeat_units * 0.75))),
                backbone_repeat_units - 1,
            ],
        )
    if pattern == "center-packed":
        return center_packed_positions(backbone_repeat_units, branch_points)
    if pattern == "start-packed":
        return normalize_branch_positions(
            backbone_repeat_units,
            branch_site_candidates(backbone_repeat_units)[:branch_points],
        )
    if pattern == "end-packed":
        return normalize_branch_positions(
            backbone_repeat_units,
            branch_site_candidates(backbone_repeat_units)[-branch_points:],
        )
    if pattern == "terminal-split":
        return terminal_split_positions(backbone_repeat_units, branch_points)
    if pattern == "inner-split":
        return inner_split_positions(backbone_repeat_units, branch_points)
    if pattern == "left-gradient":
        return unique_positions_from_scores(
            backbone_repeat_units,
            branch_points,
            [((index + 1) / branch_points) ** 1.7 for index in range(branch_points)],
        )
    if pattern == "right-gradient":
        left_gradient = branch_positions_for_pattern(
            backbone_repeat_units,
            branch_points,
            "left-gradient",
        )
        candidates = branch_site_candidates(backbone_repeat_units)
        mirror_sum = min(candidates) + max(candidates)
        return normalize_branch_positions(
            backbone_repeat_units,
            [mirror_sum - position for position in left_gradient],
        )
    if pattern == "wide-center":
        if branch_points == 1:
            scores = [0.5]
        else:
            scores = [
                0.12 + (index * (0.76 / (branch_points - 1)))
                for index in range(branch_points)
            ]
        return unique_positions_from_scores(backbone_repeat_units, branch_points, scores)
    if pattern == "inner-uniform":
        if branch_points == 1:
            scores = [0.5]
        else:
            scores = [index / (branch_points - 1) for index in range(branch_points)]
        return unique_positions_from_scores(backbone_repeat_units, branch_points, scores)
    raise ValueError(f"Unsupported LCB position pattern: {pattern}")


def lcb_macro_smiles(
    backbone_repeat_units: int,
    branch_positions: Iterable[int],
    branch_carbons: int,
) -> str:
    branch_lookup = set(branch_positions)
    branch_smiles = alkyl_smiles(branch_carbons)
    fragments: list[str] = []
    for repeat_index in range(1, backbone_repeat_units + 1):
        if repeat_index in branch_lookup:
            fragments.append(f"[C](C)({branch_smiles})C")
        else:
            fragments.append("[C@H](C)C")
    return f"*{''.join(fragments)}*"


def pe_lcb_macro_smiles(
    backbone_repeat_units: int,
    branch_positions: Iterable[int],
    branch_carbons: int,
) -> str:
    branch_lookup = set(branch_positions)
    branch_smiles = alkyl_smiles(branch_carbons)
    fragments: list[str] = []
    for repeat_index in range(1, backbone_repeat_units + 1):
        if repeat_index in branch_lookup:
            fragments.append(f"C({branch_smiles})C")
        else:
            fragments.append("CC")
    return f"*{''.join(fragments)}*"


def render_esh(meta: CaseMetadata) -> str:
    primary = ascii_safe(meta.literature_anchor["primary"])
    reason = ascii_safe(meta.literature_anchor["reason"])
    monomer_name_ascii = ascii_safe(meta.monomer_name)
    lines = [
        "#!/usr/bin/env emc.pl",
        "#",
        f"# Literature anchor: {primary}",
        f"# Notes: {reason}",
        f"# Monomer: {monomer_name_ascii}",
        f"# Side-chain carbons: {meta.sidechain_carbons}",
        f"# Target total carbons/chain: {meta.target_total_carbons_per_chain}",
        f"# Actual total carbons/chain: {meta.actual_total_carbons_per_chain}",
        f"# Repeat units/chain: {meta.repeat_units_per_chain}",
        f"# Tacticity: {meta.tacticity}",
    ]
    if meta.architecture == "lcb":
        lines.extend(
            [
                f"# Architecture: {meta.architecture}",
                f"# Backbone repeat units: {meta.backbone_repeat_units}",
                f"# Branch topology: {meta.branch_topology}",
                f"# Branch points: {meta.branch_points}",
                f"# Branch carbons: {meta.branch_carbons}",
                f"# Branch position pattern: {meta.branch_position_pattern}",
                f"# Branch positions: {','.join(str(pos) for pos in (meta.branch_positions or []))}",
                f"# ML feature coord: {ascii_safe(meta.ml_feature_coord or '')}",
            ]
        )
    lines.extend(
        [
            "",
            "ITEM\tOPTIONS",
            "replace\ttrue",
            f"field\t{meta.force_field}",
            "number\ttrue",
            f"density\t{meta.initial_density_g_per_cc}",
            f"temperature\t{meta.temperature_k}",
            f"pressure\t{meta.pressure_atm}",
            "build_dir\t.",
            "focus\ttrue",
            "ITEM\tEND",
            "",
            "ITEM\tGROUPS",
            f"monomer\t{meta.monomer_smiles},1,monomer:2,1,term:1,2,term:1",
            "term\t*C",
            "ITEM\tEND",
            "",
            "ITEM\tCLUSTERS",
            f"polymer\talternate,{meta.chain_count}",
            "ITEM\tEND",
            "",
            "ITEM\tPOLYMERS",
            "polymer",
            f"1\tmonomer,{meta.repeat_units_per_chain},term,2",
            "ITEM\tEND",
            "",
        ]
    )
    return "\n".join(lines)


def ascii_safe(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_linear_cases(args: argparse.Namespace, preset: dict) -> list[CaseMetadata]:
    sidechains = args.sidechain_carbons or preset["sidechain_carbons"]
    chain_count = args.chains if args.chains is not None else preset.get("default_chains", 150)
    cases: list[CaseMetadata] = []
    for sidechain in sidechains:
        repeat_unit_total = sidechain + 2
        repeats = choose_repeat_units(
            args.target_total_carbons,
            repeat_unit_total,
            args.repeat_mode,
        )
        actual_total = repeats * repeat_unit_total
        cases.append(
            CaseMetadata(
                case_id=case_id(sidechain, repeats, actual_total),
                preset=args.preset,
                sidechain_carbons=sidechain,
                repeat_unit_total_carbons=repeat_unit_total,
                monomer_total_carbons=repeat_unit_total,
                monomer_name=monomer_name(sidechain),
                chain_count=chain_count,
                target_total_carbons_per_chain=args.target_total_carbons,
                repeat_units_per_chain=repeats,
                actual_total_carbons_per_chain=actual_total,
                carbon_delta=actual_total - args.target_total_carbons,
                temperature_k=args.temperature,
                initial_density_g_per_cc=args.density,
                pressure_atm=args.pressure,
                force_field=args.field,
                monomer_smiles=monomer_smiles(sidechain),
                tacticity="isotactic (fixed [C@H] repeat stereocenter)",
                literature_anchor=preset["literature_anchor"],
            )
        )
    return cases


def build_lcb_cases(args: argparse.Namespace, preset: dict) -> list[CaseMetadata]:
    if args.sidechain_carbons:
        raise ValueError("--sidechain-carbons override is not supported for LCB presets")
    chain_count = args.chains if args.chains is not None else preset.get("default_chains", 150)
    polymer_family = str(preset.get("polymer_family", "PP")).upper()
    backbone_repeat_carbon_count = 2 if polymer_family == "PE" else 3
    backbone_monomer = "ethylene" if polymer_family == "PE" else "propylene"
    cases: list[CaseMetadata] = []
    lcb_specs = {
        "screen_lcb30": LCB_LEGACY30_SPEC,
        "screen_lcb100": LCB100_SPEC,
        "screen_pelcb100": LCB100_SPEC,
        "screen_lcb200": LCB200_SPEC,
        "screen_lcb300": LCB300_SPEC,
    }
    lcb_spec = lcb_specs[args.preset]
    carbon_floor, carbon_ceil = (99, 101)
    for sequence, branch_points, branch_carbons, topology, pattern, regime_tag in lcb_spec:
        backbone_repeat_units = choose_repeat_units(
            args.target_total_carbons - (branch_points * branch_carbons),
            backbone_repeat_carbon_count,
            args.repeat_mode,
        )
        actual_total = (backbone_repeat_units * backbone_repeat_carbon_count) + (branch_points * branch_carbons)
        if not carbon_floor <= actual_total <= carbon_ceil:
            raise ValueError(
                f"LCB case seq={sequence} fell outside [{carbon_floor}, {carbon_ceil}] actual carbons: {actual_total}"
            )
        branch_positions = branch_positions_for_pattern(
            backbone_repeat_units=backbone_repeat_units,
            branch_points=branch_points,
            pattern=pattern,
        )
        if len(branch_positions) != branch_points:
            raise ValueError(
                f"LCB case seq={sequence} expected {branch_points} branch positions, "
                f"got {len(branch_positions)}: {branch_positions}"
            )
        branch_smiles = alkyl_smiles(branch_carbons)
        ml_feature_coord = (
            f"b{branch_points}_L{branch_carbons}_"
            f"{LCB_TOPOLOGY_SLUG[topology]}_{pattern.replace('+', '-')}"
        )
        topology_anchor = (
            PELCB_TOPOLOGY_ANCHOR[topology]
            if polymer_family == "PE"
            else LCB_TOPOLOGY_ANCHOR[topology]
        )
        reason = (
            f"{preset['literature_anchor']['reason']} This case instantiates "
            f"{topology} with {branch_points} branch point(s), branch length "
            f"C{branch_carbons}, and pattern {pattern}. Topology anchor: "
            f"{topology_anchor}."
        )
        source_case_id = lcb_case_id(
            sequence=sequence,
            branch_points=branch_points,
            branch_carbons=branch_carbons,
            topology=topology,
            actual_total=actual_total,
        )
        if polymer_family == "PE":
            case_name = generic_lcb_case_id(
                polymer_family=polymer_family,
                sequence=sequence,
                branch_points=branch_points,
                branch_carbons=branch_carbons,
                pattern=pattern,
                actual_total=actual_total,
            )
            monomer_name_text = f"LCB-PE {topology} (b={branch_points}, L=C{branch_carbons})"
            macro_smiles = pe_lcb_macro_smiles(
                backbone_repeat_units=backbone_repeat_units,
                branch_positions=branch_positions,
                branch_carbons=branch_carbons,
            )
            tacticity = "achiral polyethylene backbone; branch-point centers retain implicit hydrogens"
        else:
            case_name = source_case_id
            monomer_name_text = lcb_monomer_name(topology, branch_points, branch_carbons)
            macro_smiles = lcb_macro_smiles(
                backbone_repeat_units=backbone_repeat_units,
                branch_positions=branch_positions,
                branch_carbons=branch_carbons,
            )
            tacticity = (
                "isotactic propylene backbone; unbranched centers keep fixed "
                "[C@H], branch-point centers are achiral quaternary carbons"
            )
        cases.append(
            CaseMetadata(
                case_id=case_name,
                preset=args.preset,
                sidechain_carbons=branch_carbons,
                repeat_unit_total_carbons=backbone_repeat_carbon_count,
                monomer_total_carbons=actual_total,
                monomer_name=monomer_name_text,
                chain_count=chain_count,
                target_total_carbons_per_chain=args.target_total_carbons,
                repeat_units_per_chain=1,
                actual_total_carbons_per_chain=actual_total,
                carbon_delta=actual_total - args.target_total_carbons,
                temperature_k=args.temperature,
                initial_density_g_per_cc=args.density,
                pressure_atm=args.pressure,
                force_field=args.field,
                monomer_smiles=macro_smiles,
                tacticity=tacticity,
                literature_anchor={
                    "primary": preset["literature_anchor"]["primary"],
                    "reason": reason,
                },
                architecture="pe_lcb" if polymer_family == "PE" else "lcb",
                backbone_monomer=backbone_monomer,
                backbone_repeat_units=backbone_repeat_units,
                branch_points=branch_points,
                branch_carbons=branch_carbons,
                branch_topology=topology,
                branch_position_pattern=pattern,
                branch_positions=branch_positions,
                branch_smiles=branch_smiles,
                regime_tag=regime_tag,
                ml_feature_coord=ml_feature_coord,
                polymer_family=polymer_family,
                source_case_id=source_case_id,
                generic_case_name=generic_lcb_case_id(
                    polymer_family=polymer_family,
                    sequence=sequence,
                    branch_points=branch_points,
                    branch_carbons=branch_carbons,
                    pattern=pattern,
                    actual_total=actual_total,
                ),
                generic_plot_label=generic_lcb_plot_label(
                    polymer_family=polymer_family,
                    sequence=sequence,
                    branch_points=branch_points,
                    branch_carbons=branch_carbons,
                    pattern=pattern,
                ),
                branch_distribution_code=generic_distribution_code(pattern),
            )
        )
    smiles_to_case: dict[str, str] = {}
    duplicates: list[str] = []
    for case in cases:
        previous = smiles_to_case.get(case.monomer_smiles)
        if previous is not None:
            duplicates.append(f"{case.case_id} duplicates {previous}")
        smiles_to_case[case.monomer_smiles] = case.case_id
    if duplicates:
        raise ValueError("LCB library contains duplicate monomer_smiles: " + "; ".join(duplicates))
    return cases


def build_cases(args: argparse.Namespace) -> list[CaseMetadata]:
    preset = PRESETS[args.preset]
    if preset.get("architecture") in {"lcb", "pe_lcb"}:
        return build_lcb_cases(args, preset)
    return build_linear_cases(args, preset)


def ensure_output(root: Path, preset: str) -> Path:
    run_dir = root / preset
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def emit_manifest(run_dir: Path, cases: Iterable[CaseMetadata], description: str) -> None:
    case_list = list(cases)
    write_json(
        run_dir / "manifest.json",
        {
            "description": description,
            "cases": [asdict(case) for case in case_list],
        },
    )
    with (run_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(case_list[0]).keys()),
        )
        writer.writeheader()
        for case in case_list:
            writer.writerow(asdict(case))


def emit_cases(run_dir: Path, cases: Iterable[CaseMetadata]) -> None:
    for case in cases:
        case_dir = run_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"{case.case_id}.esh").write_text(render_esh(case), encoding="ascii")
        write_json(case_dir / "case.json", asdict(case))


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    cases = build_cases(args)
    run_dir = ensure_output(args.output_dir, args.preset)
    emit_cases(run_dir, cases)
    emit_manifest(run_dir, cases, preset["description"])
    print(f"Wrote {len(cases)} EMC case(s) to {run_dir}")
    for case in cases:
        print(
            f"- {case.case_id}: {case.monomer_name}, "
            f"chains={case.chain_count}, sidechain={case.sidechain_carbons}, "
            f"repeats={case.repeat_units_per_chain}, "
            f"target_c={case.target_total_carbons_per_chain}, "
            f"actual_c={case.actual_total_carbons_per_chain}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
