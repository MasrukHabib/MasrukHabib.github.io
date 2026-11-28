---
title: "Graph Neural Networks, Explained: How a Node Learns from Its Neighbours"
category: Machine Learning
tags: [GNN, Graph Neural Networks, GCN, GraphSAGE, GAT, PyTorch Geometric]
excerpt: "Molecules, friendships, road maps and payments all live on graphs, and a graph has no first row. How graph neural networks learn anyway, one animated step at a time: message passing, GCN, GraphSAGE and GAT, why depth hurts, and a working model in 30 lines of PyTorch."
description: "A visual, step-by-step introduction to graph neural networks: adjacency matrices and permutation invariance, message passing, receptive fields and computation trees, the GCN layer in matrix form, GraphSAGE and graph attention, sum versus mean aggregation and the Weisfeiler-Lehman limit, oversmoothing, task heads, and runnable PyTorch and PyTorch Geometric code on Cora."
image: https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1600&auto=format&fit=crop
image_alt: "Earth at night seen from orbit, city lights strung together across a continent like the nodes and edges of a network"
math: true
---

Most of the data deep learning grew up on has a shape you can count on. An image is a grid: every pixel has a pixel above it, below it and on either side. A sentence is a line: every word has one word before it and one after. Convolutions and transformers are built around that regularity.

A lot of the data that matters has no such shape. A molecule is atoms joined by bonds. A social network is people joined by friendships. A road map is junctions joined by roads, and a payment system is accounts joined by transfers. In each case, what you know about one thing depends on what it is connected to, and the connections are irregular: one atom has a single bond, another has four; one account has two counterparties, another has two thousand. There is no first row, no left neighbour and no fixed size.

**Graph neural networks** (GNNs) are how deep learning handles that. The core idea fits in one sentence: *describe each node by its neighbours, and repeat.* Everything else, from GCN to graph attention, is a variation on how exactly to do the describing.

This post builds that idea up one step at a time, with a figure for each step: how a computer stores a graph, why ordinary networks struggle with it, message passing, how stacking layers widens what a node can see, the matrix form behind GCN, the three layers you will meet everywhere, why the choice between sum and mean matters, why deeper is not better, and finally a working model you can run on a laptop.

The math needs matrix multiplication and nothing more. Figures animate when they scroll into view: hover them for exact values and use **replay** to watch again. Figures with step buttons underneath walk through a process; press **play**, or click the steps at your own pace. The code uses PyTorch and PyTorch Geometric, and every number quoted from it comes from actually running it.
{: .callout}

## 1. What a graph looks like to a computer

A graph is a set of **nodes** (also called vertices) and a set of **edges** between them. To give one to a neural network, you need two things:

- **Node features:** a matrix $$X$$ with one row per node. For a person that row might hold age, activity and interests; for an atom, the element and its charge; for a paper, which words it uses.
- **Structure:** usually an **adjacency matrix** $$A$$. With $$n$$ nodes it is $$n \times n$$, and $$A_{uv} = 1$$ when there is an edge between $$u$$ and $$v$$.

{::nomarkdown}
<figure class="viz" id="fig-graph">
<div class="viz-head"><span class="viz-label">FIG. 1 // A GRAPH, AS A COMPUTER SEES IT</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 290" role="img" aria-labelledby="fig-graph-t fig-graph-d">
<title id="fig-graph-t">A small graph, its adjacency matrix and its feature matrix</title>
<desc id="fig-graph-d">A graph of seven nodes, A to G, with eight edges. Its adjacency matrix has a 1 wherever two nodes share an edge: row A has ones under B, C and D. The feature matrix has one row of three numbers per node.</desc>
<text class="tm ta" x="24" y="26">THE GRAPH // 7 NODES, 8 EDGES</text>
<path class="ge a-draw" pathLength="1" d="M100 160L70 85" style="--d:0.55s;--t:.35s"/>
<path class="ge a-draw" pathLength="1" d="M100 160L180 95" style="--d:0.72s;--t:.35s"/>
<path class="ge a-draw" pathLength="1" d="M100 160L140 240" style="--d:0.89s;--t:.35s"/>
<path class="ge a-draw" pathLength="1" d="M70 85L180 95" style="--d:1.06s;--t:.35s"/>
<path class="ge a-draw" pathLength="1" d="M180 95L275 115" style="--d:1.23s;--t:.35s"/>
<path class="ge a-draw" pathLength="1" d="M140 240L255 215" style="--d:1.40s;--t:.35s"/>
<path class="ge a-draw" pathLength="1" d="M275 115L255 215" style="--d:1.57s;--t:.35s"/>
<path class="ge a-draw" pathLength="1" d="M255 215L340 245" style="--d:1.74s;--t:.35s"/>
<g class="a-in" style="--d:0.05s"><circle class="gn" cx="100" cy="160" r="17"/><text class="gl" x="100" y="160" dy="0.35em">A</text></g>
<g class="a-in" style="--d:0.09s"><circle class="gn" cx="70" cy="85" r="17"/><text class="gl" x="70" y="85" dy="0.35em">B</text></g>
<g class="a-in" style="--d:0.13s"><circle class="gn" cx="180" cy="95" r="17"/><text class="gl" x="180" y="95" dy="0.35em">C</text></g>
<g class="a-in" style="--d:0.17s"><circle class="gn" cx="140" cy="240" r="17"/><text class="gl" x="140" y="240" dy="0.35em">D</text></g>
<g class="a-in" style="--d:0.21s"><circle class="gn" cx="275" cy="115" r="17"/><text class="gl" x="275" y="115" dy="0.35em">E</text></g>
<g class="a-in" style="--d:0.25s"><circle class="gn" cx="255" cy="215" r="17"/><text class="gl" x="255" y="215" dy="0.35em">F</text></g>
<g class="a-in" style="--d:0.29s"><circle class="gn" cx="340" cy="245" r="17"/><text class="gl" x="340" y="245" dy="0.35em">G</text></g>
<text class="tm ta" x="440" y="26">ADJACENCY MATRIX A</text>
<text class="tm" x="440" y="42">1 = AN EDGE, 0 = NONE</text>
<text class="tm" x="450" y="63" text-anchor="middle">A</text>
<text class="tm" x="432" y="84" text-anchor="end">A</text>
<text class="tm" x="470" y="63" text-anchor="middle">B</text>
<text class="tm" x="432" y="104" text-anchor="end">B</text>
<text class="tm" x="490" y="63" text-anchor="middle">C</text>
<text class="tm" x="432" y="124" text-anchor="end">C</text>
<text class="tm" x="510" y="63" text-anchor="middle">D</text>
<text class="tm" x="432" y="144" text-anchor="end">D</text>
<text class="tm" x="530" y="63" text-anchor="middle">E</text>
<text class="tm" x="432" y="164" text-anchor="end">E</text>
<text class="tm" x="550" y="63" text-anchor="middle">F</text>
<text class="tm" x="432" y="184" text-anchor="end">F</text>
<text class="tm" x="570" y="63" text-anchor="middle">G</text>
<text class="tm" x="432" y="204" text-anchor="end">G</text>
<rect class="cell" x="440" y="70" width="20" height="20" data-tip="A[A, A] = 0" data-tip-sub="no self-loop (GCN adds one later)"/>
<rect class="cell" x="460" y="70" width="20" height="20" data-tip="A[A, B] = 1" data-tip-sub="A and B share an edge"/>
<rect class="cell-on a-pop" x="463" y="73" width="14" height="14" rx="3" style="--d:0.70s;pointer-events:none"/>
<rect class="cell" x="480" y="70" width="20" height="20" data-tip="A[A, C] = 1" data-tip-sub="A and C share an edge"/>
<rect class="cell-on a-pop" x="483" y="73" width="14" height="14" rx="3" style="--d:0.87s;pointer-events:none"/>
<rect class="cell" x="500" y="70" width="20" height="20" data-tip="A[A, D] = 1" data-tip-sub="A and D share an edge"/>
<rect class="cell-on a-pop" x="503" y="73" width="14" height="14" rx="3" style="--d:1.04s;pointer-events:none"/>
<rect class="cell" x="520" y="70" width="20" height="20" data-tip="A[A, E] = 0" data-tip-sub="no edge between A and E"/>
<rect class="cell" x="540" y="70" width="20" height="20" data-tip="A[A, F] = 0" data-tip-sub="no edge between A and F"/>
<rect class="cell" x="560" y="70" width="20" height="20" data-tip="A[A, G] = 0" data-tip-sub="no edge between A and G"/>
<rect class="cell" x="440" y="90" width="20" height="20" data-tip="A[B, A] = 1" data-tip-sub="B and A share an edge"/>
<rect class="cell-on a-pop" x="443" y="93" width="14" height="14" rx="3" style="--d:0.70s;pointer-events:none"/>
<rect class="cell" x="460" y="90" width="20" height="20" data-tip="A[B, B] = 0" data-tip-sub="no self-loop (GCN adds one later)"/>
<rect class="cell" x="480" y="90" width="20" height="20" data-tip="A[B, C] = 1" data-tip-sub="B and C share an edge"/>
<rect class="cell-on a-pop" x="483" y="93" width="14" height="14" rx="3" style="--d:1.21s;pointer-events:none"/>
<rect class="cell" x="500" y="90" width="20" height="20" data-tip="A[B, D] = 0" data-tip-sub="no edge between B and D"/>
<rect class="cell" x="520" y="90" width="20" height="20" data-tip="A[B, E] = 0" data-tip-sub="no edge between B and E"/>
<rect class="cell" x="540" y="90" width="20" height="20" data-tip="A[B, F] = 0" data-tip-sub="no edge between B and F"/>
<rect class="cell" x="560" y="90" width="20" height="20" data-tip="A[B, G] = 0" data-tip-sub="no edge between B and G"/>
<rect class="cell" x="440" y="110" width="20" height="20" data-tip="A[C, A] = 1" data-tip-sub="C and A share an edge"/>
<rect class="cell-on a-pop" x="443" y="113" width="14" height="14" rx="3" style="--d:0.87s;pointer-events:none"/>
<rect class="cell" x="460" y="110" width="20" height="20" data-tip="A[C, B] = 1" data-tip-sub="C and B share an edge"/>
<rect class="cell-on a-pop" x="463" y="113" width="14" height="14" rx="3" style="--d:1.21s;pointer-events:none"/>
<rect class="cell" x="480" y="110" width="20" height="20" data-tip="A[C, C] = 0" data-tip-sub="no self-loop (GCN adds one later)"/>
<rect class="cell" x="500" y="110" width="20" height="20" data-tip="A[C, D] = 0" data-tip-sub="no edge between C and D"/>
<rect class="cell" x="520" y="110" width="20" height="20" data-tip="A[C, E] = 1" data-tip-sub="C and E share an edge"/>
<rect class="cell-on a-pop" x="523" y="113" width="14" height="14" rx="3" style="--d:1.38s;pointer-events:none"/>
<rect class="cell" x="540" y="110" width="20" height="20" data-tip="A[C, F] = 0" data-tip-sub="no edge between C and F"/>
<rect class="cell" x="560" y="110" width="20" height="20" data-tip="A[C, G] = 0" data-tip-sub="no edge between C and G"/>
<rect class="cell" x="440" y="130" width="20" height="20" data-tip="A[D, A] = 1" data-tip-sub="D and A share an edge"/>
<rect class="cell-on a-pop" x="443" y="133" width="14" height="14" rx="3" style="--d:1.04s;pointer-events:none"/>
<rect class="cell" x="460" y="130" width="20" height="20" data-tip="A[D, B] = 0" data-tip-sub="no edge between D and B"/>
<rect class="cell" x="480" y="130" width="20" height="20" data-tip="A[D, C] = 0" data-tip-sub="no edge between D and C"/>
<rect class="cell" x="500" y="130" width="20" height="20" data-tip="A[D, D] = 0" data-tip-sub="no self-loop (GCN adds one later)"/>
<rect class="cell" x="520" y="130" width="20" height="20" data-tip="A[D, E] = 0" data-tip-sub="no edge between D and E"/>
<rect class="cell" x="540" y="130" width="20" height="20" data-tip="A[D, F] = 1" data-tip-sub="D and F share an edge"/>
<rect class="cell-on a-pop" x="543" y="133" width="14" height="14" rx="3" style="--d:1.55s;pointer-events:none"/>
<rect class="cell" x="560" y="130" width="20" height="20" data-tip="A[D, G] = 0" data-tip-sub="no edge between D and G"/>
<rect class="cell" x="440" y="150" width="20" height="20" data-tip="A[E, A] = 0" data-tip-sub="no edge between E and A"/>
<rect class="cell" x="460" y="150" width="20" height="20" data-tip="A[E, B] = 0" data-tip-sub="no edge between E and B"/>
<rect class="cell" x="480" y="150" width="20" height="20" data-tip="A[E, C] = 1" data-tip-sub="E and C share an edge"/>
<rect class="cell-on a-pop" x="483" y="153" width="14" height="14" rx="3" style="--d:1.38s;pointer-events:none"/>
<rect class="cell" x="500" y="150" width="20" height="20" data-tip="A[E, D] = 0" data-tip-sub="no edge between E and D"/>
<rect class="cell" x="520" y="150" width="20" height="20" data-tip="A[E, E] = 0" data-tip-sub="no self-loop (GCN adds one later)"/>
<rect class="cell" x="540" y="150" width="20" height="20" data-tip="A[E, F] = 1" data-tip-sub="E and F share an edge"/>
<rect class="cell-on a-pop" x="543" y="153" width="14" height="14" rx="3" style="--d:1.72s;pointer-events:none"/>
<rect class="cell" x="560" y="150" width="20" height="20" data-tip="A[E, G] = 0" data-tip-sub="no edge between E and G"/>
<rect class="cell" x="440" y="170" width="20" height="20" data-tip="A[F, A] = 0" data-tip-sub="no edge between F and A"/>
<rect class="cell" x="460" y="170" width="20" height="20" data-tip="A[F, B] = 0" data-tip-sub="no edge between F and B"/>
<rect class="cell" x="480" y="170" width="20" height="20" data-tip="A[F, C] = 0" data-tip-sub="no edge between F and C"/>
<rect class="cell" x="500" y="170" width="20" height="20" data-tip="A[F, D] = 1" data-tip-sub="F and D share an edge"/>
<rect class="cell-on a-pop" x="503" y="173" width="14" height="14" rx="3" style="--d:1.55s;pointer-events:none"/>
<rect class="cell" x="520" y="170" width="20" height="20" data-tip="A[F, E] = 1" data-tip-sub="F and E share an edge"/>
<rect class="cell-on a-pop" x="523" y="173" width="14" height="14" rx="3" style="--d:1.72s;pointer-events:none"/>
<rect class="cell" x="540" y="170" width="20" height="20" data-tip="A[F, F] = 0" data-tip-sub="no self-loop (GCN adds one later)"/>
<rect class="cell" x="560" y="170" width="20" height="20" data-tip="A[F, G] = 1" data-tip-sub="F and G share an edge"/>
<rect class="cell-on a-pop" x="563" y="173" width="14" height="14" rx="3" style="--d:1.89s;pointer-events:none"/>
<rect class="cell" x="440" y="190" width="20" height="20" data-tip="A[G, A] = 0" data-tip-sub="no edge between G and A"/>
<rect class="cell" x="460" y="190" width="20" height="20" data-tip="A[G, B] = 0" data-tip-sub="no edge between G and B"/>
<rect class="cell" x="480" y="190" width="20" height="20" data-tip="A[G, C] = 0" data-tip-sub="no edge between G and C"/>
<rect class="cell" x="500" y="190" width="20" height="20" data-tip="A[G, D] = 0" data-tip-sub="no edge between G and D"/>
<rect class="cell" x="520" y="190" width="20" height="20" data-tip="A[G, E] = 0" data-tip-sub="no edge between G and E"/>
<rect class="cell" x="540" y="190" width="20" height="20" data-tip="A[G, F] = 1" data-tip-sub="G and F share an edge"/>
<rect class="cell-on a-pop" x="543" y="193" width="14" height="14" rx="3" style="--d:1.89s;pointer-events:none"/>
<rect class="cell" x="560" y="190" width="20" height="20" data-tip="A[G, G] = 0" data-tip-sub="no self-loop (GCN adds one later)"/>
<rect class="cell-hl a-in" x="438" y="68" width="144" height="24" rx="4" style="--d:2.3s"/>
<text class="tm ta" x="622" y="26">FEATURES X</text>
<text class="tm" x="622" y="42">ONE ROW PER NODE</text>
<rect class="cell" x="622" y="70" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="73" width="14" height="14" rx="3" style="--d:1.95s;opacity:0.92" data-tip="xA = [0.9, 0.2, 0.4]" data-tip-sub="three numbers that describe A"/>
<rect class="cell" x="642" y="70" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="73" width="14" height="14" rx="3" style="--d:1.95s;opacity:0.32" data-tip="xA = [0.9, 0.2, 0.4]" data-tip-sub="three numbers that describe A"/>
<rect class="cell" x="662" y="70" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="73" width="14" height="14" rx="3" style="--d:1.95s;opacity:0.49" data-tip="xA = [0.9, 0.2, 0.4]" data-tip-sub="three numbers that describe A"/>
<text class="tm" x="614" y="84" text-anchor="end">A</text>
<rect class="cell" x="622" y="90" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="93" width="14" height="14" rx="3" style="--d:2.00s;opacity:0.83" data-tip="xB = [0.8, 0.1, 0.5]" data-tip-sub="three numbers that describe B"/>
<rect class="cell" x="642" y="90" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="93" width="14" height="14" rx="3" style="--d:2.00s;opacity:0.23" data-tip="xB = [0.8, 0.1, 0.5]" data-tip-sub="three numbers that describe B"/>
<rect class="cell" x="662" y="90" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="93" width="14" height="14" rx="3" style="--d:2.00s;opacity:0.57" data-tip="xB = [0.8, 0.1, 0.5]" data-tip-sub="three numbers that describe B"/>
<text class="tm" x="614" y="104" text-anchor="end">B</text>
<rect class="cell" x="622" y="110" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="113" width="14" height="14" rx="3" style="--d:2.05s;opacity:0.74" data-tip="xC = [0.7, 0.3, 0.6]" data-tip-sub="three numbers that describe C"/>
<rect class="cell" x="642" y="110" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="113" width="14" height="14" rx="3" style="--d:2.05s;opacity:0.41" data-tip="xC = [0.7, 0.3, 0.6]" data-tip-sub="three numbers that describe C"/>
<rect class="cell" x="662" y="110" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="113" width="14" height="14" rx="3" style="--d:2.05s;opacity:0.66" data-tip="xC = [0.7, 0.3, 0.6]" data-tip-sub="three numbers that describe C"/>
<text class="tm" x="614" y="124" text-anchor="end">C</text>
<rect class="cell" x="622" y="130" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="133" width="14" height="14" rx="3" style="--d:2.10s;opacity:0.32" data-tip="xD = [0.2, 0.8, 0.3]" data-tip-sub="three numbers that describe D"/>
<rect class="cell" x="642" y="130" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="133" width="14" height="14" rx="3" style="--d:2.10s;opacity:0.83" data-tip="xD = [0.2, 0.8, 0.3]" data-tip-sub="three numbers that describe D"/>
<rect class="cell" x="662" y="130" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="133" width="14" height="14" rx="3" style="--d:2.10s;opacity:0.41" data-tip="xD = [0.2, 0.8, 0.3]" data-tip-sub="three numbers that describe D"/>
<text class="tm" x="614" y="144" text-anchor="end">D</text>
<rect class="cell" x="622" y="150" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="153" width="14" height="14" rx="3" style="--d:2.15s;opacity:0.41" data-tip="xE = [0.3, 0.9, 0.1]" data-tip-sub="three numbers that describe E"/>
<rect class="cell" x="642" y="150" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="153" width="14" height="14" rx="3" style="--d:2.15s;opacity:0.92" data-tip="xE = [0.3, 0.9, 0.1]" data-tip-sub="three numbers that describe E"/>
<rect class="cell" x="662" y="150" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="153" width="14" height="14" rx="3" style="--d:2.15s;opacity:0.23" data-tip="xE = [0.3, 0.9, 0.1]" data-tip-sub="three numbers that describe E"/>
<text class="tm" x="614" y="164" text-anchor="end">E</text>
<rect class="cell" x="622" y="170" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="173" width="14" height="14" rx="3" style="--d:2.20s;opacity:0.23" data-tip="xF = [0.1, 0.7, 0.2]" data-tip-sub="three numbers that describe F"/>
<rect class="cell" x="642" y="170" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="173" width="14" height="14" rx="3" style="--d:2.20s;opacity:0.74" data-tip="xF = [0.1, 0.7, 0.2]" data-tip-sub="three numbers that describe F"/>
<rect class="cell" x="662" y="170" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="173" width="14" height="14" rx="3" style="--d:2.20s;opacity:0.32" data-tip="xF = [0.1, 0.7, 0.2]" data-tip-sub="three numbers that describe F"/>
<text class="tm" x="614" y="184" text-anchor="end">F</text>
<rect class="cell" x="622" y="190" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="193" width="14" height="14" rx="3" style="--d:2.25s;opacity:0.32" data-tip="xG = [0.2, 0.6, 0.9]" data-tip-sub="three numbers that describe G"/>
<rect class="cell" x="642" y="190" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="193" width="14" height="14" rx="3" style="--d:2.25s;opacity:0.66" data-tip="xG = [0.2, 0.6, 0.9]" data-tip-sub="three numbers that describe G"/>
<rect class="cell" x="662" y="190" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="193" width="14" height="14" rx="3" style="--d:2.25s;opacity:0.92" data-tip="xG = [0.2, 0.6, 0.9]" data-tip-sub="three numbers that describe G"/>
<text class="tm" x="614" y="204" text-anchor="end">G</text>
<g class="a-in" style="--d:2.5s"><text class="t2" x="440" y="246">Row A has a 1 under B, C and D: A's neighbours.</text><text class="t2" x="440" y="266">The matrix is symmetric because edges go both ways.</text></g>
</svg>
</div>
<figcaption>Everything a GNN knows about a graph: who is connected to whom (the adjacency matrix) and what each node looks like (the feature matrix). Hover a cell to read it. This seven-node graph is the running example for the rest of the post.</figcaption>
</figure>
{:/nomarkdown}

Edges can carry features of their own (a bond type, a transfer amount, a timestamp), and they can be directed (a citation points one way) or undirected (a friendship goes both ways). Everything below works for all of these. The examples stick to undirected graphs with node features because they are the easiest to draw.

In practice, nobody stores $$A$$ as a dense matrix. Real graphs are **sparse**: in a social network with a billion users, each user has hundreds of friends, not a billion. Libraries store an **edge list** instead, a $$2 \times E$$ array of (source, target) pairs, so memory grows with the number of edges rather than with $$n^2$$. Keep the matrix picture in mind for the math and the edge list for the code.

Once you have a graph, there are three kinds of question you can ask of it.

{::nomarkdown}
<figure class="viz" id="fig-tasks">
<div class="viz-head"><span class="viz-label">FIG. 2 // THREE KINDS OF QUESTION</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 246" role="img" aria-labelledby="fig-tasks-t fig-tasks-d">
<title id="fig-tasks-t">Three questions you can ask of a graph</title>
<desc id="fig-tasks-d">The same small graph three times. Node classification predicts a label for one highlighted node. Link prediction asks whether a dashed candidate edge between two nodes should exist. Graph classification predicts one label for the whole graph, outlined.</desc>
<g class="a-in" style="--d:0.05s"><rect class="n" x="12" y="12" width="228" height="222" rx="14"/><text class="tm ta" x="26" y="32">PER NODE</text><line class="ge" x1="64" y1="74" x2="120" y2="48"/><line class="ge" x1="120" y1="48" x2="182" y2="72"/><line class="ge" x1="182" y1="72" x2="190" y2="132"/><line class="ge" x1="190" y1="132" x2="126" y2="152"/><line class="ge" x1="126" y1="152" x2="66" y2="130"/><line class="ge" x1="66" y1="130" x2="64" y2="74"/><line class="ge" x1="120" y1="48" x2="126" y2="152"/><circle class="gn" cx="64" cy="74" r="10"/><circle class="gn" cx="120" cy="48" r="10"/><circle class="gn-acc" cx="182" cy="72" r="10"/><circle class="gn" cx="190" cy="132" r="10"/><circle class="gn" cx="126" cy="152" r="10"/><circle class="gn" cx="66" cy="130" r="10"/><text class="t big" x="28" y="196">Node classification</text><text class="t2" x="28" y="216">Which topic is this paper about?</text></g>
<g class="a-in" style="--d:0.2s"><rect class="n" x="266" y="12" width="228" height="222" rx="14"/><text class="tm ta" x="280" y="32">PER PAIR</text><line class="ge" x1="318" y1="74" x2="374" y2="48"/><line class="ge" x1="374" y1="48" x2="436" y2="72"/><line class="ge" x1="436" y1="72" x2="444" y2="132"/><line class="ge" x1="444" y1="132" x2="380" y2="152"/><line class="ge" x1="380" y1="152" x2="320" y2="130"/><line class="ge" x1="320" y1="130" x2="318" y2="74"/><line class="ge" x1="374" y1="48" x2="380" y2="152"/><circle class="gn" cx="318" cy="74" r="10"/><circle class="gn" cx="374" cy="48" r="10"/><circle class="gn-acc" cx="436" cy="72" r="10"/><circle class="gn" cx="444" cy="132" r="10"/><circle class="gn-acc" cx="380" cy="152" r="10"/><circle class="gn" cx="320" cy="130" r="10"/><text class="t big" x="282" y="196">Link prediction</text><text class="t2" x="282" y="216">Will these two users connect?</text></g>
<g class="a-in" style="--d:0.35s"><rect class="n" x="520" y="12" width="228" height="222" rx="14"/><text class="tm ta" x="534" y="32">PER GRAPH</text><line class="ge" x1="572" y1="74" x2="628" y2="48"/><line class="ge" x1="628" y1="48" x2="690" y2="72"/><line class="ge" x1="690" y1="72" x2="698" y2="132"/><line class="ge" x1="698" y1="132" x2="634" y2="152"/><line class="ge" x1="634" y1="152" x2="574" y2="130"/><line class="ge" x1="574" y1="130" x2="572" y2="74"/><line class="ge" x1="628" y1="48" x2="634" y2="152"/><circle class="gn" cx="572" cy="74" r="10"/><circle class="gn" cx="628" cy="48" r="10"/><circle class="gn" cx="690" cy="72" r="10"/><circle class="gn" cx="698" cy="132" r="10"/><circle class="gn" cx="634" cy="152" r="10"/><circle class="gn" cx="574" cy="130" r="10"/><text class="t big" x="536" y="196">Graph classification</text><text class="t2" x="536" y="216">Is this molecule toxic?</text></g>
<circle class="ring a-ping" cx="182" cy="72" r="10" style="--d:1.2s"/>
<g class="a-pop" style="--d:0.8s"><circle class="gn-acc" cx="182" cy="72" r="10"/><text class="gl" x="182" y="72" dy="0.35em">?</text></g>
<path class="drop a-draw" pathLength="1" d="M436 72L380 152" style="--d:1.0s;--t:.5s"/>
<g class="a-pop" style="--d:1.5s"><circle class="gn-acc" cx="408" cy="112" r="10"/><text class="gl" x="408" y="112" dy="0.35em">?</text></g>
<rect class="q-ghost a-in" x="548" y="50" width="174" height="120" rx="16" style="--d:1.6s;stroke-dasharray:4 4"/>
<g class="a-pop" style="--d:1.9s"><circle class="gn-acc" cx="722" cy="50" r="10"/><text class="gl" x="722" y="50" dy="0.35em">?</text></g>
</svg>
</div>
<figcaption>Same graph, three kinds of question. The GNN layers are identical for all three; only the final step that turns node vectors into an answer changes.</figcaption>
</figure>
{:/nomarkdown}

- **Node classification** predicts something about each node: which topic a paper is about, whether an account is a bot, what a protein does.
- **Link prediction** predicts whether an edge should exist between two nodes: friend suggestions, product recommendations, which drug binds to which protein.
- **Graph classification** (or regression) predicts something about a whole graph: whether a molecule is toxic, how soluble it is, whether a cluster of transactions looks like a fraud ring.

The same GNN layers serve all three. Only the last step, the "head" that turns node vectors into an answer, changes. Section 9 shows how.

## 2. Why ordinary neural networks struggle with graphs

The obvious first attempt is to flatten the adjacency matrix and the features into one long vector and feed it to a multilayer perceptron. That fails for two reasons.

The first is size. An MLP has a fixed number of inputs, so it can only read graphs with exactly the number of nodes it was built for. Molecules have 9 atoms, or 90.

The second is subtler and more important. **The numbering of the nodes is arbitrary.** Nothing about a graph says which node is node 1. Renumber the nodes and you have the same graph, but a different matrix:

{::nomarkdown}
<figure class="viz" id="fig-perm">
<div class="viz-head"><span class="viz-label">FIG. 3 // THE NUMBERING PROBLEM</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 240" role="img" aria-labelledby="fig-perm-t fig-perm-d">
<title id="fig-perm-t">Renumbering the nodes changes the matrix but not the graph</title>
<desc id="fig-perm-d">The same four-node graph, a triangle with a tail, numbered two different ways. The two adjacency matrices differ in most cells, and so do the flattened vectors an ordinary network would read, even though the graph is identical.</desc>
<g class="a-in" style="--d:0.05s"><text class="tm ta" x="24" y="26">NUMBERING 1</text><line class="ge" x1="84" y1="80" x2="174" y2="64"/><line class="ge" x1="174" y1="64" x2="154" y2="150"/><line class="ge" x1="84" y1="80" x2="154" y2="150"/><line class="ge" x1="154" y1="150" x2="70" y2="176"/><circle class="gn" cx="84" cy="80" r="15"/><circle class="gn" cx="174" cy="64" r="15"/><circle class="gn" cx="154" cy="150" r="15"/><circle class="gn" cx="70" cy="176" r="15"/></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="84" y="80" dy="0.35em">1</text></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="174" y="64" dy="0.35em">2</text></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="154" y="150" dy="0.35em">3</text></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="70" y="176" dy="0.35em">4</text></g>
<g class="a-in" style="--d:0.25s"><text class="tm" x="239" y="53" text-anchor="middle">1</text><text class="tm" x="220" y="75" text-anchor="end">1</text><text class="tm" x="261" y="53" text-anchor="middle">2</text><text class="tm" x="220" y="97" text-anchor="end">2</text><text class="tm" x="283" y="53" text-anchor="middle">3</text><text class="tm" x="220" y="119" text-anchor="end">3</text><text class="tm" x="305" y="53" text-anchor="middle">4</text><text class="tm" x="220" y="141" text-anchor="end">4</text><rect class="cell" x="228" y="60" width="22" height="22"/><rect class="cell" x="250" y="60" width="22" height="22"/><rect class="cell-on" x="253" y="63" width="16" height="16" rx="3"/><rect class="cell" x="272" y="60" width="22" height="22"/><rect class="cell-on" x="275" y="63" width="16" height="16" rx="3"/><rect class="cell" x="294" y="60" width="22" height="22"/><rect class="cell" x="228" y="82" width="22" height="22"/><rect class="cell-on" x="231" y="85" width="16" height="16" rx="3"/><rect class="cell" x="250" y="82" width="22" height="22"/><rect class="cell" x="272" y="82" width="22" height="22"/><rect class="cell-on" x="275" y="85" width="16" height="16" rx="3"/><rect class="cell" x="294" y="82" width="22" height="22"/><rect class="cell" x="228" y="104" width="22" height="22"/><rect class="cell-on" x="231" y="107" width="16" height="16" rx="3"/><rect class="cell" x="250" y="104" width="22" height="22"/><rect class="cell-on" x="253" y="107" width="16" height="16" rx="3"/><rect class="cell" x="272" y="104" width="22" height="22"/><rect class="cell" x="294" y="104" width="22" height="22"/><rect class="cell-on" x="297" y="107" width="16" height="16" rx="3"/><rect class="cell" x="228" y="126" width="22" height="22"/><rect class="cell" x="250" y="126" width="22" height="22"/><rect class="cell" x="272" y="126" width="22" height="22"/><rect class="cell-on" x="275" y="129" width="16" height="16" rx="3"/><rect class="cell" x="294" y="126" width="22" height="22"/><text class="tm" x="208" y="174">AS ONE VECTOR</text><text class="tk" x="208" y="192">0110 1010 1101 0010</text></g>
<g class="a-in" style="--d:1.0s"><text class="tm ta" x="414" y="26">NUMBERING 2, SAME GRAPH</text><line class="ge" x1="474" y1="80" x2="564" y2="64"/><line class="ge" x1="564" y1="64" x2="544" y2="150"/><line class="ge" x1="474" y1="80" x2="544" y2="150"/><line class="ge" x1="544" y1="150" x2="460" y2="176"/><circle class="gn" cx="474" cy="80" r="15"/><circle class="gn" cx="564" cy="64" r="15"/><circle class="gn" cx="544" cy="150" r="15"/><circle class="gn" cx="460" cy="176" r="15"/></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="474" y="80" dy="0.35em">3</text></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="564" y="64" dy="0.35em">1</text></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="544" y="150" dy="0.35em">4</text></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="460" y="176" dy="0.35em">2</text></g>
<g class="a-in" style="--d:1.2s"><text class="tm" x="629" y="53" text-anchor="middle">1</text><text class="tm" x="610" y="75" text-anchor="end">1</text><text class="tm" x="651" y="53" text-anchor="middle">2</text><text class="tm" x="610" y="97" text-anchor="end">2</text><text class="tm" x="673" y="53" text-anchor="middle">3</text><text class="tm" x="610" y="119" text-anchor="end">3</text><text class="tm" x="695" y="53" text-anchor="middle">4</text><text class="tm" x="610" y="141" text-anchor="end">4</text><rect class="cell" x="618" y="60" width="22" height="22"/><rect class="cell" x="640" y="60" width="22" height="22"/><rect class="cell" x="662" y="60" width="22" height="22"/><rect class="cell-on" x="665" y="63" width="16" height="16" rx="3"/><rect class="cell" x="684" y="60" width="22" height="22"/><rect class="cell-on" x="687" y="63" width="16" height="16" rx="3"/><rect class="cell" x="618" y="82" width="22" height="22"/><rect class="cell" x="640" y="82" width="22" height="22"/><rect class="cell" x="662" y="82" width="22" height="22"/><rect class="cell" x="684" y="82" width="22" height="22"/><rect class="cell-on" x="687" y="85" width="16" height="16" rx="3"/><rect class="cell" x="618" y="104" width="22" height="22"/><rect class="cell-on" x="621" y="107" width="16" height="16" rx="3"/><rect class="cell" x="640" y="104" width="22" height="22"/><rect class="cell" x="662" y="104" width="22" height="22"/><rect class="cell" x="684" y="104" width="22" height="22"/><rect class="cell-on" x="687" y="107" width="16" height="16" rx="3"/><rect class="cell" x="618" y="126" width="22" height="22"/><rect class="cell-on" x="621" y="129" width="16" height="16" rx="3"/><rect class="cell" x="640" y="126" width="22" height="22"/><rect class="cell-on" x="643" y="129" width="16" height="16" rx="3"/><rect class="cell" x="662" y="126" width="22" height="22"/><rect class="cell-on" x="665" y="129" width="16" height="16" rx="3"/><rect class="cell" x="684" y="126" width="22" height="22"/><text class="tm" x="598" y="174">AS ONE VECTOR</text><text class="tk" x="598" y="192">0011 0001 1001 1110</text></g>
<rect class="a-in" x="641" y="61" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="685" y="61" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="619" y="83" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="663" y="83" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="685" y="83" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="641" y="105" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="619" y="127" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="641" y="127" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<line class="ln-faint" x1="380" y1="40" x2="380" y2="200"/>
<text class="t2 strong a-in" x="380" y="226" text-anchor="middle" style="--d:2.3s">Same graph, different input. A graph model has to give the same answer for both.</text>
</svg>
</div>
<figcaption>Node numbers are labels of convenience. Renumber them and the adjacency matrix changes in 8 of its 16 cells, so an ordinary network reading it as a flat vector sees a different input. The cells outlined in red are the ones that changed.</figcaption>
</figure>
{:/nomarkdown}

An MLP would see two unrelated inputs and could give two different answers. With $$n$$ nodes there are $$n!$$ ways to number the same graph (more than 3.6 million for just 10 nodes), and you cannot train on all of them.

So a model for graphs has to be built so that the numbering cannot matter. Formally, if $$P$$ is a permutation matrix that renumbers the nodes, a node-level model $$f$$ must be **permutation equivariant**: renumbering the input renumbers the output in the same way, and changes nothing else.

$$
f\big(PX,\; PAP^\top\big) = P\, f(X, A)
$$

A graph-level model $$g$$ must be **permutation invariant**: $$g(PX, PAP^\top) = g(X, A)$$.

That sounds abstract, but the fix is something you already know. Sums, means and maxima do not care about order: $$3 + 5 + 1$$ is the same as $$1 + 3 + 5$$. Build a model out of operations like these and the numbering stops mattering by construction. That is exactly what message passing does.

## 3. Message passing: the one idea behind every GNN

Imagine guessing something about a person you know nothing about, except who their friends are. You would look at the friends. Connected things tend to be alike, a property called **homophily**: papers cite papers on similar topics, fraudulent accounts share devices with other fraudulent accounts, and an atom next to an oxygen behaves differently from one next to a carbon.

A GNN layer turns that intuition into a computation. Every node, at the same time, runs three steps:

1. **Send.** Each node sends its current vector to its neighbours. (In general a message can be a learned function of the sender, the receiver and the edge between them.)
2. **Aggregate.** Each node combines the messages it received with a function that ignores their order: a sum, a mean or a max.
3. **Update.** Each node mixes that aggregate with its own current vector, through learned weights and a nonlinearity, to get its new vector.
{: .steps}

Step through it on the example graph. Here each node holds a single number, so you can follow the arithmetic.

{::nomarkdown}
<figure class="viz viz-steps" id="fig-mp" data-dur="4200">
<div class="viz-head"><span class="viz-label">FIG. 4 // MESSAGE PASSING, ONE STEP AT A TIME</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 290" role="img" aria-labelledby="fig-mp-t fig-mp-d">
<title id="fig-mp-t">One round of message passing on a seven-node graph</title>
<desc id="fig-mp-d">Node A has neighbours B, C and D with values 0.8, 0.2 and 0.5. They send their values to A, A takes the mean, 0.5, and updates itself to ReLU(0.5 times 0.3 plus 1.0 times 0.5) = 0.65. Then every node does the same at once: A goes from 0.30 to 0.65, B goes from 0.80 to 0.65, C goes from 0.20 to 0.77, D goes from 0.50 to 0.45, E goes from 0.90 to 0.60, F goes from 0.10 to 0.72, G goes from 0.60 to 0.40.</desc>
<text class="tm ta" x="24" y="26">EACH NODE HOLDS ONE NUMBER</text>
<line class="ge" x1="110" y1="167" x2="80" y2="92"/>
<line class="ge" x1="110" y1="167" x2="190" y2="102"/>
<line class="ge" x1="110" y1="167" x2="150" y2="247"/>
<line class="ge" x1="80" y1="92" x2="190" y2="102"/>
<line class="ge" x1="190" y1="102" x2="285" y2="122"/>
<line class="ge" x1="150" y1="247" x2="265" y2="222"/>
<line class="ge" x1="285" y1="122" x2="265" y2="222"/>
<line class="ge" x1="265" y1="222" x2="350" y2="252"/>
<line class="ge-acc is-off" x1="110" y1="167" x2="80" y2="92" data-on="0-2"/>
<line class="ge-acc is-off" x1="110" y1="167" x2="190" y2="102" data-on="0-2"/>
<line class="ge-acc is-off" x1="110" y1="167" x2="150" y2="247" data-on="0-2"/>
<line class="ge-acc" x1="110" y1="167" x2="80" y2="92" data-on="3"/>
<line class="ge-acc" x1="110" y1="167" x2="190" y2="102" data-on="3"/>
<line class="ge-acc" x1="110" y1="167" x2="150" y2="247" data-on="3"/>
<line class="ge-acc" x1="80" y1="92" x2="190" y2="102" data-on="3"/>
<line class="ge-acc" x1="190" y1="102" x2="285" y2="122" data-on="3"/>
<line class="ge-acc" x1="150" y1="247" x2="265" y2="222" data-on="3"/>
<line class="ge-acc" x1="285" y1="122" x2="265" y2="222" data-on="3"/>
<line class="ge-acc" x1="265" y1="222" x2="350" y2="252" data-on="3"/>
<circle class="q-ghost is-off" cx="110" cy="167" r="27" style="stroke-dasharray:3 3" data-on="0-2"/>
<circle class="gn" cx="110" cy="167" r="20"/>
<circle class="gn-acc" cx="110" cy="167" r="20" data-on="2-3"/>
<text class="gv" x="110" y="167" dy="0.35em" data-text="[&quot;0.30&quot;, &quot;0.30&quot;, &quot;0.65&quot;, &quot;0.65&quot;]">0.65</text>
<text class="tm" x="85" y="187.6" text-anchor="middle" style="fill:var(--text-primary)">A</text>
<circle class="gn" cx="80" cy="92" r="20"/>
<circle class="gn-acc" cx="80" cy="92" r="20" data-on="3"/>
<text class="gv" x="80" y="92" dy="0.35em" data-text="[&quot;0.80&quot;, &quot;0.80&quot;, &quot;0.80&quot;, &quot;0.65&quot;]">0.65</text>
<text class="tm" x="54.4" y="80.4" text-anchor="middle" style="fill:var(--text-primary)">B</text>
<circle class="gn" cx="190" cy="102" r="20"/>
<circle class="gn-acc" cx="190" cy="102" r="20" data-on="3"/>
<text class="gv" x="190" y="102" dy="0.35em" data-text="[&quot;0.20&quot;, &quot;0.20&quot;, &quot;0.20&quot;, &quot;0.77&quot;]">0.77</text>
<text class="tm" x="213.5" y="87.4" text-anchor="middle" style="fill:var(--text-primary)">C</text>
<circle class="gn" cx="150" cy="247" r="20"/>
<circle class="gn-acc" cx="150" cy="247" r="20" data-on="3"/>
<text class="gv" x="150" y="247" dy="0.35em" data-text="[&quot;0.50&quot;, &quot;0.50&quot;, &quot;0.50&quot;, &quot;0.45&quot;]">0.45</text>
<text class="tm" x="132.6" y="275.4" text-anchor="middle" style="fill:var(--text-primary)">D</text>
<circle class="gn" cx="285" cy="122" r="20"/>
<circle class="gn-acc" cx="285" cy="122" r="20" data-on="3"/>
<text class="gv" x="285" y="122" dy="0.35em" data-text="[&quot;0.90&quot;, &quot;0.90&quot;, &quot;0.90&quot;, &quot;0.60&quot;]">0.60</text>
<text class="tm" x="309.6" y="108.9" text-anchor="middle" style="fill:var(--text-primary)">E</text>
<circle class="gn" cx="265" cy="222" r="20"/>
<circle class="gn-acc" cx="265" cy="222" r="20" data-on="3"/>
<text class="gv" x="265" y="222" dy="0.35em" data-text="[&quot;0.10&quot;, &quot;0.10&quot;, &quot;0.10&quot;, &quot;0.72&quot;]">0.72</text>
<text class="tm" x="271.5" y="255.3" text-anchor="middle" style="fill:var(--text-primary)">F</text>
<circle class="gn" cx="350" cy="252" r="20"/>
<circle class="gn-acc" cx="350" cy="252" r="20" data-on="3"/>
<text class="gv" x="350" y="252" dy="0.35em" data-text="[&quot;0.60&quot;, &quot;0.60&quot;, &quot;0.60&quot;, &quot;0.40&quot;]">0.40</text>
<text class="tm" x="378.3" y="266" text-anchor="middle" style="fill:var(--text-primary)">G</text>
<circle class="packet a-travel is-off" cx="88.2" cy="112.4" r="4" style="--dx:12.9px;--dy:32.3px;--t:1.5s;--d:0.00s" data-on="0"/>
<circle class="packet a-travel is-off" cx="172.9" cy="115.9" r="4" style="--dx:-44.3px;--dy:36.0px;--t:1.5s;--d:0.15s" data-on="0"/>
<circle class="packet a-travel is-off" cx="140.2" cy="227.3" r="4" style="--dx:-19.4px;--dy:-38.9px;--t:1.5s;--d:0.30s" data-on="0"/>
<circle class="packet a-travel" cx="101.8" cy="146.6" r="4" style="--dx:-12.9px;--dy:-32.3px;--t:1.5s;--d:0.00s" data-on="3"/>
<circle class="packet a-travel" cx="88.2" cy="112.4" r="4" style="--dx:12.9px;--dy:32.3px;--t:1.5s;--d:0.40s" data-on="3"/>
<circle class="packet a-travel" cx="127.1" cy="153.1" r="4" style="--dx:44.3px;--dy:-36.0px;--t:1.5s;--d:0.10s" data-on="3"/>
<circle class="packet a-travel" cx="172.9" cy="115.9" r="4" style="--dx:-44.3px;--dy:36.0px;--t:1.5s;--d:0.50s" data-on="3"/>
<circle class="packet a-travel" cx="119.8" cy="186.7" r="4" style="--dx:19.4px;--dy:38.9px;--t:1.5s;--d:0.20s" data-on="3"/>
<circle class="packet a-travel" cx="140.2" cy="227.3" r="4" style="--dx:-19.4px;--dy:-38.9px;--t:1.5s;--d:0.60s" data-on="3"/>
<circle class="packet a-travel" cx="101.9" cy="94" r="4" style="--dx:64.2px;--dy:5.8px;--t:1.5s;--d:0.30s" data-on="3"/>
<circle class="packet a-travel" cx="168.1" cy="100" r="4" style="--dx:-64.2px;--dy:-5.8px;--t:1.5s;--d:0.70s" data-on="3"/>
<circle class="packet a-travel" cx="211.5" cy="106.5" r="4" style="--dx:50.0px;--dy:10.5px;--t:1.5s;--d:0.40s" data-on="3"/>
<circle class="packet a-travel" cx="263.5" cy="117.5" r="4" style="--dx:-50.0px;--dy:-10.5px;--t:1.5s;--d:0.80s" data-on="3"/>
<circle class="packet a-travel" cx="171.5" cy="242.3" r="4" style="--dx:70.0px;--dy:-15.2px;--t:1.5s;--d:0.50s" data-on="3"/>
<circle class="packet a-travel" cx="243.5" cy="226.7" r="4" style="--dx:-70.0px;--dy:15.2px;--t:1.5s;--d:0.90s" data-on="3"/>
<circle class="packet a-travel" cx="280.7" cy="143.6" r="4" style="--dx:-11.0px;--dy:54.9px;--t:1.5s;--d:0.60s" data-on="3"/>
<circle class="packet a-travel" cx="269.3" cy="200.4" r="4" style="--dx:11.0px;--dy:-54.9px;--t:1.5s;--d:1.00s" data-on="3"/>
<circle class="packet a-travel" cx="285.7" cy="229.3" r="4" style="--dx:41.6px;--dy:14.7px;--t:1.5s;--d:0.70s" data-on="3"/>
<circle class="packet a-travel" cx="329.3" cy="244.7" r="4" style="--dx:-41.6px;--dy:-14.7px;--t:1.5s;--d:1.10s" data-on="3"/>
<rect class="n" x="420" y="40" width="320" height="236" rx="12"/>
<text class="tm ta" x="440" y="68" data-text="[&quot;STEP 1 // SEND&quot;, &quot;STEP 2 // AGGREGATE&quot;, &quot;STEP 3 // UPDATE&quot;, &quot;EVERY NODE, SAME RULE, AT ONCE&quot;]">EVERY NODE, SAME RULE, AT ONCE</text>
<text class="t2 is-off" x="440" y="102" data-on="0-2">message from B</text>
<text class="t is-off" x="716" y="102" text-anchor="end" data-on="0-2">0.80</text>
<text class="t2 is-off" x="440" y="126" data-on="0-2">message from C</text>
<text class="t is-off" x="716" y="126" text-anchor="end" data-on="0-2">0.20</text>
<text class="t2 is-off" x="440" y="150" data-on="0-2">message from D</text>
<text class="t is-off" x="716" y="150" text-anchor="end" data-on="0-2">0.50</text>
<line class="ln-faint is-off" x1="440" y1="164" x2="716" y2="164" data-on="1-2"/>
<text class="t2 is-off" x="440" y="184" data-on="1-2">mean of the three</text>
<text class="t is-off" x="716" y="184" text-anchor="end" data-on="1-2">0.50</text>
<text class="tm is-off" x="440" y="220" data-on="2">ReLU(0.5 × 0.30 + 1.0 × 0.50)</text>
<text class="t2 is-off" x="440" y="248" data-on="2">new value of A</text>
<text class="t big is-off" x="716" y="248" text-anchor="end" style="fill:#e8b84e" data-on="2">0.65</text>
<text class="tm" x="440" y="96" data-on="3">NODE</text>
<text class="tm" x="550" y="96" text-anchor="end" data-on="3">BEFORE</text>
<text class="tm" x="716" y="96" text-anchor="end" data-on="3">AFTER 1 LAYER</text>
<text class="t" x="440" y="120" data-on="3">A</text>
<text class="t2" x="550" y="120" text-anchor="end" data-on="3">0.30</text>
<text class="t" x="716" y="120" text-anchor="end" data-on="3">0.65</text>
<text class="t" x="440" y="141" data-on="3">B</text>
<text class="t2" x="550" y="141" text-anchor="end" data-on="3">0.80</text>
<text class="t" x="716" y="141" text-anchor="end" data-on="3">0.65</text>
<text class="t" x="440" y="162" data-on="3">C</text>
<text class="t2" x="550" y="162" text-anchor="end" data-on="3">0.20</text>
<text class="t" x="716" y="162" text-anchor="end" data-on="3">0.77</text>
<text class="t" x="440" y="183" data-on="3">D</text>
<text class="t2" x="550" y="183" text-anchor="end" data-on="3">0.50</text>
<text class="t" x="716" y="183" text-anchor="end" data-on="3">0.45</text>
<text class="t" x="440" y="204" data-on="3">E</text>
<text class="t2" x="550" y="204" text-anchor="end" data-on="3">0.90</text>
<text class="t" x="716" y="204" text-anchor="end" data-on="3">0.60</text>
<text class="t" x="440" y="225" data-on="3">F</text>
<text class="t2" x="550" y="225" text-anchor="end" data-on="3">0.10</text>
<text class="t" x="716" y="225" text-anchor="end" data-on="3">0.72</text>
<text class="t" x="440" y="246" data-on="3">G</text>
<text class="t2" x="550" y="246" text-anchor="end" data-on="3">0.60</text>
<text class="t" x="716" y="246" text-anchor="end" data-on="3">0.40</text>
</svg>
</div>
<div class="viz-ctl" role="group" aria-label="Message passing steps"><button type="button" data-go="0" data-note="Send. Every neighbour of A passes its current value along the edge: B sends 0.80, C sends 0.20 and D sends 0.50. (Every other node is doing the same, but watch A for now.)" aria-pressed="false">1 · Send</button><button type="button" data-go="1" data-note="Aggregate. A combines what arrived with a function that ignores order. Here it is the mean: (0.80 + 0.20 + 0.50) / 3 = 0.50. Any order of arrival gives the same answer." aria-pressed="false">2 · Aggregate</button><button type="button" data-go="2" data-note="Update. A mixes the aggregate with its own old value using two learned weights and a ReLU: 0.5 × 0.30 + 1.0 × 0.50 = 0.65. That is A&#x27;s new value." aria-pressed="false">3 · Update</button><button type="button" data-go="3" data-note="Every node runs the same three steps at the same time, with the same two weights. Eight edges carry sixteen messages, and every node ends with a new value. That is one GNN layer." aria-pressed="true">4 · Every node</button></div>
<p class="viz-note" aria-live="polite">Every node runs the same three steps at the same time, with the same two weights. Eight edges carry sixteen messages, and every node ends with a new value. That is one GNN layer.</p>
<figcaption>One layer, in slow motion. Use the buttons to step through it, or press play. In a real GNN each value is a vector and each weight is a matrix, but the three steps are exactly these.</figcaption>
</figure>
{:/nomarkdown}

Written out for a node $$v$$ with neighbours $$N(v)$$, the layer in the figure is:

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( W_{\text{self}}\, \mathbf{h}_v^{(k)} \;+\; W_{\text{neigh}}\, \frac{1}{\lvert N(v) \rvert} \sum_{u \in N(v)} \mathbf{h}_u^{(k)} \Big)
$$

$$\mathbf{h}_v^{(k)}$$ is node $$v$$'s vector after $$k$$ layers, starting from its input features, $$\mathbf{h}_v^{(0)} = \mathbf{x}_v$$. $$W_{\text{self}}$$ and $$W_{\text{neigh}}$$ are weight matrices learned by gradient descent, and $$\sigma$$ is a nonlinearity such as ReLU. In the figure the vectors have one dimension, so the two matrices shrink to the numbers 0.5 and 1.0.

The general form, from the message passing framework of Gilmer et al. (2017), replaces each piece with any learnable function:

$$
\mathbf{h}_v^{(k+1)} = \phi\Big( \mathbf{h}_v^{(k)},\; \bigoplus_{u \in N(v)} \psi\big(\mathbf{h}_u^{(k)}, \mathbf{h}_v^{(k)}, \mathbf{e}_{uv}\big) \Big)
$$

Here $$\psi$$ builds a message (and can use the edge's features $$\mathbf{e}_{uv}$$), $$\bigoplus$$ is the order-independent aggregator, and $$\phi$$ is the update. Almost every GNN you will read about is a particular choice of these three functions.

<div class="key-idea">
  <span class="key-label">In one sentence</span>
  <p>A GNN layer lets every node collect its neighbours' vectors, combine them with an order-independent function, and mix the result into its own vector using weights shared by every node.</p>
</div>

The last part, **shared weights**, is what makes it work. A convolution slides the same small filter over every position of an image; a GNN layer applies the same $$W_{\text{self}}$$ and $$W_{\text{neigh}}$$ at every node. The number of parameters does not depend on the size of the graph, so a model trained on small molecules can run on large ones, and a model trained on last month's transaction graph can run on this month's.

## 4. Stacking layers: how far a node can see

One layer mixes in information from direct neighbours. Add a second layer, and the neighbours you read from have already mixed in *their* neighbours, so information from two hops away reaches you. With $$k$$ layers, a node's vector depends on everything within $$k$$ hops. That region is its **receptive field**.

The clearest way to see what a GNN computes for one node is to unroll it into a **computation tree**: the node at the root, its neighbours below it, their neighbours below them, one level per layer.

{::nomarkdown}
<figure class="viz viz-steps" id="fig-receptive" data-dur="3600">
<div class="viz-head"><span class="viz-label">FIG. 5 // EACH LAYER WIDENS THE VIEW</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="fig-receptive-t fig-receptive-d">
<title id="fig-receptive-t">How the receptive field of node A grows with each layer</title>
<desc id="fig-receptive-d">Left, the seven-node graph with the nodes A can hear from highlighted: after 0 layers only A, after 1 layer A, B, C and D, after 2 layers also E and F, after 3 layers all seven. Right, A's computation tree, with one level per layer: 1 node, then 3, then 7, then 19.</desc>
<text class="tm ta" x="20" y="26">THE GRAPH</text>
<text class="tm ta" x="430" y="26">A'S COMPUTATION TREE</text>
<line class="ge" x1="90" y1="161" x2="60" y2="86"/>
<line class="ge" x1="90" y1="161" x2="170" y2="96"/>
<line class="ge" x1="90" y1="161" x2="130" y2="241"/>
<line class="ge" x1="60" y1="86" x2="170" y2="96"/>
<line class="ge" x1="170" y1="96" x2="265" y2="116"/>
<line class="ge" x1="130" y1="241" x2="245" y2="216"/>
<line class="ge" x1="265" y1="116" x2="245" y2="216"/>
<line class="ge" x1="245" y1="216" x2="330" y2="246"/>
<line class="ge-acc" x1="90" y1="161" x2="60" y2="86" data-on="1+"/>
<line class="ge-acc" x1="90" y1="161" x2="170" y2="96" data-on="1+"/>
<line class="ge-acc" x1="90" y1="161" x2="130" y2="241" data-on="1+"/>
<line class="ge-acc" x1="60" y1="86" x2="170" y2="96" data-on="1+"/>
<line class="ge-acc" x1="170" y1="96" x2="265" y2="116" data-on="2+"/>
<line class="ge-acc" x1="130" y1="241" x2="245" y2="216" data-on="2+"/>
<line class="ge-acc" x1="265" y1="116" x2="245" y2="216" data-on="2+"/>
<line class="ge-acc" x1="245" y1="216" x2="330" y2="246" data-on="3+"/>
<circle class="gn" cx="90" cy="161" r="17"/>
<circle class="gn-acc" cx="90" cy="161" r="17" data-on="0+"/>
<text class="gl" x="90" y="161" dy="0.35em">A</text>
<circle class="gn" cx="60" cy="86" r="17"/>
<circle class="gn-acc" cx="60" cy="86" r="17" data-on="1+"/>
<text class="gl" x="60" y="86" dy="0.35em">B</text>
<circle class="gn" cx="170" cy="96" r="17"/>
<circle class="gn-acc" cx="170" cy="96" r="17" data-on="1+"/>
<text class="gl" x="170" y="96" dy="0.35em">C</text>
<circle class="gn" cx="130" cy="241" r="17"/>
<circle class="gn-acc" cx="130" cy="241" r="17" data-on="1+"/>
<text class="gl" x="130" y="241" dy="0.35em">D</text>
<circle class="gn" cx="265" cy="116" r="17"/>
<circle class="gn-acc" cx="265" cy="116" r="17" data-on="2+"/>
<text class="gl" x="265" y="116" dy="0.35em">E</text>
<circle class="gn" cx="245" cy="216" r="17"/>
<circle class="gn-acc" cx="245" cy="216" r="17" data-on="2+"/>
<text class="gl" x="245" y="216" dy="0.35em">F</text>
<circle class="gn" cx="330" cy="246" r="17"/>
<circle class="gn-acc" cx="330" cy="246" r="17" data-on="3+"/>
<text class="gl" x="330" y="246" dy="0.35em">G</text>
<text class="tm ta" x="20" y="292" data-text="[&quot;A HAS HEARD FROM 1 OF 7 NODES&quot;, &quot;A HAS HEARD FROM 4 OF 7 NODES&quot;, &quot;A HAS HEARD FROM 6 OF 7 NODES&quot;, &quot;A HAS HEARD FROM ALL 7 NODES&quot;]">A HAS HEARD FROM ALL 7 NODES</text>
<text class="tm" x="404" y="66" data-on="0+">L0</text>
<text class="tm" x="404" y="126" data-on="1+">L1</text>
<text class="tm" x="404" y="186" data-on="2+">L2</text>
<text class="tm" x="404" y="248" data-on="3+">L3</text>
<line class="ln" x1="588" y1="75" x2="482" y2="109" data-on="1+"/>
<line class="ln" x1="588" y1="75" x2="588" y2="109" data-on="1+"/>
<line class="ln" x1="588" y1="75" x2="694" y2="109" data-on="1+"/>
<line class="ln" x1="482" y1="135" x2="440" y2="170" data-on="2+"/>
<line class="ln" x1="482" y1="135" x2="489.3" y2="170" data-on="2+"/>
<line class="ln" x1="588" y1="135" x2="538.6" y2="170" data-on="2+"/>
<line class="ln" x1="588" y1="135" x2="587.9" y2="170" data-on="2+"/>
<line class="ln" x1="588" y1="135" x2="637.2" y2="170" data-on="2+"/>
<line class="ln" x1="694" y1="135" x2="686.5" y2="170" data-on="2+"/>
<line class="ln" x1="694" y1="135" x2="735.8" y2="170" data-on="2+"/>
<line class="ln" x1="440" y1="194" x2="437" y2="237" data-on="3"/>
<line class="ln" x1="440" y1="194" x2="453.6" y2="237" data-on="3"/>
<line class="ln" x1="440" y1="194" x2="470.2" y2="237" data-on="3"/>
<line class="ln" x1="489.3" y1="194" x2="486.8" y2="237" data-on="3"/>
<line class="ln" x1="489.3" y1="194" x2="503.4" y2="237" data-on="3"/>
<line class="ln" x1="489.3" y1="194" x2="520.1" y2="237" data-on="3"/>
<line class="ln" x1="538.6" y1="194" x2="536.7" y2="237" data-on="3"/>
<line class="ln" x1="538.6" y1="194" x2="553.3" y2="237" data-on="3"/>
<line class="ln" x1="538.6" y1="194" x2="569.9" y2="237" data-on="3"/>
<line class="ln" x1="587.9" y1="194" x2="586.5" y2="237" data-on="3"/>
<line class="ln" x1="587.9" y1="194" x2="603.1" y2="237" data-on="3"/>
<line class="ln" x1="637.2" y1="194" x2="619.7" y2="237" data-on="3"/>
<line class="ln" x1="637.2" y1="194" x2="636.3" y2="237" data-on="3"/>
<line class="ln" x1="686.5" y1="194" x2="652.9" y2="237" data-on="3"/>
<line class="ln" x1="686.5" y1="194" x2="669.6" y2="237" data-on="3"/>
<line class="ln" x1="686.5" y1="194" x2="686.2" y2="237" data-on="3"/>
<line class="ln" x1="735.8" y1="194" x2="702.8" y2="237" data-on="3"/>
<line class="ln" x1="735.8" y1="194" x2="719.4" y2="237" data-on="3"/>
<line class="ln" x1="735.8" y1="194" x2="736" y2="237" data-on="3"/>
<circle class="gn-acc" cx="588" cy="62" r="13"/>
<text class="gl" x="588" y="62" dy="0.35em">A</text>
<circle class="gn" cx="482" cy="122" r="13" data-on="1+"/>
<text class="gl" x="482" y="122" dy="0.35em" data-on="1+">B</text>
<circle class="gn" cx="588" cy="122" r="13" data-on="1+"/>
<text class="gl" x="588" y="122" dy="0.35em" data-on="1+">C</text>
<circle class="gn" cx="694" cy="122" r="13" data-on="1+"/>
<text class="gl" x="694" y="122" dy="0.35em" data-on="1+">D</text>
<circle class="gn-acc" cx="440" cy="182" r="12" data-on="2+"/>
<text class="gl" x="440" y="182" dy="0.35em" style="font-size:12px" data-on="2+">A</text>
<circle class="gn" cx="489.3" cy="182" r="12" data-on="2+"/>
<text class="gl" x="489.3" y="182" dy="0.35em" style="font-size:12px" data-on="2+">C</text>
<circle class="gn-acc" cx="538.6" cy="182" r="12" data-on="2+"/>
<text class="gl" x="538.6" y="182" dy="0.35em" style="font-size:12px" data-on="2+">A</text>
<circle class="gn" cx="587.9" cy="182" r="12" data-on="2+"/>
<text class="gl" x="587.9" y="182" dy="0.35em" style="font-size:12px" data-on="2+">B</text>
<circle class="gn" cx="637.2" cy="182" r="12" data-on="2+"/>
<text class="gl" x="637.2" y="182" dy="0.35em" style="font-size:12px" data-on="2+">E</text>
<circle class="gn-acc" cx="686.5" cy="182" r="12" data-on="2+"/>
<text class="gl" x="686.5" y="182" dy="0.35em" style="font-size:12px" data-on="2+">A</text>
<circle class="gn" cx="735.8" cy="182" r="12" data-on="2+"/>
<text class="gl" x="735.8" y="182" dy="0.35em" style="font-size:12px" data-on="2+">F</text>
<circle class="gn" cx="437" cy="244" r="7" data-on="3"/>
<text class="gl" x="437" y="244" dy="0.35em" style="font-size:8px" data-on="3">B</text>
<circle class="gn" cx="453.6" cy="244" r="7" data-on="3"/>
<text class="gl" x="453.6" y="244" dy="0.35em" style="font-size:8px" data-on="3">C</text>
<circle class="gn" cx="470.2" cy="244" r="7" data-on="3"/>
<text class="gl" x="470.2" y="244" dy="0.35em" style="font-size:8px" data-on="3">D</text>
<circle class="gn-acc" cx="486.8" cy="244" r="7" data-on="3"/>
<text class="gl" x="486.8" y="244" dy="0.35em" style="font-size:8px" data-on="3">A</text>
<circle class="gn" cx="503.4" cy="244" r="7" data-on="3"/>
<text class="gl" x="503.4" y="244" dy="0.35em" style="font-size:8px" data-on="3">B</text>
<circle class="gn" cx="520.1" cy="244" r="7" data-on="3"/>
<text class="gl" x="520.1" y="244" dy="0.35em" style="font-size:8px" data-on="3">E</text>
<circle class="gn" cx="536.7" cy="244" r="7" data-on="3"/>
<text class="gl" x="536.7" y="244" dy="0.35em" style="font-size:8px" data-on="3">B</text>
<circle class="gn" cx="553.3" cy="244" r="7" data-on="3"/>
<text class="gl" x="553.3" y="244" dy="0.35em" style="font-size:8px" data-on="3">C</text>
<circle class="gn" cx="569.9" cy="244" r="7" data-on="3"/>
<text class="gl" x="569.9" y="244" dy="0.35em" style="font-size:8px" data-on="3">D</text>
<circle class="gn-acc" cx="586.5" cy="244" r="7" data-on="3"/>
<text class="gl" x="586.5" y="244" dy="0.35em" style="font-size:8px" data-on="3">A</text>
<circle class="gn" cx="603.1" cy="244" r="7" data-on="3"/>
<text class="gl" x="603.1" y="244" dy="0.35em" style="font-size:8px" data-on="3">C</text>
<circle class="gn" cx="619.7" cy="244" r="7" data-on="3"/>
<text class="gl" x="619.7" y="244" dy="0.35em" style="font-size:8px" data-on="3">C</text>
<circle class="gn" cx="636.3" cy="244" r="7" data-on="3"/>
<text class="gl" x="636.3" y="244" dy="0.35em" style="font-size:8px" data-on="3">F</text>
<circle class="gn" cx="652.9" cy="244" r="7" data-on="3"/>
<text class="gl" x="652.9" y="244" dy="0.35em" style="font-size:8px" data-on="3">B</text>
<circle class="gn" cx="669.6" cy="244" r="7" data-on="3"/>
<text class="gl" x="669.6" y="244" dy="0.35em" style="font-size:8px" data-on="3">C</text>
<circle class="gn" cx="686.2" cy="244" r="7" data-on="3"/>
<text class="gl" x="686.2" y="244" dy="0.35em" style="font-size:8px" data-on="3">D</text>
<circle class="gn" cx="702.8" cy="244" r="7" data-on="3"/>
<text class="gl" x="702.8" y="244" dy="0.35em" style="font-size:8px" data-on="3">D</text>
<circle class="gn" cx="719.4" cy="244" r="7" data-on="3"/>
<text class="gl" x="719.4" y="244" dy="0.35em" style="font-size:8px" data-on="3">E</text>
<circle class="gn" cx="736" cy="244" r="7" data-on="3"/>
<text class="gl" x="736" y="244" dy="0.35em" style="font-size:8px" data-on="3">G</text>
<text class="tm" x="588" y="280" text-anchor="middle" data-text="[&quot;1 NODE&quot;, &quot;1 + 3 NODES&quot;, &quot;1 + 3 + 7 NODES&quot;, &quot;1 + 3 + 7 + 19 NODES&quot;]">1 + 3 + 7 + 19 NODES</text>
</svg>
</div>
<div class="viz-ctl" role="group" aria-label="Number of layers"><button type="button" data-go="0" data-note="With no layers, A knows only its own features. Its computation tree is a single node." aria-pressed="false">0 layers</button><button type="button" data-go="1" data-note="One layer: A mixes in B, C and D. The tree gets one level, one branch per neighbour." aria-pressed="false">1 layer</button><button type="button" data-go="2" data-note="Two layers: B, C and D were updated from their own neighbours first, so E and F now reach A. Notice A appears in its own tree three times: it is its neighbours&#x27; neighbour." aria-pressed="false">2 layers</button><button type="button" data-go="3" data-note="Three layers: G, three hops away, finally reaches A, and A has heard from the whole graph. The bottom level alone has 19 nodes, and it keeps growing with every layer." aria-pressed="true">3 layers</button></div>
<p class="viz-note" aria-live="polite">Three layers: G, three hops away, finally reaches A, and A has heard from the whole graph. The bottom level alone has 19 nodes, and it keeps growing with every layer.</p>
<figcaption>A GNN with k layers computes each node's vector from a tree of depth k. The receptive field on the left and the tree on the right are the same thing seen two ways. Highlighted tree nodes are A itself.</figcaption>
</figure>
{:/nomarkdown}

Two things in that tree are worth noticing. First, **A appears inside its own tree.** Two layers down, A is its neighbours' neighbour, so a node hears an echo of itself. That is harmless, and it is one reason the self-loops in section 5 are natural. Second, **the tree grows fast**: 1, 3, 7, 19 nodes per level here. In a social network where people have hundreds of friends, three layers can reach a large share of the whole graph. That growth is behind both of the problems in section 8, and it is why large-graph methods sample neighbours instead of using all of them.

In practice, most GNNs use **two or three layers**. For most node-level tasks the useful signal is local, and that is enough to capture it.

## 5. The same idea in matrix form: GCN

Message passing node by node is easy to follow but tedious to write down. With an adjacency matrix, the whole layer, for every node at once, becomes two matrix multiplications. Start with the simplest version: sum your neighbours, then transform.

$$
H^{(k+1)} = \sigma\big(A\, H^{(k)}\, W^{(k)}\big)
$$

$$H^{(k)}$$ stacks every node's vector as a row. Row $$v$$ of $$AH$$ is the sum of the rows of $$H$$ that belong to $$v$$'s neighbours, because $$A_{vu}$$ is 1 for exactly those $$u$$. Multiplying by $$W$$ then applies the same linear map to every node. That is message passing with sum aggregation, written as linear algebra.

It has two problems, and the **Graph Convolutional Network** (GCN) of Kipf and Welling (2017) fixes both:

- **A node forgets itself.** $$A_{vv} = 0$$, so a node's own vector is missing from its update. The fix is to add self-loops and use $$A + I$$.
- **Sums grow with degree.** A node with 500 neighbours gets a vector hundreds of times larger than a node with 2, and after a few layers the scale is unmanageable. The fix is to normalise by degree.

GCN uses a symmetric normalisation, where $$\tilde{D}$$ is the diagonal matrix of node degrees counting the self-loop:

$$
H^{(k+1)} = \sigma\big(\hat{A}\, H^{(k)}\, W^{(k)}\big), \qquad \hat{A} = \tilde{D}^{-1/2}\, (A + I)\, \tilde{D}^{-1/2}
$$

Per node, that gives each neighbour $$u$$ of $$v$$ the weight $$1/\sqrt{\tilde{d}_u \tilde{d}_v}$$:

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( \sum_{u \in N(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_u\, \tilde{d}_v}}\; W^{(k)}\, \mathbf{h}_u^{(k)} \Big)
$$

{::nomarkdown}
<figure class="viz" id="fig-matrix">
<div class="viz-head"><span class="viz-label">FIG. 6 // ONE GCN LAYER, AS MATRICES</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 270" role="img" aria-labelledby="fig-matrix-t fig-matrix-d">
<title id="fig-matrix-t">The GCN layer as two matrix multiplications</title>
<desc id="fig-matrix-d">The normalised adjacency matrix A-hat, seven by seven, times the feature matrix H, seven by three, times a shared weight matrix W, three by two, gives the new feature matrix H-prime, seven by two. Row A of A-hat is nonzero for A, B, C and D, so A's new row mixes exactly those four rows of H. A-hat of A and B is 1 over root 12, about 0.29.</desc>
<text class="tm ta" x="24" y="26">H′ = σ( Â · H · W )</text>
<text class="tm" x="61" y="50" text-anchor="middle">A</text>
<text class="tm" x="45" y="69" text-anchor="end">A</text>
<text class="tm" x="79" y="50" text-anchor="middle">B</text>
<text class="tm" x="45" y="87" text-anchor="end">B</text>
<text class="tm" x="97" y="50" text-anchor="middle">C</text>
<text class="tm" x="45" y="105" text-anchor="end">C</text>
<text class="tm" x="115" y="50" text-anchor="middle">D</text>
<text class="tm" x="45" y="123" text-anchor="end">D</text>
<text class="tm" x="133" y="50" text-anchor="middle">E</text>
<text class="tm" x="45" y="141" text-anchor="end">E</text>
<text class="tm" x="151" y="50" text-anchor="middle">F</text>
<text class="tm" x="45" y="159" text-anchor="end">F</text>
<text class="tm" x="169" y="50" text-anchor="middle">G</text>
<text class="tm" x="45" y="177" text-anchor="end">G</text>
<rect class="cell" x="52" y="56" width="18" height="18" data-tip="Â[A, A] = 0.25" data-tip-sub="1 / √(4 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="54" y="58" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="70" y="56" width="18" height="18" data-tip="Â[A, B] = 0.29" data-tip-sub="1 / √(4 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="72" y="58" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="88" y="56" width="18" height="18" data-tip="Â[A, C] = 0.25" data-tip-sub="1 / √(4 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="90" y="58" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="106" y="56" width="18" height="18" data-tip="Â[A, D] = 0.29" data-tip-sub="1 / √(4 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="108" y="58" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="124" y="56" width="18" height="18"/>
<rect class="cell" x="142" y="56" width="18" height="18"/>
<rect class="cell" x="160" y="56" width="18" height="18"/>
<rect class="cell" x="52" y="74" width="18" height="18" data-tip="Â[B, A] = 0.29" data-tip-sub="1 / √(3 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="54" y="76" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="70" y="74" width="18" height="18" data-tip="Â[B, B] = 0.33" data-tip-sub="1 / √(3 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="72" y="76" width="14" height="14" rx="2" style="opacity:0.75;pointer-events:none"/>
<rect class="cell" x="88" y="74" width="18" height="18" data-tip="Â[B, C] = 0.29" data-tip-sub="1 / √(3 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="90" y="76" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="106" y="74" width="18" height="18"/>
<rect class="cell" x="124" y="74" width="18" height="18"/>
<rect class="cell" x="142" y="74" width="18" height="18"/>
<rect class="cell" x="160" y="74" width="18" height="18"/>
<rect class="cell" x="52" y="92" width="18" height="18" data-tip="Â[C, A] = 0.25" data-tip-sub="1 / √(4 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="54" y="94" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="70" y="92" width="18" height="18" data-tip="Â[C, B] = 0.29" data-tip-sub="1 / √(4 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="72" y="94" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="88" y="92" width="18" height="18" data-tip="Â[C, C] = 0.25" data-tip-sub="1 / √(4 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="90" y="94" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="106" y="92" width="18" height="18"/>
<rect class="cell" x="124" y="92" width="18" height="18" data-tip="Â[C, E] = 0.29" data-tip-sub="1 / √(4 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="126" y="94" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="142" y="92" width="18" height="18"/>
<rect class="cell" x="160" y="92" width="18" height="18"/>
<rect class="cell" x="52" y="110" width="18" height="18" data-tip="Â[D, A] = 0.29" data-tip-sub="1 / √(3 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="54" y="112" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="70" y="110" width="18" height="18"/>
<rect class="cell" x="88" y="110" width="18" height="18"/>
<rect class="cell" x="106" y="110" width="18" height="18" data-tip="Â[D, D] = 0.33" data-tip-sub="1 / √(3 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="108" y="112" width="14" height="14" rx="2" style="opacity:0.75;pointer-events:none"/>
<rect class="cell" x="124" y="110" width="18" height="18"/>
<rect class="cell" x="142" y="110" width="18" height="18" data-tip="Â[D, F] = 0.29" data-tip-sub="1 / √(3 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="144" y="112" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="160" y="110" width="18" height="18"/>
<rect class="cell" x="52" y="128" width="18" height="18"/>
<rect class="cell" x="70" y="128" width="18" height="18"/>
<rect class="cell" x="88" y="128" width="18" height="18" data-tip="Â[E, C] = 0.29" data-tip-sub="1 / √(3 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="90" y="130" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="106" y="128" width="18" height="18"/>
<rect class="cell" x="124" y="128" width="18" height="18" data-tip="Â[E, E] = 0.33" data-tip-sub="1 / √(3 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="126" y="130" width="14" height="14" rx="2" style="opacity:0.75;pointer-events:none"/>
<rect class="cell" x="142" y="128" width="18" height="18" data-tip="Â[E, F] = 0.29" data-tip-sub="1 / √(3 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="144" y="130" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="160" y="128" width="18" height="18"/>
<rect class="cell" x="52" y="146" width="18" height="18"/>
<rect class="cell" x="70" y="146" width="18" height="18"/>
<rect class="cell" x="88" y="146" width="18" height="18"/>
<rect class="cell" x="106" y="146" width="18" height="18" data-tip="Â[F, D] = 0.29" data-tip-sub="1 / √(4 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="108" y="148" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="124" y="146" width="18" height="18" data-tip="Â[F, E] = 0.29" data-tip-sub="1 / √(4 × 3), degrees counting the self-loop"/>
<rect class="cell-on" x="126" y="148" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="142" y="146" width="18" height="18" data-tip="Â[F, F] = 0.25" data-tip-sub="1 / √(4 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="144" y="148" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="160" y="146" width="18" height="18" data-tip="Â[F, G] = 0.35" data-tip-sub="1 / √(4 × 2), degrees counting the self-loop"/>
<rect class="cell-on" x="162" y="148" width="14" height="14" rx="2" style="opacity:0.78;pointer-events:none"/>
<rect class="cell" x="52" y="164" width="18" height="18"/>
<rect class="cell" x="70" y="164" width="18" height="18"/>
<rect class="cell" x="88" y="164" width="18" height="18"/>
<rect class="cell" x="106" y="164" width="18" height="18"/>
<rect class="cell" x="124" y="164" width="18" height="18"/>
<rect class="cell" x="142" y="164" width="18" height="18" data-tip="Â[G, F] = 0.35" data-tip-sub="1 / √(2 × 4), degrees counting the self-loop"/>
<rect class="cell-on" x="144" y="166" width="14" height="14" rx="2" style="opacity:0.78;pointer-events:none"/>
<rect class="cell" x="160" y="164" width="18" height="18" data-tip="Â[G, G] = 0.50" data-tip-sub="1 / √(2 × 2), degrees counting the self-loop"/>
<rect class="cell-on" x="162" y="166" width="14" height="14" rx="2" style="opacity:1.00;pointer-events:none"/>
<rect class="cell-hl a-in" x="50" y="54" width="130" height="22" rx="3" style="--d:0.6s"/>
<text class="t big" x="115" y="208" text-anchor="middle">Â</text>
<text class="tm" x="115" y="226" text-anchor="middle">7 × 7, WHO LISTENS</text>
<text class="t big" x="196" y="125" text-anchor="middle">×</text>
<rect class="cell" x="214" y="56" width="18" height="18"/>
<rect class="cell-on" x="216" y="58" width="14" height="14" rx="2" style="opacity:0.92"/>
<rect class="cell" x="232" y="56" width="18" height="18"/>
<rect class="cell-on" x="234" y="58" width="14" height="14" rx="2" style="opacity:0.32"/>
<rect class="cell" x="250" y="56" width="18" height="18"/>
<rect class="cell-on" x="252" y="58" width="14" height="14" rx="2" style="opacity:0.49"/>
<rect class="cell" x="214" y="74" width="18" height="18"/>
<rect class="cell-on" x="216" y="76" width="14" height="14" rx="2" style="opacity:0.83"/>
<rect class="cell" x="232" y="74" width="18" height="18"/>
<rect class="cell-on" x="234" y="76" width="14" height="14" rx="2" style="opacity:0.23"/>
<rect class="cell" x="250" y="74" width="18" height="18"/>
<rect class="cell-on" x="252" y="76" width="14" height="14" rx="2" style="opacity:0.57"/>
<rect class="cell" x="214" y="92" width="18" height="18"/>
<rect class="cell-on" x="216" y="94" width="14" height="14" rx="2" style="opacity:0.74"/>
<rect class="cell" x="232" y="92" width="18" height="18"/>
<rect class="cell-on" x="234" y="94" width="14" height="14" rx="2" style="opacity:0.41"/>
<rect class="cell" x="250" y="92" width="18" height="18"/>
<rect class="cell-on" x="252" y="94" width="14" height="14" rx="2" style="opacity:0.66"/>
<rect class="cell" x="214" y="110" width="18" height="18"/>
<rect class="cell-on" x="216" y="112" width="14" height="14" rx="2" style="opacity:0.32"/>
<rect class="cell" x="232" y="110" width="18" height="18"/>
<rect class="cell-on" x="234" y="112" width="14" height="14" rx="2" style="opacity:0.83"/>
<rect class="cell" x="250" y="110" width="18" height="18"/>
<rect class="cell-on" x="252" y="112" width="14" height="14" rx="2" style="opacity:0.41"/>
<rect class="cell" x="214" y="128" width="18" height="18"/>
<rect class="cell-on" x="216" y="130" width="14" height="14" rx="2" style="opacity:0.41"/>
<rect class="cell" x="232" y="128" width="18" height="18"/>
<rect class="cell-on" x="234" y="130" width="14" height="14" rx="2" style="opacity:0.92"/>
<rect class="cell" x="250" y="128" width="18" height="18"/>
<rect class="cell-on" x="252" y="130" width="14" height="14" rx="2" style="opacity:0.23"/>
<rect class="cell" x="214" y="146" width="18" height="18"/>
<rect class="cell-on" x="216" y="148" width="14" height="14" rx="2" style="opacity:0.23"/>
<rect class="cell" x="232" y="146" width="18" height="18"/>
<rect class="cell-on" x="234" y="148" width="14" height="14" rx="2" style="opacity:0.74"/>
<rect class="cell" x="250" y="146" width="18" height="18"/>
<rect class="cell-on" x="252" y="148" width="14" height="14" rx="2" style="opacity:0.32"/>
<rect class="cell" x="214" y="164" width="18" height="18"/>
<rect class="cell-on" x="216" y="166" width="14" height="14" rx="2" style="opacity:0.32"/>
<rect class="cell" x="232" y="164" width="18" height="18"/>
<rect class="cell-on" x="234" y="166" width="14" height="14" rx="2" style="opacity:0.66"/>
<rect class="cell" x="250" y="164" width="18" height="18"/>
<rect class="cell-on" x="252" y="166" width="14" height="14" rx="2" style="opacity:0.92"/>
<rect class="cell-hl a-in" x="212" y="55" width="58" height="20" rx="3" style="--d:1.10s"/>
<rect class="cell-hl a-in" x="212" y="73" width="58" height="20" rx="3" style="--d:1.22s"/>
<rect class="cell-hl a-in" x="212" y="91" width="58" height="20" rx="3" style="--d:1.34s"/>
<rect class="cell-hl a-in" x="212" y="109" width="58" height="20" rx="3" style="--d:1.46s"/>
<text class="t big" x="241" y="208" text-anchor="middle">H</text>
<text class="tm" x="241" y="226" text-anchor="middle">7 × 3</text>
<text class="t big" x="286" y="125" text-anchor="middle">×</text>
<rect class="cell" x="304" y="92" width="18" height="18"/>
<rect class="cell-on" x="306" y="94" width="14" height="14" rx="2" style="opacity:0.66"/>
<rect class="cell" x="322" y="92" width="18" height="18"/>
<rect class="cell-on" x="324" y="94" width="14" height="14" rx="2" style="opacity:0.32"/>
<rect class="cell" x="304" y="110" width="18" height="18"/>
<rect class="cell-on" x="306" y="112" width="14" height="14" rx="2" style="opacity:0.23"/>
<rect class="cell" x="322" y="110" width="18" height="18"/>
<rect class="cell-on" x="324" y="112" width="14" height="14" rx="2" style="opacity:0.92"/>
<rect class="cell" x="304" y="128" width="18" height="18"/>
<rect class="cell-on" x="306" y="130" width="14" height="14" rx="2" style="opacity:0.49"/>
<rect class="cell" x="322" y="128" width="18" height="18"/>
<rect class="cell-on" x="324" y="130" width="14" height="14" rx="2" style="opacity:0.57"/>
<rect class="cell-hl a-in" x="302" y="90" width="40" height="58" rx="3" style="--d:2.0s"/>
<text class="t big" x="322" y="208" text-anchor="middle">W</text>
<text class="tm" x="322" y="226" text-anchor="middle">3 × 2</text>
<text class="t big" x="362" y="125" text-anchor="middle">=</text>
<rect class="cell" x="382" y="56" width="18" height="18"/>
<rect class="cell" x="400" y="56" width="18" height="18"/>
<rect class="cell" x="382" y="74" width="18" height="18"/>
<rect class="cell" x="400" y="74" width="18" height="18"/>
<rect class="cell" x="382" y="92" width="18" height="18"/>
<rect class="cell" x="400" y="92" width="18" height="18"/>
<rect class="cell" x="382" y="110" width="18" height="18"/>
<rect class="cell" x="400" y="110" width="18" height="18"/>
<rect class="cell" x="382" y="128" width="18" height="18"/>
<rect class="cell" x="400" y="128" width="18" height="18"/>
<rect class="cell" x="382" y="146" width="18" height="18"/>
<rect class="cell" x="400" y="146" width="18" height="18"/>
<rect class="cell" x="382" y="164" width="18" height="18"/>
<rect class="cell" x="400" y="164" width="18" height="18"/>
<rect class="cell-on a-pop" x="384" y="58" width="14" height="14" rx="2" style="--d:2.50s;opacity:0.35"/>
<rect class="cell-on a-pop" x="402" y="58" width="14" height="14" rx="2" style="--d:2.50s;opacity:0.50"/>
<rect class="cell-on a-pop" x="384" y="76" width="14" height="14" rx="2" style="--d:3.06s;opacity:0.50"/>
<rect class="cell-on a-pop" x="402" y="76" width="14" height="14" rx="2" style="--d:3.06s;opacity:0.65"/>
<rect class="cell-on a-pop" x="384" y="94" width="14" height="14" rx="2" style="--d:3.12s;opacity:0.65"/>
<rect class="cell-on a-pop" x="402" y="94" width="14" height="14" rx="2" style="--d:3.12s;opacity:0.35"/>
<rect class="cell-on a-pop" x="384" y="112" width="14" height="14" rx="2" style="--d:3.18s;opacity:0.35"/>
<rect class="cell-on a-pop" x="402" y="112" width="14" height="14" rx="2" style="--d:3.18s;opacity:0.50"/>
<rect class="cell-on a-pop" x="384" y="130" width="14" height="14" rx="2" style="--d:3.24s;opacity:0.50"/>
<rect class="cell-on a-pop" x="402" y="130" width="14" height="14" rx="2" style="--d:3.24s;opacity:0.65"/>
<rect class="cell-on a-pop" x="384" y="148" width="14" height="14" rx="2" style="--d:3.30s;opacity:0.65"/>
<rect class="cell-on a-pop" x="402" y="148" width="14" height="14" rx="2" style="--d:3.30s;opacity:0.35"/>
<rect class="cell-on a-pop" x="384" y="166" width="14" height="14" rx="2" style="--d:3.36s;opacity:0.35"/>
<rect class="cell-on a-pop" x="402" y="166" width="14" height="14" rx="2" style="--d:3.36s;opacity:0.50"/>
<rect class="glow" x="380" y="54" width="40" height="22" rx="3" style="--d:2.6s;--p:4.6s"/>
<text class="t big" x="400" y="208" text-anchor="middle">H′</text>
<text class="tm" x="400" y="226" text-anchor="middle">7 × 2</text>
<text class="tm ta a-in" x="460" y="64" style="--d:0.7s">1 // READ ROW A OF Â</text>
<text class="t2 a-in" x="460" y="84" style="--d:0.8s">Nonzero for A, B, C and D: A itself</text>
<text class="t2 a-in" x="460" y="101" style="--d:0.8s">plus its three neighbours.</text>
<text class="tm ta a-in" x="460" y="132" style="--d:1.4s">2 // WEIGHT BY 1 / √(DEGREE × DEGREE)</text>
<text class="t2 a-in" x="460" y="152" style="--d:1.5s">A has degree 4 with its self-loop, B has 3,</text>
<text class="t2 a-in" x="460" y="169" style="--d:1.5s">so Â[A, B] = 1/√12 = 0.29. Hover a cell.</text>
<text class="tm ta a-in" x="460" y="200" style="--d:2.0s">3 // ONE W, SHARED BY EVERY NODE</text>
<text class="t2 a-in" x="460" y="220" style="--d:2.1s">It maps each node's 3 features to 2.</text>
<text class="t2 strong a-in" x="460" y="251" style="--d:2.6s">Row A of H′ is A's new vector.</text>
</svg>
</div>
<figcaption>The whole layer for every node at once. Row A of Â picks out the rows of H that A listens to and weights them; W then transforms every node with the same weights. Cell brightness in Â shows the weight; hover a cell for the exact value.</figcaption>
</figure>
{:/nomarkdown}

The square root on both sides is a compromise. Dividing by $$\tilde{d}_v$$ alone would give a plain mean. Dividing by $$\sqrt{\tilde{d}_u}$$ as well also turns down messages from busy hubs, which send the same message to everyone they are connected to. The word "convolution" comes from spectral graph theory, where this layer is a cheap first-order approximation of a filter on the graph's frequencies, but you do not need any of that to use it. Read it as a weighted average over yourself and your neighbours, followed by a linear layer shared by every node.

Here is a GCN from scratch in PyTorch, with a dense $$\hat{A}$$. For a graph of a few thousand nodes that is perfectly fine:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def gcn_norm_dense(edge_index, num_nodes):
    """Â = D^-1/2 (A + I) D^-1/2 as a dense matrix. Fine for a few thousand nodes."""
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    A = A + torch.eye(num_nodes)                   # self-loops: a node keeps its own vector
    d_inv_sqrt = A.sum(dim=1).pow(-0.5)            # degrees, counting the self-loop
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)   # W, shared by every node

    def forward(self, H, A_hat):
        return A_hat @ self.lin(H)                 # transform every node, then mix neighbours

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, p=0.5):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden)
        self.layer2 = GCNLayer(hidden, n_classes)
        self.p = p

    def forward(self, X, A_hat):
        H = F.dropout(X, self.p, self.training)
        H = F.relu(self.layer1(H, A_hat))
        H = F.dropout(H, self.p, self.training)
        return self.layer2(H, A_hat)               # one row of class scores per node
```

Computing $$HW$$ before multiplying by $$\hat{A}$$ is the same maths in a cheaper order: it shrinks the features from 1,433 columns to 16 before the $$n \times n$$ product.

For large graphs a dense $$n \times n$$ matrix is out of the question, so libraries compute the same thing straight from the edge list, as a **gather** (read each edge's source vector) followed by a **scatter** (add it into the edge's target). Mean aggregation over a $$2 \times E$$ edge list takes four lines, and its cost grows with the number of edges:

```python
def mean_neighbours(h, edge_index):
    src, dst = edge_index                                    # one column per directed edge src -> dst
    total = torch.zeros_like(h).index_add_(0, dst, h[src])   # gather from src, scatter-add into dst
    count = torch.zeros(h.size(0), dtype=h.dtype).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    return total / count.clamp(min=1).unsqueeze(-1)          # sum -> mean; isolated nodes stay zero
```

On Cora (section 10) this returns exactly what the dense version computes, using memory for 10,556 edges instead of 7.3 million matrix cells. PyTorch Geometric, DGL and the other graph libraries are, at heart, fast and careful versions of that function.

## 6. Three layers you will meet everywhere: GCN, GraphSAGE and GAT

Most GNN layers differ in one thing: **how much each neighbour's message counts.** Switch between the three classic answers on the same neighbourhood.

{::nomarkdown}
<figure class="viz viz-steps" id="fig-layers" data-dur="4600">
<div class="viz-head"><span class="viz-label">FIG. 7 // GCN, GRAPHSAGE AND GAT ON THE SAME NEIGHBOURHOOD</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 290" role="img" aria-labelledby="fig-layers-t fig-layers-d">
<title id="fig-layers-t">How GCN, GraphSAGE and GAT weight the same four neighbours</title>
<desc id="fig-layers-d">A node v with four neighbours of degree 1, 2, 9 and 4. Edge thickness shows each neighbour's weight. GCN uses 1 over root of degree products: 0.32, 0.26, 0.14, 0.20, with self-loop 0.20. GraphSAGE gives every neighbour 0.25 and keeps v's own vector separate. GAT learns attention weights, here 0.45, 0.08, 0.05, 0.22 with self 0.20.</desc>
<text class="tm ta" x="24" y="26">ONE NODE v, FOUR NEIGHBOURS</text>
<line class="ge" x1="322" y1="236" x2="313.2" y2="203.2" style="stroke-opacity:.6"/>
<circle class="hn" cx="313.2" cy="203.2" r="2.5"/>
<line class="ge" x1="322" y1="236" x2="323.2" y2="202" style="stroke-opacity:.6"/>
<circle class="hn" cx="323.2" cy="202" r="2.5"/>
<line class="ge" x1="322" y1="236" x2="333.1" y2="203.9" style="stroke-opacity:.6"/>
<circle class="hn" cx="333.1" cy="203.9" r="2.5"/>
<line class="ge" x1="322" y1="236" x2="342" y2="208.5" style="stroke-opacity:.6"/>
<circle class="hn" cx="342" cy="208.5" r="2.5"/>
<line class="ge" x1="322" y1="236" x2="349.2" y2="215.5" style="stroke-opacity:.6"/>
<circle class="hn" cx="349.2" cy="215.5" r="2.5"/>
<line class="ge" x1="322" y1="236" x2="353.9" y2="224.4" style="stroke-opacity:.6"/>
<circle class="hn" cx="353.9" cy="224.4" r="2.5"/>
<line class="ge" x1="322" y1="236" x2="356" y2="234.2" style="stroke-opacity:.6"/>
<circle class="hn" cx="356" cy="234.2" r="2.5"/>
<line class="ge" x1="322" y1="236" x2="355" y2="244.2" style="stroke-opacity:.6"/>
<circle class="hn" cx="355" cy="244.2" r="2.5"/>
<line class="ge-acc" x1="171.5" y1="138.1" x2="80.3" y2="79.2" style="stroke-width:8.15" data-sw="[6.14, 5.15, 8.15]"/>
<line class="ge-acc" x1="208" y1="137.3" x2="304.1" y2="69.8" style="stroke-width:2.6" data-sw="[5.27, 5.15, 2.6]"/>
<line class="ge-acc" x1="208.4" y1="162" x2="307.8" y2="226.7" style="stroke-width:2.15" data-sw="[3.52, 5.15, 2.15]"/>
<line class="ge-acc" x1="171.4" y1="161.7" x2="74.4" y2="222.9" style="stroke-width:4.7" data-sw="[4.4, 5.15, 4.7]"/>
<path class="ge-acc" d="M176 132 C 150 84, 230 84, 204 132" style="fill:none;stroke-width:4.4" data-sw="[4.4, 1.4, 4.4]"/>
<text class="tm" x="190" y="84" text-anchor="middle" data-text="[&quot;self 0.20&quot;, &quot;self: kept apart&quot;, &quot;self 0.20&quot;]">self 0.20</text>
<circle class="gn-acc" cx="190" cy="150" r="22"/>
<text class="gl" x="190" y="150" dy="0.35em" style="font-size:15px">v</text>
<circle class="gn" cx="66" cy="70" r="17"/>
<text class="gl" x="66" y="70" dy="0.35em" style="font-size:12px">u1</text>
<text class="tm" x="66" y="44" text-anchor="middle">deg 1</text>
<text class="t" x="130.5" y="96.6" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.32&quot;, &quot;0.25&quot;, &quot;0.45&quot;]">0.45</text>
<circle class="gn" cx="318" cy="60" r="17"/>
<text class="gl" x="318" y="60" dy="0.35em" style="font-size:12px">u2</text>
<text class="tm" x="318" y="34" text-anchor="middle">deg 2</text>
<text class="t" x="269.6" y="117.6" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.26&quot;, &quot;0.25&quot;, &quot;0.08&quot;]">0.08</text>
<circle class="gn" cx="322" cy="236" r="17"/>
<text class="gl" x="322" y="236" dy="0.35em" style="font-size:12px">u3</text>
<text class="tm" x="322" y="267" text-anchor="middle">deg 9</text>
<text class="t" x="271.3" y="187.9" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.14&quot;, &quot;0.25&quot;, &quot;0.05&quot;]">0.05</text>
<circle class="gn" cx="60" cy="232" r="17"/>
<text class="gl" x="60" y="232" dy="0.35em" style="font-size:12px">u4</text>
<text class="tm" x="60" y="263" text-anchor="middle">deg 4</text>
<text class="t" x="127" y="212.6" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.20&quot;, &quot;0.25&quot;, &quot;0.22&quot;]">0.22</text>
<rect class="n" x="420" y="34" width="322" height="240" rx="12"/>
<text class="tm ta" x="440" y="64" data-text="[&quot;KIPF &amp; WELLING, 2017&quot;, &quot;HAMILTON, YING &amp; LESKOVEC, 2017&quot;, &quot;VELIČKOVIĆ ET AL., 2018&quot;]">VELIČKOVIĆ ET AL., 2018</text>
<text class="t big" x="440" y="96" style="font-size:22px" data-text="[&quot;GCN&quot;, &quot;GraphSAGE&quot;, &quot;GAT&quot;]">GAT</text>
<text class="tm" x="440" y="126" style="fill:#e8b84e" data-text="[&quot;WEIGHT = 1 / √((DEG u + 1)(DEG v + 1))&quot;, &quot;WEIGHT = 1 / (NUMBER OF NEIGHBOURS)&quot;, &quot;WEIGHT = SOFTMAX OF A LEARNED SCORE&quot;]">WEIGHT = SOFTMAX OF A LEARNED SCORE</text>
<text class="t2" x="440" y="160" data-text="[&quot;Weights come from the graph alone.&quot;, &quot;Every neighbour counts the same.&quot;, &quot;Weights are learned from features.&quot;]">Weights are learned from features.</text>
<text class="t2" x="440" y="182" data-text="[&quot;The hub u3, with 9 links, is turned&quot;, &quot;v&#x27;s own vector is kept apart and&quot;, &quot;Here the model decided u1 matters&quot;]">Here the model decided u1 matters</text>
<text class="t2" x="440" y="204" data-text="[&quot;down, so it cannot dominate.&quot;, &quot;joined on: [ own ‖ mean ].&quot;, &quot;most and the hub u3 hardly at all.&quot;]">most and the hub u3 hardly at all.</text>
<text class="t2" x="440" y="226" data-text="[&quot;No extra parameters, very fast.&quot;, &quot;Samples neighbours, so it scales.&quot;, &quot;More compute per edge, more to fit.&quot;]">More compute per edge, more to fit.</text>
<text class="tm" x="440" y="256" data-text="[&quot;WEIGHTS INCLUDE THE SELF-LOOP&quot;, &quot;NEIGHBOUR WEIGHTS SUM TO 1&quot;, &quot;EXAMPLE WEIGHTS, ALL FIVE SUM TO 1&quot;]">EXAMPLE WEIGHTS, ALL FIVE SUM TO 1</text>
</svg>
</div>
<div class="viz-ctl" role="group" aria-label="Choose a GNN layer"><button type="button" data-go="0" data-note="GCN: fixed weights from degrees. u1 has only one link, so its message counts most (0.32); the hub u3 talks to everyone, so it counts least (0.14)." aria-pressed="false">GCN</button><button type="button" data-go="1" data-note="GraphSAGE: a plain mean, so every neighbour gets 0.25, and v&#x27;s own vector is concatenated alongside instead of being averaged in." aria-pressed="false">GraphSAGE</button><button type="button" data-go="2" data-note="GAT: attention. The weights are computed from the nodes&#x27; features, so they can differ for every node and every edge. These values are an illustration of what a trained model might learn." aria-pressed="true">GAT</button></div>
<p class="viz-note" aria-live="polite">GAT: attention. The weights are computed from the nodes' features, so they can differ for every node and every edge. These values are an illustration of what a trained model might learn.</p>
<figcaption>Same neighbourhood, three rules for how much each message counts. Edge thickness is the weight. The GCN and GraphSAGE numbers follow from the degrees; the GAT numbers are made up to show the idea, since real attention weights depend on the features.</figcaption>
</figure>
{:/nomarkdown}

**GCN** (Kipf and Welling, 2017) uses the fixed weights from the previous section. They depend only on the structure of the graph, so they cost nothing to learn. The limitation is that two neighbours with the same degree always count the same, whatever they actually say.

**GraphSAGE** (Hamilton, Ying and Leskovec, 2017) makes two changes. It keeps a node's own vector apart from its neighbours' mean, joining the two with a concatenation ($$\Vert$$) instead of averaging them together. And it **samples** a fixed number of neighbours instead of using all of them:

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( W^{(k)} \big[\, \mathbf{h}_v^{(k)} \;\Vert\; \operatorname{mean}_{u \in S(v)} \mathbf{h}_u^{(k)} \,\big] \Big)
$$

$$S(v)$$ is a random sample of $$v$$'s neighbours; the paper used 25 at the first hop and 10 at the second. Sampling caps the size of the computation tree, which is what lets GraphSAGE train in mini-batches on graphs with millions of nodes. And because it learns a *function* of a neighbourhood rather than a vector per node, it is **inductive**: it can embed nodes that did not exist at training time, such as a new user or a new transaction.

**GAT**, the graph attention network (Veličković et al., 2018), lets the model learn the weights. It scores each neighbour with a small attention function of both nodes' features, and a softmax turns the scores into weights:

$$
e_{vu} = \operatorname{LeakyReLU}\big( \mathbf{a}^\top [\, W\mathbf{h}_v \,\Vert\, W\mathbf{h}_u \,] \big), \qquad
\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{w \in N(v) \cup \{v\}} \exp(e_{vw})}
$$

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( \sum_{u \in N(v) \cup \{v\}} \alpha_{vu}\, W \mathbf{h}_u^{(k)} \Big)
$$

Like a transformer, GAT runs several attention heads in parallel and concatenates their outputs; the paper used 8 heads on Cora. Because the weights depend on content, a node can learn to listen to the one neighbour that matters and ignore the rest. The price is more computation per edge and more parameters to fit.

{::nomarkdown}
<div class="table-wrap">
  <table>
    <thead>
      <tr><th></th><th>GCN</th><th>GraphSAGE</th><th>GAT</th></tr>
    </thead>
    <tbody>
      <tr><td>Neighbour weight</td><td>Fixed, from degrees</td><td>Equal (mean)</td><td>Learned attention</td></tr>
      <tr><td>Own vector</td><td>Averaged in via a self-loop</td><td>Kept apart, concatenated</td><td>Attended to via a self-loop</td></tr>
      <tr><td>Training in the paper</td><td>Full graph, transductive</td><td>Sampled mini-batches, inductive</td><td>Full graph, both settings</td></tr>
      <tr><td>Cost per edge</td><td>Lowest</td><td>Low</td><td>Highest</td></tr>
      <tr><td>Reach for it when</td><td>You want a fast, strong baseline</td><td>The graph is large or keeps growing</td><td>Some neighbours matter far more than others</td></tr>
    </tbody>
  </table>
</div>
{:/nomarkdown}

## 7. Sum, mean or max: why the aggregator matters

The aggregator looks like a detail, but it decides what a GNN can and cannot tell apart. Xu et al. made this precise in "How Powerful are Graph Neural Networks?" (2019): when does a GNN give two different neighbourhoods two different vectors?

Take three neighbourhoods with one-hot features, where gold is $$[1, 0]$$ and blue is $$[0, 1]$$:

{::nomarkdown}
<figure class="viz" id="fig-agg">
<div class="viz-head"><span class="viz-label">FIG. 8 // WHAT SUM, MEAN AND MAX CAN TELL APART</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 268" role="img" aria-labelledby="fig-agg-t fig-agg-d">
<title id="fig-agg-t">Sum, mean and max aggregation on three neighbourhoods</title>
<desc id="fig-agg-d">Three neighbourhoods with one-hot features, gold [1, 0] and blue [0, 1]: one gold and one blue; two of each; two gold and one blue. Sum gives [1, 1], [2, 2] and [2, 1], all different. Mean gives [0.5, 0.5] twice and [0.67, 0.33], so the first two collide. Max gives [1, 1] for all three.</desc>
<g class="a-in" style="--d:0.05s"><text class="tm ta" x="170" y="24" text-anchor="middle">NEIGHBOURHOOD 1</text><line class="ge" x1="170" y1="108" x2="111.7" y2="86.8"/><line class="ge" x1="170" y1="108" x2="228.3" y2="86.8"/><circle class="gn" cx="170" cy="108" r="13"/><text class="gl" x="170" y="108" dy="0.35em">?</text><circle class="s1" cx="111.7" cy="86.8" r="11"/><circle class="s2" cx="228.3" cy="86.8" r="11"/></g>
<g class="a-in" style="--d:0.17s"><text class="tm ta" x="350" y="24" text-anchor="middle">NEIGHBOURHOOD 2</text><line class="ge" x1="350" y1="108" x2="291.7" y2="86.8"/><line class="ge" x1="350" y1="108" x2="325.4" y2="51.1"/><line class="ge" x1="350" y1="108" x2="374.6" y2="51.1"/><line class="ge" x1="350" y1="108" x2="408.3" y2="86.8"/><circle class="gn" cx="350" cy="108" r="13"/><text class="gl" x="350" y="108" dy="0.35em">?</text><circle class="s1" cx="291.7" cy="86.8" r="11"/><circle class="s1" cx="325.4" cy="51.1" r="11"/><circle class="s2" cx="374.6" cy="51.1" r="11"/><circle class="s2" cx="408.3" cy="86.8" r="11"/></g>
<g class="a-in" style="--d:0.29s"><text class="tm ta" x="530" y="24" text-anchor="middle">NEIGHBOURHOOD 3</text><line class="ge" x1="530" y1="108" x2="471.7" y2="86.8"/><line class="ge" x1="530" y1="108" x2="530" y2="46"/><line class="ge" x1="530" y1="108" x2="588.3" y2="86.8"/><circle class="gn" cx="530" cy="108" r="13"/><text class="gl" x="530" y="108" dy="0.35em">?</text><circle class="s1" cx="471.7" cy="86.8" r="11"/><circle class="s1" cx="530" cy="46" r="11"/><circle class="s2" cx="588.3" cy="86.8" r="11"/></g>
<g class="a-in" style="--d:0.8s"><text class="tm ta" x="24" y="168">SUM</text><text class="t ok-v" x="170" y="168" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text><text class="t ok-v" x="350" y="168" text-anchor="middle" style="font-family:var(--font-mono)">[2, 2]</text><text class="t ok-v" x="530" y="168" text-anchor="middle" style="font-family:var(--font-mono)">[2, 1]</text></g>
<g class="a-pop" style="--d:1.1s"><rect class="ok-badge" x="628" y="154" width="118" height="20" rx="10"/><text class="tm ok-t" x="687" y="168" text-anchor="middle" style="font-size:9.5px">ALL 3 DIFFER</text></g>
<g class="a-in" style="--d:1.35s"><text class="tm ta" x="24" y="198">MEAN</text><text class="t bad-t" x="170" y="198" text-anchor="middle" style="font-family:var(--font-mono)">[0.5, 0.5]</text><text class="t bad-t" x="350" y="198" text-anchor="middle" style="font-family:var(--font-mono)">[0.5, 0.5]</text><text class="t ok-v" x="530" y="198" text-anchor="middle" style="font-family:var(--font-mono)">[0.67, 0.33]</text></g>
<g class="a-pop" style="--d:1.65s"><rect class="evil-bg" x="628" y="184" width="118" height="20" rx="10" style="stroke:#e0736d;stroke-opacity:.6"/><text class="tm bad-t" x="687" y="198" text-anchor="middle" style="font-size:9.5px">1 AND 2 COLLIDE</text></g>
<g class="a-in" style="--d:1.9s"><text class="tm ta" x="24" y="228">MAX</text><text class="t bad-t" x="170" y="228" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text><text class="t bad-t" x="350" y="228" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text><text class="t bad-t" x="530" y="228" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text></g>
<g class="a-pop" style="--d:2.2s"><rect class="evil-bg" x="628" y="214" width="118" height="20" rx="10" style="stroke:#e0736d;stroke-opacity:.6"/><text class="tm bad-t" x="687" y="228" text-anchor="middle" style="font-size:9.5px">ALL 3 COLLIDE</text></g>
<line class="ln-faint" x1="24" y1="146" x2="746" y2="146"/>
<text class="tm" x="24" y="256">VECTORS ARE [GOLD, BLUE]: GOLD = [1, 0], BLUE = [0, 1]</text>
</svg>
</div>
<figcaption>What each aggregator can see. Max keeps only which features are present, mean keeps proportions, sum keeps counts. Only sum gives three different neighbourhoods three different results.</figcaption>
</figure>
{:/nomarkdown}

- **Max** sees only *which* features are present. All three neighbourhoods contain gold and blue, so max cannot tell them apart.
- **Mean** sees *proportions*. It separates the third neighbourhood, which is two-thirds gold, but not the first two, which are both half and half.
- **Sum** sees *counts*, so it separates all three.

Mean and max are not wrong. They are the better choice when the number of neighbours is noise rather than signal, and the mean stays stable when degrees vary wildly. But when counting matters, as in chemistry, where a carbon with three hydrogens is not a carbon with one, you want a sum. Xu et al. proved that sum aggregation followed by an MLP is as powerful as message passing can be, and called the result the **Graph Isomorphism Network** (GIN):

$$
\mathbf{h}_v^{(k+1)} = \operatorname{MLP}^{(k)}\Big( \big(1 + \epsilon^{(k)}\big)\, \mathbf{h}_v^{(k)} + \sum_{u \in N(v)} \mathbf{h}_u^{(k)} \Big)
$$

"As powerful as message passing can be" has a ceiling: the **Weisfeiler-Lehman test** (1-WL), a classic algorithm for checking whether two graphs might be the same. Message passing cannot tell apart any pair of graphs that 1-WL cannot. The textbook example is a ring of six nodes versus two separate triangles. Every node in both graphs has exactly two neighbours, which look exactly like it, so every round of message passing gives every node the same vector, and the two graphs come out identical, even though one contains triangles and the other does not. When that matters, as with counting rings in molecules, the usual fixes are extra input features such as cycle counts or positional encodings, or more expressive architectures.

## 8. Why deeper is not better: oversmoothing

In computer vision, deeper networks usually do better. GNNs typically peak at two or three layers and then get worse. The main reason shows up if you take the learned weights away and just repeat the averaging step. Step through the rounds:

{::nomarkdown}
<figure class="viz viz-steps" id="fig-smooth" data-dur="3400">
<div class="viz-head"><span class="viz-label">FIG. 9 // OVERSMOOTHING, ROUND BY ROUND</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="fig-smooth-t fig-smooth-d">
<title id="fig-smooth-t">Repeated neighbourhood averaging pushes every node to the same value</title>
<desc id="fig-smooth-d">The seven-node graph, with each node coloured from blue (0) to gold (1) by its value, and the same values on a number line split by true class. Values after 0, 1, 2, 4, 8 and 32 rounds of averaging with neighbours: after 0: A 0.90, B 0.35, C 0.80, D 0.60, E 0.10, F 0.20, G 0.15; after 2: A 0.61, B 0.63, C 0.56, D 0.50, E 0.39, F 0.34, G 0.22; after 32: A 0.48, B 0.48, C 0.48, D 0.48, E 0.48, F 0.48, G 0.48. The gap between the class averages goes from 0.42 to 0.00.</desc>
<text class="tm ta" x="20" y="26">THE GRAPH, COLOURED BY VALUE</text>
<line class="ge" x1="90" y1="161" x2="60" y2="86"/>
<line class="ge" x1="90" y1="161" x2="170" y2="96"/>
<line class="ge" x1="90" y1="161" x2="130" y2="241"/>
<line class="ge" x1="60" y1="86" x2="170" y2="96"/>
<line class="ge" x1="170" y1="96" x2="265" y2="116"/>
<line class="ge" x1="130" y1="241" x2="245" y2="216"/>
<line class="ge" x1="265" y1="116" x2="245" y2="216"/>
<line class="ge" x1="245" y1="216" x2="330" y2="246"/>
<circle cx="90" cy="161" r="17" style="stroke:rgba(255,248,230,.35);stroke-width:1.5;fill:#7e8677" data-fill="[&quot;#bb8517&quot;, &quot;#98864d&quot;, &quot;#918659&quot;, &quot;#888667&quot;, &quot;#828671&quot;, &quot;#7e8677&quot;]"/>
<text class="gl" x="90" y="161" dy="0.35em">A</text>
<circle cx="60" cy="86" r="17" style="stroke:rgba(255,248,230,.35);stroke-width:1.5;fill:#7e8677" data-fill="[&quot;#6b8695&quot;, &quot;#9b8649&quot;, &quot;#938655&quot;, &quot;#8c8661&quot;, &quot;#83866f&quot;, &quot;#7e8677&quot;]"/>
<text class="gl" x="60" y="86" dy="0.35em">B</text>
<circle cx="170" cy="96" r="17" style="stroke:rgba(255,248,230,.35);stroke-width:1.5;fill:#7e8677" data-fill="[&quot;#ac852e&quot;, &quot;#86866a&quot;, &quot;#8a8664&quot;, &quot;#87866a&quot;, &quot;#828671&quot;, &quot;#7e8677&quot;]"/>
<text class="gl" x="170" y="96" dy="0.35em">C</text>
<circle cx="130" cy="241" r="17" style="stroke:rgba(255,248,230,.35);stroke-width:1.5;fill:#7e8677" data-fill="[&quot;#8f865c&quot;, &quot;#8b8663&quot;, &quot;#818673&quot;, &quot;#7d8679&quot;, &quot;#7d8679&quot;, &quot;#7e8677&quot;]"/>
<text class="gl" x="130" y="241" dy="0.35em">D</text>
<circle cx="265" cy="116" r="17" style="stroke:rgba(255,248,230,.35);stroke-width:1.5;fill:#7e8677" data-fill="[&quot;#4787ce&quot;, &quot;#6e8691&quot;, &quot;#71868c&quot;, &quot;#79867f&quot;, &quot;#7d8679&quot;, &quot;#7e8677&quot;]"/>
<text class="gl" x="265" y="116" dy="0.35em">E</text>
<circle cx="245" cy="216" r="17" style="stroke:rgba(255,248,230,.35);stroke-width:1.5;fill:#7e8677" data-fill="[&quot;#5687b7&quot;, &quot;#5f86a9&quot;, &quot;#6a8697&quot;, &quot;#71868c&quot;, &quot;#79867f&quot;, &quot;#7e8677&quot;]"/>
<text class="gl" x="245" y="216" dy="0.35em">F</text>
<circle cx="330" cy="246" r="17" style="stroke:rgba(255,248,230,.35);stroke-width:1.5;fill:#7e8677" data-fill="[&quot;#4f87c3&quot;, &quot;#5287bd&quot;, &quot;#5887b3&quot;, &quot;#67869b&quot;, &quot;#758685&quot;, &quot;#7e8677&quot;]"/>
<text class="gl" x="330" y="246" dy="0.35em">G</text>
<rect x="101" y="141" width="9" height="9" rx="2" style="fill:#c98500;stroke:#161412;stroke-width:1.5"/>
<rect x="71" y="66" width="9" height="9" rx="2" style="fill:#c98500;stroke:#161412;stroke-width:1.5"/>
<rect x="181" y="76" width="9" height="9" rx="2" style="fill:#c98500;stroke:#161412;stroke-width:1.5"/>
<rect x="141" y="221" width="9" height="9" rx="2" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/>
<rect x="276" y="96" width="9" height="9" rx="2" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/>
<rect x="256" y="196" width="9" height="9" rx="2" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/>
<rect x="341" y="226" width="9" height="9" rx="2" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/>
<rect x="20" y="282" width="9" height="9" rx="2" style="fill:#c98500"/>
<text class="tm" x="34" y="290">TRUE CLASS GOLD</text>
<rect x="150" y="282" width="9" height="9" rx="2" style="fill:#3987e5"/>
<text class="tm" x="164" y="290">TRUE CLASS BLUE</text>
<text class="tm ta" x="430" y="26">EACH NODE'S VALUE, BY TRUE CLASS</text>
<text class="tm" x="430" y="62" style="fill:#c98500">GOLD</text>
<text class="tm" x="430" y="156" style="fill:#3987e5">BLUE</text>
<line class="grid-strong" x1="452" y1="236" x2="732" y2="236"/>
<line class="grid" x1="452" y1="44" x2="452" y2="236"/>
<text class="axis" x="452" y="252" text-anchor="middle">0</text>
<line class="grid" x1="592" y1="44" x2="592" y2="236"/>
<text class="axis" x="592" y="252" text-anchor="middle">0.5</text>
<line class="grid" x1="732" y1="44" x2="732" y2="236"/>
<text class="axis" x="732" y="252" text-anchor="middle">1</text>
<line x1="452" y1="66" x2="452" y2="128" style="stroke:#c98500;stroke-width:2;stroke-dasharray:3 3;transform:translateX(134.5px)" data-tx="[191.3, 175.8, 168.3, 155.4, 142.6, 134.5]"/>
<g data-tx="[252.0, 185.5, 171.5, 154.6, 141.7, 134.5]" style="transform:translateX(134.5px)"><circle cx="452" cy="78" r="8" style="fill:#c98500;stroke:#161412;stroke-width:1.5"/><text class="gl" x="452" y="78" dy="0.35em" style="font-size:9.5px">A</text></g>
<g data-tx="[98.0, 191.3, 175.8, 160.9, 144.8, 134.6]" style="transform:translateX(134.6px)"><circle cx="452" cy="98" r="8" style="fill:#c98500;stroke:#161412;stroke-width:1.5"/><text class="gl" x="452" y="98" dy="0.35em" style="font-size:9.5px">B</text></g>
<g data-tx="[224.0, 150.5, 157.5, 150.9, 141.4, 134.5]" style="transform:translateX(134.5px)"><circle cx="452" cy="118" r="8" style="fill:#c98500;stroke:#161412;stroke-width:1.5"/><text class="gl" x="452" y="118" dy="0.35em" style="font-size:9.5px">C</text></g>
<line x1="452" y1="156" x2="452" y2="232" style="stroke:#3987e5;stroke-width:2;stroke-dasharray:3 3;transform:translateX(134.5px)" data-tx="[73.5, 96.0, 101.3, 114.2, 126.7, 134.5]"/>
<g data-tx="[168.0, 158.7, 139.2, 132.6, 132.6, 134.5]" style="transform:translateX(134.5px)"><circle cx="452" cy="168" r="8" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/><text class="gl" x="452" y="168" dy="0.35em" style="font-size:9.5px">D</text></g>
<g data-tx="[28.0, 102.7, 108.9, 125.2, 132.1, 134.5]" style="transform:translateX(134.5px)"><circle cx="452" cy="186" r="8" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/><text class="gl" x="452" y="186" dy="0.35em" style="font-size:9.5px">E</text></g>
<g data-tx="[56.0, 73.5, 96.0, 109.1, 124.6, 134.5]" style="transform:translateX(134.5px)"><circle cx="452" cy="204" r="8" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/><text class="gl" x="452" y="204" dy="0.35em" style="font-size:9.5px">F</text></g>
<g data-tx="[42.0, 49.0, 61.2, 90.0, 117.3, 134.5]" style="transform:translateX(134.5px)"><circle cx="452" cy="222" r="8" style="fill:#3987e5;stroke:#161412;stroke-width:1.5"/><text class="gl" x="452" y="222" dy="0.35em" style="font-size:9.5px">G</text></g>
<text class="tm ta" x="430" y="270" data-text="[&quot;GAP BETWEEN CLASS AVERAGES: 0.42&quot;, &quot;GAP BETWEEN CLASS AVERAGES: 0.29&quot;, &quot;GAP BETWEEN CLASS AVERAGES: 0.24&quot;, &quot;GAP BETWEEN CLASS AVERAGES: 0.15&quot;, &quot;GAP BETWEEN CLASS AVERAGES: 0.06&quot;, &quot;GAP BETWEEN CLASS AVERAGES: 0.00&quot;]">GAP BETWEEN CLASS AVERAGES: 0.00</text>
<text class="tm" x="430" y="289" data-text="[&quot;ONE THRESHOLD SEPARATES THE CLASSES: NO&quot;, &quot;ONE THRESHOLD SEPARATES THE CLASSES: NO&quot;, &quot;ONE THRESHOLD SEPARATES THE CLASSES: YES&quot;, &quot;ONE THRESHOLD SEPARATES THE CLASSES: YES&quot;, &quot;ONE THRESHOLD SEPARATES THE CLASSES: YES&quot;, &quot;ONE THRESHOLD SEPARATES THE CLASSES: NO&quot;]">ONE THRESHOLD SEPARATES THE CLASSES: NO</text>
</svg>
</div>
<div class="viz-ctl" role="group" aria-label="Number of averaging rounds"><button type="button" data-go="0" data-note="Layer 0: raw, noisy features. B is a gold node with a low value and D is a blue node with a high one, so no single threshold puts every node on the right side." aria-pressed="false">0</button><button type="button" data-go="1" data-note="After one round, every node moves towards the average of its neighbourhood. B is pulled up by A and C, its gold neighbours. D is still held up by A, so the classes still overlap." aria-pressed="false">1</button><button type="button" data-go="2" data-note="After two rounds the classes are cleanly apart: one threshold would now classify every node correctly. This is the useful part of smoothing, and why two or three layers is the usual choice." aria-pressed="false">2</button><button type="button" data-go="3" data-note="After four rounds, gold leaks into blue through the A-D and C-E edges. The gap between the class averages has shrunk from 0.42 to 0.15." aria-pressed="false">4</button><button type="button" data-go="4" data-note="After eight rounds the classes have nearly merged. They are still technically apart, but the closest gold and blue nodes differ by 0.03, a margin any real noise would wipe out." aria-pressed="false">8</button><button type="button" data-go="5" data-note="After 32 rounds every node holds the same value, to two decimal places. No classifier can tell the classes apart. This is oversmoothing." aria-pressed="true">32 layers</button></div>
<p class="viz-note" aria-live="polite">After 32 rounds every node holds the same value, to two decimal places. No classifier can tell the classes apart. This is oversmoothing.</p>
<figcaption>Each round replaces a node's value with the mean over itself and its neighbours, a GNN layer with the learned weights taken out. Dashed lines mark each class's average. A little smoothing cleans up noise; a lot erases the difference between the classes.</figcaption>
</figure>
{:/nomarkdown}

Each layer replaces a node's vector with an average over its neighbourhood, which is a smoothing operation, like blurring an image. A little smoothing helps: neighbours of the same class pull a noisy node back towards its class, which is exactly why GNNs work. Too much, and the averages of averages converge to the same vector everywhere in a connected graph. Li, Han and Wu (2018) showed that a GCN layer is a form of Laplacian smoothing, and that stacking many of them makes node vectors indistinguishable. This is **oversmoothing**. (With GCN's symmetric normalisation, the limit differs from node to node only by a factor that depends on degree, which is no more useful for telling classes apart.)

A second, related problem is **oversquashing** (Alon and Yahav, 2021). Remember the computation tree: the number of nodes within $$k$$ hops can grow exponentially, but all of their information has to squeeze through one fixed-size vector at the root. Signals from far away arrive too compressed to use. This mostly hurts tasks that genuinely need long-range information.

What helps:

- **Stay shallow.** Two or three layers is the right default. If you need more capacity, make the layers wider, not the stack deeper.
- **Residual connections**, $$\mathbf{h}^{(k+1)} = \mathbf{h}^{(k)} + \operatorname{layer}\big(\mathbf{h}^{(k)}\big)$$, so the original signal survives.
- **Jumping knowledge** (Xu et al., 2018): give the output layer every intermediate representation, so the model can choose the right depth for each node.
- **Normalisation designed for graphs**, such as PairNorm (Zhao and Akoglu, 2020), which stops node vectors from collapsing towards each other.
- **DropEdge** and similar methods, which remove random edges during training and slow the smoothing down.

## 9. From node vectors to answers: the three heads

After the GNN layers, every node has a vector that summarises its neighbourhood. A small **head** turns those vectors into the answer for each kind of question from section 1.

{::nomarkdown}
<figure class="viz" id="fig-heads">
<div class="viz-head"><span class="viz-label">FIG. 10 // FROM NODE VECTORS TO ANSWERS</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 268" role="img" aria-labelledby="fig-heads-t fig-heads-d">
<title id="fig-heads-t">One GNN, three heads</title>
<desc id="fig-heads-d">An input graph goes through two or three GNN layers, producing one vector per node. A node head applies a linear layer and softmax to each node, an edge head scores a pair of nodes with a dot product, and a graph head pools all node vectors and applies an MLP.</desc>
<defs><marker id="fig-heads-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<g class="a-in" style="--d:0.05s"><text class="tm ta" x="24" y="40">INPUT GRAPH</text><line class="ge" x1="40" y1="112" x2="88" y2="84"/><line class="ge" x1="88" y1="84" x2="110" y2="146"/><line class="ge" x1="40" y1="112" x2="110" y2="146"/><line class="ge" x1="110" y1="146" x2="58" y2="178"/><line class="ge" x1="110" y1="146" x2="132" y2="196"/><line class="ge" x1="58" y1="178" x2="132" y2="196"/><circle class="gn" cx="40" cy="112" r="9"/><circle class="gn" cx="88" cy="84" r="9"/><circle class="gn" cx="110" cy="146" r="9"/><circle class="gn" cx="58" cy="178" r="9"/><circle class="gn" cx="132" cy="196" r="9"/></g>
<path class="flow" d="M150 142H186" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:0.2s"><rect class="n-acc" x="194" y="98" width="124" height="88" rx="12"/><rect class="glow" x="194" y="98" width="124" height="88" rx="12" style="--d:1.0s;--p:5s"/><text class="t" x="256" y="132" text-anchor="middle">GNN layers</text><text class="t2" x="256" y="150" text-anchor="middle">message passing</text><text class="tm" x="256" y="170" text-anchor="middle">× 2 OR 3</text></g>
<path class="flow" d="M318 142H344" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:0.45s"><text class="tm ta" x="352" y="78">ONE VECTOR</text><text class="tm ta" x="352" y="92">PER NODE</text><rect class="cell-on" x="354" y="106" width="12" height="12" rx="2" style="opacity:0.92"/><rect class="cell-on" x="368" y="106" width="12" height="12" rx="2" style="opacity:0.44"/><rect class="cell-on" x="382" y="106" width="12" height="12" rx="2" style="opacity:0.68"/><rect class="cell-on" x="396" y="106" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="354" y="122" width="12" height="12" rx="2" style="opacity:0.52"/><rect class="cell-on" x="368" y="122" width="12" height="12" rx="2" style="opacity:0.84"/><rect class="cell-on" x="382" y="122" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="396" y="122" width="12" height="12" rx="2" style="opacity:0.60"/><rect class="cell-on" x="354" y="138" width="12" height="12" rx="2" style="opacity:0.76"/><rect class="cell-on" x="368" y="138" width="12" height="12" rx="2" style="opacity:0.68"/><rect class="cell-on" x="382" y="138" width="12" height="12" rx="2" style="opacity:0.92"/><rect class="cell-on" x="396" y="138" width="12" height="12" rx="2" style="opacity:0.44"/><rect class="cell-on" x="354" y="154" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="368" y="154" width="12" height="12" rx="2" style="opacity:0.52"/><rect class="cell-on" x="382" y="154" width="12" height="12" rx="2" style="opacity:0.44"/><rect class="cell-on" x="396" y="154" width="12" height="12" rx="2" style="opacity:0.92"/><rect class="cell-on" x="354" y="170" width="12" height="12" rx="2" style="opacity:0.60"/><rect class="cell-on" x="368" y="170" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="382" y="170" width="12" height="12" rx="2" style="opacity:0.76"/><rect class="cell-on" x="396" y="170" width="12" height="12" rx="2" style="opacity:0.68"/></g>
<path class="flow" d="M418 142L464 58" marker-end="url(#fig-heads-ah)"/>
<circle class="packet a-travel" cx="418" cy="142" r="3.5" style="--dx:42px;--dy:-84px;--t:1.7s;--d:1.40s"/>
<g class="a-in" style="--d:0.8s"><rect class="n" x="474" y="30" width="140" height="56" rx="10"/><text class="t" x="490" y="54">Node head</text><text class="t2" x="490" y="72">linear + softmax</text></g>
<path class="flow" d="M614 58H636" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:1.1s"><text class="tm ta" x="644" y="52">PER NODE</text><text class="t2 strong" x="644" y="70">paper topic</text></g>
<path class="flow" d="M418 142L464 142" marker-end="url(#fig-heads-ah)"/>
<circle class="packet a-travel" cx="418" cy="142" r="3.5" style="--dx:42px;--dy:0px;--t:1.7s;--d:1.75s"/>
<g class="a-in" style="--d:1.0s"><rect class="n" x="474" y="114" width="140" height="56" rx="10"/><text class="t" x="490" y="138">Edge head</text><text class="t2" x="490" y="156">score = hᵤ · hᵥ</text></g>
<path class="flow" d="M614 142H636" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:1.3s"><text class="tm ta" x="644" y="136">PER PAIR</text><text class="t2 strong" x="644" y="154">link score</text></g>
<path class="flow" d="M418 142L464 226" marker-end="url(#fig-heads-ah)"/>
<circle class="packet a-travel" cx="418" cy="142" r="3.5" style="--dx:42px;--dy:84px;--t:1.7s;--d:2.10s"/>
<g class="a-in" style="--d:1.2s"><rect class="n" x="474" y="198" width="140" height="56" rx="10"/><text class="t" x="490" y="222">Graph head</text><text class="t2" x="490" y="240">pool, then MLP</text></g>
<path class="flow" d="M614 226H636" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:1.5s"><text class="tm ta" x="644" y="220">PER GRAPH</text><text class="t2 strong" x="644" y="238">toxic or not</text></g>
</svg>
</div>
<figcaption>The layers that compute node vectors are shared across tasks. Only the small head at the end decides whether the answer is per node, per pair or per graph.</figcaption>
</figure>
{:/nomarkdown}

- **Node head.** A linear layer and a softmax on each node's vector, trained with cross-entropy on the nodes that have labels.
- **Edge head.** A score for a pair of nodes, most simply the dot product $$\mathbf{h}_u^\top \mathbf{h}_v$$ through a sigmoid. It is trained on existing edges as positives and randomly sampled non-edges as negatives.
- **Graph head.** A **readout** pools every node vector into one, with a sum, mean or max (order-independent again, for the reason in section 2), and an MLP predicts the label from that.

## 10. A working GNN on Cora in 30 lines

Cora is the MNIST of graph learning: a citation network of 2,708 machine learning papers. Each paper is described by a 1,433-dimensional bag-of-words vector and belongs to one of 7 topics, and there are 5,278 citation links between them (stored as 10,556 directed edges, one in each direction). The standard split, from Yang, Cohen and Salakhutdinov (2016), is deliberately harsh: only **140 papers are labelled for training**, 20 per topic, with 500 for validation and 1,000 for testing. The model has to infer the other topics from the words and the citations.

Here is a two-layer GCN in PyTorch Geometric, with the hyperparameters from the GCN paper:

```python
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root="data/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0]          # one graph: x [2708, 1433], edge_index [2, 10556], y, train/val/test masks

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, n_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)                               # every node, every step
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])  # but only 140 labels
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
print(f"test accuracy: {acc:.3f}")
```

It trains in well under a minute on a laptop CPU. Over ten random seeds it scored 81.7% test accuracy on average (between 80.4% and 82.6%), in line with the 81.5% in the paper. The from-scratch version from section 5 scores 81.4%: same model, same result.

Look at the loss line. Only 140 nodes contribute to it, but the forward pass runs over all 2,708, and the unlabelled papers still shape every prediction through their words and their citations. That setting, where the test nodes are in the graph during training but their labels are hidden, is called **transductive** learning.

How much is the graph worth? Here are the same 140 labels used four ways:

{::nomarkdown}
<figure class="viz" id="fig-cora">
<div class="viz-head"><span class="viz-label">FIG. 11 // WHAT THE GRAPH IS WORTH ON CORA</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 262" role="img" aria-labelledby="fig-cora-t fig-cora-d">
<title id="fig-cora-t">Test accuracy on Cora with 140 labelled papers</title>
<desc id="fig-cora-d">Test accuracy on the Cora citation graph, standard split. An MLP on the words alone: 55.1%. Label propagation on the citations alone: 68.0%. GCN on both: 81.5%. GAT on both: 83.0% plus or minus 0.7. The GCN code in this post: 81.7% averaged over 10 seeds.</desc>
<line class="grid-strong" x1="300" y1="24" x2="300" y2="226"/>
<text class="axis" x="300" y="242" text-anchor="middle">0%</text>
<line class="grid" x1="406" y1="24" x2="406" y2="226"/>
<text class="axis" x="406" y="242" text-anchor="middle">25%</text>
<line class="grid" x1="512" y1="24" x2="512" y2="226"/>
<text class="axis" x="512" y="242" text-anchor="middle">50%</text>
<line class="grid" x1="618" y1="24" x2="618" y2="226"/>
<text class="axis" x="618" y="242" text-anchor="middle">75%</text>
<line class="grid" x1="724" y1="24" x2="724" y2="226"/>
<text class="axis" x="724" y="242" text-anchor="middle">100%</text>
<g class="row" tabindex="0" data-tip="55.1% test accuracy" data-tip-sub="MLP: GAT paper, Table 2" aria-label="MLP, 55.1% test accuracy"><rect class="hit" x="12" y="26" width="736" height="36"/><text class="t" x="20" y="42">MLP</text><text class="t2" x="20" y="58">words only, no graph</text><rect class="hn a-grow" x="300" y="34" width="233.6" height="20" rx="3" style="--d:0.15s"/><text class="axis" x="541.6" y="49" style="fill:var(--text-primary)">55.1%</text></g>
<g class="row" tabindex="0" data-tip="68.0% test accuracy" data-tip-sub="Label propagation: GCN paper, Table 2" aria-label="Label propagation, 68.0% test accuracy"><rect class="hit" x="12" y="66" width="736" height="36"/><text class="t" x="20" y="82">Label propagation</text><text class="t2" x="20" y="98">citations only, no words</text><rect class="hn a-grow" x="300" y="74" width="288.3" height="20" rx="3" style="--d:0.33s"/><text class="axis" x="596.3" y="89" style="fill:var(--text-primary)">68.0%</text></g>
<g class="row" tabindex="0" data-tip="81.5% test accuracy" data-tip-sub="GCN: Kipf &amp; Welling, 2017" aria-label="GCN, 81.5% test accuracy"><rect class="hit" x="12" y="106" width="736" height="36"/><text class="t" x="20" y="122">GCN</text><text class="t2" x="20" y="138">words + citations</text><rect class="s1 a-grow" x="300" y="114" width="345.6" height="20" rx="3" style="--d:0.51s"/><text class="axis" x="653.6" y="129" style="fill:var(--text-primary)">81.5%</text></g>
<g class="row" tabindex="0" data-tip="83.0% test accuracy ± 0.7" data-tip-sub="GAT: Veličković et al., 2018, ± std over 100 runs" aria-label="GAT, 83.0% test accuracy ± 0.7"><rect class="hit" x="12" y="146" width="736" height="36"/><text class="t" x="20" y="162">GAT</text><text class="t2" x="20" y="178">words + citations, learned weights</text><rect class="s1 a-grow" x="300" y="154" width="351.9" height="20" rx="3" style="--d:0.69s"/><text class="axis" x="659.9" y="169" style="fill:var(--text-primary)">83.0%</text></g>
<g class="row" tabindex="0" data-tip="81.7% test accuracy ± 0.7" data-tip-sub="GCN, this post&#x27;s code: std 0.7 over 10 seeds" aria-label="GCN, this post&#x27;s code, 81.7% test accuracy ± 0.7"><rect class="hit" x="12" y="186" width="736" height="36"/><text class="t" x="20" y="202">GCN, this post's code</text><text class="t2" x="20" y="218">mean of 10 seeds on a CPU</text><rect class="s2 a-grow" x="300" y="194" width="346.3" height="20" rx="3" style="--d:0.87s"/><text class="axis" x="654.3" y="209" style="fill:var(--text-primary)">81.7%</text></g>
</svg>
</div>
<figcaption>Same 140 labels, four ways of using them. The published numbers come from the GCN and GAT papers; the last bar is the code above, run with ten different random seeds. Hover a bar for its source.</figcaption>
</figure>
{:/nomarkdown}

Words alone get about 55%. Citations alone, spreading the known labels along the edges with label propagation, get 68%. A GCN that combines the two reaches about 81%. Cora is strongly homophilous: 81% of its edges connect two papers on the same topic, which is exactly the setting GNNs are built for.

To try the other layers, swap the convolution. GraphSAGE is a drop-in change; GAT needs its heads, a higher dropout and an ELU activation:

```python
from torch_geometric.nn import GATConv, SAGEConv

# GraphSAGE: the node's own vector alongside the mean of its neighbours
self.conv1, self.conv2 = SAGEConv(in_dim, 16), SAGEConv(16, n_classes)

# GAT: 8 heads of 8 features, concatenated, then one averaged head at the output
self.conv1 = GATConv(in_dim, 8, heads=8, dropout=0.6)
self.conv2 = GATConv(8 * 8, n_classes, heads=1, concat=False, dropout=0.6)
# in forward(): dropout p=0.6 and F.elu instead of p=0.5 and F.relu; train with lr=0.005
```

With the same 200-epoch loop and ten seeds each, GraphSAGE scored 80.9% and GAT 81.8%. (The GAT paper's 83.0% comes from a longer run that stops early on the validation set.) On a graph this small and this homophilous the three layers land within about a point of each other, which is a useful lesson in itself: the graph does most of the work, and the choice of layer matters less than people expect.

## 11. Before you reach for a GNN

GNNs are powerful when the connections carry information the features do not. They are not automatically better, and the same few mistakes come up again and again.

- **Check homophily first.** GCN-style averaging assumes neighbours are alike. On **heterophilous** graphs, where connected nodes tend to differ, such as fraudsters who deliberately transact with ordinary customers, or the WebKB graphs of university web pages, a plain MLP can beat a GCN (Zhu et al., 2020). Measure the share of training edges that join two nodes with the same label before you design the model.
- **Beat a strong baseline that ignores the graph.** On tabular problems, gradient-boosted trees with a few hand-made neighbourhood features (how many transactions share this card, and how many of those were fraud) are hard to beat. A GNN has to earn its extra complexity.
- **Do not leak labels through the edges in link prediction.** If the edges you test on are still in the graph the GNN passes messages over, the model can simply see the answer. Split the edges first and pass messages only over the training edges. PyTorch Geometric's `RandomLinkSplit` does this for you.
- **Respect time.** In a graph that grows (transactions, citations, friendships), a node at training time should only see edges that existed before it, the same rule you would apply to any time series.
- **Sample when the graph does not fit.** Full-graph training stops being practical somewhere in the millions of nodes. `NeighborLoader` trains GraphSAGE-style on sampled computation trees, one mini-batch of target nodes at a time. It needs one of PyG's optional compiled packages, `pyg-lib` or `torch-sparse`, installed to match your PyTorch version:

```python
from torch_geometric.loader import NeighborLoader   # sampling needs pyg-lib or torch-sparse

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],          # sample 25 neighbours at hop 1, then 10 at hop 2
    batch_size=1024,
    input_nodes=data.train_mask,
    shuffle=True,
)
for batch in loader:                 # each batch: a small subgraph around up to 1,024 target nodes
    out = model(batch.x, batch.edge_index)[: batch.batch_size]   # targets come first
    loss = F.cross_entropy(out, batch.y[: batch.batch_size])
```

## Final thoughts

Strip away the notation and a graph neural network is a simple machine. Every node looks at its neighbours, summarises what it sees with a function that ignores order, and updates itself with weights shared by every node. Do that two or three times, and each node carries a description of its neighbourhood that a small head can turn into a label, a link score or a verdict on the whole graph.

Most of the design space lives in the details this post stepped through: how much each neighbour counts (GCN, GraphSAGE or GAT), whether to sum or average (GIN), and how many rounds to run before every node starts to look the same. If you keep one practical rule, make it this: start with two layers of GCN or GraphSAGE next to a strong baseline that ignores the graph, and let the gap between them tell you what the connections are worth.

## References

- Kipf, T. N. and Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- Hamilton, W. L., Ying, R. and Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
- Veličković, P. et al. (2018). *Graph Attention Networks.* [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- Gilmer, J. et al. (2017). *Neural Message Passing for Quantum Chemistry.* [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
- Xu, K., Hu, W., Leskovec, J. and Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- Li, Q., Han, Z. and Wu, X.-M. (2018). *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning.* [arXiv:1801.07606](https://arxiv.org/abs/1801.07606)
- Alon, U. and Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* [arXiv:2006.05205](https://arxiv.org/abs/2006.05205)
- Xu, K. et al. (2018). *Representation Learning on Graphs with Jumping Knowledge Networks.* [arXiv:1806.03536](https://arxiv.org/abs/1806.03536)
- Zhao, L. and Akoglu, L. (2020). *PairNorm: Tackling Oversmoothing in GNNs.* [arXiv:1909.12223](https://arxiv.org/abs/1909.12223)
- Zhu, J. et al. (2020). *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs.* [arXiv:2006.11468](https://arxiv.org/abs/2006.11468)
- Yang, Z., Cohen, W. W. and Salakhutdinov, R. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings.* [arXiv:1603.08861](https://arxiv.org/abs/1603.08861)
- Sanchez-Lengeling, B. et al. (2021). *A Gentle Introduction to Graph Neural Networks.* Distill. [distill.pub/2021/gnn-intro](https://distill.pub/2021/gnn-intro/)
- Hamilton, W. L. (2020). *Graph Representation Learning.* [cs.mcgill.ca/~wlh/grl_book](https://www.cs.mcgill.ca/~wlh/grl_book/)
- PyTorch Geometric documentation: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
