---
title: "Fraud Detection on IEEE-CIS: LightGBM, GraphSAGE, and What the Graph Was Worth"
category: Machine Learning
tags: [Fraud Detection, GNN, GraphSAGE, LightGBM, IEEE-CIS, FDB]
excerpt: "A relational GraphSAGE model beat its own no-graph control by 2 AUC points on IEEE-CIS and still lost to tuned LightGBM by 3. A walkthrough of the benchmark, the leakage guards, and the honest negative result."
description: "A complete fraud-detection study on IEEE-CIS through Amazon's Fraud Dataset Benchmark: the out-of-time split, label-free feature engineering, a 32-billion-edge implicit transaction graph, relational GraphSAGE, bootstrap intervals, ablations, and why gradient-boosted trees still win."
image: https://images.unsplash.com/photo-1563013544-824ae1b704d3?q=80&w=1600&auto=format&fit=crop
image_alt: "A person holding a credit card while typing on a laptop"
math: true
---

A fraud model that scores one transaction at a time is missing most of the evidence. The same card turns up on a new device. The same address receives orders from twenty freshly created accounts. An email domain that was clean for a year suddenly appears next to a burst of declined payments. Fraud is not a property of a row. It is a property of a neighbourhood.

That is the argument for graph learning, and it is a good argument. So I built the graph: 590,540 transactions as nodes, linked through shared clients, cards, addresses, email domains and devices, with roughly **32 billion implicit edges**. Then I trained a relational GraphSAGE model on it and compared it against the same network with the graph removed, a tuned LightGBM, and an ensemble of the two.

The graph was real. It moved the neural network **+0.021 AUC** over its no-graph control, with a bootstrap interval that excludes zero. It still lost to LightGBM by **0.032 AUC**. The likely reason is that the hand-built client features had already captured most of what the graph knew.

This post is the full walkthrough: the benchmark and its out-of-time split, the features that never touch a label, the graph construction that avoids materialising a single edge, the model, the ablations, and the checks that decide whether a result like this is trustworthy. It is also an argument that a benchmark is most useful when it gives you the result you did not want.

## 1. Why use a benchmark at all

The IEEE-CIS Fraud Detection data is a competition dataset: about 590,000 e-commerce transactions over six months, split into a transaction table and an identity table with device, browser, operating system and network information. The target is binary, fraud is rare, and the interesting structure is relational.

The problem with doing this on your own is that every choice flatters you. A random split puts the same client on both sides, so the model can memorise a card and look brilliant. A different metric hides the fact that the operating point is unusable. A different preprocessing pipeline quietly changes the features.

Amazon's [Fraud Dataset Benchmark (FDB)](https://github.com/amazon-science/fraud-dataset-benchmark) exists to remove those degrees of freedom. One loader returns a fixed feature set and a fixed **out-of-time** split; one evaluation function returns AUC-ROC and recall at a 1% false-positive rate. The [paper](https://arxiv.org/abs/2208.14417) also publishes baselines from Amazon Fraud Detector, AutoGluon, H2O and Auto-sklearn on the same split. That gives a number to beat that is not your own imagination.

The loader joins the transaction and identity tables, derives `ENTITY_ID` from card, address and account-start day, normalises the raw `D` columns, keeps 67 features, sorts by time and takes the last 5% as test. It is not the Kaggle leaderboard split; it is a holdout from the labelled data, which is exactly what you want for an honest offline estimate.

## 2. The data, in six numbers

{::nomarkdown}
<div class="table-wrap">
  <table>
    <thead>
      <tr><th>Quantity</th><th>Value</th><th>Why it matters</th></tr>
    </thead>
    <tbody>
      <tr><td>Transactions</td><td>590,540</td><td>561,013 train, 29,527 test</td></tr>
      <tr><td>Fraud rate</td><td>3.47% train, 3.96% test</td><td>Accuracy is meaningless; predicting "legit" scores 96.5%</td></tr>
      <tr><td>Features</td><td>67 (61 numeric, 6 categorical)</td><td>FDB's fixed feature set, before engineering</td></tr>
      <tr><td>Identity coverage</td><td>24.7% of transactions</td><td>75% of rows have no device or browser data</td></tr>
      <tr><td>Columns &gt;50% missing</td><td>32 of 67</td><td>Missingness itself carries signal</td></tr>
      <tr><td>Clients</td><td>208,962</td><td>2.7 transactions per client on average</td></tr>
    </tbody>
  </table>
</div>
{:/nomarkdown}

The split is 170 days of training and 11 days of test. There are only **1,169 frauds** in the test period, so every AUC in this post has a confidence interval of roughly ±0.007. That matters later, when the differences between models are smaller than the noise floor of a single-run comparison.

The most important pattern shows up before any modelling. Among clients with at least two transactions and at least one fraud, **39% have every transaction labelled fraud**. Fraud often behaves like an all-or-nothing property of a client, not a coin flip per transaction. Half of the test transactions come from a client that already appears in train, which is why client history is so useful at prediction time and why a random split would be so misleading.

## 3. Two metrics and one operating point

A fraud team cannot review 20% of traffic. It can review roughly 1% of legitimate traffic before the queue becomes useless. That is why FDB reports **recall at 1% FPR**: set the threshold so that 1% of legitimate transactions are flagged, then measure how many frauds you catch.

For the engineered LightGBM, that threshold catches about **705 of the 1,169 test frauds** (60.3%) while flagging at most roughly 284 legitimate transactions. Seven in ten alerts are real. That is the number an operations team feels, and it moves much faster than AUC when you are already near the top of the ranking.

The training protocol is part of the metric. The last 10% of the training period is held out for early stopping. The model is then refit on all of training with the chosen number of trees. Each model is scored on test once, after every choice has been made.

## 4. Features that never touch a label

The engineering is deliberately boring. Three families, none of which uses the target:

1. **Transaction-level:** log amount, the cents part of the amount, hour of day, and whether identity data exists at all.
2. **Frequency encoding:** how common a card, address, email domain, device string or client is. Rare values are often risky values.
3. **Client aggregates:** how many transactions a client has, the mean and spread of its amounts, its average distance from home, its average `C` values, how the current amount compares to its usual amount, and how many email domains it has used.

```python
def add_features(df, stats_source):
    out = df.copy()

    amt = out["transactionamt"].astype("float64")
    out["amt_log"] = np.log1p(amt)
    out["amt_cents"] = (amt - np.floor(amt)).round(2)
    out["hour"] = pd.to_datetime(out["EVENT_TIMESTAMP"]).dt.hour
    out["has_identity"] = out["id_01"].notna().astype(int)

    for col in FREQ_COLS:                       # card1, addr1, email, device, client...
        freq = stats_source[col].astype(str).value_counts(normalize=True)
        out[f"{col}_freq"] = out[col].astype(str).map(freq).astype(float)

    grouped = stats_source.groupby("ENTITY_ID")
    out["client_n_tx"] = out["ENTITY_ID"].map(grouped.size()).astype(float)
    for col, funcs in AGG_SPEC.items():          # amount, dist1, c1, c13, c14
        stats = grouped[col].agg(funcs)
        for f in funcs:
            out[f"client_{col}_{f}"] = out["ENTITY_ID"].map(stats[f]).astype(float)

    out["amt_vs_client_mean"] = amt / out["client_transactionamt_mean"]
    out["client_n_emails"] = out["ENTITY_ID"].map(grouped["p_emaildomain"].nunique()).astype(float)
    return out
```

The subtle decision is **which rows feed the statistics**. The default here is train only: compute the frequencies and client aggregates on train, then look them up for test. The alternative, common in Kaggle solutions, pools the unlabelled test rows into the statistics. It uses no labels, so it is not label leakage, but it does use information from the future. I kept the realistic version, because a production model cannot compute next month's frequencies before next month happens.

The 21 new features are worth **+0.0041 AUC** and **+1.7 points of recall** at the operating point, which is a small average gain and a meaningful operational one.

## 5. LightGBM is a hard baseline to beat

Gradient-boosted trees are the default for a reason. They handle the 32 mostly-missing columns natively, split categorical columns without one-hot encoding, find interactions without being told, and do not care that the amount distribution is heavy-tailed. On tabular data with mixed types and missingness, they are still the model to beat.

The configuration is deliberately conventional: 256 leaves, learning rate 0.05, 60% column sampling, L2 regularisation, a 3,000-tree cap and early stopping after 100 rounds. The only unusual choice is the refit: the best iteration count is multiplied by 1.1 and the model is retrained on the full training window.

{::nomarkdown}
<div class="table-wrap">
  <table>
    <thead>
      <tr><th>Model</th><th>Test AUC-ROC</th><th>Recall @ 1% FPR</th></tr>
    </thead>
    <tbody>
      <tr><td>AFD OFI (FDB paper)</td><td>0.938</td><td>—</td></tr>
      <tr><td>AFD TFI (FDB paper)</td><td>0.940</td><td>—</td></tr>
      <tr><td>Auto-sklearn (FDB paper)</td><td>0.932</td><td>—</td></tr>
      <tr><td>H2O (FDB paper)</td><td>0.890</td><td>—</td></tr>
      <tr><td>AutoGluon (FDB paper)</td><td>0.855</td><td>—</td></tr>
      <tr><td>LightGBM baseline</td><td>0.9468</td><td>0.5860</td></tr>
      <tr><td><strong>LightGBM engineered</strong></td><td><strong>0.9509</strong></td><td><strong>0.6031</strong></td></tr>
    </tbody>
  </table>
</div>
{:/nomarkdown}

The paper's baselines are from 2022 and were not tuned for this notebook, so the comparison is directional rather than a fair fight. The useful part is that a straightforward LightGBM with 21 label-free features is competitive with the strongest published baseline on this split. That is the bar the graph has to clear.

## 6. Building a graph without building the graph

Every transaction becomes a node. Two transactions are neighbours under a relation when they share that relation's value:

- **client** — the same `ENTITY_ID`
- **card** — the same `card1`
- **card + address** — the same `card1` and `addr1`
- **email** — the same purchaser email domain
- **device** — the same device string

{::nomarkdown}
<figure class="viz" id="fig-graph">
<div class="viz-head"><span class="viz-label">FIG. 1 // TRANSACTIONS AS NODES</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="fig-graph-t fig-graph-d">
<title id="fig-graph-t">Transactions linked through five shared values</title>
<desc id="fig-graph-d">Four transactions, two of them fraudulent, connected to five shared-value groups: client, card, card plus address, email domain and device. The graph has 590,540 transaction nodes, five relations and about 32.3 billion implicit edges.</desc>
<text class="tm ta" x="24" y="24">TRANSACTIONS LINKED BY SHARED VALUES</text>
<line class="ln-faint" x1="24" y1="38" x2="470" y2="38"/>
<line class="ln-faint" x1="500" y1="44" x2="500" y2="266"/>
<path class="edge" d="M75 78 L272 58"/>
<path class="edge" d="M75 78 L272 104"/>
<path class="edge" d="M75 128 L272 58"/>
<path class="edge" d="M75 128 L272 150"/>
<path class="edge" d="M75 128 L272 196"/>
<path class="edge" d="M75 178 L272 104"/>
<path class="edge" d="M75 178 L272 242"/>
<path class="edge" d="M75 228 L272 58"/>
<path class="edge" d="M75 228 L272 196"/>
<path class="edge" d="M75 228 L272 242"/>
<circle cx="62" cy="78" r="13" fill="#3d9e5c"/>
<text class="tm" x="62" y="82" text-anchor="middle" style="fill:#f0ebe0">T1</text>
<circle cx="62" cy="128" r="13" fill="#b83c38"/>
<text class="tm" x="62" y="132" text-anchor="middle" style="fill:#f0ebe0">T2</text>
<circle cx="62" cy="178" r="13" fill="#3d9e5c"/>
<text class="tm" x="62" y="182" text-anchor="middle" style="fill:#f0ebe0">T3</text>
<circle cx="62" cy="228" r="13" fill="#b83c38"/>
<text class="tm" x="62" y="232" text-anchor="middle" style="fill:#f0ebe0">T4</text>
<rect class="n-acc" x="272" y="50" width="16" height="16" rx="3"/>
<rect class="n-acc" x="272" y="96" width="16" height="16" rx="3"/>
<rect class="n-acc" x="272" y="142" width="16" height="16" rx="3"/>
<rect class="n-acc" x="272" y="188" width="16" height="16" rx="3"/>
<rect class="n-acc" x="272" y="234" width="16" height="16" rx="3"/>
<text class="t2" x="296" y="62"><tspan class="t2 strong">client</tspan><tspan class="tm" dx="8">ENTITY_ID</tspan></text>
<text class="t2" x="296" y="108"><tspan class="t2 strong">card</tspan><tspan class="tm" dx="8">card1</tspan></text>
<text class="t2" x="296" y="154"><tspan class="t2 strong">card + address</tspan><tspan class="tm" dx="8">card1, addr1</tspan></text>
<text class="t2" x="296" y="200"><tspan class="t2 strong">email domain</tspan><tspan class="tm" dx="8">p_emaildomain</tspan></text>
<text class="t2" x="296" y="246"><tspan class="t2 strong">device</tspan><tspan class="tm" dx="8">deviceinfo</tspan></text>
<circle cx="30" cy="280" r="5" fill="#3d9e5c"/>
<text class="tm" x="42" y="284">LEGIT</text>
<circle cx="96" cy="280" r="5" fill="#b83c38"/>
<text class="tm" x="108" y="284">FRAUD</text>
<text class="t" x="530" y="82" style="font-size:30px;fill:#e8b84e">590,540</text>
<text class="tm" x="530" y="102">TRANSACTIONS (NODES)</text>
<text class="t" x="530" y="156" style="font-size:30px;fill:#e8b84e">32.3B</text>
<text class="tm" x="530" y="176">IMPLICIT EDGES, NEVER BUILT</text>
<text class="t" x="530" y="230" style="font-size:30px;fill:#e8b84e">5</text>
<text class="tm" x="530" y="250">RELATIONS</text>
</svg>
</div>
<figcaption>Two fraudulent transactions and two legitimate ones, linked by the values they share. The graph is stored as a group index per relation, not as an edge list, which is how 32.3 billion implicit edges fit in memory.</figcaption>
</figure>
{:/nomarkdown}

The obvious implementation builds an edge list. On this data that is a bad idea. Summing $$n(n-1)/2$$ over every group gives **32,306,434,550 implicit edges**, mostly from giant hubs such as `gmail.com`. Materialising them would need hundreds of gigabytes and would tell the model nothing that a group index cannot.

So the graph is stored as one integer per node per relation: which group am I in? A GraphSAGE mean over neighbours is then a group sum minus the node itself, divided by the group size minus one:

$$
\text{mean}_{j \in N_r(i)} h_j = \frac{\sum_{j \in g_r(i)} h_j - h_i}{\left|g_r(i)\right| - 1}
$$

That is exactly what the mean aggregator would compute on the full edge list, in $$O(N)$$ memory. Missing values get their own group that never sends messages, so two transactions with an unknown device are not treated as sharing one.

Before training anything, the notebook measures whether the graph is worth modelling. For each relation it computes the fraud rate among a transaction's neighbours, separately for fraud and legitimate transactions. The strongest is the client relation, where the neighbour fraud rate around a fraud transaction is **61.5×** the rate around a legitimate one. Fraud clusters along the graph, and that is measured on train only. Labels are used for this diagnostic, never as model input.

## 7. Relational GraphSAGE

The model is a two-layer relational GraphSAGE with one weight matrix per relation, in the spirit of R-GCN. Each layer computes:

$$
h_i^{(l+1)} = \text{ReLU}\Big(\text{LN}\big(W_{self} h_i^{(l)} + \sum_{r} W_r\,[\text{mean}_{j \in N_r(i)} h_j^{(l)} \,\|\, \log(1+|N_r(i)|)]\big)\Big) + h_i^{(l)}
$$

Three details matter. The **degree scalar** lets the model discount enormous hubs. **Layer normalisation** keeps the summed messages from five relations at a stable scale. The **residual connection** means a node with no neighbours still has a usable representation, which is the majority case for devices and email domains.

```python
def neighbour_mean(h, g):
    """Mean of h over the OTHER senders in my group (GraphSAGE mean, no edge list)."""
    hw = h * g["w"].unsqueeze(1)
    group_sum = torch.zeros(g["n_groups"] + 1, h.size(1),
                            device=h.device, dtype=h.dtype).index_add(0, g["group"], hw)
    return (group_sum[g["group"]] - hw) / g["cnt"].clamp(min=1).unsqueeze(1)


class RelationalSAGELayer(nn.Module):
    def forward(self, h, graph):
        out = self.self_lin(h)
        for rel, g in graph.items():
            out = out + self.nbr_lin[rel](
                torch.cat([neighbour_mean(h, g), g["log_cnt"].unsqueeze(1)], dim=1))
        return self.drop(torch.relu(self.norm(out))) + h
```

The whole network has **63,169 parameters**: a linear encoder to 64 dimensions, two message-passing layers, and a small classification head. Training is full-batch, because the graph fits in memory, with AdamW at 5e-3, gradient clipping, early stopping on the last 10% of train, and three seeds averaged for the final model. One training run takes about ten minutes on a T4 and about twenty times longer on CPU.

The leakage rules are explicit:

- Node features are standardised with **train-only** statistics.
- Test labels are set to -1 and never enter the loss.
- With the realistic setting, test transactions **receive** messages from train but never **send** them, so a test score cannot depend on another test row's future.
- Early stopping uses the same time-based validation slice as LightGBM.

The control model is the same network with `graph = {}`: identical depth, width and optimiser, no message passing. Without that control, a GNN result is just a neural-network result with extra steps.

## 8. The results

{::nomarkdown}
<div class="table-wrap">
  <table>
    <thead>
      <tr><th>Model</th><th>Test AUC-ROC</th><th>Δ vs LightGBM engineered</th><th>95% CI of Δ</th></tr>
    </thead>
    <tbody>
      <tr><td>LightGBM baseline</td><td>0.9468</td><td>−0.0041</td><td>—</td></tr>
      <tr><td><strong>LightGBM engineered</strong></td><td><strong>0.9509</strong></td><td>—</td><td>—</td></tr>
      <tr><td>MLP (no graph)</td><td>0.8980</td><td>−0.0529</td><td>—</td></tr>
      <tr><td>GraphSAGE GNN</td><td>0.9186</td><td>−0.0324</td><td>[−0.0388, −0.0258]</td></tr>
      <tr><td>LightGBM + GNN ensemble</td><td>0.9422</td><td>−0.0087</td><td>[−0.0117, −0.0056]</td></tr>
    </tbody>
  </table>
</div>
{:/nomarkdown}

{::nomarkdown}
<figure class="viz" id="fig-models">
<div class="viz-head"><span class="viz-label">FIG. 2 // TEST AUC-ROC ON THE FDB HOLDOUT</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="fig-models-t fig-models-d">
<title id="fig-models-t">Test AUC-ROC for five models</title>
<desc id="fig-models-d">LightGBM with engineered features scores 0.951, the LightGBM baseline 0.947, the LightGBM and GNN ensemble 0.942, the relational GraphSAGE 0.919, and the identical network without the graph 0.898. The axis starts at 0.88.</desc>
<text class="tm" x="212" y="30" text-anchor="middle">0.88</text>
<text class="tm" x="337" y="30" text-anchor="middle">0.90</text>
<text class="tm" x="462" y="30" text-anchor="middle">0.92</text>
<text class="tm" x="587" y="30" text-anchor="middle">0.94</text>
<text class="tm" x="712" y="30" text-anchor="middle">0.96</text>
<line class="ln-faint" x1="212" y1="44" x2="212" y2="266"/>
<line class="ln-faint" x1="337" y1="44" x2="337" y2="266"/>
<line class="ln-faint" x1="462" y1="44" x2="462" y2="266"/>
<line class="ln-faint" x1="587" y1="44" x2="587" y2="266"/>
<line class="ln-faint" x1="712" y1="44" x2="712" y2="266"/>
<text class="t2" x="200" y="72" text-anchor="end">LightGBM engineered</text>
<rect x="212" y="57" width="443" height="22" rx="3" fill="#d4a03c"/>
<text class="tm" x="663" y="72" style="fill:#f0ebe0">0.9509</text>
<text class="t2" x="200" y="116" text-anchor="end">LightGBM baseline</text>
<rect x="212" y="101" width="418" height="22" rx="3" fill="#a49a88"/>
<text class="tm" x="638" y="116" style="fill:#f0ebe0">0.9468</text>
<text class="t2" x="200" y="160" text-anchor="end">LightGBM + GNN</text>
<rect x="212" y="145" width="389" height="22" rx="3" fill="#3d9e5c"/>
<text class="tm" x="609" y="160" style="fill:#f0ebe0">0.9422</text>
<text class="t2" x="200" y="204" text-anchor="end">GraphSAGE GNN</text>
<rect x="212" y="189" width="241" height="22" rx="3" fill="#b4c0d4"/>
<text class="tm" x="461" y="204" style="fill:#f0ebe0">0.9186</text>
<text class="t2" x="200" y="248" text-anchor="end">MLP (no graph)</text>
<rect x="212" y="233" width="113" height="22" rx="3" fill="#6f675a"/>
<text class="tm" x="333" y="248" style="fill:#f0ebe0">0.8980</text>
<text class="tm" x="212" y="288">TEST AUC-ROC // 29,527 TRANSACTIONS // AXIS STARTS AT 0.88</text>
</svg>
</div>
<figcaption>The graph lifts the neural network by more than two AUC points over its identical no-graph control. It still does not reach the boosted trees, and a 50/50 rank ensemble lands between the two rather than above both.</figcaption>
</figure>
{:/nomarkdown}

The first result is the one the graph was built for. The GNN scores **0.9186**; the MLP with the same architecture, features and training schedule scores **0.8980**. The paired bootstrap gives **+0.0206 AUC**, with a 95% interval whose lower bound is **+0.0148**. The graph is not decoration. Whatever the transactions say about each other, the message-passing layers are using it.

The second result is less convenient. LightGBM engineered is **0.0324 AUC higher**, with an interval of **[−0.0388, −0.0258]** that sits entirely below zero. The ensemble is closer, but still **0.0087 below**, interval **[−0.0117, −0.0056]**. Both differences are larger than the resampling noise and both point the same way.

The seed spread is worth reporting on its own. The three GNN seeds scored 0.9058, 0.9128 and 0.9178 on test, a standard deviation of 0.0049. Averaging their predictions brought the model to 0.9186, which is better than the mean of the individual scores. Seed averaging is not cosmetic; on a test set this small it is worth about 0.0065 AUC.

So the honest summary is: **the graph helped the network, but it did not help the system.**

<div class="key-idea">
  <span class="key-label">In one sentence</span>
  <p>A graph neural network earns its complexity only when it knows something your features do not. Here, most of the relational signal was already in the columns.</p>
</div>

## 9. Why the trees won

There are five explanations that fit the evidence, and they are not mutually exclusive.

**The hand-built features already encode the graph.** `client_n_tx`, `client_transactionamt_mean`, `amt_vs_client_mean` and the frequency encodings are one-hop summaries of the same neighbourhoods the GNN learns to aggregate. The difference is that LightGBM receives the statistic directly, while the GNN has to learn a 64-dimensional version of it from 63,000 parameters and 561,000 rows at a 3.5% positive rate. In this setup, the learned version was not better than the hand-designed one.

**The strongest relational signal is probably the label, and the graph is label-free.** The 39% all-or-nothing pattern says that for a large group of clients, one confirmed fraud comes with many more. That is label propagation, and it is exactly what a label-free aggregator cannot use. Doing it safely would need time-aware labels and a strict cutoff, not a feature column.

**Missingness is a feature, and trees are better at it.** Thirty-two of the 67 columns are more than half missing, and `has_identity` is one of the engineered features. LightGBM routes missing values natively. The GNN fills them with zero and appends a missing flag, which is a reasonable encoding but a lossy one.

**Test nodes are receivers only.** In the realistic setting, test transactions receive messages from train but never send them. That is the safer production rule, and it also means the test graph is thinner than the training graph. A transductive setup would score higher and would be a different, less honest experiment.

**The network starts five points behind.** The MLP is 0.0529 AUC below LightGBM. The graph closes two of those five points. The remaining three are a mix of architecture, optimisation and the mismatch between a small full-batch GNN and a gradient-boosted ensemble on tabular data. Relations were not the bottleneck.

That last framing is the useful one. It is tempting to read "GNN loses to LightGBM" as "graphs do not work for fraud". The experiment does not say that. It says that **on this feature set**, the graph's contribution was smaller than the gap between a neural network and a tuned tree ensemble.

## 10. The graph works as a committee

Removing one relation at a time is the cleanest way to see what each link contributes. The ablation runs on the validation period with a fixed epoch count and one seed, so treat the ordering as indicative rather than exact.

{::nomarkdown}
<figure class="viz" id="fig-ablation">
<div class="viz-head"><span class="viz-label">FIG. 3 // WHAT EACH RELATION CONTRIBUTES</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="fig-ablation-t fig-ablation-d">
<title id="fig-ablation-t">Relation ablation and single-relation models</title>
<desc id="fig-ablation-d">Left: validation AUC lost when each relation is removed. Card loses 0.0225, email 0.0198, client 0.0193, device 0.0118 and card plus address 0.0087. Right: no single relation alone reaches the full graph, and the device relation alone is below the no-graph MLP.</desc>
<text class="tm ta" x="24" y="24">AUC LOST WHEN REMOVED</text>
<text class="tm ta" x="420" y="24">SINGLE RELATION ONLY</text>
<line class="ln-faint" x1="400" y1="44" x2="400" y2="270"/>
<line class="ln-faint" x1="110" y1="48" x2="110" y2="248"/>
<text class="t2" x="100" y="72" text-anchor="end">card</text>
<text class="t2" x="100" y="112" text-anchor="end">email</text>
<text class="t2" x="100" y="152" text-anchor="end">client</text>
<text class="t2" x="100" y="192" text-anchor="end">device</text>
<text class="t2" x="100" y="232" text-anchor="end">card + addr</text>
<rect x="110" y="57" width="189" height="22" rx="3" fill="#d4a03c"/>
<text class="tm" x="307" y="72" style="fill:#f0ebe0">0.0225</text>
<rect x="110" y="97" width="166" height="22" rx="3" fill="#c9a77a"/>
<text class="tm" x="284" y="112" style="fill:#f0ebe0">0.0198</text>
<rect x="110" y="137" width="162" height="22" rx="3" fill="#b4c0d4"/>
<text class="tm" x="280" y="152" style="fill:#f0ebe0">0.0193</text>
<rect x="110" y="177" width="99" height="22" rx="3" fill="#8a8070"/>
<text class="tm" x="217" y="192" style="fill:#f0ebe0">0.0118</text>
<rect x="110" y="217" width="73" height="22" rx="3" fill="#6f675a"/>
<text class="tm" x="191" y="232" style="fill:#f0ebe0">0.0087</text>
<text class="tm" x="110" y="264">0</text>
<text class="tm" x="299" y="264">0.025</text>
<line x1="706" y1="48" x2="706" y2="248" stroke="#d4a03c" stroke-width="1.2"/>
<line x1="644" y1="48" x2="644" y2="248" stroke="rgba(255,248,230,0.45)" stroke-width="1" stroke-dasharray="3 4"/>
<circle cx="583" cy="68" r="5" fill="#d4a03c"/>
<line x1="583" y1="68" x2="706" y2="68" stroke="rgba(212,160,60,0.35)" stroke-width="1"/>
<circle cx="584" cy="108" r="5" fill="#c9a77a"/>
<line x1="584" y1="108" x2="706" y2="108" stroke="rgba(212,160,60,0.35)" stroke-width="1"/>
<circle cx="636" cy="148" r="5" fill="#b4c0d4"/>
<line x1="636" y1="148" x2="706" y2="148" stroke="rgba(212,160,60,0.35)" stroke-width="1"/>
<circle cx="475" cy="188" r="5" fill="#8a8070"/>
<line x1="475" y1="188" x2="706" y2="188" stroke="rgba(212,160,60,0.35)" stroke-width="1"/>
<circle cx="599" cy="228" r="5" fill="#6f675a"/>
<line x1="599" y1="228" x2="706" y2="228" stroke="rgba(212,160,60,0.35)" stroke-width="1"/>
<line class="ln-faint" x1="430" y1="248" x2="730" y2="248"/>
<text class="tm" x="430" y="264">0.87</text>
<text class="tm" x="516" y="264" text-anchor="middle">0.88</text>
<text class="tm" x="601" y="264" text-anchor="middle">0.89</text>
<text class="tm" x="687" y="264" text-anchor="middle">0.90</text>
</svg>
</div>
<figcaption>Left: every relation is needed. Card, email and client each cost about two AUC points when removed. Right: no single relation on its own beats the no-graph MLP (dashed), and device alone is clearly worse. The solid amber line is the full graph. The value is in the combination.</figcaption>
</figure>
{:/nomarkdown}

Removing any one relation lowers validation AUC, so the model is not leaning on a single shortcut. The largest drops come from **card (0.0225)**, **email (0.0198)** and **client (0.0193)**. The email result is the surprise: a domain is a coarse identifier, but shared domains combined with client history are informative in a way a frequency encoding does not capture.

The right-hand panel is the more interesting one. **No single relation alone reaches the full graph**, and the device relation alone (0.8753) is below the no-graph MLP (0.8950). Device strings are sparse and hubby; they help only when card and client messages provide context. The graph is a committee, not a set of independent features.

## 11. How I checked the numbers

A negative result only counts if the pipeline could have produced a positive one. The notebook therefore records **17 sanity checks** and **six hypothesis checks**, and prints the whole list at the end. All 17 sanity checks passed; four of the six hypotheses were supported. The two failures are the two headline claims above, which is the point of recording them alongside the results.

The checks that did the most work:

- **No transaction in both splits**, labels aligned with test rows, and a strictly out-of-time boundary.
- **Leak detector:** the AUC of every numeric feature against the label. The strongest was `c4` at 0.686, far below the 0.95 threshold that would suggest a label had slipped into a column.
- **Shuffled-label test:** retrain the GNN on permuted labels. Validation AUC 0.5102 and test AUC 0.5034, both inside the chance band. If the graph leaked, this model would score above chance.
- **Bootstrap intervals:** 1,000 resamples of the test set, with paired differences against LightGBM rather than two independent intervals.
- **Relation ablation:** the figure above, run as a separate experiment rather than inferred from the model.
- **t-SNE of the learned embedding** against the raw inputs, as a qualitative check that message passing changed the representation.

None of this proves the model would make money. It proves that the comparison is measuring what it claims to measure, which is the part that most fraud notebooks skip.

## 12. What I would try next

**Stack the GNN instead of averaging it.** A 50/50 rank average gives the GNN equal weight at every point in the ranking, including where it is weakest. A better composition is to add the GNN logit or its embedding as a feature to LightGBM, generated out-of-fold, and let the tree learn where the graph is trustworthy. That is a different hypothesis from the one tested here, and it is the most promising next step.

**Use labels across time, carefully.** The 39% all-or-nothing pattern says the strongest signal is a client's confirmed history. A time-aware label propagation, where a transaction can only receive labels from strictly earlier transactions, is the honest way to test it. The cutoff has to be enforced in the graph construction, not in a comment.

**Make the edges typed and weighted.** The current model treats five relations as five independent channels with equal treatment inside each. Typed attention, edge recency, or a learned mixture over relations could let it prefer a shared card over a shared email domain when both exist.

**Optimise for the operating point.** AUC is a global ranking metric; the review queue lives at 1% FPR. Training with a ranking or cost-sensitive objective, or selecting on partial AUC, could improve the number that matters even when full AUC does not move.

**Calibrate the alert volume.** A score is not a decision. The threshold should come from review capacity and the cost of a missed fraud, and it should be recalibrated as the fraud rate drifts between periods.

## 13. Reproducing it

The notebook runs in three ways: a Docker image with every package and the FDB loader pinned to a fixed commit, a local Python 3.11 environment, or Google Colab with a T4 GPU. The data load needs about 12 GB of RAM; the GNN takes roughly ten minutes on a T4 and about twenty times longer on CPU. Figures, predictions and both models are written to a working directory that survives restarts: `data/` locally, Google Drive on Colab.

```bash
git clone https://github.com/MasrukHabib/FDB.git
cd FDB
cp .env.example .env          # Kaggle username, key, Jupyter token
docker compose up --build -d
```

<!-- NOTE: the repo is currently private. Publish it, or remove the clone link above and the link below before publishing this post. -->

The repository is [github.com/MasrukHabib/FDB](https://github.com/MasrukHabib/FDB). The README covers the GPU override, the Colab setup, the memory settings for WSL 2, and the sync rules if you edit the notebook in both places. The notebook is the source of truth; every score in this post comes from a cell in it.

## Final thoughts

Three things stayed with me after this project.

**A graph needs to add information, not just structure.** The GNN learned real relational signal, and that signal was already compressed into a handful of client aggregates. The model was not wrong; the features were already right. Before adding message passing, ask what your columns already know.

**A control model is not optional.** Without the no-graph MLP, "GNN 0.919" would look like a success next to a 0.5 baseline. With it, the result becomes a measurement: the graph is worth two points, and the architecture is missing three more.

**Report the comparisons you lose.** The two failed hypotheses in the notebook are the most useful lines in it. They say where the next improvement has to come from, and they make the wins that remain believable.

## References

- Amazon Science. [Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) and [the FDB paper](https://arxiv.org/abs/2208.14417) (2022).
- Kaggle. [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection).
- Hamilton, Ying and Leskovec (2017). [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216). GraphSAGE.
- Schlichtkrull et al. (2018). [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103). R-GCN.
- Ke et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree).
- Efron and Tibshirani (1993). *An Introduction to the Bootstrap*. The resampling method behind the confidence intervals.
