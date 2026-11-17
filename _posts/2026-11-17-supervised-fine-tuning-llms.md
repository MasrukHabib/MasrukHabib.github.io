---
title: "Supervised Fine-Tuning of LLMs: From the Loss Function to LoRA"
category: LLMs
tags: [SFT, Fine-Tuning, LoRA, QLoRA, PyTorch, Transformers]
excerpt: "What supervised fine-tuning actually optimizes, why the loss is masked, how one gradient step reshapes the next-token distribution, and how LoRA and QLoRA fit a 7B model on one GPU. With the math and the code."
description: "A technical guide to supervised fine-tuning (SFT) of large language models: the next-token objective, the cross-entropy gradient, loss masking and chat templates, packing and gradient accumulation, the memory math of full fine-tuning, LoRA and QLoRA, hyperparameters, data quality and evaluation, with PyTorch and Hugging Face code."
image: https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=1600&auto=format&fit=crop
image_alt: "Abstract 3D render of the letters AI"
math: true
---

Ask a pretrained base model “What is the capital of Bangladesh?” and there is a fair chance it replies with another question. It has read millions of quiz pages, and on those pages a question is usually followed by more questions. Nothing is broken. The model is doing exactly what it was trained to do: continue text.

**Supervised fine-tuning (SFT)** is the step that turns that text-continuer into something that answers. What surprises most people is how little changes. The architecture is the same. The objective is the same next-token prediction. What changes is the data, and one detail of the loss: the model is graded only on the answer.

This post takes SFT apart from the inside: what the objective optimizes and why that makes data quality so important, what one gradient step does to the next-token distribution, loss masking and chat templates, packing and gradient accumulation, where the memory goes in full fine-tuning, and how LoRA and QLoRA bring a 7B model down to one GPU. Each section has the math and the PyTorch or Hugging Face code that goes with it.

The code uses PyTorch, Hugging Face `transformers` and `peft`. The math needs the chain rule and a softmax, nothing more. Hover any figure for exact values, and use its replay button to watch it again.
{: .callout}

## 1. Where SFT sits in the pipeline

A modern chat model is usually trained in three stages. They share an architecture and, for the first two, an objective.

{::nomarkdown}
<figure class="viz" id="fig-stages">
<div class="viz-head"><span class="viz-label">FIG. 1 // WHERE SFT SITS</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 292" role="img" aria-labelledby="fig-stages-t fig-stages-d">
<title id="fig-stages-t">The three stages of training a chat model</title>
<desc id="fig-stages-d">Pretraining teaches next-token prediction on trillions of tokens of raw text. Supervised fine-tuning uses the same objective on thousands to a million curated prompt and response pairs. Preference tuning with RLHF or DPO then learns from comparisons. Asked for the capital of Bangladesh, the base model continues with more questions, the SFT model answers, and the chat model answers in a more helpful style.</desc>
<defs><marker id="fig-stages-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<g class="a-in" style="--d:0.05s"><rect class="n" x="20" y="34" width="200" height="108" rx="12"/><rect class="glow" x="20" y="34" width="200" height="108" rx="12" style="--d:0.40s;--p:4.5s"/><text class="tm ta" x="120.0" y="56" text-anchor="middle">STAGE 1</text><text class="t big" x="120.0" y="80" text-anchor="middle">Pretraining</text><text class="t2" x="120.0" y="100" text-anchor="middle">predict the next token</text><text class="t2" x="120.0" y="116" text-anchor="middle">of raw web text, code, books</text><text class="tm" x="120.0" y="134" text-anchor="middle">TRILLIONS OF TOKENS</text></g>
<path class="flow" d="M226 88H274" marker-end="url(#fig-stages-ah)"/>
<circle class="packet a-travel" cx="226" cy="88" r="3.5" style="--dx:40px;--t:1.6s;--d:0.90s"/>
<g class="a-in" style="--d:0.20s"><rect class="n-acc" x="280" y="34" width="200" height="108" rx="12"/><rect class="glow" x="280" y="34" width="200" height="108" rx="12" style="--d:1.50s;--p:4.5s"/><text class="tm ta" x="380.0" y="56" text-anchor="middle">STAGE 2</text><text class="t big" x="380.0" y="80" text-anchor="middle">Supervised fine-tuning</text><text class="t2" x="380.0" y="100" text-anchor="middle">predict the next token</text><text class="t2" x="380.0" y="116" text-anchor="middle">of curated answers only</text><text class="tm" x="380.0" y="134" text-anchor="middle">10K TO 1M DEMONSTRATIONS</text></g>
<path class="flow" d="M486 88H534" marker-end="url(#fig-stages-ah)"/>
<circle class="packet a-travel" cx="486" cy="88" r="3.5" style="--dx:40px;--t:1.6s;--d:2.00s"/>
<g class="a-in" style="--d:0.35s"><rect class="n" x="540" y="34" width="200" height="108" rx="12"/><rect class="glow" x="540" y="34" width="200" height="108" rx="12" style="--d:2.60s;--p:4.5s"/><text class="tm ta" x="640.0" y="56" text-anchor="middle">STAGE 3</text><text class="t big" x="640.0" y="80" text-anchor="middle">Preference tuning</text><text class="t2" x="640.0" y="100" text-anchor="middle">prefer the better answer</text><text class="t2" x="640.0" y="116" text-anchor="middle">RLHF (PPO) or DPO</text><text class="tm" x="640.0" y="134" text-anchor="middle">10K TO 1M COMPARISONS</text></g>
<text class="a-in" x="20" y="186" style="--d:0.9s"><tspan class="tm ta">ASK EACH MODEL</tspan><tspan class="t2 strong" dx="10">“What is the capital of Bangladesh?”</tspan></text>
<g class="a-in" style="--d:1.30s"><rect class="untrusted" x="20" y="200" width="200" height="78" rx="10"/><text class="tm" x="34" y="222">BASE MODEL</text><text class="t2" x="34" y="244">What is the capital of India?</text><text class="t2" x="34" y="262">What is the capital of Nepal?…</text></g>
<g class="a-in" style="--d:1.75s"><rect class="n-acc" x="280" y="200" width="200" height="78" rx="10"/><text class="tm" x="294" y="222">SFT MODEL</text><text class="t2 strong" x="294" y="244">The capital of Bangladesh</text><text class="t2 strong" x="294" y="262">is Dhaka.</text></g>
<g class="a-in" style="--d:2.20s"><rect class="n" x="540" y="200" width="200" height="78" rx="10"/><text class="tm" x="554" y="222">CHAT MODEL</text><text class="t2 strong" x="554" y="244">Dhaka. It is also the</text><text class="t2 strong" x="554" y="262">country's largest city.</text></g>
</svg>
</div>
<figcaption>Same architecture, same next-token objective, very different behaviour. Pretraining makes a model that continues text; SFT on a much smaller set of demonstrations turns it into one that answers. The base model's reply is typical: it treats your question as the start of a list of questions.</figcaption>
</figure>
{:/nomarkdown}

**Pretraining** reads trillions of tokens of raw text and learns to predict the next one. It is where the model's knowledge and language ability come from, and almost all of the compute is spent here. **Supervised fine-tuning** runs the same next-token objective on a far smaller, curated set of demonstrations: prompts paired with the responses you want. **Preference tuning** (RLHF with PPO, or DPO) then learns from comparisons between two responses rather than from a single correct one.

The order matters. SFT is cheap and stable, and it gets the model close to the behaviour you want. Preference tuning then refines it, and it uses the SFT model as its starting point and its reference. The effect can be large: in the InstructGPT paper, labelers preferred outputs from a 1.3B-parameter model trained this way over outputs from the 175B-parameter GPT-3.

<div class="key-idea">
  <span class="key-label">In one sentence</span>
  <p>SFT is next-token prediction on curated prompt and response pairs, with the loss counted only on the response.</p>
</div>

What SFT is good at teaching is **behaviour**: following instructions, the turn-taking format of a chat, when to stop, tone, refusals, a tool-call syntax, a fixed JSON schema for an extraction task. What it is bad at is **adding knowledge**. A model fine-tuned on facts it did not know learns them slowly and starts to hallucinate more on everything else (Gekhman et al., 2024). If the problem is “the model doesn't know our documents”, the answer is retrieval, which I covered in [Production RAG on PostgreSQL](/blog/production-rag-on-postgresql-with-pgvector/). If the problem is “the model knows enough but answers in the wrong way”, the answer is SFT.

## 2. The objective, and what minimizing it really means

A training example is a prompt $$x$$ and a response $$y = (y_1, \dots, y_T)$$. A decoder-only model factorizes the probability of the response one token at a time:

$$
p_\theta(y \mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid x, y_{<t})
$$

SFT maximizes the likelihood of the demonstrations, which is the same as minimizing their negative log-likelihood:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x,y)\sim\mathcal{D}} \sum_{t=1}^{T} \log p_\theta(y_t \mid x, y_{<t})
$$

That is the whole objective. It is worth asking what minimizing it actually does. For a fixed prompt, the expected negative log-likelihood splits into two terms:

$$
\mathbb{E}_{y \sim p_{\mathcal{D}}(\cdot \mid x)}\big[-\log p_\theta(y \mid x)\big] = H\big(p_{\mathcal{D}}(\cdot \mid x)\big) + \mathrm{KL}\big(p_{\mathcal{D}}(\cdot \mid x) \,\Vert\, p_\theta(\cdot \mid x)\big)
$$

The entropy of the data does not depend on $$\theta$$, so SFT is minimizing the **forward KL divergence** from the data distribution to the model. Forward KL is *mass-covering*: it punishes the model hard for giving near-zero probability to anything that appears in the data, and barely punishes it for spreading probability too widely. In plain words, the model is pushed to reproduce **every** behaviour in your dataset, including the inconsistent formats, the wrong answers and the handful of examples an annotator wrote at 2 a.m. That is the mathematical reason data quality dominates SFT, and it is why section 12 matters more than any hyperparameter.

Training uses **teacher forcing**: the model always conditions on the true prefix $$y_{<t}$$, never on its own earlier guesses. Combined with the causal attention mask, that lets one forward pass compute all $$T$$ predictions in parallel. The cost is *exposure bias*: at inference the model conditions on its own output, so a small early mistake can put it in a state it never saw in training. Preference tuning and other on-policy methods are partly a response to that.

## 3. One gradient step, up close

At each position the model produces a vector of logits $$\mathbf{z} \in \mathbb{R}^{V}$$, one per vocabulary entry (around 128,000 for Llama 3). The softmax turns them into probabilities:

$$
p_v = \frac{e^{z_v}}{\sum_{u=1}^{V} e^{z_u}}
$$

For a target token $$y$$, the per-token loss is the negative log-probability, which expands to:

$$
\ell = -\log p_y = -z_y + \log \sum_{u=1}^{V} e^{z_u}
$$

Differentiate with respect to one logit $$z_v$$. The log-sum-exp term gives back the softmax, $$p_v$$, and the first term contributes $$-1$$ only when $$v = y$$:

$$
\frac{\partial \ell}{\partial z_v} = p_v - \mathbb{1}[v = y]
\qquad\Longrightarrow\qquad
\nabla_{\mathbf{z}}\,\ell = \mathbf{p} - \mathbf{e}_y
$$

This one line explains most of what SFT does at the level of a single token. Gradient descent moves the logits against the gradient, so the target's logit goes **up** by an amount proportional to $$1 - p_y$$, its missing probability, and every other logit goes **down** in proportion to its own probability $$p_v$$. The tokens that were stealing the most probability get pushed down the hardest.

{::nomarkdown}
<figure class="viz" id="fig-grad">
<div class="viz-head"><span class="viz-label">FIG. 3 // ONE GRADIENT STEP ON THE NEXT-TOKEN DISTRIBUTION</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 308" role="img" aria-labelledby="fig-grad-t fig-grad-d">
<title id="fig-grad-t">How the cross-entropy gradient reshapes the next-token distribution</title>
<desc id="fig-grad-d">For the prompt 'The capital of Bangladesh is', a toy six-token vocabulary gives Dhaka probability 0.34. The gradient of the loss with respect to the logits is p minus the one-hot target: -0.66 for Dhaka and plus each token's own probability for the others. One step against the gradient raises Dhaka to 0.63 and lowers every other token.</desc>
<defs><marker id="fig-grad-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<text class="tm ta" x="18" y="30">NEXT TOKEN</text>
<text class="tm ta" x="150" y="30">p BEFORE</text>
<text class="tm ta" x="432" y="30" text-anchor="middle">GRADIENT ∂ℓ/∂z = p − onehot(y)</text>
<text class="tm ta" x="560" y="30">p AFTER ONE STEP</text>
<text class="t2" x="18" y="48">“The capital of Bangladesh is”</text>
<line class="grid-strong" x1="432" y1="52" x2="432" y2="236"/>
<g class="row" tabindex="0" data-tip="p 0.335 → 0.628" data-tip-sub="target token: gradient -0.665, so the logit is pushed up" aria-label="Dhaka: p 0.335 → 0.628"><rect class="hit" x="12" y="58" width="736" height="26"/><text class="t" x="18" y="75" style="font-family:var(--font-mono)">·Dhaka</text><rect class="s1 a-grow" x="150" y="62" width="76.6" height="16" rx="3" style="--d:0.10s"/><text class="axis" x="232.6" y="74">0.34</text>
<rect class="s1 a-grow" x="370.8" y="62" width="61.2" height="16" rx="3" style="--d:0.90s;transform-origin:100% 50%"/><text class="axis" x="364.8" y="74" text-anchor="end">-0.66</text>
<rect class="s1 a-grow" x="560" y="62" width="143.5" height="16" rx="3" style="--d:1.80s"/><text class="axis" x="709.5" y="74">0.63</text></g>
<g class="row" tabindex="0" data-tip="p 0.184 → 0.096" data-tip-sub="gradient +0.184, so the logit is pushed down" aria-label="Chittagong: p 0.184 → 0.096"><rect class="hit" x="12" y="88" width="736" height="26"/><text class="t2" x="18" y="105" style="font-family:var(--font-mono)">·Chittagong</text><rect class="hn a-grow" x="150" y="92" width="42.0" height="16" rx="3" style="--d:0.16s"/><text class="axis" x="198.0" y="104">0.18</text>
<rect class="evil a-grow" x="432" y="92" width="16.9" height="16" rx="3" style="--d:0.96s"/><text class="axis" x="454.9" y="104">+0.18</text>
<rect class="hn a-grow" x="560" y="92" width="22.1" height="16" rx="3" style="--d:1.86s"/><text class="axis" x="588.1" y="104">0.10</text></g>
<g class="row" tabindex="0" data-tip="p 0.151 → 0.083" data-tip-sub="gradient +0.151, so the logit is pushed down" aria-label="a: p 0.151 → 0.083"><rect class="hit" x="12" y="118" width="736" height="26"/><text class="t2" x="18" y="135" style="font-family:var(--font-mono)">·a</text><rect class="hn a-grow" x="150" y="122" width="34.4" height="16" rx="3" style="--d:0.22s"/><text class="axis" x="190.4" y="134">0.15</text>
<rect class="evil a-grow" x="432" y="122" width="13.9" height="16" rx="3" style="--d:1.02s"/><text class="axis" x="451.9" y="134">+0.15</text>
<rect class="hn a-grow" x="560" y="122" width="19.0" height="16" rx="3" style="--d:1.92s"/><text class="axis" x="585.0" y="134">0.08</text></g>
<g class="row" tabindex="0" data-tip="p 0.136 → 0.077" data-tip-sub="gradient +0.136, so the logit is pushed down" aria-label="the: p 0.136 → 0.077"><rect class="hit" x="12" y="148" width="736" height="26"/><text class="t2" x="18" y="165" style="font-family:var(--font-mono)">·the</text><rect class="hn a-grow" x="150" y="152" width="31.1" height="16" rx="3" style="--d:0.28s"/><text class="axis" x="187.1" y="164">0.14</text>
<rect class="evil a-grow" x="432" y="152" width="12.5" height="16" rx="3" style="--d:1.08s"/><text class="axis" x="450.5" y="164">+0.14</text>
<rect class="hn a-grow" x="560" y="152" width="17.5" height="16" rx="3" style="--d:1.98s"/><text class="axis" x="583.5" y="164">0.08</text></g>
<g class="row" tabindex="0" data-tip="p 0.112 → 0.065" data-tip-sub="gradient +0.112, so the logit is pushed down" aria-label="located: p 0.112 → 0.065"><rect class="hit" x="12" y="178" width="736" height="26"/><text class="t2" x="18" y="195" style="font-family:var(--font-mono)">·located</text><rect class="hn a-grow" x="150" y="182" width="25.5" height="16" rx="3" style="--d:0.34s"/><text class="axis" x="181.5" y="194">0.11</text>
<rect class="evil a-grow" x="432" y="182" width="10.3" height="16" rx="3" style="--d:1.14s"/><text class="axis" x="448.3" y="194">+0.11</text>
<rect class="hn a-grow" x="560" y="182" width="14.9" height="16" rx="3" style="--d:2.04s"/><text class="axis" x="580.9" y="194">0.07</text></g>
<g class="row" tabindex="0" data-tip="p 0.083 → 0.050" data-tip-sub="gradient +0.083, so the logit is pushed down" aria-label="home: p 0.083 → 0.050"><rect class="hit" x="12" y="208" width="736" height="26"/><text class="t2" x="18" y="225" style="font-family:var(--font-mono)">·home</text><rect class="hn a-grow" x="150" y="212" width="18.9" height="16" rx="3" style="--d:0.40s"/><text class="axis" x="174.9" y="224">0.08</text>
<rect class="evil a-grow" x="432" y="212" width="7.6" height="16" rx="3" style="--d:1.20s"/><text class="axis" x="445.6" y="224">+0.08</text>
<rect class="hn a-grow" x="560" y="212" width="11.5" height="16" rx="3" style="--d:2.10s"/><text class="axis" x="577.5" y="224">0.05</text></g>
<text class="a-in" x="432" y="264" text-anchor="middle" style="--d:0.9s"><tspan class="t2">pushed up ←</tspan><tspan class="t2" dx="14">→ pushed down</tspan></text>
<text class="a-in" x="18" y="294" style="--d:2.3s"><tspan class="t2">Loss on “Dhaka”: −log p = </tspan><tspan class="t">1.09 → 0.47</tspan><tspan class="t2" dx="8">after one step of size η = 1.5 on the logits.</tspan></text>
</svg>
</div>
<details class="viz-data"><summary>View the numbers</summary><table><thead><tr><th>Token</th><th>p before</th><th>∂ℓ/∂z</th><th>p after</th></tr></thead><tbody><tr><td>Dhaka</td><td>0.335</td><td>-0.665</td><td>0.628</td></tr><tr><td>Chittagong</td><td>0.184</td><td>+0.184</td><td>0.096</td></tr><tr><td>a</td><td>0.151</td><td>+0.151</td><td>0.083</td></tr><tr><td>the</td><td>0.136</td><td>+0.136</td><td>0.077</td></tr><tr><td>located</td><td>0.112</td><td>+0.112</td><td>0.065</td></tr><tr><td>home</td><td>0.083</td><td>+0.083</td><td>0.050</td></tr></tbody></table></details>
<figcaption>A toy six-token vocabulary, so the numbers are easy to check. The target's logit moves up by its missing probability, 1 − p; every other logit moves down in proportion to how much probability it was stealing. For illustration the step is applied to the logits directly; in training the same signal flows back through the network into the weights.</figcaption>
</figure>
{:/nomarkdown}

Three consequences follow, and all three show up in practice:

1. **Easy tokens stop teaching.** When the model already assigns the target $$p_y \approx 1$$, the gradient $$\mathbf{p} - \mathbf{e}_y$$ is almost zero. Most tokens in a typical answer are predictable, so a small number of surprising tokens carry most of the update. A falling average loss can hide that the model is learning only a few things.
2. **The signal reaches every weight.** The logits are a function of all the parameters, so the chain rule carries the same vector back through the network: $$\nabla_\theta\,\ell = J_\theta(\mathbf{z})^{\top}(\mathbf{p} - \mathbf{e}_y)$$, where $$J_\theta(\mathbf{z})$$ is the Jacobian of the logits with respect to the weights.
3. **Loss has a readable scale.** The exponential of the mean loss is the **perplexity**. A loss of 0.95 means a perplexity of $$e^{0.95} \approx 2.6$$: on average the model is as unsure as if it were choosing between 2.6 equally likely tokens.

## 4. Loss masking: train on the answer, not the question

A chat example is not a prompt and a response in two neat fields. It is one sequence of tokens, laid out by the model's **chat template**. Qwen models use this layout, called ChatML (shown here without the system turn):

```text
<|im_start|>user
Capital of Bangladesh?<|im_end|>
<|im_start|>assistant
Dhaka.<|im_end|>
```

The model reads the whole sequence, but it should only be graded on the assistant's reply. The standard way to do that is to copy the input ids into a `labels` tensor and overwrite every non-answer position with `-100`, the value PyTorch's cross-entropy ignores.

{::nomarkdown}
<figure class="viz" id="fig-mask">
<div class="viz-head"><span class="viz-label">FIG. 2 // LOSS ONLY ON THE ANSWER</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 304" role="img" aria-labelledby="fig-mask-t fig-mask-d">
<title id="fig-mask-t">Loss masking and the one-token shift in supervised fine-tuning</title>
<desc id="fig-mask-d">A chat example is tokenized into 15 tokens. The 12 tokens of the template and the user's question get the label -100 and contribute no loss. The three answer tokens, Dhaka, the full stop and the end-of-turn token, are the labels. Each label is predicted from the token before it. Their losses are 2.31, 0.42 and 0.11, and the example's loss is their mean, 0.95.</desc>
<defs><marker id="fig-mask-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#e8b84e"/></marker></defs>
<text class="tm ta" x="14" y="20">INPUT TOKENS // WHAT THE MODEL READS</text>
<text class="tm ta" x="14" y="108">LABELS // WHAT IT IS GRADED ON</text>
<text class="tm ta" x="14" y="196">PER-TOKEN LOSS // −log p(label)</text>
<g class="a-in" style="--d:0.00s"><rect class="n" x="10.0" y="30" width="78.4" height="28" rx="6"/><text class="tk" x="49.2" y="48" text-anchor="middle">&lt;|im_start|&gt;</text></g>
<g class="a-in" style="--d:0.50s"><rect class="pad" x="10.0" y="118" width="78.4" height="28" rx="6"/><text class="tk dim" x="49.2" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="49.2" y="254" text-anchor="middle" style="--d:2.00s">0</text>
<g class="a-in" style="--d:0.03s"><rect class="n" x="90.4" y="30" width="32.8" height="28" rx="6"/><text class="tk" x="106.8" y="48" text-anchor="middle">user</text></g>
<g class="a-in" style="--d:0.53s"><rect class="pad" x="90.4" y="118" width="32.8" height="28" rx="6"/><text class="tk dim" x="106.8" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="106.8" y="254" text-anchor="middle" style="--d:2.02s">0</text>
<g class="a-in" style="--d:0.06s"><rect class="n" x="125.2" y="30" width="27.0" height="28" rx="6"/><text class="tk" x="138.7" y="48" text-anchor="middle">\n</text></g>
<g class="a-in" style="--d:0.56s"><rect class="pad" x="125.2" y="118" width="27.0" height="28" rx="6"/><text class="tk dim" x="138.7" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="138.7" y="254" text-anchor="middle" style="--d:2.04s">0</text>
<g class="a-in" style="--d:0.09s"><rect class="n" x="154.2" y="30" width="49.9" height="28" rx="6"/><text class="tk" x="179.1" y="48" text-anchor="middle">Capital</text></g>
<g class="a-in" style="--d:0.59s"><rect class="pad" x="154.2" y="118" width="49.9" height="28" rx="6"/><text class="tk dim" x="179.1" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="179.1" y="254" text-anchor="middle" style="--d:2.06s">0</text>
<g class="a-in" style="--d:0.12s"><rect class="n" x="206.1" y="30" width="27.1" height="28" rx="6"/><text class="tk" x="219.7" y="48" text-anchor="middle">·of</text></g>
<g class="a-in" style="--d:0.62s"><rect class="pad" x="206.1" y="118" width="27.1" height="28" rx="6"/><text class="tk dim" x="219.7" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="219.7" y="254" text-anchor="middle" style="--d:2.08s">0</text>
<g class="a-in" style="--d:0.15s"><rect class="n" x="235.2" y="30" width="72.7" height="28" rx="6"/><text class="tk" x="271.6" y="48" text-anchor="middle">·Bangladesh</text></g>
<g class="a-in" style="--d:0.65s"><rect class="pad" x="235.2" y="118" width="72.7" height="28" rx="6"/><text class="tk dim" x="271.6" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="271.6" y="254" text-anchor="middle" style="--d:2.10s">0</text>
<g class="a-in" style="--d:0.18s"><rect class="n" x="309.9" y="30" width="27.0" height="28" rx="6"/><text class="tk" x="323.4" y="48" text-anchor="middle">?</text></g>
<g class="a-in" style="--d:0.68s"><rect class="pad" x="309.9" y="118" width="27.0" height="28" rx="6"/><text class="tk dim" x="323.4" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="323.4" y="254" text-anchor="middle" style="--d:2.12s">0</text>
<g class="a-in" style="--d:0.21s"><rect class="n" x="338.9" y="30" width="67.0" height="28" rx="6"/><text class="tk" x="372.4" y="48" text-anchor="middle">&lt;|im_end|&gt;</text></g>
<g class="a-in" style="--d:0.71s"><rect class="pad" x="338.9" y="118" width="67.0" height="28" rx="6"/><text class="tk dim" x="372.4" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="372.4" y="254" text-anchor="middle" style="--d:2.14s">0</text>
<g class="a-in" style="--d:0.24s"><rect class="n" x="407.9" y="30" width="27.0" height="28" rx="6"/><text class="tk" x="421.4" y="48" text-anchor="middle">\n</text></g>
<g class="a-in" style="--d:0.74s"><rect class="pad" x="407.9" y="118" width="27.0" height="28" rx="6"/><text class="tk dim" x="421.4" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="421.4" y="254" text-anchor="middle" style="--d:2.16s">0</text>
<g class="a-in" style="--d:0.27s"><rect class="n" x="436.9" y="30" width="78.4" height="28" rx="6"/><text class="tk" x="476.1" y="48" text-anchor="middle">&lt;|im_start|&gt;</text></g>
<g class="a-in" style="--d:0.77s"><rect class="pad" x="436.9" y="118" width="78.4" height="28" rx="6"/><text class="tk dim" x="476.1" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="476.1" y="254" text-anchor="middle" style="--d:2.18s">0</text>
<g class="a-in" style="--d:0.30s"><rect class="n" x="517.3" y="30" width="61.3" height="28" rx="6"/><text class="tk" x="547.9" y="48" text-anchor="middle">assistant</text></g>
<g class="a-in" style="--d:0.80s"><rect class="pad" x="517.3" y="118" width="61.3" height="28" rx="6"/><text class="tk dim" x="547.9" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="547.9" y="254" text-anchor="middle" style="--d:2.20s">0</text>
<g class="a-in" style="--d:0.33s"><rect class="n" x="580.6" y="30" width="27.0" height="28" rx="6"/><text class="tk" x="594.1" y="48" text-anchor="middle">\n</text></g>
<g class="a-in" style="--d:0.83s"><rect class="pad" x="580.6" y="118" width="27.0" height="28" rx="6"/><text class="tk dim" x="594.1" y="136" text-anchor="middle">-100</text></g><text class="tm a-in" x="594.1" y="254" text-anchor="middle" style="--d:2.22s">0</text>
<g class="a-in" style="--d:0.36s"><rect class="n-acc" x="609.6" y="30" width="38.5" height="28" rx="6"/><text class="tk" x="628.8" y="48" text-anchor="middle">Dhaka</text></g>
<g class="pt" tabindex="0" data-tip="−log p = 2.31" data-tip-sub="p = 0.10 for this token, given everything before it" aria-label="label Dhaka, loss 2.31"><rect class="hit" x="609.6" y="118" width="38.5" height="28"/><g class="a-pop" style="--d:1.20s"><rect class="n-acc" x="609.6" y="118" width="38.5" height="28" rx="6"/><text class="tk" x="628.8" y="136" text-anchor="middle">Dhaka</text></g></g>
<g class="a-in" style="--d:2.20s"><rect class="s1" x="619.8" y="207.2" width="18" height="50.8" rx="3"/><text class="t" x="628.8" y="199.2" text-anchor="middle">2.31</text></g>
<path class="hop a-draw" pathLength="1" d="M594.1 60C594.1 92 628.8 84 628.8 114" marker-end="url(#fig-mask-ah)" style="--d:0.90s;--t:.5s"/>
<g class="a-in" style="--d:0.39s"><rect class="n-acc" x="650.1" y="30" width="27.0" height="28" rx="6"/><text class="tk" x="663.6" y="48" text-anchor="middle">.</text></g>
<g class="pt" tabindex="0" data-tip="−log p = 0.42" data-tip-sub="p = 0.66 for this token, given everything before it" aria-label="label ., loss 0.42"><rect class="hit" x="650.1" y="118" width="27.0" height="28"/><g class="a-pop" style="--d:1.45s"><rect class="n-acc" x="650.1" y="118" width="27.0" height="28" rx="6"/><text class="tk" x="663.6" y="136" text-anchor="middle">.</text></g></g>
<g class="a-in" style="--d:2.45s"><rect class="s1" x="654.6" y="248.8" width="18" height="9.2" rx="3"/><text class="t" x="663.6" y="240.8" text-anchor="middle">0.42</text></g>
<path class="hop a-draw" pathLength="1" d="M628.8 60C628.8 92 663.6 84 663.6 114" marker-end="url(#fig-mask-ah)" style="--d:1.15s;--t:.5s"/>
<g class="a-in" style="--d:0.42s"><rect class="n-acc" x="679.1" y="30" width="67.0" height="28" rx="6"/><text class="tk" x="712.6" y="48" text-anchor="middle">&lt;|im_end|&gt;</text></g>
<g class="pt" tabindex="0" data-tip="−log p = 0.11" data-tip-sub="p = 0.90 for this token, given everything before it" aria-label="label &lt;|im_end|&gt;, loss 0.11"><rect class="hit" x="679.1" y="118" width="67.0" height="28"/><g class="a-pop" style="--d:1.70s"><rect class="n-acc" x="679.1" y="118" width="67.0" height="28" rx="6"/><text class="tk" x="712.6" y="136" text-anchor="middle">&lt;|im_end|&gt;</text></g></g>
<g class="a-in" style="--d:2.70s"><rect class="s1" x="703.6" y="255.6" width="18" height="2.4" rx="3"/><text class="t" x="712.6" y="247.6" text-anchor="middle">0.11</text></g>
<path class="hop a-draw" pathLength="1" d="M663.6 60C663.6 92 712.6 84 712.6 114" marker-end="url(#fig-mask-ah)" style="--d:1.40s;--t:.5s"/>
<line class="grid" x1="14" y1="258.5" x2="746" y2="258.5"/>
<text class="a-in" x="14" y="290" style="--d:3.0s"><tspan class="t2">Loss for this example: </tspan><tspan class="t">(2.31 + 0.42 + 0.11) / 3 = 0.95</tspan><tspan class="t2" dx="8">The 12 prompt tokens add nothing to the loss or the gradient.</tspan></text>
</svg>
</div>
<figcaption>The labels are a copy of the input with the prompt replaced by <code>-100</code>, the value PyTorch's cross-entropy ignores. Arrows show the shift: position <em>t</em> predicts token <em>t</em>+1, so the token before the answer predicts its first word. The end-of-turn token is a label too; that is how the model learns to stop. Hover an answer token for its probability.</figcaption>
</figure>
{:/nomarkdown}

Written with a mask $$m_t$$ over the full token sequence $$s_1, \dots, s_N$$, the loss for one example becomes:

$$
\mathcal{L} = -\frac{\sum_{t=1}^{N} m_t \log p_\theta(s_t \mid s_{<t})}{\sum_{t=1}^{N} m_t},
\qquad
m_t = \begin{cases} 1 & \text{if } s_t \text{ belongs to an assistant turn} \\ 0 & \text{otherwise} \end{cases}
$$

Why mask the prompt? It is given at inference time, so the model never has to produce it. User messages are noisy and varied, and gradient spent imitating them is gradient not spent on answers. And a model trained on full conversations learns to write the user's next message too, which is exactly the rambling you get from a badly fine-tuned model. Masking is a strong default, not a law: Shi et al. (2024) found that also training on the instruction can help when responses are short and the dataset is small. If your data looks like that, measure both.

Two details are easy to get wrong:

- **Train on the end-of-turn token.** <code>&lt;&#124;im_end&#124;&gt;</code> (or <code>&lt;&#124;eot_id&#124;&gt;</code>, or whatever your model uses) must be a label. It is the only way the model learns where an answer ends. Models fine-tuned without it never stop talking.
- **Multi-turn conversations:** mask every system and user token and train on every assistant turn, not only the last one. Each earlier assistant turn is free extra supervision.

The labels are easiest to build from two renderings of the same conversation: the prompt alone, ending with the assistant header, and the full conversation. Everything after the prompt is the answer.

```python
def build_example(messages, tokenizer):
    """messages: [{"role": "user", "content": ...}, ..., {"role": "assistant", "content": ...}]"""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    # the prompt must tokenize as an exact prefix of the full conversation
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not render the prompt as a prefix")

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
```

Inside the model, the shift and the masked mean are a few lines. Hugging Face models do this for you when you pass `labels`, but it is worth seeing once:

```python
import torch.nn.functional as F

def sft_loss(logits, labels):
    # logits: [batch, seq, vocab]; labels: [batch, seq], -100 on prompt and padding
    shift_logits = logits[:, :-1, :]      # position t ...
    shift_labels = labels[:, 1:]          # ... predicts token t + 1
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,                # masked tokens add no loss and no gradient
        reduction="mean",                 # mean over the unmasked tokens only
    )
```

## 5. Chat templates: the format is part of the model

To the model, the template's special tokens are just more tokens. If you fine-tune with one layout and serve with another, every request arrives in a format the model was never trained on, and quality drops in ways that are hard to trace. Three rules prevent nearly all of it:

1. **Render with `tokenizer.apply_chat_template`** at training time and at inference time. Never hand-build the special tokens in an f-string.
2. **Keep the model's own template.** Fine-tuning an instruct model? Use its template. Starting from a base model with no template? Pick one, set `tokenizer.chat_template`, and ship that tokenizer with the weights.
3. **Watch the pad token.** Many tokenizers have no pad token, and the usual fix is `tokenizer.pad_token = tokenizer.eos_token`. That is safe only if your labels are built as above. If you instead derive labels by masking every position where `input_ids == pad_token_id`, you also mask the real end-of-sequence token, and the model never learns to stop.

The cheapest test in all of fine-tuning is to decode exactly what the loss will see, before you train:

```python
ex = build_example(messages, tokenizer)
target_ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
print(repr(tokenizer.decode(target_ids)))
# 'Dhaka.<|im_end|>\n'   answer + end-of-turn token, no prompt
```

## 6. Padding, packing and attention boundaries

A GPU processes a batch as a rectangle: rows times sequence length. SFT examples vary wildly in length, so a batch padded to its longest row is mostly padding. If $$R$$ rows of length $$L$$ hold examples of lengths $$n_i$$, the fraction of compute doing real work is:

$$
u = \frac{\sum_i n_i}{R \cdot L}
$$

**Packing** concatenates several examples into each row so that $$u$$ approaches 1. A simple first-fit-decreasing pass (sort by length, drop each example into the first row it fits) gets close.

{::nomarkdown}
<figure class="viz" id="fig-pack">
<div class="viz-head"><span class="viz-label">FIG. 4 // PADDING VS PACKING</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 426" role="img" aria-labelledby="fig-pack-t fig-pack-d">
<title id="fig-pack-t">Padding wastes compute; packing needs a block-diagonal attention mask</title>
<desc id="fig-pack-d">Eight examples of 512, 140, 260, 90, 300, 180, 200, 330 tokens. Padded to 512 tokens each, only 49% of the 4096 token slots hold real tokens. Packed back to back with first-fit decreasing into 4 rows, 98% are used. With a plain causal mask, a packed row lets tokens attend to earlier examples in the same row; a block-diagonal mask with position ids that restart at each example keeps them separate.</desc>
<text class="tm ta" x="18" y="20">PADDED // ONE EXAMPLE PER ROW</text>
<rect class="pad" x="18" y="32" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="32" width="330.0" height="16" rx="3" style="--d:0.00s"/><text class="tm" x="342.0" y="44" text-anchor="end">512</text>
<rect class="pad" x="18" y="55" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="55" width="90.2" height="16" rx="3" style="--d:0.05s"/><text class="tm" x="102.2" y="67" text-anchor="end">140</text>
<rect class="pad" x="18" y="78" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="78" width="167.6" height="16" rx="3" style="--d:0.10s"/><text class="tm" x="179.6" y="90" text-anchor="end">260</text>
<rect class="pad" x="18" y="101" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="101" width="58.0" height="16" rx="3" style="--d:0.15s"/><text class="tm" x="70.0" y="113" text-anchor="end">90</text>
<rect class="pad" x="18" y="124" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="124" width="193.4" height="16" rx="3" style="--d:0.20s"/><text class="tm" x="205.4" y="136" text-anchor="end">300</text>
<rect class="pad" x="18" y="147" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="147" width="116.0" height="16" rx="3" style="--d:0.25s"/><text class="tm" x="128.0" y="159" text-anchor="end">180</text>
<rect class="pad" x="18" y="170" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="170" width="128.9" height="16" rx="3" style="--d:0.30s"/><text class="tm" x="140.9" y="182" text-anchor="end">200</text>
<rect class="pad" x="18" y="193" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="193" width="212.7" height="16" rx="3" style="--d:0.35s"/><text class="tm" x="224.7" y="205" text-anchor="end">330</text>
<text class="a-in" x="18" y="230" style="--d:0.6s"><tspan class="t">49% useful</tspan><tspan class="t2" dx="8">2,012 of 4,096 slots, 8 rows</tspan></text>
<text class="tm ta" x="410" y="20">PACKED // EXAMPLES BACK TO BACK</text>
<rect class="pad" x="410" y="32" width="330" height="16" rx="3"/>
<rect class="chunk a-in" x="411.0" y="32" width="328.0" height="16" rx="3" style="--d:0.90s"/><text class="tm a-in" x="733.0" y="44" text-anchor="end" style="--d:0.90s">512</text>
<rect class="pad" x="410" y="55" width="330" height="16" rx="3"/>
<rect class="chunk a-in" x="411.0" y="55" width="210.7" height="16" rx="3" style="--d:1.02s"/><text class="tm a-in" x="615.7" y="67" text-anchor="end" style="--d:1.02s">330</text>
<rect class="chunk a-in" x="623.7" y="55" width="114.0" height="16" rx="3" style="--d:1.14s"/><text class="tm a-in" x="731.7" y="67" text-anchor="end" style="--d:1.14s">180</text>
<rect class="pad" x="410" y="78" width="330" height="16" rx="3"/>
<rect class="chunk a-in" x="411.0" y="78" width="191.4" height="16" rx="3" style="--d:1.26s"/><text class="tm a-in" x="596.4" y="90" text-anchor="end" style="--d:1.26s">300</text>
<rect class="chunk a-in" x="604.4" y="78" width="126.9" height="16" rx="3" style="--d:1.38s"/><text class="tm a-in" x="725.3" y="90" text-anchor="end" style="--d:1.38s">200</text>
<rect class="pad" x="410" y="101" width="330" height="16" rx="3"/>
<rect class="chunk a-in" x="411.0" y="101" width="165.6" height="16" rx="3" style="--d:1.50s"/><text class="tm a-in" x="570.6" y="113" text-anchor="end" style="--d:1.50s">260</text>
<rect class="chunk a-in" x="578.6" y="101" width="88.2" height="16" rx="3" style="--d:1.62s"/><text class="tm a-in" x="660.8" y="113" text-anchor="end" style="--d:1.62s">140</text>
<rect class="chunk a-in" x="668.8" y="101" width="56.0" height="16" rx="3" style="--d:1.74s"/><text class="tm a-in" x="718.8" y="113" text-anchor="end" style="--d:1.74s">90</text>
<text class="a-in" x="410" y="138" style="--d:2.0s"><tspan class="t">98% useful</tspan><tspan class="t2" dx="8">2,012 of 2,048 slots, 4 rows</tspan></text>
<text class="tm ta" x="18" y="262">NAIVE CAUSAL MASK</text>
<rect class="att a-in" x="18" y="276" width="9" height="9" style="--d:2.20s"/>
<rect class="att a-in" x="18" y="286" width="9" height="9" style="--d:2.24s"/>
<rect class="att a-in" x="28" y="286" width="9" height="9" style="--d:2.24s"/>
<rect class="att a-in" x="18" y="296" width="9" height="9" style="--d:2.27s"/>
<rect class="att a-in" x="28" y="296" width="9" height="9" style="--d:2.27s"/>
<rect class="att a-in" x="38" y="296" width="9" height="9" style="--d:2.27s"/>
<rect class="att a-in" x="18" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="28" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="38" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="48" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="18" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="28" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="38" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="48" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="58" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="leak a-in" x="18" y="326" width="9" height="9" style="--d:2.38s"/>
<rect class="leak a-in" x="28" y="326" width="9" height="9" style="--d:2.38s"/>
<rect class="leak a-in" x="38" y="326" width="9" height="9" style="--d:2.38s"/>
<rect class="leak a-in" x="48" y="326" width="9" height="9" style="--d:2.38s"/>
<rect class="leak a-in" x="58" y="326" width="9" height="9" style="--d:2.38s"/>
<rect class="att a-in" x="68" y="326" width="9" height="9" style="--d:2.38s"/>
<rect class="leak a-in" x="18" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="leak a-in" x="28" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="leak a-in" x="38" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="leak a-in" x="48" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="leak a-in" x="58" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="att a-in" x="68" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="att a-in" x="78" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="leak a-in" x="18" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="leak a-in" x="28" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="leak a-in" x="38" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="leak a-in" x="48" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="leak a-in" x="58" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="att a-in" x="68" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="att a-in" x="78" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="att a-in" x="88" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="leak a-in" x="18" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="leak a-in" x="28" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="leak a-in" x="38" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="leak a-in" x="48" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="leak a-in" x="58" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="68" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="78" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="88" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="98" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="leak a-in" x="18" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="28" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="38" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="48" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="58" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="68" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="78" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="88" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="98" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="att a-in" x="108" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="leak a-in" x="18" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="28" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="38" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="48" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="58" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="68" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="78" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="88" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="98" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="att a-in" x="108" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="att a-in" x="118" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="leak a-in" x="18" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="28" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="38" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="48" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="58" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="68" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="78" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="88" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="leak a-in" x="98" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="att a-in" x="108" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="att a-in" x="118" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="att a-in" x="128" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="ln-faint" x="17.5" y="275.5" width="120" height="120" fill="none"/>
<text class="t2" x="152" y="296">Red cells: tokens attend</text><text class="t2" x="152" y="314">to other examples in</text>
<text class="t2" x="152" y="332">the same row.</text>
<text class="tm ta" x="410" y="262">BLOCK-DIAGONAL MASK</text>
<rect class="att a-in" x="410" y="276" width="9" height="9" style="--d:2.20s"/>
<rect class="att a-in" x="410" y="286" width="9" height="9" style="--d:2.24s"/>
<rect class="att a-in" x="420" y="286" width="9" height="9" style="--d:2.24s"/>
<rect class="att a-in" x="410" y="296" width="9" height="9" style="--d:2.27s"/>
<rect class="att a-in" x="420" y="296" width="9" height="9" style="--d:2.27s"/>
<rect class="att a-in" x="430" y="296" width="9" height="9" style="--d:2.27s"/>
<rect class="att a-in" x="410" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="420" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="430" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="440" y="306" width="9" height="9" style="--d:2.31s"/>
<rect class="att a-in" x="410" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="420" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="430" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="440" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="450" y="316" width="9" height="9" style="--d:2.34s"/>
<rect class="att a-in" x="460" y="326" width="9" height="9" style="--d:2.38s"/>
<rect class="att a-in" x="460" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="att a-in" x="470" y="336" width="9" height="9" style="--d:2.41s"/>
<rect class="att a-in" x="460" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="att a-in" x="470" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="att a-in" x="480" y="346" width="9" height="9" style="--d:2.45s"/>
<rect class="att a-in" x="460" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="470" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="480" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="490" y="356" width="9" height="9" style="--d:2.48s"/>
<rect class="att a-in" x="500" y="366" width="9" height="9" style="--d:2.52s"/>
<rect class="att a-in" x="500" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="att a-in" x="510" y="376" width="9" height="9" style="--d:2.55s"/>
<rect class="att a-in" x="500" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="att a-in" x="510" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="att a-in" x="520" y="386" width="9" height="9" style="--d:2.58s"/>
<rect class="ln-faint" x="409.5" y="275.5" width="120" height="120" fill="none"/>
<text class="t2" x="544" y="296">Each example sees only</text><text class="t2" x="544" y="314">itself. Position ids</text>
<text class="t2" x="544" y="332">restart at every example.</text>
<text class="tm a-in" x="410" y="414" style="--d:3.0s">position_ids 0 1 2 3 4 0 1 2 3 0 1 2</text>
</svg>
</div>
<details class="viz-data"><summary>View the numbers</summary><table><thead><tr><th></th><th>Rows</th><th>Token slots</th><th>Real tokens</th><th>Useful</th></tr></thead><tbody><tr><td>Padded</td><td>8</td><td>4096</td><td>2012</td><td>49.1%</td></tr><tr><td>Packed</td><td>4</td><td>2048</td><td>2012</td><td>98.2%</td></tr></tbody></table></details>
<figcaption>Top: the same 8 examples padded (left) and packed with first-fit decreasing (right). Packing halves the number of rows, so it roughly halves the compute. Bottom: a packed row of three short examples. With a plain causal mask (left) the red cells let the second and third example read the ones before them; a block-diagonal mask (right) prevents it.</figcaption>
</figure>
{:/nomarkdown}

Packing has a trap. With a plain causal mask, a token in the second example of a row can attend to the first example, which is an unrelated conversation. The model learns from context it will never have at inference, and some of the gradient goes into exploiting it. The fix is a **block-diagonal** mask, so each example sees only itself, and **position ids that restart at zero** for every example.

You rarely build that mask by hand. In Hugging Face `transformers`, `DataCollatorWithFlattening` concatenates a batch into a single row and returns the restarting `position_ids`, and FlashAttention 2 kernels use them to keep examples apart. Load the model with `attn_implementation="flash_attention_2"` to use it. A side benefit: attention cost within a row drops from $$\big(\sum_i n_i\big)^2$$ to $$\sum_i n_i^2$$, because examples never attend to each other.

## 7. Normalizing the loss, and the gradient accumulation trap

Once a batch holds examples of different lengths, you have to decide how they are weighted. Two choices are common:

$$
\mathcal{L}_{\text{token}} = \frac{\sum_{i}\sum_{t} m_{i,t}\,\ell_{i,t}}{\sum_{i}\sum_{t} m_{i,t}}
\qquad\qquad
\mathcal{L}_{\text{seq}} = \frac{1}{B}\sum_{i=1}^{B} \frac{\sum_{t} m_{i,t}\,\ell_{i,t}}{\sum_{t} m_{i,t}}
$$

The token mean gives every answer token equal weight, so long answers dominate. The sequence mean gives every example equal weight, so each token of a short answer counts more. Most libraries default to the token mean. Either is defensible; the bug is getting a mixture of the two by accident.

That accident happens with **gradient accumulation**, where one optimizer step is assembled from several micro-batches. If each micro-batch computes its own token mean and the means are averaged, the result is neither formula. Take two micro-batches: the first has 1,000 answer tokens with summed loss 500 (mean 0.5), the second has 10 tokens with summed loss 20 (mean 2.0). Then:

$$
\underbrace{\frac{500 + 20}{1000 + 10} = 0.515}_{\text{token mean of the full batch}}
\qquad\text{vs}\qquad
\underbrace{\tfrac{1}{2}\left(0.5 + 2.0\right) = 1.25}_{\text{mean of micro-batch means}}
$$

Each token of the short micro-batch gets 100 times the weight of a token in the long one, and the effective objective now depends on how your data happened to be split. This exact mismatch was found in widely used training code, including the Hugging Face Trainer, and fixed in late 2024. If you write the loop yourself, sum the losses and divide once by the total token count of the whole step:

```python
import torch.nn.functional as F

def accumulation_step(model, optimizer, micro_batches):
    # count every answer token in the whole optimizer step first
    # (with several GPUs, all-reduce this count so each rank divides by the same number)
    n_tokens = sum((mb["labels"][:, 1:] != -100).sum() for mb in micro_batches)
    for mb in micro_batches:
        logits = model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"]).logits
        loss_sum = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            mb["labels"][:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        (loss_sum / n_tokens).backward()   # gradients add up to the true token mean
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

## 8. Full fine-tuning: where the memory goes

Updating every weight sounds like it needs memory for the weights. It needs about eight times that. With mixed-precision training and Adam, each of the $$\Psi$$ parameters carries:

$$
M_{\text{state}} = \underbrace{2\Psi}_{\text{bf16 weights}} + \underbrace{2\Psi}_{\text{bf16 grads}} + \underbrace{4\Psi}_{\text{fp32 master}} + \underbrace{4\Psi + 4\Psi}_{\text{Adam } m,\ v} = 16\,\Psi \ \text{bytes}
$$

The fp32 master copy exists because bf16 cannot represent small updates to large weights. The two Adam tensors come straight from its update rule, which keeps a running mean and a running mean of squares of each gradient:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2,
\qquad
\theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

Here $$\hat m_t$$ and $$\hat v_t$$ are the bias-corrected moments, and AdamW adds a decoupled weight decay term $$-\eta\lambda\theta_{t-1}$$. For a 7B model, $$16\Psi$$ is **112 GB before a single activation is stored**. That is more than one 80 GB GPU.

Activations come on top and scale with batch size, sequence length, hidden size and depth. **Gradient checkpointing** keeps only each layer's input and recomputes the rest during the backward pass, trading roughly a third more compute for a large cut in activation memory. To fit the model state itself you have two options: shard it across GPUs with ZeRO-3 or FSDP, which divides the $$16\Psi$$ by the number of GPUs, or stop training most of the parameters. That second option is LoRA.

## 9. LoRA: learn a low-rank update

**LoRA** (Hu et al., 2021) freezes the pretrained weight $$W_0 \in \mathbb{R}^{d \times k}$$ and learns the update as a product of two thin matrices:

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A x,
\qquad
B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k},\quad r \ll \min(d, k)
$$

The bet behind it is that the change fine-tuning makes to a weight matrix has low *intrinsic rank*: a handful of directions do most of the work. $$BA$$ has rank at most $$r$$, and it needs $$r(d + k)$$ parameters instead of $$dk$$.

{::nomarkdown}
<figure class="viz" id="fig-lora">
<div class="viz-head"><span class="viz-label">FIG. 5 // A LORA ADAPTER ON ONE LINEAR LAYER</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 314" role="img" aria-labelledby="fig-lora-t fig-lora-d">
<title id="fig-lora-t">LoRA adds a trainable low-rank path next to a frozen weight matrix</title>
<desc id="fig-lora-d">The input x goes through the frozen pretrained matrix W0 of size d by k, and in parallel through a trainable matrix A that projects it down to r = 16 numbers, then a trainable matrix B that projects back up to d. The low-rank path is scaled by alpha over r and added to W0 x. With d = k = 4096, W0 has 16,777,216 parameters and A plus B have 131,072, which is 0.78%.</desc>
<defs><marker id="fig-lora-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<g class="a-in" style="--d:0s"><rect class="n" x="26" y="96" width="16" height="128" rx="4"/><text class="t big" x="34" y="246" text-anchor="middle">x</text><text class="tm" x="34" y="262" text-anchor="middle">k = 4096</text></g>
<path class="flow" d="M42 160H82V92H122" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.15s"><rect class="n" x="126" y="28" width="128" height="128" rx="8"/><rect class="glow" x="126" y="28" width="128" height="128" rx="8" style="--d:0.6s;--p:4s"/><text class="t big" x="190" y="86" text-anchor="middle">W₀</text><text class="t2" x="190" y="106" text-anchor="middle">d × k</text><text class="tm" x="190" y="140" text-anchor="middle">FROZEN</text></g>
<path class="flow" d="M254 92H566V140" marker-end="url(#fig-lora-ah)"/>
<circle class="packet a-travel" cx="262" cy="92" r="3.5" style="--dx:296px;--t:1.4s;--d:1.0s"/>
<text class="t2 a-in" x="410" y="84" text-anchor="middle" style="--d:0.5s">W₀x: the pretrained path, untouched</text>
<path class="flow" d="M82 160V240H122" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.35s"><rect class="n-acc" x="126" y="232" width="128" height="16" rx="4"/><rect class="glow" x="126" y="232" width="128" height="16" rx="4" style="--d:1.5s;--p:4s"/><text class="t" x="190" y="222" text-anchor="middle">A</text><text class="t2" x="190" y="270" text-anchor="middle">r × k, random init</text></g>
<path class="flow" d="M254 240H282" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.5s"><rect class="n-acc" x="286" y="232" width="16" height="16" rx="3"/><text class="tm ta" x="294" y="222" text-anchor="middle">r = 16</text></g>
<path class="flow" d="M302 240H330" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.65s"><rect class="n-acc" x="334" y="176" width="16" height="128" rx="4"/><rect class="glow" x="334" y="176" width="16" height="128" rx="4" style="--d:2.0s;--p:4s"/><text class="t" x="364" y="198">B</text><text class="t2" x="364" y="216">d × r</text><text class="t2" x="364" y="232">starts at 0</text></g>
<path class="flow" d="M350 286H436" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.8s"><rect class="n" x="440" y="270" width="64" height="32" rx="8"/><text class="t" x="472" y="291" text-anchor="middle">× α/r</text></g>
<path class="flow" d="M504 286H566V180" marker-end="url(#fig-lora-ah)"/>
<circle class="packet a-travel" cx="512" cy="286" r="3.5" style="--dx:46px;--t:.8s;--d:2.3s"/>
<g class="a-in" style="--d:0.95s"><circle class="n-acc" cx="566" cy="160" r="16"/><path class="ln" d="M558 160H574M566 152V168" style="stroke:#e8b84e"/></g>
<path class="flow" d="M582 160H628" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:1.1s"><rect class="n-acc" x="632" y="96" width="16" height="128" rx="4"/><text class="t big" x="640" y="246" text-anchor="middle">h</text><text class="tm" x="640" y="262" text-anchor="middle">d = 4096</text></g>
<text class="a-in" x="744" y="30" text-anchor="end" style="--d:1.3s"><tspan class="t">h = W₀x + (α/r)·BAx</tspan></text>
<text class="a-in" x="744" y="52" text-anchor="end" style="--d:1.4s"><tspan class="tm">W₀ 16,777,216 FROZEN</tspan></text>
<text class="a-in" x="744" y="68" text-anchor="end" style="--d:1.5s"><tspan class="tm ta">A + B 131,072 TRAINABLE (0.78%)</tspan></text>
</svg>
</div>
<figcaption>The pretrained path is untouched. The trainable path squeezes the input through r = 16 numbers and expands it back, so the update BA can have rank at most 16. B starts at zero, so the adapted model is exactly the pretrained model at step 0. For a 4096 × 4096 projection that is 131,072 trainable parameters instead of 16,777,216.</figcaption>
</figure>
{:/nomarkdown}

The initialization is deliberate. $$A$$ starts random and $$B$$ starts at **zero**, so $$\Delta W = 0$$ and the model at step 0 is exactly the pretrained model. That has a neat consequence you can read off the gradients. Write $$u = Ax$$ for the $$r$$-dimensional bottleneck, $$s = \alpha / r$$, and $$G = \partial L / \partial h$$ for the gradient arriving at the layer's output:

$$
\frac{\partial L}{\partial B} = s\, G\, u^{\top} = s\, G\, (Ax)^{\top},
\qquad
\frac{\partial L}{\partial A} = s\, B^{\top} G\, x^{\top}
$$

At step 0, $$B = 0$$, so $$\partial L / \partial A = 0$$: only $$B$$ moves on the first step, and $$A$$ starts learning once $$B$$ is nonzero. $$W_0$$ gets no gradient at all, which means no gradient buffer, no master copy and no Adam state for it. That is where the memory saving comes from.

The scale $$\alpha / r$$ keeps the size of the update roughly stable when you change $$r$$, so you retune the learning rate less. At higher ranks, rsLoRA (Kalajdzievski, 2023) argues for $$\alpha / \sqrt{r}$$ instead, available in `peft` as `use_rslora=True`. After training, the update folds back into the weight, $$W = W_0 + \frac{\alpha}{r} BA$$, so a merged LoRA model runs at exactly the speed of the original.

The whole method fits in a dozen lines of PyTorch:

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                                    # W0 stays frozen
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))     # B = 0, so the update starts at 0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale
```

In practice you use `peft`, which does the same thing for every matching layer:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

On a Llama-2-7B-style model, $$r = 16$$ on all seven projections of all 32 layers gives 39,976,960 trainable parameters, about **0.59%** of the model. Two findings are worth keeping in mind. The QLoRA paper found that adapting **all** linear layers, not just the attention projections, is needed to match full fine-tuning. And Biderman et al. (2024) found that LoRA *learns less and forgets less*: it underperforms full fine-tuning on large new domains such as code and maths, but it damages the model's general abilities less. For most SFT, which is about behaviour rather than new knowledge, that trade is a good one.

## 10. QLoRA: a 4-bit base under 16-bit adapters

LoRA removes the optimizer state, but the frozen weights still sit in bf16: 14 GB for 7B parameters. **QLoRA** (Dettmers et al., 2023) stores them in 4 bits instead, and trains bf16 LoRA adapters on top. It was enough to fine-tune a 65B model on a single 48 GB GPU.

The 4-bit format is **NF4**. Weights are quantized in blocks of 64. Each block is scaled by its largest absolute value, and each weight is rounded to the nearest of 16 levels $$Q_{\text{NF4}}$$, placed at quantiles of a standard normal distribution, because pretrained weights are roughly normal:

$$
c_b = \max_{j \in b} \lvert w_j \rvert,
\qquad
q_j = \arg\min_{q \in Q_{\text{NF4}}} \left\lvert \frac{w_j}{c_b} - q \right\rvert,
\qquad
\hat w_j = c_b\, q_j
$$

Equal-probability levels mean each of the 16 values is used about equally often, which is the best a 4-bit code can do for normally distributed data. The block constants cost memory too: one fp32 number per 64 weights is 0.5 bits per parameter. **Double quantization** stores those constants in 8 bits, with their own fp32 constant per 256 blocks:

$$
\frac{8}{64} + \frac{32}{64 \cdot 256} \approx 0.127 \ \text{bits per parameter}
$$

So a 7B model's frozen weights take about $$7 \times 10^9 \times 4.127 / 8 \approx 3.6$$ GB. During the forward and backward pass each block is dequantized to bf16 on the fly, so the math still runs in 16-bit, and gradients flow through the frozen weights into the adapters. QLoRA also uses paged optimizers, which move optimizer state to CPU memory during spikes instead of crashing with an out-of-memory error.

{::nomarkdown}
<figure class="viz" id="fig-mem">
<div class="viz-head"><span class="viz-label">FIG. 6 // GPU MEMORY TO FINE-TUNE A 7B MODEL</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 274" role="img" aria-labelledby="fig-mem-t fig-mem-d">
<title id="fig-mem-t">Memory for full fine-tuning, LoRA and QLoRA of a 7-billion-parameter model</title>
<desc id="fig-mem-d">Full fine-tuning with mixed-precision Adam needs about 16 bytes per parameter, about 112 GB for 7 billion parameters, before activations. LoRA keeps the bf16 weights frozen, about 14 GB, plus about 0.6 GB for adapters and their optimizer state. QLoRA stores the frozen weights in 4-bit NF4, about 3.6 GB, plus the same adapters. Dashed lines mark 24 GB and 80 GB GPUs.</desc>
<line class="grid-strong" x1="295.6" y1="34" x2="295.6" y2="232" style="stroke-dasharray:3 4"/><text class="axis" x="295.6" y="26" text-anchor="middle">24 GB GPU</text>
<line class="grid-strong" x1="542.0" y1="34" x2="542.0" y2="232" style="stroke-dasharray:3 4"/><text class="axis" x="542.0" y="26" text-anchor="middle">80 GB GPU</text>
<text class="t" x="18" y="64">Full fine-tuning</text><text class="t2" x="18" y="81">bf16 + Adam</text>
<g class="row" tabindex="0" data-tip="14.0 GB" data-tip-sub="bf16 weights: 2 bytes × 7B" aria-label="Full fine-tuning, bf16 weights: 14.0 GB"><rect class="hit" x="190.0" y="50" width="61.6" height="34"/><rect class="m1 a-grow" x="190.0" y="52" width="60.1" height="30" rx="3" style="--d:0.15s"/></g>
<g class="row" tabindex="0" data-tip="14.0 GB" data-tip-sub="bf16 gradients: 2 bytes × 7B" aria-label="Full fine-tuning, bf16 gradients: 14.0 GB"><rect class="hit" x="251.6" y="50" width="61.6" height="34"/><rect class="m2 a-grow" x="251.6" y="52" width="60.1" height="30" rx="3" style="--d:0.33s"/></g>
<g class="row" tabindex="0" data-tip="28.0 GB" data-tip-sub="fp32 master weights: 4 bytes × 7B" aria-label="Full fine-tuning, fp32 master weights: 28.0 GB"><rect class="hit" x="313.2" y="50" width="123.2" height="34"/><rect class="m3 a-grow" x="313.2" y="52" width="121.7" height="30" rx="3" style="--d:0.51s"/></g>
<g class="row" tabindex="0" data-tip="28.0 GB" data-tip-sub="Adam first moment m: 4 bytes × 7B" aria-label="Full fine-tuning, Adam first moment m: 28.0 GB"><rect class="hit" x="436.4" y="50" width="123.2" height="34"/><rect class="m4 a-grow" x="436.4" y="52" width="121.7" height="30" rx="3" style="--d:0.69s"/></g>
<g class="row" tabindex="0" data-tip="28.0 GB" data-tip-sub="Adam second moment v: 4 bytes × 7B" aria-label="Full fine-tuning, Adam second moment v: 28.0 GB"><rect class="hit" x="559.6" y="50" width="123.2" height="34"/><rect class="m5 a-grow" x="559.6" y="52" width="121.7" height="30" rx="3" style="--d:0.87s"/></g>
<text class="a-in" x="690.8" y="72" style="--d:1.40s"><tspan class="t">≈112 GB</tspan></text>
<text class="t" x="18" y="128">LoRA</text><text class="t2" x="18" y="145">bf16 base, r = 16</text>
<g class="row" tabindex="0" data-tip="14.0 GB" data-tip-sub="frozen bf16 weights: 2 bytes × 7B" aria-label="LoRA, frozen bf16 weights: 14.0 GB"><rect class="hit" x="190.0" y="114" width="61.6" height="34"/><rect class="m1 a-grow" x="190.0" y="116" width="60.1" height="30" rx="3" style="--d:0.55s"/></g>
<g class="row" tabindex="0" data-tip="0.6 GB" data-tip-sub="adapters + their Adam state: 40M params × 16 bytes" aria-label="LoRA, adapters + their Adam state: 0.6 GB"><rect class="hit" x="251.6" y="114" width="8.0" height="34"/><rect class="m6 a-grow" x="251.6" y="116" width="1.5" height="30" rx="3" style="--d:0.73s"/></g>
<text class="a-in" x="262.4" y="136" style="--d:1.26s"><tspan class="t">≈14.6 GB</tspan></text>
<text class="t" x="18" y="192">QLoRA</text><text class="t2" x="18" y="209">NF4 base, r = 16</text>
<g class="row" tabindex="0" data-tip="3.6 GB" data-tip-sub="frozen NF4 weights: ≈4.13 bits × 7B" aria-label="QLoRA, frozen NF4 weights: 3.6 GB"><rect class="hit" x="190.0" y="178" width="15.9" height="34"/><rect class="m7 a-grow" x="190.0" y="180" width="14.4" height="30" rx="3" style="--d:0.95s"/></g>
<g class="row" tabindex="0" data-tip="0.6 GB" data-tip-sub="adapters + their Adam state: 40M params × 16 bytes" aria-label="QLoRA, adapters + their Adam state: 0.6 GB"><rect class="hit" x="205.9" y="178" width="8.0" height="34"/><rect class="m6 a-grow" x="205.9" y="180" width="1.5" height="30" rx="3" style="--d:1.13s"/></g>
<text class="a-in" x="216.7" y="200" style="--d:1.66s"><tspan class="t">≈4.3 GB</tspan></text>
<rect class="m1" x="18" y="252" width="10" height="10" rx="2"/><text class="t2" x="33" y="261">bf16 weights</text>
<rect class="m2" x="126.6" y="252" width="10" height="10" rx="2"/><text class="t2" x="141.6" y="261">gradients</text>
<rect class="m3" x="216.29999999999998" y="252" width="10" height="10" rx="2"/><text class="t2" x="231.29999999999998" y="261">fp32 master</text>
<rect class="m4" x="318.59999999999997" y="252" width="10" height="10" rx="2"/><text class="t2" x="333.59999999999997" y="261">Adam m</text>
<rect class="m5" x="389.4" y="252" width="10" height="10" rx="2"/><text class="t2" x="404.4" y="261">Adam v</text>
<rect class="m7" x="460.2" y="252" width="10" height="10" rx="2"/><text class="t2" x="475.2" y="261">NF4 weights</text>
<rect class="m6" x="562.5" y="252" width="10" height="10" rx="2"/><text class="t2" x="577.5" y="261">adapters + state</text>
</svg>
</div>
<details class="viz-data"><summary>View the numbers</summary><table><thead><tr><th>Method</th><th>Component</th><th>GB</th></tr></thead><tbody><tr><td>Full fine-tuning</td><td>bf16 weights</td><td>14.00</td></tr><tr><td>Full fine-tuning</td><td>bf16 gradients</td><td>14.00</td></tr><tr><td>Full fine-tuning</td><td>fp32 master weights</td><td>28.00</td></tr><tr><td>Full fine-tuning</td><td>Adam first moment m</td><td>28.00</td></tr><tr><td>Full fine-tuning</td><td>Adam second moment v</td><td>28.00</td></tr><tr><td>Full fine-tuning</td><td>total</td><td>112.0</td></tr><tr><td>LoRA</td><td>frozen bf16 weights</td><td>14.00</td></tr><tr><td>LoRA</td><td>adapters + their Adam state</td><td>0.64</td></tr><tr><td>LoRA</td><td>total</td><td>14.6</td></tr><tr><td>QLoRA</td><td>frozen NF4 weights</td><td>3.61</td></tr><tr><td>QLoRA</td><td>adapters + their Adam state</td><td>0.64</td></tr><tr><td>QLoRA</td><td>total</td><td>4.3</td></tr></tbody></table></details>
<figcaption>Model state only, for 7 billion parameters; activations come on top and grow with batch size and sequence length. Full fine-tuning spends most of its memory on optimizer state, not weights. LoRA removes almost all of it, and QLoRA shrinks the frozen weights fourfold, which is what moves fine-tuning from a cluster to one consumer GPU.</figcaption>
</figure>
{:/nomarkdown}

Two practical notes. QLoRA is slower than LoRA, because every layer is dequantized on every step; you are buying memory with time. And when you merge the adapters, merge them into a **bf16 copy** of the base model, not into the 4-bit one, or you round the merged weights a second time.

## 11. Hyperparameters that matter

SFT is forgiving, but a few settings decide most outcomes. These are reasonable starting points, not rules:

| Setting | Full fine-tuning | LoRA / QLoRA |
|---|---|---|
| Learning rate | 1e-5 to 2e-5 | 1e-4 to 3e-4 |
| Epochs | 1 to 3 | 1 to 3 |
| Warmup | about 3% of steps | about 3% of steps |
| Schedule | cosine or linear decay | cosine or linear decay |
| Effective batch | 64 to 256 sequences | 16 to 128 sequences |
| Rank and alpha | not used | r = 8 to 64, alpha = r or 2r |
| Weight decay | 0 to 0.1 | 0 |

LoRA wants a learning rate roughly ten times higher than full fine-tuning, because the update starts at zero and lives in a much smaller space. The usual schedule is a linear warmup over $$T_w$$ steps, then cosine decay to a floor:

$$
\eta_t =
\begin{cases}
\eta_{\max}\, \dfrac{t}{T_w} & t < T_w \\[8pt]
\eta_{\min} + \tfrac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)\left(1 + \cos\!\left(\pi\, \dfrac{t - T_w}{T - T_w}\right)\right) & t \ge T_w
\end{cases}
$$

The effective batch is what the optimizer sees per step, $$B_{\text{eff}} = B_{\text{device}} \times \text{accumulation steps} \times \text{GPUs}$$. Think of it in tokens as well as sequences: with packing to 2,048 tokens, 32 sequences per step is about 65,000 tokens, prompts included.

**Epochs** deserve the most care. SFT datasets are small, and a large model memorizes them quickly. The tell-tale sign is in the loss curves:

{::nomarkdown}
<figure class="viz" id="fig-loss">
<div class="viz-head"><span class="viz-label">FIG. 7 // WHAT OVERFITTING LOOKS LIKE IN SFT (SCHEMATIC)</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 312" role="img" aria-labelledby="fig-loss-t fig-loss-d">
<title id="fig-loss-t">Schematic training and validation loss over three epochs of supervised fine-tuning</title>
<desc id="fig-loss-d">A schematic, not a measured run. Training loss falls through epoch 1, then drops sharply at the start of epochs 2 and 3 as the model sees examples it has already memorized. Validation loss falls, bottoms out near step 300 in epoch 2, then rises. The best checkpoint is at that minimum.</desc>
<line class="grid" x1="64" y1="240.0" x2="730" y2="240.0"/><text class="axis" x="54" y="244.0" text-anchor="end">0.0</text>
<line class="grid" x1="64" y1="188.5" x2="730" y2="188.5"/><text class="axis" x="54" y="192.5" text-anchor="end">0.5</text>
<line class="grid" x1="64" y1="137.0" x2="730" y2="137.0"/><text class="axis" x="54" y="141.0" text-anchor="end">1.0</text>
<line class="grid" x1="64" y1="85.5" x2="730" y2="85.5"/><text class="axis" x="54" y="89.5" text-anchor="end">1.5</text>
<line class="grid" x1="64" y1="34.0" x2="730" y2="34.0"/><text class="axis" x="54" y="38.0" text-anchor="end">2.0</text>
<text class="axis" x="64.0" y="260" text-anchor="middle">0</text>
<text class="axis" x="286.0" y="260" text-anchor="middle">200</text>
<text class="axis" x="508.0" y="260" text-anchor="middle">400</text>
<text class="axis" x="730.0" y="260" text-anchor="middle">600</text>
<line class="grid-strong" x1="286.0" y1="34" x2="286.0" y2="240"/>
<line class="grid-strong" x1="508.0" y1="34" x2="508.0" y2="240"/>
<text class="tm" x="175.0" y="26" text-anchor="middle">EPOCH 1</text>
<text class="tm" x="397.0" y="26" text-anchor="middle">EPOCH 2</text>
<text class="tm" x="619.0" y="26" text-anchor="middle">EPOCH 3</text>
<text class="t2" x="397.0" y="282" text-anchor="middle">optimizer steps</text>
<path class="s1-line a-draw" pathLength="1" d="M66.2 41.4L70.7 47.0L75.1 54.2L79.5 60.1L84.0 63.9L88.4 68.5L92.9 75.2L97.3 81.1L101.7 84.5L106.2 87.4L110.6 91.9L115.1 96.2L119.5 97.7L123.9 97.9L128.4 99.8L132.8 102.8L137.3 103.9L141.7 103.8L146.1 105.4L150.6 109.1L155.0 111.7L159.5 112.3L163.9 113.7L168.3 117.0L172.8 119.4L177.2 119.1L181.7 118.4L186.1 119.6L190.5 120.9L195.0 119.9L199.4 118.3L203.9 119.0L208.3 121.1L212.7 121.7L217.2 121.2L221.6 122.4L226.1 125.3L230.5 126.8L234.9 126.1L239.4 125.8L243.8 127.3L248.3 127.9L252.7 126.0L257.1 124.2L261.6 124.7L266.0 125.7L270.5 124.9L274.9 123.8L279.3 125.1L283.8 127.7L288.2 152.8L292.7 152.2L297.1 153.0L301.5 155.1L306.0 155.5L310.4 153.6L314.9 152.4L319.3 153.3L323.7 153.6L328.2 151.9L332.6 150.7L337.1 152.2L341.5 154.3L345.9 154.4L350.4 154.1L354.8 155.8L359.3 158.3L363.7 158.5L368.1 157.0L372.6 156.9L377.0 158.0L381.5 157.7L385.9 155.4L390.3 154.4L394.8 155.8L399.2 156.9L403.7 156.2L408.1 156.1L412.5 158.4L417.0 160.8L421.4 160.8L425.9 160.0L430.3 161.0L434.7 162.5L439.2 161.7L443.6 159.4L448.1 158.8L452.5 160.0L456.9 160.0L461.4 158.5L465.8 158.5L470.3 161.0L474.7 162.8L479.1 162.6L483.6 162.6L488.0 164.5L492.5 166.2L496.9 165.2L501.3 163.5L505.8 163.6L510.2 184.0L514.7 183.0L519.1 180.9L523.5 181.0L528.0 183.0L532.4 183.9L536.9 183.2L541.3 183.8L545.7 186.4L550.2 187.9L554.6 186.9L559.1 185.9L563.5 186.8L567.9 187.5L572.4 185.7L576.8 183.6L581.3 183.9L585.7 185.3L590.1 185.1L594.6 184.1L599.0 185.1L603.5 187.8L607.9 189.0L612.3 188.2L616.8 188.3L621.2 190.0L625.7 190.5L630.1 188.5L634.5 186.8L639.0 187.4L643.4 188.0L647.9 186.7L652.3 185.5L656.7 186.8L661.2 189.2L665.6 189.7L670.1 189.1L674.5 190.2L678.9 192.5L683.4 192.9L687.8 191.1L692.3 190.2L696.7 191.1L701.1 191.1L705.6 189.0L710.0 187.7L714.5 189.0L718.9 190.6L723.3 190.3L727.8 189.9" style="--d:.2s;--t:2.2s"/>
<rect class="zone a-in" x="397.0" y="34" width="333.0" height="206" style="--d:2.4s"/>
<text class="t2 a-in" x="722" y="52" text-anchor="end" style="--d:2.5s">validation loss rising: memorizing, not learning</text>
<path class="s2-line a-draw" pathLength="1" d="M119.5 73.1L175.0 99.9L230.5 112.3L286.0 119.5L341.5 123.6L397.0 125.2L452.5 124.6L508.0 122.6L563.5 117.4L619.0 111.2L674.5 105.1L730.0 98.9" style="--d:.6s;--t:2.0s"/>
<g class="pt" tabindex="0" data-tip="1.62" data-tip-sub="validation loss at step 50" aria-label="step 50, validation loss 1.62"><circle class="hit" cx="119.5" cy="73.1" r="12"/><circle class="mk2 a-pop" cx="119.5" cy="73.1" r="4" style="--d:0.60s"/></g>
<g class="pt" tabindex="0" data-tip="1.36" data-tip-sub="validation loss at step 100" aria-label="step 100, validation loss 1.36"><circle class="hit" cx="175.0" cy="99.9" r="12"/><circle class="mk2 a-pop" cx="175.0" cy="99.9" r="4" style="--d:0.76s"/></g>
<g class="pt" tabindex="0" data-tip="1.24" data-tip-sub="validation loss at step 150" aria-label="step 150, validation loss 1.24"><circle class="hit" cx="230.5" cy="112.3" r="12"/><circle class="mk2 a-pop" cx="230.5" cy="112.3" r="4" style="--d:0.92s"/></g>
<g class="pt" tabindex="0" data-tip="1.17" data-tip-sub="validation loss at step 200" aria-label="step 200, validation loss 1.17"><circle class="hit" cx="286.0" cy="119.5" r="12"/><circle class="mk2 a-pop" cx="286.0" cy="119.5" r="4" style="--d:1.08s"/></g>
<g class="pt" tabindex="0" data-tip="1.13" data-tip-sub="validation loss at step 250" aria-label="step 250, validation loss 1.13"><circle class="hit" cx="341.5" cy="123.6" r="12"/><circle class="mk2 a-pop" cx="341.5" cy="123.6" r="4" style="--d:1.24s"/></g>
<g class="pt" tabindex="0" data-tip="1.11" data-tip-sub="validation loss at step 300" aria-label="step 300, validation loss 1.11"><circle class="hit" cx="397.0" cy="125.2" r="12"/><circle class="mk2 a-pop" cx="397.0" cy="125.2" r="4" style="--d:1.40s"/></g>
<g class="pt" tabindex="0" data-tip="1.12" data-tip-sub="validation loss at step 350" aria-label="step 350, validation loss 1.12"><circle class="hit" cx="452.5" cy="124.6" r="12"/><circle class="mk2 a-pop" cx="452.5" cy="124.6" r="4" style="--d:1.56s"/></g>
<g class="pt" tabindex="0" data-tip="1.14" data-tip-sub="validation loss at step 400" aria-label="step 400, validation loss 1.14"><circle class="hit" cx="508.0" cy="122.6" r="12"/><circle class="mk2 a-pop" cx="508.0" cy="122.6" r="4" style="--d:1.72s"/></g>
<g class="pt" tabindex="0" data-tip="1.19" data-tip-sub="validation loss at step 450" aria-label="step 450, validation loss 1.19"><circle class="hit" cx="563.5" cy="117.4" r="12"/><circle class="mk2 a-pop" cx="563.5" cy="117.4" r="4" style="--d:1.88s"/></g>
<g class="pt" tabindex="0" data-tip="1.25" data-tip-sub="validation loss at step 500" aria-label="step 500, validation loss 1.25"><circle class="hit" cx="619.0" cy="111.2" r="12"/><circle class="mk2 a-pop" cx="619.0" cy="111.2" r="4" style="--d:2.04s"/></g>
<g class="pt" tabindex="0" data-tip="1.31" data-tip-sub="validation loss at step 550" aria-label="step 550, validation loss 1.31"><circle class="hit" cx="674.5" cy="105.1" r="12"/><circle class="mk2 a-pop" cx="674.5" cy="105.1" r="4" style="--d:2.20s"/></g>
<g class="pt" tabindex="0" data-tip="1.37" data-tip-sub="validation loss at step 600" aria-label="step 600, validation loss 1.37"><circle class="hit" cx="730.0" cy="98.9" r="12"/><circle class="mk2 a-pop" cx="730.0" cy="98.9" r="4" style="--d:2.36s"/></g>
<circle class="ring a-ping" cx="397.0" cy="125.2" r="7" style="--d:2.6s"/>
<text class="a-in" x="397.0" y="109.2" text-anchor="middle" style="--d:2.6s"><tspan class="t">best checkpoint</tspan></text>
<text class="a-in" x="516.0" y="211.2" style="--d:2.3s"><tspan class="t2">training loss drops at each new epoch</tspan></text>
<g class="a-in" style="--d:0.3s"><rect class="s1" x="64" y="296" width="14" height="4" rx="2"/><text class="t2" x="84" y="301">training loss</text><rect class="s2" x="194" y="296" width="14" height="4" rx="2"/><text class="t2" x="214" y="301">validation loss</text></g>
</svg>
</div>
<details class="viz-data"><summary>View the numbers</summary><table><thead><tr><th>Step</th><th>Validation loss</th></tr></thead><tbody><tr><td>50</td><td>1.62</td></tr><tr><td>100</td><td>1.36</td></tr><tr><td>150</td><td>1.24</td></tr><tr><td>200</td><td>1.17</td></tr><tr><td>250</td><td>1.13</td></tr><tr><td>300</td><td>1.11</td></tr><tr><td>350</td><td>1.12</td></tr><tr><td>400</td><td>1.14</td></tr><tr><td>450</td><td>1.19</td></tr><tr><td>500</td><td>1.25</td></tr><tr><td>550</td><td>1.31</td></tr><tr><td>600</td><td>1.37</td></tr></tbody></table></details>
<figcaption>A schematic of a pattern you will see often, not a measured run. Each time an epoch starts, training loss steps down because the model is now predicting text it has already seen. Validation loss is the honest number: keep the checkpoint at its minimum and stop when it turns up.</figcaption>
</figure>
{:/nomarkdown}

Evaluate on a held-out split every few hundred steps, keep the checkpoint with the lowest validation loss, and treat a rising validation loss as a stop sign, not noise. If you need more signal from the same data, more diverse data beats more epochs.

## 12. Data: quality beats quantity

Section 2 showed that SFT pushes the model to cover every behaviour in the data. The practical version of that result is LIMA (Zhou et al., 2023): a 65B model fine-tuned on just **1,000** carefully chosen examples produced answers that people often rated as good as those from models trained on far more. A thousand great examples teach the format and the style; a hundred thousand mediocre ones teach the mediocrity.

What a good SFT set looks like:

<ul class="checklist">
  <li><strong>Correct.</strong> Every response is one you would be happy to ship. The model will copy the mistakes as faithfully as the good answers.</li>
  <li><strong>Consistent.</strong> One format, one tone, one way of refusing. Mixed conventions produce a model that switches between them at random.</li>
  <li><strong>Diverse in prompts.</strong> Cover the range of requests you expect, including vague, long and adversarial ones, not twenty paraphrases of the same question.</li>
  <li><strong>Deduplicated.</strong> Remove exact and near duplicates (MinHash works well). Repeats act like extra epochs on those examples.</li>
  <li><strong>Decontaminated.</strong> Remove anything that overlaps your evaluation sets, or your evaluation measures memory.</li>
  <li><strong>Mixed with general data</strong> when the task is narrow. A slice of general instruction data helps the model keep the abilities you are not training.</li>
</ul>

## 13. Evaluating a fine-tuned model

Validation loss tells you when to stop. It does not tell you whether the model is better. For that you need evaluations that look like the job:

- **Task metrics** on a held-out set: exact match or F1 for extraction, schema validity for JSON output, unit tests (pass@k) for code.
- **Side-by-side comparisons** against the base model and against a strong prompt on the base model. If careful prompting or retrieval gets you 95% of the way, fine-tuning may not be worth maintaining.
- **LLM-as-judge** with a written rubric, for open-ended answers. Check the judge against a few dozen human labels before you trust its numbers.
- **Regression checks** on general ability: instruction following, a general knowledge benchmark and your safety prompts. This is where forgetting shows up.

Fix the decoding settings (greedy, or a fixed temperature and seed) so differences come from the model and not from sampling.

## 14. A complete QLoRA SFT script

Putting it together: a 4-bit base model, LoRA on every linear layer, masked labels, and a validation split that picks the best checkpoint. The data is a JSONL file with one conversation per line:

```json
{"messages": [{"role": "user", "content": "Capital of Bangladesh?"}, {"role": "assistant", "content": "Dhaka."}]}
```

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

MODEL = "Qwen/Qwen2.5-7B-Instruct"   # any causal LM whose tokenizer has a chat template
MAX_LEN = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # safe: labels are built separately

def build_example(row):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not render the prompt as a prefix")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})
ds = ds.map(build_example, remove_columns=ds["train"].column_names)
ds = ds.filter(lambda r: len(r["input_ids"]) <= MAX_LEN)   # drop, don't truncate: truncation can cut off the answer

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)   # also enables gradient checkpointing
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="sft-qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,        # 32 sequences per optimizer step
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=2,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,          # keep the checkpoint with the lowest validation loss
    metric_for_best_model="eval_loss",
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
)
trainer.train()
model.save_pretrained("sft-qlora/adapter")
```

`DataCollatorForSeq2Seq` works for causal models too: it pads `input_ids` with the pad token and `labels` with `-100`, so padding never reaches the loss. To serve the result, merge the adapter into a bf16 copy of the base model:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)   # torch_dtype= on transformers before 4.56
merged = PeftModel.from_pretrained(base, "sft-qlora/adapter").merge_and_unload()
merged.save_pretrained("sft-merged")
AutoTokenizer.from_pretrained(MODEL).save_pretrained("sft-merged")
```

## Final thoughts

Supervised fine-tuning is the least exotic step in training a language model. The objective is the one the model was pretrained with, and the code is a masked cross-entropy. The leverage is everywhere around it: which tokens carry the loss, how examples are weighted, which parameters are allowed to move, and above all which examples you show it.

If I had to keep three ideas from this post, they would be these. SFT minimizes forward KL, so the model will imitate everything in your data, and curating the data is the job. Mask the prompt, keep the end-of-turn token, and decode your labels before you train. And start with QLoRA on all linear layers, a validation split and one or two epochs; reach for full fine-tuning only when an evaluation tells you LoRA is the bottleneck.

## References

- Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). The InstructGPT paper.
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
- Zhou et al. (2023). [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206).
- Biderman et al. (2024). [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673).
- Gekhman et al. (2024). [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904)
- Shi et al. (2024). [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394).
- Kalajdzievski (2023). [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732).
- Rajbhandari et al. (2020). [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
- Hugging Face. [PEFT documentation](https://huggingface.co/docs/peft) and [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating).
