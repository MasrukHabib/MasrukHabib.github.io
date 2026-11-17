---
title: "LLM-এর সুপারভাইজড ফাইন-টিউনিং: লস ফাংশন থেকে LoRA পর্যন্ত"
category: "LLM"
tags: [SFT, "ফাইন-টিউনিং", LoRA, QLoRA, PyTorch, Transformers]
excerpt: "সুপারভাইজড ফাইন-টিউনিং আসলে কী অপটিমাইজ করে, লস কেন মাস্ক করা হয়, একটি গ্রেডিয়েন্ট ধাপ কীভাবে পরের টোকেনের ডিস্ট্রিবিউশন নতুন করে গড়ে দেয়, আর LoRA ও QLoRA কীভাবে একটি 7B মডেলকে একটিমাত্র GPU-তে এঁটে দেয়। গণিত আর কোডসহ।"
description: "লার্জ ল্যাঙ্গুয়েজ মডেলের সুপারভাইজড ফাইন-টিউনিং (SFT) নিয়ে একটি টেকনিক্যাল গাইড: পরের টোকেন প্রেডিকশনের অবজেক্টিভ, ক্রস-এনট্রপির গ্রেডিয়েন্ট, লস মাস্কিং ও চ্যাট টেমপ্লেট, প্যাকিং ও গ্রেডিয়েন্ট অ্যাকুমুলেশন, ফুল ফাইন-টিউনিংয়ের মেমরির হিসাব, LoRA ও QLoRA, হাইপারপ্যারামিটার, ডেটার মান ও মূল্যায়ন, সাথে PyTorch আর Hugging Face কোড।"
image: https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=1600&auto=format&fit=crop
image_alt: "AI অক্ষর দুটির একটি বিমূর্ত 3D রেন্ডার"
math: true
---

প্রিট্রেন করা একটি বেস মডেলকে “বাংলাদেশের রাজধানী কী?” জিজ্ঞেস করে দেখুন, বেশ সম্ভাবনা আছে যে সে উত্তরে আরেকটা প্রশ্নই করে বসবে। সে লাখ লাখ কুইজের পাতা পড়েছে, আর সেসব পাতায় একটা প্রশ্নের পরে সাধারণত আরও প্রশ্নই থাকে। এতে কোনো গোলমাল নেই। মডেলটি ঠিক সেটাই করছে, যা করার জন্য তাকে ট্রেইন করা হয়েছে: টেক্সট চালিয়ে যাওয়া।

**সুপারভাইজড ফাইন-টিউনিং (SFT)** হলো সেই ধাপ, যা এই টেক্সট চালিয়ে যাওয়া মডেলটিকে উত্তর দিতে পারে এমন মডেলে বদলে দেয়। বেশিরভাগ মানুষ অবাক হন এটা দেখে যে আসলে কত কম জিনিস বদলায়। আর্কিটেকচার একই থাকে। অবজেক্টিভও সেই একই পরের টোকেন প্রেডিকশন (next-token prediction)। বদলায় শুধু ডেটা, আর লসের একটি খুঁটিনাটি: মডেলকে নম্বর দেওয়া হয় কেবল উত্তরের ওপর।

এই পোস্টে SFT-কে ভেতর থেকে খুলে দেখব: অবজেক্টিভটি কী অপটিমাইজ করে আর সে কারণে ডেটার মান কেন এত গুরুত্বপূর্ণ, একটি গ্রেডিয়েন্ট ধাপ পরের টোকেনের ডিস্ট্রিবিউশনে কী করে, লস মাস্কিং ও চ্যাট টেমপ্লেট, প্যাকিং ও গ্রেডিয়েন্ট অ্যাকুমুলেশন, ফুল ফাইন-টিউনিংয়ে মেমরি কোথায় খরচ হয়, আর LoRA ও QLoRA কীভাবে একটি 7B মডেলকে একটিমাত্র GPU-তে নামিয়ে আনে। প্রতিটি সেকশনে আছে তার গণিত, আর সাথে মানানসই PyTorch বা Hugging Face কোড।

কোডে ব্যবহার করা হয়েছে PyTorch, Hugging Face `transformers` আর `peft`। গণিতের জন্য লাগবে শুধু চেইন রুল (chain rule) আর সফটম্যাক্স, এর বেশি কিছু না। যেকোনো চিত্রের ওপর মাউস রাখলে সঠিক মানগুলো দেখতে পাবেন, আর রিপ্লে বাটন চেপে অ্যানিমেশনটি আবার দেখতে পারবেন।
{: .callout}

## ১. পাইপলাইনে SFT-এর জায়গা কোথায়
{: id="1-where-sft-sits-in-the-pipeline"}

একটি আধুনিক চ্যাট মডেলকে সাধারণত তিনটি ধাপে ট্রেইন করা হয়। তিনটি ধাপেই আর্কিটেকচার একই, আর প্রথম দুটিতে অবজেক্টিভও একই।

{::nomarkdown}
<figure class="viz" id="fig-stages">
<div class="viz-head"><span class="viz-label">চিত্র ১ // SFT-এর জায়গা</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 292" role="img" aria-labelledby="fig-stages-t fig-stages-d">
<title id="fig-stages-t">একটি চ্যাট মডেল ট্রেইন করার তিনটি ধাপ</title>
<desc id="fig-stages-d">প্রিট্রেনিং ট্রিলিয়ন ট্রিলিয়ন টোকেনের কাঁচা টেক্সটে পরের টোকেন প্রেডিক্ট করা শেখায়। সুপারভাইজড ফাইন-টিউনিং একই অবজেক্টিভ ব্যবহার করে কয়েক হাজার থেকে দশ লাখ বাছাই করা প্রম্পট আর উত্তরের জোড়ার ওপর। এরপর RLHF বা DPO দিয়ে প্রেফারেন্স টিউনিং শেখে তুলনা থেকে। বাংলাদেশের রাজধানী জানতে চাইলে বেস মডেল আরও প্রশ্ন লিখে চলে, SFT মডেল উত্তর দেয়, আর চ্যাট মডেল উত্তর দেয় আরও সহায়ক ভঙ্গিতে।</desc>
<defs><marker id="fig-stages-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<g class="a-in" style="--d:0.05s"><rect class="n" x="20" y="34" width="200" height="108" rx="12"/><rect class="glow" x="20" y="34" width="200" height="108" rx="12" style="--d:0.40s;--p:4.5s"/><text class="tm ta" x="120.0" y="56" text-anchor="middle">ধাপ ১</text><text class="t big" x="120.0" y="80" text-anchor="middle">প্রিট্রেনিং</text><text class="t2" x="120.0" y="100" text-anchor="middle">কাঁচা ওয়েব টেক্সট, কোড, বইয়ে</text><text class="t2" x="120.0" y="116" text-anchor="middle">পরের টোকেন প্রেডিক্ট করা</text><text class="tm" x="120.0" y="134" text-anchor="middle">ট্রিলিয়ন ট্রিলিয়ন টোকেন</text></g>
<path class="flow" d="M226 88H274" marker-end="url(#fig-stages-ah)"/>
<circle class="packet a-travel" cx="226" cy="88" r="3.5" style="--dx:40px;--t:1.6s;--d:0.90s"/>
<g class="a-in" style="--d:0.20s"><rect class="n-acc" x="280" y="34" width="200" height="108" rx="12"/><rect class="glow" x="280" y="34" width="200" height="108" rx="12" style="--d:1.50s;--p:4.5s"/><text class="tm ta" x="380.0" y="56" text-anchor="middle">ধাপ ২</text><text class="t big" x="380.0" y="80" text-anchor="middle">সুপারভাইজড ফাইন-টিউনিং</text><text class="t2" x="380.0" y="100" text-anchor="middle">শুধু বাছাই করা উত্তরে</text><text class="t2" x="380.0" y="116" text-anchor="middle">পরের টোকেন প্রেডিক্ট করা</text><text class="tm" x="380.0" y="134" text-anchor="middle">১০ হাজার থেকে ১০ লাখ ডেমোনস্ট্রেশন</text></g>
<path class="flow" d="M486 88H534" marker-end="url(#fig-stages-ah)"/>
<circle class="packet a-travel" cx="486" cy="88" r="3.5" style="--dx:40px;--t:1.6s;--d:2.00s"/>
<g class="a-in" style="--d:0.35s"><rect class="n" x="540" y="34" width="200" height="108" rx="12"/><rect class="glow" x="540" y="34" width="200" height="108" rx="12" style="--d:2.60s;--p:4.5s"/><text class="tm ta" x="640.0" y="56" text-anchor="middle">ধাপ ৩</text><text class="t big" x="640.0" y="80" text-anchor="middle">প্রেফারেন্স টিউনিং</text><text class="t2" x="640.0" y="100" text-anchor="middle">ভালো উত্তরটিকে বেছে নেওয়া</text><text class="t2" x="640.0" y="116" text-anchor="middle">RLHF (PPO) বা DPO</text><text class="tm" x="640.0" y="134" text-anchor="middle">১০ হাজার থেকে ১০ লাখ তুলনা</text></g>
<text class="a-in" x="20" y="186" style="--d:0.9s"><tspan class="tm ta">প্রতিটি মডেলকে জিজ্ঞেস করুন</tspan><tspan class="t2 strong" dx="10">“বাংলাদেশের রাজধানী কী?”</tspan></text>
<g class="a-in" style="--d:1.30s"><rect class="untrusted" x="20" y="200" width="200" height="78" rx="10"/><text class="tm" x="34" y="222">বেস মডেল</text><text class="t2" x="34" y="244">ভারতের রাজধানী কী?</text><text class="t2" x="34" y="262">নেপালের রাজধানী কী?…</text></g>
<g class="a-in" style="--d:1.75s"><rect class="n-acc" x="280" y="200" width="200" height="78" rx="10"/><text class="tm" x="294" y="222">SFT মডেল</text><text class="t2 strong" x="294" y="244">বাংলাদেশের রাজধানী</text><text class="t2 strong" x="294" y="262">ঢাকা।</text></g>
<g class="a-in" style="--d:2.20s"><rect class="n" x="540" y="200" width="200" height="78" rx="10"/><text class="tm" x="554" y="222">চ্যাট মডেল</text><text class="t2 strong" x="554" y="244">ঢাকা। এটি দেশের</text><text class="t2 strong" x="554" y="262">সবচেয়ে বড় শহরও।</text></g>
</svg>
</div>
<figcaption>একই আর্কিটেকচার, একই পরের টোকেন অবজেক্টিভ, অথচ আচরণ একেবারে আলাদা। প্রিট্রেনিং এমন মডেল তৈরি করে যা টেক্সট চালিয়ে যায়; এর চেয়ে অনেক ছোট এক সেট ডেমোনস্ট্রেশন (উদাহরণ) দিয়ে SFT সেটিকে এমন মডেলে বদলে দেয়, যা উত্তর দেয়। বেস মডেলের জবাবটি খুবই সাধারণ: সে আপনার প্রশ্নকে একগাদা প্রশ্নের তালিকার শুরু বলে ধরে নেয়।</figcaption>
</figure>
{:/nomarkdown}

**প্রিট্রেনিং** ট্রিলিয়ন ট্রিলিয়ন টোকেনের কাঁচা টেক্সট পড়ে আর পরের টোকেনটি প্রেডিক্ট করতে শেখে। মডেলের জ্ঞান আর ভাষার দক্ষতা আসে এখান থেকেই, আর প্রায় সব কম্পিউটই (compute) খরচ হয় এই ধাপে। **সুপারভাইজড ফাইন-টিউনিং** সেই একই পরের টোকেন অবজেক্টিভ চালায় অনেক ছোট, বাছাই করা একটি ডেমোনস্ট্রেশনের সেটে: প্রতিটি প্রম্পটের সাথে জোড়া বাঁধা থাকে আপনি যে উত্তরটি চান। এরপর **প্রেফারেন্স টিউনিং** (PPO দিয়ে RLHF, অথবা DPO) শেখে একটিমাত্র সঠিক উত্তর থেকে নয়, বরং দুটি উত্তরের তুলনা থেকে।

ক্রমটা গুরুত্বপূর্ণ। SFT সস্তা আর স্থিতিশীল, আর এটি মডেলকে আপনার কাঙ্ক্ষিত আচরণের কাছাকাছি নিয়ে যায়। এরপর প্রেফারেন্স টিউনিং সেটাকে আরও ঘষেমেজে নেয়, আর SFT মডেলটিকেই সে শুরুর বিন্দু ও রেফারেন্স হিসেবে ব্যবহার করে। এর প্রভাব বিশাল হতে পারে: InstructGPT পেপারে লেবেলাররা 175B প্যারামিটারের GPT-3-এর আউটপুটের চেয়ে এভাবে ট্রেইন করা 1.3B প্যারামিটারের একটি মডেলের আউটপুটই বেশি পছন্দ করেছিলেন।

<div class="key-idea">
  <span class="key-label">এক বাক্যে</span>
  <p>SFT হলো বাছাই করা প্রম্পট আর উত্তরের জোড়ায় পরের টোকেন প্রেডিকশন, যেখানে লস গোনা হয় শুধু উত্তরের ওপর।</p>
</div>

SFT সবচেয়ে ভালো শেখাতে পারে **আচরণ**: নির্দেশনা মেনে চলা, চ্যাটে পালা করে কথা বলার ফরম্যাট, কখন থামতে হবে, কথার সুর, কখন না বলতে হবে, টুল-কলের সিনট্যাক্স, কোনো এক্সট্র্যাকশন কাজের জন্য নির্দিষ্ট একটি JSON স্কিমা। আর যেখানে সে দুর্বল, সেটা হলো **নতুন জ্ঞান যোগ করা**। মডেল আগে জানত না এমন তথ্যের ওপর ফাইন-টিউন করলে সে সেগুলো শেখে ধীরে, আর বাকি সবকিছুতে আগের চেয়ে বেশি হ্যালুসিনেট করতে শুরু করে (Gekhman et al., 2024)। সমস্যাটা যদি হয় “মডেল আমাদের ডকুমেন্টগুলো জানে না”, তাহলে সমাধান রিট্রিভাল, যা নিয়ে আমি লিখেছি [PostgreSQL-এ প্রোডাকশন RAG](/bn/blog/production-rag-on-postgresql-with-pgvector/) পোস্টে। আর সমস্যাটা যদি হয় “মডেল যথেষ্ট জানে, কিন্তু ভুল ভঙ্গিতে উত্তর দেয়”, তাহলে সমাধান SFT।

## ২. অবজেক্টিভ, আর একে মিনিমাইজ করার আসল মানে
{: id="2-the-objective-and-what-minimizing-it-really-means"}

একটি ট্রেনিং উদাহরণ মানে একটি প্রম্পট $$x$$ আর একটি উত্তর $$y = (y_1, \dots, y_T)$$। একটি ডিকোডার-অনলি (decoder-only) মডেল উত্তরের সম্ভাবনাকে এক এক টোকেন করে ভেঙে গুণফল হিসেবে লেখে:

$$
p_\theta(y \mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid x, y_{<t})
$$

SFT ডেমোনস্ট্রেশনগুলোর লাইকলিহুড ম্যাক্সিমাইজ করে, যা তাদের নেগেটিভ লগ-লাইকলিহুড মিনিমাইজ করার সমান:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x,y)\sim\mathcal{D}} \sum_{t=1}^{T} \log p_\theta(y_t \mid x, y_{<t})
$$

পুরো অবজেক্টিভ এটুকুই। তবে একে মিনিমাইজ করলে আসলে কী ঘটে, সেটা একবার ভেবে দেখা দরকার। একটি নির্দিষ্ট প্রম্পটের জন্য প্রত্যাশিত (expected) নেগেটিভ লগ-লাইকলিহুড দুটি পদে ভাগ হয়ে যায়:

$$
\mathbb{E}_{y \sim p_{\mathcal{D}}(\cdot \mid x)}\big[-\log p_\theta(y \mid x)\big] = H\big(p_{\mathcal{D}}(\cdot \mid x)\big) + \mathrm{KL}\big(p_{\mathcal{D}}(\cdot \mid x) \,\Vert\, p_\theta(\cdot \mid x)\big)
$$

ডেটার এনট্রপি $$\theta$$-এর ওপর নির্ভর করে না, তাই SFT আসলে ডেটা ডিস্ট্রিবিউশন থেকে মডেলের দিকে **ফরোয়ার্ড KL ডাইভারজেন্স** মিনিমাইজ করছে। ফরোয়ার্ড KL হলো *mass-covering* (সব আচরণ ঢেকে ফেলার প্রবণতা): ডেটায় আছে এমন কোনো কিছুকে মডেল প্রায় শূন্য সম্ভাবনা দিলে সে কঠোর শাস্তি পায়, কিন্তু সম্ভাবনা বেশি ছড়িয়ে দিলে শাস্তি প্রায় পায়ই না। সহজ কথায়, আপনার ডেটাসেটের **প্রতিটি** আচরণ হুবহু তুলে আনতে মডেলকে ঠেলে দেওয়া হয়, যার মধ্যে আছে অসংগত ফরম্যাট, ভুল উত্তর, আর রাত ২টায় কোনো অ্যানোটেটরের লেখা সেই হাতে গোনা কয়েকটা উদাহরণও। SFT-তে ডেটার মানই কেন সবকিছুর ওপরে, এটাই তার গাণিতিক কারণ, আর এ কারণেই সেকশন ১২ যেকোনো হাইপারপ্যারামিটারের চেয়ে বেশি গুরুত্বপূর্ণ।

ট্রেনিংয়ে ব্যবহার হয় **টিচার ফোর্সিং** (teacher forcing): মডেল সবসময় আসল প্রিফিক্স $$y_{<t}$$-এর ওপর ভর করে প্রেডিক্ট করে, কখনোই নিজের আগের অনুমানের ওপর নয়। কজাল (causal) অ্যাটেনশন মাস্কের সাথে মিলে এটি একটিমাত্র ফরোয়ার্ড পাসে সব $$T$$টি প্রেডিকশন একসাথে হিসাব করতে দেয়। এর দাম হলো *এক্সপোজার বায়াস*: ইনফারেন্সের সময় মডেল চলে নিজের আউটপুটের ওপর ভর করে, তাই শুরুর দিকের ছোট একটি ভুল তাকে এমন অবস্থায় নিয়ে যেতে পারে, যা সে ট্রেনিংয়ে কখনো দেখেনি। প্রেফারেন্স টিউনিং আর অন্যান্য on-policy পদ্ধতি আংশিকভাবে এই সমস্যারই জবাব।

## ৩. একটি গ্রেডিয়েন্ট ধাপ, কাছ থেকে দেখা
{: id="3-one-gradient-step-up-close"}

প্রতিটি পজিশনে মডেল লজিটের একটি ভেক্টর $$\mathbf{z} \in \mathbb{R}^{V}$$ তৈরি করে, ভোকাবুলারির প্রতিটি এন্ট্রির জন্য একটি করে (Llama 3-এর ক্ষেত্রে প্রায় ১ লাখ ২৮ হাজার)। সফটম্যাক্স এগুলোকে সম্ভাবনায় রূপ দেয়:

$$
p_v = \frac{e^{z_v}}{\sum_{u=1}^{V} e^{z_u}}
$$

একটি টার্গেট টোকেন $$y$$-এর জন্য প্রতি টোকেনের লস হলো নেগেটিভ লগ-সম্ভাবনা, যাকে খুলে লিখলে দাঁড়ায়:

$$
\ell = -\log p_y = -z_y + \log \sum_{u=1}^{V} e^{z_u}
$$

এবার একটি লজিট $$z_v$$-এর সাপেক্ষে ডিফারেনশিয়েট করুন। log-sum-exp পদটি ফিরিয়ে দেয় সফটম্যাক্সকেই, অর্থাৎ $$p_v$$, আর প্রথম পদটি $$-1$$ যোগ করে কেবল তখন, যখন $$v = y$$:

$$
\frac{\partial \ell}{\partial z_v} = p_v - \mathbb{1}[v = y]
\qquad\Longrightarrow\qquad
\nabla_{\mathbf{z}}\,\ell = \mathbf{p} - \mathbf{e}_y
$$

একটিমাত্র টোকেনের স্তরে SFT যা করে, তার বেশিরভাগই এই এক লাইনে ব্যাখ্যা হয়ে যায়। গ্রেডিয়েন্ট ডিসেন্ট লজিটগুলোকে গ্রেডিয়েন্টের উল্টো দিকে সরায়, তাই টার্গেটের লজিট **বাড়ে** $$1 - p_y$$-এর সমানুপাতে, অর্থাৎ তার যতটুকু সম্ভাবনা কম পড়েছে সেই অনুপাতে, আর বাকি প্রতিটি লজিট **কমে** তার নিজের সম্ভাবনা $$p_v$$-এর অনুপাতে। যে টোকেনগুলো সবচেয়ে বেশি সম্ভাবনা কেড়ে নিচ্ছিল, সেগুলোকেই সবচেয়ে জোরে নিচে ঠেলে দেওয়া হয়।

{::nomarkdown}
<figure class="viz" id="fig-grad">
<div class="viz-head"><span class="viz-label">চিত্র ৩ // পরের টোকেনের ডিস্ট্রিবিউশনে একটি গ্রেডিয়েন্ট ধাপ</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 308" role="img" aria-labelledby="fig-grad-t fig-grad-d">
<title id="fig-grad-t">ক্রস-এনট্রপি গ্রেডিয়েন্ট কীভাবে পরের টোকেনের ডিস্ট্রিবিউশন বদলে দেয়</title>
<desc id="fig-grad-d">'The capital of Bangladesh is' প্রম্পটের জন্য ছয় টোকেনের একটি ছোট কাল্পনিক ভোকাবুলারি Dhaka-কে 0.34 সম্ভাবনা দেয়। লজিটের সাপেক্ষে লসের গ্রেডিয়েন্ট হলো p বিয়োগ one-hot টার্গেট: Dhaka-র জন্য -0.66, আর বাকিদের জন্য প্রতিটি টোকেনের নিজের সম্ভাবনার সমান ধনাত্মক মান। গ্রেডিয়েন্টের উল্টো দিকে এক ধাপ নিলে Dhaka বেড়ে 0.63 হয়, আর বাকি প্রতিটি টোকেন কমে যায়।</desc>
<defs><marker id="fig-grad-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<text class="tm ta" x="18" y="30">পরের টোকেন</text>
<text class="tm ta" x="150" y="30">আগের p</text>
<text class="tm ta" x="432" y="30" text-anchor="middle">গ্রেডিয়েন্ট ∂ℓ/∂z = p − onehot(y)</text>
<text class="tm ta" x="560" y="30">এক ধাপ পরের p</text>
<text class="t2" x="18" y="48">“The capital of Bangladesh is”</text>
<line class="grid-strong" x1="432" y1="52" x2="432" y2="236"/>
<g class="row" tabindex="0" data-tip="p 0.335 → 0.628" data-tip-sub="টার্গেট টোকেন: গ্রেডিয়েন্ট -0.665, তাই লজিট ওপরে ঠেলা হয়" aria-label="Dhaka: p 0.335 → 0.628"><rect class="hit" x="12" y="58" width="736" height="26"/><text class="t" x="18" y="75" style="font-family:var(--font-mono)">·Dhaka</text><rect class="s1 a-grow" x="150" y="62" width="76.6" height="16" rx="3" style="--d:0.10s"/><text class="axis" x="232.6" y="74">0.34</text>
<rect class="s1 a-grow" x="370.8" y="62" width="61.2" height="16" rx="3" style="--d:0.90s;transform-origin:100% 50%"/><text class="axis" x="364.8" y="74" text-anchor="end">-0.66</text>
<rect class="s1 a-grow" x="560" y="62" width="143.5" height="16" rx="3" style="--d:1.80s"/><text class="axis" x="709.5" y="74">0.63</text></g>
<g class="row" tabindex="0" data-tip="p 0.184 → 0.096" data-tip-sub="গ্রেডিয়েন্ট +0.184, তাই লজিট নিচে ঠেলা হয়" aria-label="Chittagong: p 0.184 → 0.096"><rect class="hit" x="12" y="88" width="736" height="26"/><text class="t2" x="18" y="105" style="font-family:var(--font-mono)">·Chittagong</text><rect class="hn a-grow" x="150" y="92" width="42.0" height="16" rx="3" style="--d:0.16s"/><text class="axis" x="198.0" y="104">0.18</text>
<rect class="evil a-grow" x="432" y="92" width="16.9" height="16" rx="3" style="--d:0.96s"/><text class="axis" x="454.9" y="104">+0.18</text>
<rect class="hn a-grow" x="560" y="92" width="22.1" height="16" rx="3" style="--d:1.86s"/><text class="axis" x="588.1" y="104">0.10</text></g>
<g class="row" tabindex="0" data-tip="p 0.151 → 0.083" data-tip-sub="গ্রেডিয়েন্ট +0.151, তাই লজিট নিচে ঠেলা হয়" aria-label="a: p 0.151 → 0.083"><rect class="hit" x="12" y="118" width="736" height="26"/><text class="t2" x="18" y="135" style="font-family:var(--font-mono)">·a</text><rect class="hn a-grow" x="150" y="122" width="34.4" height="16" rx="3" style="--d:0.22s"/><text class="axis" x="190.4" y="134">0.15</text>
<rect class="evil a-grow" x="432" y="122" width="13.9" height="16" rx="3" style="--d:1.02s"/><text class="axis" x="451.9" y="134">+0.15</text>
<rect class="hn a-grow" x="560" y="122" width="19.0" height="16" rx="3" style="--d:1.92s"/><text class="axis" x="585.0" y="134">0.08</text></g>
<g class="row" tabindex="0" data-tip="p 0.136 → 0.077" data-tip-sub="গ্রেডিয়েন্ট +0.136, তাই লজিট নিচে ঠেলা হয়" aria-label="the: p 0.136 → 0.077"><rect class="hit" x="12" y="148" width="736" height="26"/><text class="t2" x="18" y="165" style="font-family:var(--font-mono)">·the</text><rect class="hn a-grow" x="150" y="152" width="31.1" height="16" rx="3" style="--d:0.28s"/><text class="axis" x="187.1" y="164">0.14</text>
<rect class="evil a-grow" x="432" y="152" width="12.5" height="16" rx="3" style="--d:1.08s"/><text class="axis" x="450.5" y="164">+0.14</text>
<rect class="hn a-grow" x="560" y="152" width="17.5" height="16" rx="3" style="--d:1.98s"/><text class="axis" x="583.5" y="164">0.08</text></g>
<g class="row" tabindex="0" data-tip="p 0.112 → 0.065" data-tip-sub="গ্রেডিয়েন্ট +0.112, তাই লজিট নিচে ঠেলা হয়" aria-label="located: p 0.112 → 0.065"><rect class="hit" x="12" y="178" width="736" height="26"/><text class="t2" x="18" y="195" style="font-family:var(--font-mono)">·located</text><rect class="hn a-grow" x="150" y="182" width="25.5" height="16" rx="3" style="--d:0.34s"/><text class="axis" x="181.5" y="194">0.11</text>
<rect class="evil a-grow" x="432" y="182" width="10.3" height="16" rx="3" style="--d:1.14s"/><text class="axis" x="448.3" y="194">+0.11</text>
<rect class="hn a-grow" x="560" y="182" width="14.9" height="16" rx="3" style="--d:2.04s"/><text class="axis" x="580.9" y="194">0.07</text></g>
<g class="row" tabindex="0" data-tip="p 0.083 → 0.050" data-tip-sub="গ্রেডিয়েন্ট +0.083, তাই লজিট নিচে ঠেলা হয়" aria-label="home: p 0.083 → 0.050"><rect class="hit" x="12" y="208" width="736" height="26"/><text class="t2" x="18" y="225" style="font-family:var(--font-mono)">·home</text><rect class="hn a-grow" x="150" y="212" width="18.9" height="16" rx="3" style="--d:0.40s"/><text class="axis" x="174.9" y="224">0.08</text>
<rect class="evil a-grow" x="432" y="212" width="7.6" height="16" rx="3" style="--d:1.20s"/><text class="axis" x="445.6" y="224">+0.08</text>
<rect class="hn a-grow" x="560" y="212" width="11.5" height="16" rx="3" style="--d:2.10s"/><text class="axis" x="577.5" y="224">0.05</text></g>
<text class="a-in" x="432" y="264" text-anchor="middle" style="--d:0.9s"><tspan class="t2">ওপরে ঠেলা হয় ←</tspan><tspan class="t2" dx="14">→ নিচে ঠেলা হয়</tspan></text>
<text class="a-in" x="18" y="294" style="--d:2.3s"><tspan class="t2">“Dhaka”-র লস: −log p = </tspan><tspan class="t">1.09 → 0.47</tspan><tspan class="t2" dx="8">লজিটের ওপর η = 1.5 আকারের এক ধাপের পরে।</tspan></text>
</svg>
</div>
<details class="viz-data"><summary>সংখ্যাগুলো দেখুন</summary><table><thead><tr><th>টোকেন</th><th>আগের p</th><th>∂ℓ/∂z</th><th>পরের p</th></tr></thead><tbody><tr><td>Dhaka</td><td>0.335</td><td>-0.665</td><td>0.628</td></tr><tr><td>Chittagong</td><td>0.184</td><td>+0.184</td><td>0.096</td></tr><tr><td>a</td><td>0.151</td><td>+0.151</td><td>0.083</td></tr><tr><td>the</td><td>0.136</td><td>+0.136</td><td>0.077</td></tr><tr><td>located</td><td>0.112</td><td>+0.112</td><td>0.065</td></tr><tr><td>home</td><td>0.083</td><td>+0.083</td><td>0.050</td></tr></tbody></table></details>
<figcaption>ছয় টোকেনের একটি ছোট কাল্পনিক ভোকাবুলারি, যাতে সংখ্যাগুলো সহজে মিলিয়ে দেখা যায়। টার্গেটের লজিট বাড়ে তার ঘাটতি সম্ভাবনা, অর্থাৎ 1 − p পরিমাণে; বাকি প্রতিটি লজিট কমে, সে কতটা সম্ভাবনা কেড়ে নিচ্ছিল তার অনুপাতে। বোঝানোর সুবিধার জন্য এখানে ধাপটি সরাসরি লজিটের ওপর প্রয়োগ করা হয়েছে; আসল ট্রেনিংয়ে একই সিগন্যাল নেটওয়ার্কের ভেতর দিয়ে পেছনে গিয়ে ওয়েটগুলোতে পৌঁছায়।</figcaption>
</figure>
{:/nomarkdown}

এ থেকে তিনটি ফল পাওয়া যায়, আর তিনটিই বাস্তবে চোখে পড়ে:

1. **সহজ টোকেন আর কিছু শেখায় না।** মডেল যখন আগেই টার্গেটকে $$p_y \approx 1$$ দিয়ে রাখে, তখন গ্রেডিয়েন্ট $$\mathbf{p} - \mathbf{e}_y$$ প্রায় শূন্য। সাধারণ একটি উত্তরের বেশিরভাগ টোকেনই আগে থেকে আন্দাজ করা যায়, তাই আপডেটের বেশিরভাগটাই আসে অল্প কয়েকটি অপ্রত্যাশিত টোকেন থেকে। গড় লস কমতে থাকলেও তার আড়ালে লুকিয়ে থাকতে পারে যে মডেল আসলে শিখছে মাত্র কয়েকটি জিনিস।
2. **সিগন্যাল প্রতিটি ওয়েটে পৌঁছায়।** লজিটগুলো সব প্যারামিটারের একটি ফাংশন, তাই চেইন রুল একই ভেক্টরকে নেটওয়ার্কের ভেতর দিয়ে পেছনে বয়ে নিয়ে যায়: $$\nabla_\theta\,\ell = J_\theta(\mathbf{z})^{\top}(\mathbf{p} - \mathbf{e}_y)$$, যেখানে $$J_\theta(\mathbf{z})$$ হলো ওয়েটের সাপেক্ষে লজিটের জ্যাকোবিয়ান (Jacobian)।
3. **লসের মান সহজে পড়া যায়।** গড় লসের এক্সপোনেনশিয়াল হলো **পারপ্লেক্সিটি**। লস 0.95 মানে পারপ্লেক্সিটি $$e^{0.95} \approx 2.6$$: অর্থাৎ গড়ে মডেল ততটাই অনিশ্চিত, যেন সে সমান সম্ভাবনার 2.6টি টোকেনের মধ্য থেকে একটি বেছে নিচ্ছে।

## ৪. লস মাস্কিং: প্রশ্নে নয়, উত্তরে ট্রেইন করুন
{: id="4-loss-masking-train-on-the-answer-not-the-question"}

একটি চ্যাট উদাহরণ মানে পরিপাটি দুটি ঘরে রাখা একটি প্রম্পট আর একটি উত্তর নয়। এটি টোকেনের একটিই সিকোয়েন্স, যা সাজানো হয় মডেলের **চ্যাট টেমপ্লেট** অনুযায়ী। Qwen মডেলগুলো এই লেআউট ব্যবহার করে, যার নাম ChatML (এখানে সিস্টেম টার্ন ছাড়া দেখানো হলো):

```text
<|im_start|>user
Capital of Bangladesh?<|im_end|>
<|im_start|>assistant
Dhaka.<|im_end|>
```

মডেল পুরো সিকোয়েন্সটাই পড়ে, কিন্তু তাকে নম্বর দেওয়া উচিত শুধু অ্যাসিস্ট্যান্টের উত্তরের ওপর। এর প্রচলিত উপায় হলো ইনপুট আইডিগুলো একটি `labels` টেনসরে কপি করা, আর উত্তরের বাইরের প্রতিটি পজিশনকে `-100` দিয়ে ওভাররাইট করা, যে মানটিকে PyTorch-এর ক্রস-এনট্রপি উপেক্ষা করে।

{::nomarkdown}
<figure class="viz" id="fig-mask">
<div class="viz-head"><span class="viz-label">চিত্র ২ // লস শুধু উত্তরের ওপর</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 304" role="img" aria-labelledby="fig-mask-t fig-mask-d">
<title id="fig-mask-t">সুপারভাইজড ফাইন-টিউনিংয়ে লস মাস্কিং আর এক টোকেনের শিফট</title>
<desc id="fig-mask-d">একটি চ্যাট উদাহরণকে টোকেনাইজ করলে ১৫টি টোকেন হয়। টেমপ্লেট আর ব্যবহারকারীর প্রশ্নের ১২টি টোকেন লেবেল হিসেবে পায় -100, আর লসে এদের কোনো অবদান নেই। উত্তরের তিনটি টোকেন, Dhaka, ফুল স্টপ আর এন্ড-অফ-টার্ন টোকেন, হলো লেবেল। প্রতিটি লেবেল প্রেডিক্ট করা হয় তার আগের টোকেন থেকে। এদের লস 2.31, 0.42 আর 0.11, আর উদাহরণটির লস হলো এদের গড়, 0.95।</desc>
<defs><marker id="fig-mask-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#e8b84e"/></marker></defs>
<text class="tm ta" x="14" y="20">ইনপুট টোকেন // মডেল যা পড়ে</text>
<text class="tm ta" x="14" y="108">লেবেল // যার ওপর নম্বর দেওয়া হয়</text>
<text class="tm ta" x="14" y="196">প্রতি টোকেনের লস // −log p(label)</text>
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
<g class="pt" tabindex="0" data-tip="−log p = 2.31" data-tip-sub="আগের সবকিছু জানা থাকলে এই টোকেনের p = 0.10" aria-label="লেবেল Dhaka, লস 2.31"><rect class="hit" x="609.6" y="118" width="38.5" height="28"/><g class="a-pop" style="--d:1.20s"><rect class="n-acc" x="609.6" y="118" width="38.5" height="28" rx="6"/><text class="tk" x="628.8" y="136" text-anchor="middle">Dhaka</text></g></g>
<g class="a-in" style="--d:2.20s"><rect class="s1" x="619.8" y="207.2" width="18" height="50.8" rx="3"/><text class="t" x="628.8" y="199.2" text-anchor="middle">2.31</text></g>
<path class="hop a-draw" pathLength="1" d="M594.1 60C594.1 92 628.8 84 628.8 114" marker-end="url(#fig-mask-ah)" style="--d:0.90s;--t:.5s"/>
<g class="a-in" style="--d:0.39s"><rect class="n-acc" x="650.1" y="30" width="27.0" height="28" rx="6"/><text class="tk" x="663.6" y="48" text-anchor="middle">.</text></g>
<g class="pt" tabindex="0" data-tip="−log p = 0.42" data-tip-sub="আগের সবকিছু জানা থাকলে এই টোকেনের p = 0.66" aria-label="লেবেল ., লস 0.42"><rect class="hit" x="650.1" y="118" width="27.0" height="28"/><g class="a-pop" style="--d:1.45s"><rect class="n-acc" x="650.1" y="118" width="27.0" height="28" rx="6"/><text class="tk" x="663.6" y="136" text-anchor="middle">.</text></g></g>
<g class="a-in" style="--d:2.45s"><rect class="s1" x="654.6" y="248.8" width="18" height="9.2" rx="3"/><text class="t" x="663.6" y="240.8" text-anchor="middle">0.42</text></g>
<path class="hop a-draw" pathLength="1" d="M628.8 60C628.8 92 663.6 84 663.6 114" marker-end="url(#fig-mask-ah)" style="--d:1.15s;--t:.5s"/>
<g class="a-in" style="--d:0.42s"><rect class="n-acc" x="679.1" y="30" width="67.0" height="28" rx="6"/><text class="tk" x="712.6" y="48" text-anchor="middle">&lt;|im_end|&gt;</text></g>
<g class="pt" tabindex="0" data-tip="−log p = 0.11" data-tip-sub="আগের সবকিছু জানা থাকলে এই টোকেনের p = 0.90" aria-label="লেবেল &lt;|im_end|&gt;, লস 0.11"><rect class="hit" x="679.1" y="118" width="67.0" height="28"/><g class="a-pop" style="--d:1.70s"><rect class="n-acc" x="679.1" y="118" width="67.0" height="28" rx="6"/><text class="tk" x="712.6" y="136" text-anchor="middle">&lt;|im_end|&gt;</text></g></g>
<g class="a-in" style="--d:2.70s"><rect class="s1" x="703.6" y="255.6" width="18" height="2.4" rx="3"/><text class="t" x="712.6" y="247.6" text-anchor="middle">0.11</text></g>
<path class="hop a-draw" pathLength="1" d="M663.6 60C663.6 92 712.6 84 712.6 114" marker-end="url(#fig-mask-ah)" style="--d:1.40s;--t:.5s"/>
<line class="grid" x1="14" y1="258.5" x2="746" y2="258.5"/>
<text class="a-in" x="14" y="290" style="--d:3.0s"><tspan class="t2">এই উদাহরণের লস: </tspan><tspan class="t">(2.31 + 0.42 + 0.11) / 3 = 0.95</tspan><tspan class="t2" dx="8">প্রম্পটের ১২টি টোকেন লস বা গ্রেডিয়েন্টে কিছুই যোগ করে না।</tspan></text>
</svg>
</div>
<figcaption>লেবেলগুলো হলো ইনপুটেরই একটি কপি, যেখানে প্রম্পটের জায়গায় বসানো হয়েছে <code>-100</code>, যে মানটিকে PyTorch-এর ক্রস-এনট্রপি উপেক্ষা করে। তীরচিহ্নগুলো শিফটটা দেখায়: পজিশন <em>t</em> প্রেডিক্ট করে টোকেন <em>t</em>+1, তাই উত্তরের ঠিক আগের টোকেনটি উত্তরের প্রথম শব্দ প্রেডিক্ট করে। এন্ড-অফ-টার্ন টোকেনও একটি লেবেল; মডেল এভাবেই থামতে শেখে। উত্তরের কোনো টোকেনের ওপর মাউস রাখলে তার সম্ভাবনা দেখতে পাবেন।</figcaption>
</figure>
{:/nomarkdown}

মাস্ক $$m_t$$ ব্যবহার করে পুরো টোকেন সিকোয়েন্স $$s_1, \dots, s_N$$-এর ওপর লিখলে একটি উদাহরণের লস দাঁড়ায়:

$$
\mathcal{L} = -\frac{\sum_{t=1}^{N} m_t \log p_\theta(s_t \mid s_{<t})}{\sum_{t=1}^{N} m_t},
\qquad
m_t = \begin{cases} 1 & \text{if } s_t \text{ belongs to an assistant turn} \\ 0 & \text{otherwise} \end{cases}
$$

প্রম্পট মাস্ক করবেন কেন? ইনফারেন্সের সময় প্রম্পট তো দেওয়াই থাকে, তাই মডেলকে কখনো সেটা তৈরি করতে হয় না। ব্যবহারকারীর বার্তা এলোমেলো আর নানা রকমের, আর সেগুলো নকল করতে যে গ্রেডিয়েন্ট খরচ হয়, সেটা আর উত্তরের পেছনে খরচ হয় না। তার ওপর পুরো কথোপকথনে ট্রেইন করা মডেল ব্যবহারকারীর পরের বার্তাটিও লিখতে শিখে ফেলে, আর খারাপভাবে ফাইন-টিউন করা মডেলের সেই অসংলগ্ন বকবকানি ঠিক এখান থেকেই আসে। মাস্কিং একটি শক্ত ডিফল্ট, অলঙ্ঘনীয় নিয়ম নয়: Shi et al. (2024) দেখিয়েছেন, উত্তর ছোট আর ডেটাসেট ছোট হলে নির্দেশনার (instruction) ওপরও ট্রেইন করলে উপকার হতে পারে। আপনার ডেটা যদি এমন হয়, দুটোই মেপে দেখুন।

দুটি খুঁটিনাটিতে ভুল করা খুব সহজ:

- **এন্ড-অফ-টার্ন টোকেনেও ট্রেইন করুন।** <code>&lt;&#124;im_end&#124;&gt;</code> (অথবা <code>&lt;&#124;eot_id&#124;&gt;</code>, কিংবা আপনার মডেল যেটাই ব্যবহার করুক) অবশ্যই একটি লেবেল হতে হবে। উত্তর কোথায় শেষ হয়, মডেল সেটা শেখে কেবল এভাবেই। এটি বাদ দিয়ে ফাইন-টিউন করা মডেল কথা বলা আর থামায় না।
- **মাল্টি-টার্ন কথোপকথন:** সিস্টেম আর ব্যবহারকারীর প্রতিটি টোকেন মাস্ক করুন, আর শুধু শেষটিতে নয়, অ্যাসিস্ট্যান্টের প্রতিটি টার্নে ট্রেইন করুন। আগের প্রতিটি অ্যাসিস্ট্যান্টের টার্ন হলো বিনা খরচে পাওয়া বাড়তি সুপারভিশন।

লেবেল বানানোর সবচেয়ে সহজ উপায় হলো একই কথোপকথনকে দুইভাবে রেন্ডার করা: একবার শুধু প্রম্পট, যা শেষ হয় অ্যাসিস্ট্যান্ট হেডার দিয়ে, আরেকবার পুরো কথোপকথন। প্রম্পটের পরে যা কিছু আছে, সবই উত্তর।

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

মডেলের ভেতরে শিফট আর মাস্ক করা গড় মাত্র কয়েক লাইনের কাজ। আপনি `labels` পাস করলে Hugging Face মডেলগুলো কাজটা নিজেরাই করে দেয়, তবু একবার নিজের চোখে দেখে নেওয়া ভালো:

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

## ৫. চ্যাট টেমপ্লেট: ফরম্যাটও মডেলেরই অংশ
{: id="5-chat-templates-the-format-is-part-of-the-model"}

মডেলের কাছে টেমপ্লেটের স্পেশাল টোকেনগুলো আর দশটা টোকেনের মতোই। আপনি যদি এক লেআউটে ফাইন-টিউন করে আরেক লেআউটে সার্ভ করেন, তাহলে প্রতিটি রিকোয়েস্ট আসবে এমন ফরম্যাটে, যাতে মডেলকে কখনো ট্রেইন করা হয়নি, আর মান এমনভাবে পড়ে যাবে যে কারণ খুঁজে বের করাই কঠিন। তিনটি নিয়ম মানলে এর প্রায় পুরোটাই এড়ানো যায়:

1. ট্রেনিং আর ইনফারেন্স, দুই সময়েই **`tokenizer.apply_chat_template` দিয়ে রেন্ডার করুন**। কখনো f-string-এ হাতে স্পেশাল টোকেন বসাবেন না।
2. **মডেলের নিজের টেমপ্লেটই রাখুন।** কোনো instruct মডেল ফাইন-টিউন করছেন? তার টেমপ্লেটই ব্যবহার করুন। টেমপ্লেট নেই এমন কোনো বেস মডেল থেকে শুরু করছেন? একটি বেছে নিন, `tokenizer.chat_template` সেট করুন, আর ওয়েটের সাথে ঠিক সেই টোকেনাইজারটিই রিলিজ করুন।
3. **প্যাড টোকেনের দিকে নজর রাখুন।** অনেক টোকেনাইজারে কোনো প্যাড টোকেন থাকে না, আর সাধারণ সমাধান হলো `tokenizer.pad_token = tokenizer.eos_token`। এটা নিরাপদ কেবল তখনই, যখন আপনার লেবেলগুলো ওপরের মতো করে বানানো হয়। এর বদলে আপনি যদি `input_ids == pad_token_id` এমন প্রতিটি পজিশন মাস্ক করে লেবেল বানান, তাহলে আসল এন্ড-অফ-সিকোয়েন্স টোকেনটিও মাস্ক হয়ে যাবে, আর মডেল কখনো থামতে শিখবে না।

পুরো ফাইন-টিউনিংয়ের সবচেয়ে সস্তা পরীক্ষাটি হলো ট্রেইন শুরুর আগেই লস ঠিক কী দেখবে, সেটা ডিকোড করে দেখা:

```python
ex = build_example(messages, tokenizer)
target_ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
print(repr(tokenizer.decode(target_ids)))
# 'Dhaka.<|im_end|>\n'   answer + end-of-turn token, no prompt
```

## ৬. প্যাডিং, প্যাকিং আর অ্যাটেনশনের সীমানা
{: id="6-padding-packing-and-attention-boundaries"}

GPU একটি ব্যাচকে প্রসেস করে একটা আয়তক্ষেত্রের মতো: রো গুণ সিকোয়েন্সের দৈর্ঘ্য। SFT-এর উদাহরণগুলোর দৈর্ঘ্যে আকাশ-পাতাল তফাত থাকে, তাই সবচেয়ে লম্বা রো পর্যন্ত প্যাড করা একটি ব্যাচের বেশির ভাগটাই প্যাডিং। ধরুন $$R$$টি রো আছে, প্রতিটির দৈর্ঘ্য $$L$$, আর তাতে রাখা উদাহরণগুলোর দৈর্ঘ্য $$n_i$$। তাহলে কম্পিউটের যে অংশ আসল কাজে লাগছে, তা হলো:

$$
u = \frac{\sum_i n_i}{R \cdot L}
$$

**প্যাকিং** প্রতিটি রোতে কয়েকটি উদাহরণ পরপর জুড়ে দেয়, যাতে $$u$$ প্রায় 1-এর কাছে পৌঁছায়। একটা সাধারণ first-fit-decreasing পাসই (দৈর্ঘ্য অনুযায়ী সাজান, তারপর প্রতিটি উদাহরণকে প্রথম যে রোতে জায়গা হয় সেখানে বসিয়ে দিন) এর বেশ কাছাকাছি নিয়ে যায়।

{::nomarkdown}
<figure class="viz" id="fig-pack">
<div class="viz-head"><span class="viz-label">চিত্র ৪ // প্যাডিং বনাম প্যাকিং</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 426" role="img" aria-labelledby="fig-pack-t fig-pack-d">
<title id="fig-pack-t">প্যাডিং কম্পিউট নষ্ট করে; প্যাকিংয়ের জন্য দরকার ব্লক-ডায়াগোনাল অ্যাটেনশন মাস্ক</title>
<desc id="fig-pack-d">আটটি উদাহরণ, দৈর্ঘ্য 512, 140, 260, 90, 300, 180, 200 ও 330 টোকেন। প্রতিটিকে 512 টোকেন পর্যন্ত প্যাড করলে 4096টি টোকেন স্লটের মাত্র ৪৯%-এ আসল টোকেন থাকে। first-fit decreasing দিয়ে পরপর সাজিয়ে ৪টি রোতে প্যাক করলে ৯৮% স্লট কাজে লাগে। সাধারণ কজাল মাস্কে প্যাক করা রোয়ের টোকেনগুলো একই রোয়ের আগের উদাহরণগুলোতে অ্যাটেন্ড করতে পারে; ব্লক-ডায়াগোনাল মাস্ক আর প্রতিটি উদাহরণে নতুন করে শুরু হওয়া পজিশন আইডি মিলে সেগুলোকে আলাদা রাখে।</desc>
<text class="tm ta" x="18" y="20">প্যাড করা // প্রতি রোতে একটি উদাহরণ</text>
<rect class="pad" x="18" y="32" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="32" width="330.0" height="16" rx="3" style="--d:0.00s"/><text class="tm" x="342.0" y="44" text-anchor="end">512</text>
<rect class="pad" x="18" y="55" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="55" width="90.2" height="16" rx="3" style="--d:0.05s"/><text class="tm" x="102.2" y="67" text-anchor="end">140</text>
<rect class="pad" x="18" y="78" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="78" width="167.6" height="16" rx="3" style="--d:0.10s"/><text class="tm" x="179.6" y="90" text-anchor="end">260</text>
<rect class="pad" x="18" y="101" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="101" width="58.0" height="16" rx="3" style="--d:0.15s"/><text class="tm" x="70.0" y="113" text-anchor="end">90</text>
<rect class="pad" x="18" y="124" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="124" width="193.4" height="16" rx="3" style="--d:0.20s"/><text class="tm" x="205.4" y="136" text-anchor="end">300</text>
<rect class="pad" x="18" y="147" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="147" width="116.0" height="16" rx="3" style="--d:0.25s"/><text class="tm" x="128.0" y="159" text-anchor="end">180</text>
<rect class="pad" x="18" y="170" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="170" width="128.9" height="16" rx="3" style="--d:0.30s"/><text class="tm" x="140.9" y="182" text-anchor="end">200</text>
<rect class="pad" x="18" y="193" width="330" height="16" rx="3"/><rect class="chunk a-grow" x="18" y="193" width="212.7" height="16" rx="3" style="--d:0.35s"/><text class="tm" x="224.7" y="205" text-anchor="end">330</text>
<text class="a-in" x="18" y="230" style="--d:0.6s"><tspan class="t">৪৯% কাজে লাগে</tspan><tspan class="t2" dx="8">৪,০৯৬ স্লটের ২,০১২টি, ৮টি রো</tspan></text>
<text class="tm ta" x="410" y="20">প্যাক করা // উদাহরণ পরপর</text>
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
<text class="a-in" x="410" y="138" style="--d:2.0s"><tspan class="t">৯৮% কাজে লাগে</tspan><tspan class="t2" dx="8">২,০৪৮ স্লটের ২,০১২টি, ৪টি রো</tspan></text>
<text class="tm ta" x="18" y="262">সাধারণ কজাল মাস্ক</text>
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
<text class="t2" x="152" y="296">লাল ঘর: এখানে টোকেন</text><text class="t2" x="152" y="314">একই রোয়ের অন্য উদাহরণে</text>
<text class="t2" x="152" y="332">অ্যাটেন্ড করছে।</text>
<text class="tm ta" x="410" y="262">ব্লক-ডায়াগোনাল মাস্ক</text>
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
<text class="t2" x="544" y="296">প্রতিটি উদাহরণ শুধু</text><text class="t2" x="544" y="314">নিজেকে দেখে। পজিশন আইডি</text>
<text class="t2" x="544" y="332">প্রতি উদাহরণে নতুন করে শুরু।</text>
<text class="tm a-in" x="410" y="414" style="--d:3.0s">position_ids 0 1 2 3 4 0 1 2 3 0 1 2</text>
</svg>
</div>
<details class="viz-data"><summary>সংখ্যাগুলো দেখুন</summary><table><thead><tr><th></th><th>রো</th><th>টোকেন স্লট</th><th>আসল টোকেন</th><th>কাজের অংশ</th></tr></thead><tbody><tr><td>প্যাড করা</td><td>8</td><td>4096</td><td>2012</td><td>49.1%</td></tr><tr><td>প্যাক করা</td><td>4</td><td>2048</td><td>2012</td><td>98.2%</td></tr></tbody></table></details>
<figcaption>ওপরে: একই ৮টি উদাহরণ, বাঁয়ে প্যাড করা আর ডানে first-fit decreasing দিয়ে প্যাক করা। প্যাকিংয়ে রোয়ের সংখ্যা অর্ধেক হয়ে যায়, তাই কম্পিউটও মোটামুটি অর্ধেক লাগে। নিচে: তিনটি ছোট উদাহরণ দিয়ে প্যাক করা একটি রো। সাধারণ কজাল মাস্কে (বাঁয়ে) লাল ঘরগুলোর কারণে দ্বিতীয় আর তৃতীয় উদাহরণ তাদের আগের উদাহরণগুলো পড়তে পারে; ব্লক-ডায়াগোনাল মাস্ক (ডানে) সেটা আটকে দেয়।</figcaption>
</figure>
{:/nomarkdown}

প্যাকিংয়ে একটা ফাঁদ আছে। সাধারণ কজাল (causal) মাস্কে একটি রোয়ের দ্বিতীয় উদাহরণের কোনো টোকেন প্রথম উদাহরণে অ্যাটেন্ড করতে পারে, অথচ সেটা সম্পূর্ণ আলাদা একটা কথোপকথন। মডেল এমন কনটেক্সট থেকে শেখে যা ইনফারেন্সের সময় সে কখনো পাবে না, আর গ্রেডিয়েন্টের একটা অংশ খরচ হয় সেই বাড়তি কনটেক্সট কাজে লাগানো শিখতে। সমাধান হলো একটি **ব্লক-ডায়াগোনাল** মাস্ক, যাতে প্রতিটি উদাহরণ শুধু নিজেকে দেখে, আর প্রতিটি উদাহরণে **শূন্য থেকে নতুন করে শুরু হওয়া পজিশন আইডি**।

এই মাস্ক সাধারণত হাতে বানাতে হয় না। Hugging Face `transformers`-এ `DataCollatorWithFlattening` একটি ব্যাচকে জুড়ে একটিমাত্র রো বানায় এবং নতুন করে শুরু হওয়া `position_ids` ফেরত দেয়, আর FlashAttention 2 কার্নেলগুলো সেগুলো দেখে উদাহরণগুলোকে আলাদা রাখে। এটা ব্যবহার করতে মডেল লোড করুন `attn_implementation="flash_attention_2"` দিয়ে। বাড়তি একটা লাভও আছে: একটি রোয়ের ভেতরে অ্যাটেনশনের খরচ $$\big(\sum_i n_i\big)^2$$ থেকে কমে $$\sum_i n_i^2$$ হয়ে যায়, কারণ উদাহরণগুলো কখনো একে অপরের দিকে অ্যাটেন্ড করে না।

## ৭. লস নর্মালাইজ করা, আর গ্রেডিয়েন্ট অ্যাকুমুলেশনের ফাঁদ
{: id="7-normalizing-the-loss-and-the-gradient-accumulation-trap"}

একটি ব্যাচে যখন ভিন্ন ভিন্ন দৈর্ঘ্যের উদাহরণ থাকে, তখন ঠিক করতে হয় কোন উদাহরণের ওজন কত হবে। সাধারণত দুটি পথ দেখা যায়:

$$
\mathcal{L}_{\text{token}} = \frac{\sum_{i}\sum_{t} m_{i,t}\,\ell_{i,t}}{\sum_{i}\sum_{t} m_{i,t}}
\qquad\qquad
\mathcal{L}_{\text{seq}} = \frac{1}{B}\sum_{i=1}^{B} \frac{\sum_{t} m_{i,t}\,\ell_{i,t}}{\sum_{t} m_{i,t}}
$$

টোকেন গড় (token mean) উত্তরের প্রতিটি টোকেনকে সমান ওজন দেয়, তাই লম্বা উত্তরগুলোর প্রভাবই বেশি থাকে। সিকোয়েন্স গড় (sequence mean) প্রতিটি উদাহরণকে সমান ওজন দেয়, তাই ছোট উত্তরের প্রতিটি টোকেন বেশি গুরুত্ব পায়। বেশির ভাগ লাইব্রেরির ডিফল্ট হলো টোকেন গড়। দুটোরই পক্ষে যুক্তি আছে; বাগটা হলো অজান্তেই দুটোর একটা মিশ্রণ তৈরি হয়ে যাওয়া।

এই দুর্ঘটনাটা ঘটে **গ্রেডিয়েন্ট অ্যাকুমুলেশনে**, যেখানে কয়েকটি মাইক্রো-ব্যাচ মিলিয়ে অপটিমাইজারের একটি স্টেপ তৈরি হয়। প্রতিটি মাইক্রো-ব্যাচ যদি নিজের টোকেন গড় হিসাব করে আর তারপর সেই গড়গুলোর গড় নেওয়া হয়, তাহলে ফলটা ওপরের কোনো সূত্রের সঙ্গেই মেলে না। ধরুন দুটি মাইক্রো-ব্যাচ আছে: প্রথমটিতে উত্তরের 1,000টি টোকেন, মোট লস 500 (গড় 0.5); দ্বিতীয়টিতে 10টি টোকেন, মোট লস 20 (গড় 2.0)। তাহলে:

$$
\underbrace{\frac{500 + 20}{1000 + 10} = 0.515}_{\text{token mean of the full batch}}
\qquad\text{vs}\qquad
\underbrace{\tfrac{1}{2}\left(0.5 + 2.0\right) = 1.25}_{\text{mean of micro-batch means}}
$$

ছোট মাইক্রো-ব্যাচের প্রতিটি টোকেন লম্বা মাইক্রো-ব্যাচের একটি টোকেনের ১০০ গুণ ওজন পায়, আর কার্যকর অবজেক্টিভটা তখন নির্ভর করে আপনার ডেটা ঘটনাচক্রে কীভাবে ভাগ হয়েছে তার ওপর। ঠিক এই গরমিলটা বহুল ব্যবহৃত ট্রেনিং কোডে পাওয়া গিয়েছিল, Hugging Face Trainer-ও তার মধ্যে ছিল, আর ২০২৪-এর শেষ দিকে তা ঠিক করা হয়। লুপ যদি নিজে লেখেন, তাহলে লসগুলো যোগ করুন এবং পুরো স্টেপের মোট টোকেন সংখ্যা দিয়ে একবারই ভাগ করুন:

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

## ৮. ফুল ফাইন-টিউনিং: মেমরি কোথায় যায়
{: id="8-full-fine-tuning-where-the-memory-goes"}

প্রতিটি ওয়েট আপডেট করার কথা শুনলে মনে হয়, মেমরি লাগবে শুধু ওয়েটগুলোর জন্য। আসলে লাগে তার প্রায় আট গুণ। মিক্সড-প্রিসিশন ট্রেনিং আর Adam ব্যবহার করলে মোট $$\Psi$$ প্যারামিটারের প্রতিটির সঙ্গে থাকে:

$$
M_{\text{state}} = \underbrace{2\Psi}_{\text{bf16 weights}} + \underbrace{2\Psi}_{\text{bf16 grads}} + \underbrace{4\Psi}_{\text{fp32 master}} + \underbrace{4\Psi + 4\Psi}_{\text{Adam } m,\ v} = 16\,\Psi \ \text{bytes}
$$

fp32 মাস্টার কপি রাখতে হয়, কারণ বড় ওয়েটের ওপর ছোট আপডেট bf16 ঠিকমতো ধরে রাখতে পারে না। Adam-এর দুটি টেনসর আসে সরাসরি তার আপডেট নিয়ম থেকে, যা প্রতিটি গ্রেডিয়েন্টের একটি চলমান গড় (running mean) আর গ্রেডিয়েন্টের বর্গের একটি চলমান গড় রাখে:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2,
\qquad
\theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

এখানে $$\hat m_t$$ আর $$\hat v_t$$ হলো বায়াস-সংশোধিত মোমেন্ট (bias-corrected moments), আর AdamW এতে আলাদা (decoupled) একটি ওয়েট ডিকে (weight decay) টার্ম $$-\eta\lambda\theta_{t-1}$$ যোগ করে। একটি 7B মডেলের ক্ষেত্রে $$16\Psi$$ মানে **একটিও অ্যাক্টিভেশন রাখার আগেই 112 GB**। এটা একটি 80 GB GPU-র চেয়েও বেশি।

অ্যাক্টিভেশন আসে এর ওপরে, আর তা বাড়ে ব্যাচ সাইজ, সিকোয়েন্সের দৈর্ঘ্য, হিডেন সাইজ আর মডেলের গভীরতার সঙ্গে। **গ্রেডিয়েন্ট চেকপয়েন্টিং** প্রতিটি লেয়ারের শুধু ইনপুটটা রেখে দেয় এবং ব্যাকওয়ার্ড পাসের সময় বাকিটা আবার হিসাব করে নেয়; মোটামুটি এক-তৃতীয়াংশ বাড়তি কম্পিউটের বিনিময়ে অ্যাক্টিভেশন মেমরি অনেকখানি কমে যায়। আর মডেল স্টেটটাকেই মেমরিতে আঁটাতে হলে আপনার সামনে দুটি পথ: ZeRO-3 বা FSDP দিয়ে একে কয়েকটি GPU-তে ভাগ (shard) করে দেওয়া, যাতে $$16\Psi$$ GPU-র সংখ্যা দিয়ে ভাগ হয়ে যায়, অথবা বেশির ভাগ প্যারামিটার ট্রেইন করাই বন্ধ করে দেওয়া। দ্বিতীয় পথটির নামই LoRA।

## ৯. LoRA: একটি লো-র‍্যাঙ্ক আপডেট শেখা
{: id="9-lora-learn-a-low-rank-update"}

**LoRA** (Hu et al., 2021) প্রিট্রেইনড ওয়েট $$W_0 \in \mathbb{R}^{d \times k}$$ ফ্রিজ করে রাখে, আর আপডেটটা শেখে দুটি সরু ম্যাট্রিক্সের গুণফল হিসেবে:

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A x,
\qquad
B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k},\quad r \ll \min(d, k)
$$

এর পেছনে মূল বাজিটা হলো: ফাইন-টিউনিং একটি ওয়েট ম্যাট্রিক্সে যে পরিবর্তন আনে, তার *অন্তর্নিহিত র‍্যাঙ্ক* (intrinsic rank) কম, অর্থাৎ হাতে গোনা কয়েকটি দিকই (direction) বেশির ভাগ কাজটা করে। $$BA$$-এর র‍্যাঙ্ক সর্বোচ্চ $$r$$, আর এর জন্য প্যারামিটার লাগে $$r(d + k)$$টি, $$dk$$টি নয়।

{::nomarkdown}
<figure class="viz" id="fig-lora">
<div class="viz-head"><span class="viz-label">চিত্র ৫ // একটি লিনিয়ার লেয়ারে LoRA অ্যাডাপ্টার</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 314" role="img" aria-labelledby="fig-lora-t fig-lora-d">
<title id="fig-lora-t">LoRA একটি ফ্রোজেন ওয়েট ম্যাট্রিক্সের পাশে একটি ট্রেইনযোগ্য লো-র‍্যাঙ্ক পথ যোগ করে</title>
<desc id="fig-lora-d">ইনপুট x যায় d গুণ k আকারের ফ্রোজেন প্রিট্রেইনড ম্যাট্রিক্স W0-এর ভেতর দিয়ে, আর সমান্তরালভাবে যায় ট্রেইনযোগ্য ম্যাট্রিক্স A-এর ভেতর দিয়ে, যা একে r = 16টি সংখ্যায় নামিয়ে আনে, তারপর ট্রেইনযোগ্য ম্যাট্রিক্স B দিয়ে, যা আবার d-তে তুলে আনে। লো-র‍্যাঙ্ক পথটিকে alpha ভাগ r দিয়ে স্কেল করে W0 x-এর সঙ্গে যোগ করা হয়। d = k = 4096 হলে W0-এর প্যারামিটার 16,777,216টি, আর A ও B মিলিয়ে 131,072টি, অর্থাৎ 0.78%।</desc>
<defs><marker id="fig-lora-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<g class="a-in" style="--d:0s"><rect class="n" x="26" y="96" width="16" height="128" rx="4"/><text class="t big" x="34" y="246" text-anchor="middle">x</text><text class="tm" x="34" y="262" text-anchor="middle">k = 4096</text></g>
<path class="flow" d="M42 160H82V92H122" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.15s"><rect class="n" x="126" y="28" width="128" height="128" rx="8"/><rect class="glow" x="126" y="28" width="128" height="128" rx="8" style="--d:0.6s;--p:4s"/><text class="t big" x="190" y="86" text-anchor="middle">W₀</text><text class="t2" x="190" y="106" text-anchor="middle">d × k</text><text class="tm" x="190" y="140" text-anchor="middle">ফ্রোজেন</text></g>
<path class="flow" d="M254 92H566V140" marker-end="url(#fig-lora-ah)"/>
<circle class="packet a-travel" cx="262" cy="92" r="3.5" style="--dx:296px;--t:1.4s;--d:1.0s"/>
<text class="t2 a-in" x="410" y="84" text-anchor="middle" style="--d:0.5s">W₀x: প্রিট্রেইনড পথ, অপরিবর্তিত</text>
<path class="flow" d="M82 160V240H122" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.35s"><rect class="n-acc" x="126" y="232" width="128" height="16" rx="4"/><rect class="glow" x="126" y="232" width="128" height="16" rx="4" style="--d:1.5s;--p:4s"/><text class="t" x="190" y="222" text-anchor="middle">A</text><text class="t2" x="190" y="270" text-anchor="middle">r × k, শুরুতে র‍্যান্ডম</text></g>
<path class="flow" d="M254 240H282" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.5s"><rect class="n-acc" x="286" y="232" width="16" height="16" rx="3"/><text class="tm ta" x="294" y="222" text-anchor="middle">r = 16</text></g>
<path class="flow" d="M302 240H330" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.65s"><rect class="n-acc" x="334" y="176" width="16" height="128" rx="4"/><rect class="glow" x="334" y="176" width="16" height="128" rx="4" style="--d:2.0s;--p:4s"/><text class="t" x="364" y="198">B</text><text class="t2" x="364" y="216">d × r</text><text class="t2" x="364" y="232">শুরুতে 0</text></g>
<path class="flow" d="M350 286H436" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:0.8s"><rect class="n" x="440" y="270" width="64" height="32" rx="8"/><text class="t" x="472" y="291" text-anchor="middle">× α/r</text></g>
<path class="flow" d="M504 286H566V180" marker-end="url(#fig-lora-ah)"/>
<circle class="packet a-travel" cx="512" cy="286" r="3.5" style="--dx:46px;--t:.8s;--d:2.3s"/>
<g class="a-in" style="--d:0.95s"><circle class="n-acc" cx="566" cy="160" r="16"/><path class="ln" d="M558 160H574M566 152V168" style="stroke:#e8b84e"/></g>
<path class="flow" d="M582 160H628" marker-end="url(#fig-lora-ah)"/>
<g class="a-in" style="--d:1.1s"><rect class="n-acc" x="632" y="96" width="16" height="128" rx="4"/><text class="t big" x="640" y="246" text-anchor="middle">h</text><text class="tm" x="640" y="262" text-anchor="middle">d = 4096</text></g>
<text class="a-in" x="744" y="30" text-anchor="end" style="--d:1.3s"><tspan class="t">h = W₀x + (α/r)·BAx</tspan></text>
<text class="a-in" x="744" y="52" text-anchor="end" style="--d:1.4s"><tspan class="tm">W₀ 16,777,216 ফ্রোজেন</tspan></text>
<text class="a-in" x="744" y="68" text-anchor="end" style="--d:1.5s"><tspan class="tm ta">A + B 131,072 ট্রেইনযোগ্য (0.78%)</tspan></text>
</svg>
</div>
<figcaption>প্রিট্রেইনড পথটি অপরিবর্তিত থাকে। ট্রেইনযোগ্য পথটি ইনপুটকে r = 16টি সংখ্যায় চেপে ছোট করে, তারপর আবার বড় করে, তাই আপডেট BA-এর র‍্যাঙ্ক সর্বোচ্চ 16 হতে পারে। B শুরু হয় শূন্য থেকে, তাই স্টেপ 0-তে অ্যাডাপ্ট করা মডেলটি হুবহু প্রিট্রেইনড মডেলই। 4096 × 4096 আকারের একটি প্রজেকশনের জন্য এর মানে 16,777,216টির বদলে মাত্র 131,072টি ট্রেইনযোগ্য প্যারামিটার।</figcaption>
</figure>
{:/nomarkdown}

ইনিশিয়ালাইজেশনটা ইচ্ছাকৃত। $$A$$ শুরু হয় র‍্যান্ডম মান দিয়ে আর $$B$$ শুরু হয় **শূন্য** থেকে, তাই $$\Delta W = 0$$, অর্থাৎ স্টেপ 0-তে মডেলটি হুবহু প্রিট্রেইনড মডেল। এর একটা চমৎকার ফল সরাসরি গ্রেডিয়েন্ট থেকেই বোঝা যায়। ধরা যাক $$u = Ax$$ হলো $$r$$-মাত্রিক বটলনেক, $$s = \alpha / r$$, আর $$G = \partial L / \partial h$$ হলো লেয়ারের আউটপুটে এসে পৌঁছানো গ্রেডিয়েন্ট:

$$
\frac{\partial L}{\partial B} = s\, G\, u^{\top} = s\, G\, (Ax)^{\top},
\qquad
\frac{\partial L}{\partial A} = s\, B^{\top} G\, x^{\top}
$$

স্টেপ 0-তে $$B = 0$$, তাই $$\partial L / \partial A = 0$$: প্রথম স্টেপে শুধু $$B$$ নড়ে, আর $$A$$ শিখতে শুরু করে তখনই, যখন $$B$$ আর শূন্য থাকে না। $$W_0$$ কোনো গ্রেডিয়েন্টই পায় না, তার মানে এর জন্য কোনো গ্রেডিয়েন্ট বাফার, মাস্টার কপি বা Adam স্টেট রাখতে হয় না। মেমরির সাশ্রয়টা আসে এখান থেকেই।

স্কেল $$\alpha / r$$ থাকার কারণে $$r$$ বদলালেও আপডেটের আকার মোটামুটি একই থাকে, তাই লার্নিং রেট নতুন করে টিউন করার ঝামেলা কমে। র‍্যাঙ্ক বেশি হলে rsLoRA (Kalajdzievski, 2023) বলছে, তার বদলে $$\alpha / \sqrt{r}$$ ব্যবহার করা উচিত; `peft`-এ এটা `use_rslora=True` হিসেবে পাওয়া যায়। ট্রেনিংয়ের পর আপডেটটি ওয়েটের সঙ্গেই মিশিয়ে দেওয়া যায়, $$W = W_0 + \frac{\alpha}{r} BA$$, তাই মার্জ করা LoRA মডেল হুবহু আসল মডেলের গতিতেই চলে।

পুরো পদ্ধতিটা PyTorch-এর ডজনখানেক লাইনেই এঁটে যায়:

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

বাস্তবে আপনি `peft` ব্যবহার করবেন, যা মিলে যাওয়া প্রতিটি লেয়ারের জন্য একই কাজ করে:

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

Llama-2-7B ধাঁচের একটি মডেলে ৩২টি লেয়ারের প্রতিটির সাতটি প্রজেকশনেই $$r = 16$$ দিলে 39,976,960টি ট্রেইনযোগ্য প্যারামিটার পাওয়া যায়, যা মডেলের প্রায় **0.59%**। দুটি গবেষণার ফল মনে রাখার মতো। QLoRA পেপারে দেখা গেছে, ফুল ফাইন-টিউনিংয়ের সমান ফল পেতে শুধু অ্যাটেনশন প্রজেকশন নয়, **সব** লিনিয়ার লেয়ার অ্যাডাপ্ট করতে হয়। আর Biderman et al. (2024) দেখিয়েছেন যে LoRA *কম শেখে, আবার কম ভোলেও*: কোড বা গণিতের মতো বড় নতুন ডোমেইনে এটি ফুল ফাইন-টিউনিংয়ের চেয়ে পিছিয়ে থাকে, কিন্তু মডেলের সাধারণ সক্ষমতার ক্ষতি করে কম। বেশির ভাগ SFT-এর লক্ষ্য যেহেতু নতুন জ্ঞান নয়, আচরণ, তাই এই বিনিময়টা ভালোই।

## ১০. QLoRA: ১৬-বিট অ্যাডাপ্টারের নিচে ৪-বিট বেস
{: id="10-qlora-a-4-bit-base-under-16-bit-adapters"}

LoRA অপটিমাইজার স্টেট সরিয়ে দেয়, কিন্তু ফ্রোজেন ওয়েটগুলো তখনও bf16-এই বসে থাকে: 7B প্যারামিটারের জন্য ১৪ GB। **QLoRA** (Dettmers et al., 2023) এই ওয়েটগুলো রাখে ৪ বিটে, আর তার ওপরে bf16 LoRA অ্যাডাপ্টার ট্রেইন করে। এতেই একটিমাত্র ৪৮ GB GPU-তে একটি 65B মডেল ফাইন-টিউন করা সম্ভব হয়েছিল।

এই ৪-বিট ফরম্যাটের নাম **NF4**। ওয়েটগুলো ৬৪টি করে ব্লকে কোয়ান্টাইজ করা হয়। প্রতিটি ব্লককে তার সবচেয়ে বড় পরম মান (absolute value) দিয়ে স্কেল করা হয়, তারপর প্রতিটি ওয়েটকে রাউন্ড করা হয় ১৬টি লেভেলের সেট $$Q_{\text{NF4}}$$-এর নিকটতম লেভেলে। লেভেলগুলো বসানো থাকে স্ট্যান্ডার্ড নরমাল ডিস্ট্রিবিউশনের কোয়ান্টাইলে (quantile), কারণ প্রিট্রেইনড ওয়েটগুলো মোটামুটি নরমাল ডিস্ট্রিবিউশন মেনে চলে:

$$
c_b = \max_{j \in b} \lvert w_j \rvert,
\qquad
q_j = \arg\min_{q \in Q_{\text{NF4}}} \left\lvert \frac{w_j}{c_b} - q \right\rvert,
\qquad
\hat w_j = c_b\, q_j
$$

লেভেলগুলোর সম্ভাবনা সমান হওয়ায় ১৬টি মানের প্রতিটি মোটামুটি সমান হারে ব্যবহৃত হয়, আর নরমাল ডিস্ট্রিবিউশনের ডেটার জন্য একটি ৪-বিট কোড এর চেয়ে ভালো কিছু করতে পারে না। ব্লকের কনস্ট্যান্টগুলোও মেমরি নেয়: প্রতি ৬৪টি ওয়েটে একটি করে fp32 সংখ্যা মানে প্রতি প্যারামিটারে ০.৫ বিট। **ডাবল কোয়ান্টাইজেশন** (double quantization) এই কনস্ট্যান্টগুলোকে ৮ বিটে রাখে, আর প্রতি ২৫৬টি ব্লকের জন্য থাকে তাদের নিজস্ব একটি fp32 কনস্ট্যান্ট:

$$
\frac{8}{64} + \frac{32}{64 \cdot 256} \approx 0.127 \ \text{bits per parameter}
$$

ফলে একটি 7B মডেলের ফ্রোজেন ওয়েট জায়গা নেয় প্রায় $$7 \times 10^9 \times 4.127 / 8 \approx 3.6$$ GB। ফরোয়ার্ড আর ব্যাকওয়ার্ড পাসের সময় প্রতিটি ব্লককে চলতে চলতেই (on the fly) bf16-এ ডিকোয়ান্টাইজ করা হয়, তাই হিসাব তখনও চলে ১৬-বিটেই, আর গ্রেডিয়েন্ট ফ্রোজেন ওয়েটের ভেতর দিয়ে অ্যাডাপ্টার পর্যন্ত পৌঁছায়। QLoRA পেজড অপটিমাইজারও (paged optimizer) ব্যবহার করে: মেমরির ব্যবহার হঠাৎ বেড়ে গেলে এটি out-of-memory এরর দিয়ে ক্র্যাশ না করে অপটিমাইজার স্টেটকে CPU মেমরিতে সরিয়ে দেয়।

{::nomarkdown}
<figure class="viz" id="fig-mem">
<div class="viz-head"><span class="viz-label">চিত্র ৬ // একটি 7B মডেল ফাইন-টিউন করতে GPU মেমরি</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 274" role="img" aria-labelledby="fig-mem-t fig-mem-d">
<title id="fig-mem-t">৭০০ কোটি প্যারামিটারের একটি মডেলের ফুল ফাইন-টিউনিং, LoRA আর QLoRA-তে কত মেমরি লাগে</title>
<desc id="fig-mem-d">মিক্সড-প্রিসিশন Adam দিয়ে ফুল ফাইন-টিউনিংয়ে প্রতি প্যারামিটারে প্রায় ১৬ বাইট লাগে, অর্থাৎ ৭০০ কোটি প্যারামিটারের জন্য প্রায় ১১২ GB, তাও অ্যাক্টিভেশন হিসাবে ধরার আগেই। LoRA bf16 ওয়েটগুলো ফ্রিজ করে রাখে, যা প্রায় ১৪ GB, তার সাথে অ্যাডাপ্টার ও তাদের অপটিমাইজার স্টেটের জন্য প্রায় ০.৬ GB। QLoRA ফ্রোজেন ওয়েটগুলো ৪-বিট NF4-এ রাখে, প্রায় ৩.৬ GB, সাথে একই অ্যাডাপ্টার। ড্যাশ দেওয়া রেখাগুলো ২৪ GB আর ৮০ GB GPU-র সীমা দেখায়।</desc>
<line class="grid-strong" x1="295.6" y1="34" x2="295.6" y2="232" style="stroke-dasharray:3 4"/><text class="axis" x="295.6" y="26" text-anchor="middle">24 GB GPU</text>
<line class="grid-strong" x1="542.0" y1="34" x2="542.0" y2="232" style="stroke-dasharray:3 4"/><text class="axis" x="542.0" y="26" text-anchor="middle">80 GB GPU</text>
<text class="t" x="18" y="64">ফুল ফাইন-টিউনিং</text><text class="t2" x="18" y="81">bf16 + Adam</text>
<g class="row" tabindex="0" data-tip="14.0 GB" data-tip-sub="bf16 ওয়েট: 2 বাইট × 7B" aria-label="ফুল ফাইন-টিউনিং, bf16 ওয়েট: 14.0 GB"><rect class="hit" x="190.0" y="50" width="61.6" height="34"/><rect class="m1 a-grow" x="190.0" y="52" width="60.1" height="30" rx="3" style="--d:0.15s"/></g>
<g class="row" tabindex="0" data-tip="14.0 GB" data-tip-sub="bf16 গ্রেডিয়েন্ট: 2 বাইট × 7B" aria-label="ফুল ফাইন-টিউনিং, bf16 গ্রেডিয়েন্ট: 14.0 GB"><rect class="hit" x="251.6" y="50" width="61.6" height="34"/><rect class="m2 a-grow" x="251.6" y="52" width="60.1" height="30" rx="3" style="--d:0.33s"/></g>
<g class="row" tabindex="0" data-tip="28.0 GB" data-tip-sub="fp32 মাস্টার ওয়েট: 4 বাইট × 7B" aria-label="ফুল ফাইন-টিউনিং, fp32 মাস্টার ওয়েট: 28.0 GB"><rect class="hit" x="313.2" y="50" width="123.2" height="34"/><rect class="m3 a-grow" x="313.2" y="52" width="121.7" height="30" rx="3" style="--d:0.51s"/></g>
<g class="row" tabindex="0" data-tip="28.0 GB" data-tip-sub="Adam-এর প্রথম মোমেন্ট m: 4 বাইট × 7B" aria-label="ফুল ফাইন-টিউনিং, Adam-এর প্রথম মোমেন্ট m: 28.0 GB"><rect class="hit" x="436.4" y="50" width="123.2" height="34"/><rect class="m4 a-grow" x="436.4" y="52" width="121.7" height="30" rx="3" style="--d:0.69s"/></g>
<g class="row" tabindex="0" data-tip="28.0 GB" data-tip-sub="Adam-এর দ্বিতীয় মোমেন্ট v: 4 বাইট × 7B" aria-label="ফুল ফাইন-টিউনিং, Adam-এর দ্বিতীয় মোমেন্ট v: 28.0 GB"><rect class="hit" x="559.6" y="50" width="123.2" height="34"/><rect class="m5 a-grow" x="559.6" y="52" width="121.7" height="30" rx="3" style="--d:0.87s"/></g>
<text class="a-in" x="690.8" y="72" style="--d:1.40s"><tspan class="t">≈112 GB</tspan></text>
<text class="t" x="18" y="128">LoRA</text><text class="t2" x="18" y="145">bf16 বেস, r = 16</text>
<g class="row" tabindex="0" data-tip="14.0 GB" data-tip-sub="ফ্রোজেন bf16 ওয়েট: 2 বাইট × 7B" aria-label="LoRA, ফ্রোজেন bf16 ওয়েট: 14.0 GB"><rect class="hit" x="190.0" y="114" width="61.6" height="34"/><rect class="m1 a-grow" x="190.0" y="116" width="60.1" height="30" rx="3" style="--d:0.55s"/></g>
<g class="row" tabindex="0" data-tip="0.6 GB" data-tip-sub="অ্যাডাপ্টার + তাদের Adam স্টেট: 40M প্যারামিটার × 16 বাইট" aria-label="LoRA, অ্যাডাপ্টার + তাদের Adam স্টেট: 0.6 GB"><rect class="hit" x="251.6" y="114" width="8.0" height="34"/><rect class="m6 a-grow" x="251.6" y="116" width="1.5" height="30" rx="3" style="--d:0.73s"/></g>
<text class="a-in" x="262.4" y="136" style="--d:1.26s"><tspan class="t">≈14.6 GB</tspan></text>
<text class="t" x="18" y="192">QLoRA</text><text class="t2" x="18" y="209">NF4 বেস, r = 16</text>
<g class="row" tabindex="0" data-tip="3.6 GB" data-tip-sub="ফ্রোজেন NF4 ওয়েট: ≈4.13 বিট × 7B" aria-label="QLoRA, ফ্রোজেন NF4 ওয়েট: 3.6 GB"><rect class="hit" x="190.0" y="178" width="15.9" height="34"/><rect class="m7 a-grow" x="190.0" y="180" width="14.4" height="30" rx="3" style="--d:0.95s"/></g>
<g class="row" tabindex="0" data-tip="0.6 GB" data-tip-sub="অ্যাডাপ্টার + তাদের Adam স্টেট: 40M প্যারামিটার × 16 বাইট" aria-label="QLoRA, অ্যাডাপ্টার + তাদের Adam স্টেট: 0.6 GB"><rect class="hit" x="205.9" y="178" width="8.0" height="34"/><rect class="m6 a-grow" x="205.9" y="180" width="1.5" height="30" rx="3" style="--d:1.13s"/></g>
<text class="a-in" x="216.7" y="200" style="--d:1.66s"><tspan class="t">≈4.3 GB</tspan></text>
<rect class="m1" x="18" y="252" width="10" height="10" rx="2"/><text class="t2" x="33" y="261">bf16 ওয়েট</text>
<rect class="m2" x="126.6" y="252" width="10" height="10" rx="2"/><text class="t2" x="141.6" y="261">গ্রেডিয়েন্ট</text>
<rect class="m3" x="216.29999999999998" y="252" width="10" height="10" rx="2"/><text class="t2" x="231.29999999999998" y="261">fp32 মাস্টার</text>
<rect class="m4" x="318.59999999999997" y="252" width="10" height="10" rx="2"/><text class="t2" x="333.59999999999997" y="261">Adam m</text>
<rect class="m5" x="389.4" y="252" width="10" height="10" rx="2"/><text class="t2" x="404.4" y="261">Adam v</text>
<rect class="m7" x="460.2" y="252" width="10" height="10" rx="2"/><text class="t2" x="475.2" y="261">NF4 ওয়েট</text>
<rect class="m6" x="562.5" y="252" width="10" height="10" rx="2"/><text class="t2" x="577.5" y="261">অ্যাডাপ্টার + স্টেট</text>
</svg>
</div>
<details class="viz-data"><summary>সংখ্যাগুলো দেখুন</summary><table><thead><tr><th>পদ্ধতি</th><th>অংশ</th><th>GB</th></tr></thead><tbody><tr><td>ফুল ফাইন-টিউনিং</td><td>bf16 ওয়েট</td><td>14.00</td></tr><tr><td>ফুল ফাইন-টিউনিং</td><td>bf16 গ্রেডিয়েন্ট</td><td>14.00</td></tr><tr><td>ফুল ফাইন-টিউনিং</td><td>fp32 মাস্টার ওয়েট</td><td>28.00</td></tr><tr><td>ফুল ফাইন-টিউনিং</td><td>Adam-এর প্রথম মোমেন্ট m</td><td>28.00</td></tr><tr><td>ফুল ফাইন-টিউনিং</td><td>Adam-এর দ্বিতীয় মোমেন্ট v</td><td>28.00</td></tr><tr><td>ফুল ফাইন-টিউনিং</td><td>মোট</td><td>112.0</td></tr><tr><td>LoRA</td><td>ফ্রোজেন bf16 ওয়েট</td><td>14.00</td></tr><tr><td>LoRA</td><td>অ্যাডাপ্টার + তাদের Adam স্টেট</td><td>0.64</td></tr><tr><td>LoRA</td><td>মোট</td><td>14.6</td></tr><tr><td>QLoRA</td><td>ফ্রোজেন NF4 ওয়েট</td><td>3.61</td></tr><tr><td>QLoRA</td><td>অ্যাডাপ্টার + তাদের Adam স্টেট</td><td>0.64</td></tr><tr><td>QLoRA</td><td>মোট</td><td>4.3</td></tr></tbody></table></details>
<figcaption>শুধু মডেল স্টেট, ৭০০ কোটি প্যারামিটারের জন্য; অ্যাক্টিভেশন এর ওপরে আলাদা করে যোগ হয়, আর তা ব্যাচ সাইজ ও সিকোয়েন্সের দৈর্ঘ্যের সাথে বাড়ে। ফুল ফাইন-টিউনিংয়ের বেশিরভাগ মেমরি যায় অপটিমাইজার স্টেটে, ওয়েটে নয়। LoRA এর প্রায় পুরোটাই সরিয়ে দেয়, আর QLoRA ফ্রোজেন ওয়েটকে চার ভাগের এক ভাগে নামিয়ে আনে। ঠিক এ কারণেই ফাইন-টিউনিং একটি পুরো ক্লাস্টার থেকে নেমে আসে একটিমাত্র সাধারণ কনজিউমার GPU-তে।</figcaption>
</figure>
{:/nomarkdown}

দুটি ব্যবহারিক কথা। LoRA-র তুলনায় QLoRA ধীর, কারণ প্রতিটি স্টেপে প্রতিটি লেয়ার ডিকোয়ান্টাইজ করতে হয়; আপনি আসলে সময় দিয়ে মেমরি কিনছেন। আর অ্যাডাপ্টার মার্জ করার সময় সেগুলো বেস মডেলের একটি **bf16 কপিতে** মার্জ করুন, ৪-বিট কপিতে নয়; নইলে মার্জ করা ওয়েটগুলো দ্বিতীয়বার রাউন্ড হয়ে যাবে।

## ১১. যে হাইপারপ্যারামিটারগুলো সত্যিই গুরুত্বপূর্ণ
{: id="11-hyperparameters-that-matter"}

SFT ছোটখাটো ভুল অনেকটাই মাফ করে দেয়, কিন্তু অল্প কয়েকটি সেটিংই বেশিরভাগ ফলাফল ঠিক করে দেয়। নিচের মানগুলো শুরু করার জন্য যুক্তিসংগত, কোনো বাঁধাধরা নিয়ম নয়:

| সেটিং | ফুল ফাইন-টিউনিং | LoRA / QLoRA |
|---|---|---|
| লার্নিং রেট | 1e-5 থেকে 2e-5 | 1e-4 থেকে 3e-4 |
| ইপক | 1 থেকে 3 | 1 থেকে 3 |
| ওয়ার্মআপ | মোট স্টেপের প্রায় 3% | মোট স্টেপের প্রায় 3% |
| শিডিউল | কোসাইন বা লিনিয়ার ডিকে | কোসাইন বা লিনিয়ার ডিকে |
| ইফেক্টিভ ব্যাচ | 64 থেকে 256 সিকোয়েন্স | 16 থেকে 128 সিকোয়েন্স |
| র‍্যাঙ্ক ও alpha | ব্যবহৃত হয় না | r = 8 থেকে 64, alpha = r বা 2r |
| ওয়েট ডিকে | 0 থেকে 0.1 | 0 |

LoRA-র লার্নিং রেট লাগে ফুল ফাইন-টিউনিংয়ের মোটামুটি দশ গুণ, কারণ আপডেট শুরু হয় শূন্য থেকে এবং থাকে অনেক ছোট একটি স্পেসে। সাধারণ শিডিউল হলো প্রথমে $$T_w$$ স্টেপ ধরে লিনিয়ার ওয়ার্মআপ, তারপর একটি নিম্নসীমা পর্যন্ত কোসাইন ডিকে (cosine decay):

$$
\eta_t =
\begin{cases}
\eta_{\max}\, \dfrac{t}{T_w} & t < T_w \\[8pt]
\eta_{\min} + \tfrac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)\left(1 + \cos\!\left(\pi\, \dfrac{t - T_w}{T - T_w}\right)\right) & t \ge T_w
\end{cases}
$$

ইফেক্টিভ ব্যাচ হলো প্রতি স্টেপে অপটিমাইজার যতটুকু দেখে, $$B_{\text{eff}} = B_{\text{device}} \times \text{accumulation steps} \times \text{GPUs}$$। এটাকে শুধু সিকোয়েন্সের হিসাবে নয়, টোকেনের হিসাবেও ভাবুন: 2,048 টোকেনে প্যাকিং করলে প্রতি স্টেপে ৩২টি সিকোয়েন্স মানে প্রম্পটসহ প্রায় ৬৫,০০০ টোকেন।

সবচেয়ে বেশি সাবধান থাকতে হয় **ইপক** নিয়ে। SFT ডেটাসেট ছোট, আর একটি বড় মডেল খুব দ্রুত সেগুলো মুখস্থ করে ফেলে। এর স্পষ্ট লক্ষণ ধরা পড়ে লস কার্ভে:

{::nomarkdown}
<figure class="viz" id="fig-loss">
<div class="viz-head"><span class="viz-label">চিত্র ৭ // SFT-তে ওভারফিটিং দেখতে কেমন (স্কিম্যাটিক)</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 312" role="img" aria-labelledby="fig-loss-t fig-loss-d">
<title id="fig-loss-t">সুপারভাইজড ফাইন-টিউনিংয়ের তিন ইপক জুড়ে ট্রেনিং ও ভ্যালিডেশন লসের একটি স্কিম্যাটিক ছবি</title>
<desc id="fig-loss-d">এটি একটি স্কিম্যাটিক, সত্যিকারের কোনো রান থেকে মাপা নয়। ইপক ১ জুড়ে ট্রেনিং লস কমতে থাকে, তারপর ইপক ২ ও ৩-এর শুরুতে হঠাৎ অনেকটা নেমে যায়, কারণ মডেল তখন এমন উদাহরণ দেখছে যা সে আগেই মুখস্থ করে ফেলেছে। ভ্যালিডেশন লস কমে, ইপক ২-এ স্টেপ ৩০০-এর কাছে সর্বনিম্নে পৌঁছায়, তারপর বাড়তে থাকে। সেরা চেকপয়েন্টটি ঠিক সেই সর্বনিম্ন বিন্দুতে।</desc>
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
<text class="tm" x="175.0" y="26" text-anchor="middle">ইপক ১</text>
<text class="tm" x="397.0" y="26" text-anchor="middle">ইপক ২</text>
<text class="tm" x="619.0" y="26" text-anchor="middle">ইপক ৩</text>
<text class="t2" x="397.0" y="282" text-anchor="middle">অপটিমাইজার স্টেপ</text>
<path class="s1-line a-draw" pathLength="1" d="M66.2 41.4L70.7 47.0L75.1 54.2L79.5 60.1L84.0 63.9L88.4 68.5L92.9 75.2L97.3 81.1L101.7 84.5L106.2 87.4L110.6 91.9L115.1 96.2L119.5 97.7L123.9 97.9L128.4 99.8L132.8 102.8L137.3 103.9L141.7 103.8L146.1 105.4L150.6 109.1L155.0 111.7L159.5 112.3L163.9 113.7L168.3 117.0L172.8 119.4L177.2 119.1L181.7 118.4L186.1 119.6L190.5 120.9L195.0 119.9L199.4 118.3L203.9 119.0L208.3 121.1L212.7 121.7L217.2 121.2L221.6 122.4L226.1 125.3L230.5 126.8L234.9 126.1L239.4 125.8L243.8 127.3L248.3 127.9L252.7 126.0L257.1 124.2L261.6 124.7L266.0 125.7L270.5 124.9L274.9 123.8L279.3 125.1L283.8 127.7L288.2 152.8L292.7 152.2L297.1 153.0L301.5 155.1L306.0 155.5L310.4 153.6L314.9 152.4L319.3 153.3L323.7 153.6L328.2 151.9L332.6 150.7L337.1 152.2L341.5 154.3L345.9 154.4L350.4 154.1L354.8 155.8L359.3 158.3L363.7 158.5L368.1 157.0L372.6 156.9L377.0 158.0L381.5 157.7L385.9 155.4L390.3 154.4L394.8 155.8L399.2 156.9L403.7 156.2L408.1 156.1L412.5 158.4L417.0 160.8L421.4 160.8L425.9 160.0L430.3 161.0L434.7 162.5L439.2 161.7L443.6 159.4L448.1 158.8L452.5 160.0L456.9 160.0L461.4 158.5L465.8 158.5L470.3 161.0L474.7 162.8L479.1 162.6L483.6 162.6L488.0 164.5L492.5 166.2L496.9 165.2L501.3 163.5L505.8 163.6L510.2 184.0L514.7 183.0L519.1 180.9L523.5 181.0L528.0 183.0L532.4 183.9L536.9 183.2L541.3 183.8L545.7 186.4L550.2 187.9L554.6 186.9L559.1 185.9L563.5 186.8L567.9 187.5L572.4 185.7L576.8 183.6L581.3 183.9L585.7 185.3L590.1 185.1L594.6 184.1L599.0 185.1L603.5 187.8L607.9 189.0L612.3 188.2L616.8 188.3L621.2 190.0L625.7 190.5L630.1 188.5L634.5 186.8L639.0 187.4L643.4 188.0L647.9 186.7L652.3 185.5L656.7 186.8L661.2 189.2L665.6 189.7L670.1 189.1L674.5 190.2L678.9 192.5L683.4 192.9L687.8 191.1L692.3 190.2L696.7 191.1L701.1 191.1L705.6 189.0L710.0 187.7L714.5 189.0L718.9 190.6L723.3 190.3L727.8 189.9" style="--d:.2s;--t:2.2s"/>
<rect class="zone a-in" x="397.0" y="34" width="333.0" height="206" style="--d:2.4s"/>
<text class="t2 a-in" x="722" y="52" text-anchor="end" style="--d:2.5s">ভ্যালিডেশন লস বাড়ছে: শেখা নয়, মুখস্থ</text>
<path class="s2-line a-draw" pathLength="1" d="M119.5 73.1L175.0 99.9L230.5 112.3L286.0 119.5L341.5 123.6L397.0 125.2L452.5 124.6L508.0 122.6L563.5 117.4L619.0 111.2L674.5 105.1L730.0 98.9" style="--d:.6s;--t:2.0s"/>
<g class="pt" tabindex="0" data-tip="1.62" data-tip-sub="স্টেপ 50-এ ভ্যালিডেশন লস" aria-label="স্টেপ 50, ভ্যালিডেশন লস 1.62"><circle class="hit" cx="119.5" cy="73.1" r="12"/><circle class="mk2 a-pop" cx="119.5" cy="73.1" r="4" style="--d:0.60s"/></g>
<g class="pt" tabindex="0" data-tip="1.36" data-tip-sub="স্টেপ 100-এ ভ্যালিডেশন লস" aria-label="স্টেপ 100, ভ্যালিডেশন লস 1.36"><circle class="hit" cx="175.0" cy="99.9" r="12"/><circle class="mk2 a-pop" cx="175.0" cy="99.9" r="4" style="--d:0.76s"/></g>
<g class="pt" tabindex="0" data-tip="1.24" data-tip-sub="স্টেপ 150-এ ভ্যালিডেশন লস" aria-label="স্টেপ 150, ভ্যালিডেশন লস 1.24"><circle class="hit" cx="230.5" cy="112.3" r="12"/><circle class="mk2 a-pop" cx="230.5" cy="112.3" r="4" style="--d:0.92s"/></g>
<g class="pt" tabindex="0" data-tip="1.17" data-tip-sub="স্টেপ 200-এ ভ্যালিডেশন লস" aria-label="স্টেপ 200, ভ্যালিডেশন লস 1.17"><circle class="hit" cx="286.0" cy="119.5" r="12"/><circle class="mk2 a-pop" cx="286.0" cy="119.5" r="4" style="--d:1.08s"/></g>
<g class="pt" tabindex="0" data-tip="1.13" data-tip-sub="স্টেপ 250-এ ভ্যালিডেশন লস" aria-label="স্টেপ 250, ভ্যালিডেশন লস 1.13"><circle class="hit" cx="341.5" cy="123.6" r="12"/><circle class="mk2 a-pop" cx="341.5" cy="123.6" r="4" style="--d:1.24s"/></g>
<g class="pt" tabindex="0" data-tip="1.11" data-tip-sub="স্টেপ 300-এ ভ্যালিডেশন লস" aria-label="স্টেপ 300, ভ্যালিডেশন লস 1.11"><circle class="hit" cx="397.0" cy="125.2" r="12"/><circle class="mk2 a-pop" cx="397.0" cy="125.2" r="4" style="--d:1.40s"/></g>
<g class="pt" tabindex="0" data-tip="1.12" data-tip-sub="স্টেপ 350-এ ভ্যালিডেশন লস" aria-label="স্টেপ 350, ভ্যালিডেশন লস 1.12"><circle class="hit" cx="452.5" cy="124.6" r="12"/><circle class="mk2 a-pop" cx="452.5" cy="124.6" r="4" style="--d:1.56s"/></g>
<g class="pt" tabindex="0" data-tip="1.14" data-tip-sub="স্টেপ 400-এ ভ্যালিডেশন লস" aria-label="স্টেপ 400, ভ্যালিডেশন লস 1.14"><circle class="hit" cx="508.0" cy="122.6" r="12"/><circle class="mk2 a-pop" cx="508.0" cy="122.6" r="4" style="--d:1.72s"/></g>
<g class="pt" tabindex="0" data-tip="1.19" data-tip-sub="স্টেপ 450-এ ভ্যালিডেশন লস" aria-label="স্টেপ 450, ভ্যালিডেশন লস 1.19"><circle class="hit" cx="563.5" cy="117.4" r="12"/><circle class="mk2 a-pop" cx="563.5" cy="117.4" r="4" style="--d:1.88s"/></g>
<g class="pt" tabindex="0" data-tip="1.25" data-tip-sub="স্টেপ 500-এ ভ্যালিডেশন লস" aria-label="স্টেপ 500, ভ্যালিডেশন লস 1.25"><circle class="hit" cx="619.0" cy="111.2" r="12"/><circle class="mk2 a-pop" cx="619.0" cy="111.2" r="4" style="--d:2.04s"/></g>
<g class="pt" tabindex="0" data-tip="1.31" data-tip-sub="স্টেপ 550-এ ভ্যালিডেশন লস" aria-label="স্টেপ 550, ভ্যালিডেশন লস 1.31"><circle class="hit" cx="674.5" cy="105.1" r="12"/><circle class="mk2 a-pop" cx="674.5" cy="105.1" r="4" style="--d:2.20s"/></g>
<g class="pt" tabindex="0" data-tip="1.37" data-tip-sub="স্টেপ 600-এ ভ্যালিডেশন লস" aria-label="স্টেপ 600, ভ্যালিডেশন লস 1.37"><circle class="hit" cx="730.0" cy="98.9" r="12"/><circle class="mk2 a-pop" cx="730.0" cy="98.9" r="4" style="--d:2.36s"/></g>
<circle class="ring a-ping" cx="397.0" cy="125.2" r="7" style="--d:2.6s"/>
<text class="a-in" x="397.0" y="109.2" text-anchor="middle" style="--d:2.6s"><tspan class="t">সেরা চেকপয়েন্ট</tspan></text>
<text class="a-in" x="516.0" y="211.2" style="--d:2.3s"><tspan class="t2">প্রতি নতুন ইপকে ট্রেনিং লস নেমে যায়</tspan></text>
<g class="a-in" style="--d:0.3s"><rect class="s1" x="64" y="296" width="14" height="4" rx="2"/><text class="t2" x="84" y="301">ট্রেনিং লস</text><rect class="s2" x="194" y="296" width="14" height="4" rx="2"/><text class="t2" x="214" y="301">ভ্যালিডেশন লস</text></g>
</svg>
</div>
<details class="viz-data"><summary>সংখ্যাগুলো দেখুন</summary><table><thead><tr><th>স্টেপ</th><th>ভ্যালিডেশন লস</th></tr></thead><tbody><tr><td>50</td><td>1.62</td></tr><tr><td>100</td><td>1.36</td></tr><tr><td>150</td><td>1.24</td></tr><tr><td>200</td><td>1.17</td></tr><tr><td>250</td><td>1.13</td></tr><tr><td>300</td><td>1.11</td></tr><tr><td>350</td><td>1.12</td></tr><tr><td>400</td><td>1.14</td></tr><tr><td>450</td><td>1.19</td></tr><tr><td>500</td><td>1.25</td></tr><tr><td>550</td><td>1.31</td></tr><tr><td>600</td><td>1.37</td></tr></tbody></table></details>
<figcaption>প্রায়ই চোখে পড়বে এমন একটি প্যাটার্নের স্কিম্যাটিক, মাপা কোনো রান নয়। প্রতিবার নতুন ইপক শুরু হলে ট্রেনিং লস এক ধাপ নেমে যায়, কারণ মডেল তখন এমন টেক্সট প্রেডিক্ট করছে যা সে আগেই দেখেছে। সৎ সংখ্যাটি হলো ভ্যালিডেশন লস: এর সর্বনিম্ন বিন্দুর চেকপয়েন্টটি রেখে দিন, আর এটি ওপরের দিকে উঠতে শুরু করলেই থামুন।</figcaption>
</figure>
{:/nomarkdown}

কয়েকশো স্টেপ পরপর আলাদা করে রাখা অংশে (held-out split) মূল্যায়ন করুন, সবচেয়ে কম ভ্যালিডেশন লসের চেকপয়েন্টটি রেখে দিন, আর ভ্যালিডেশন লস বাড়তে দেখলে সেটাকে নয়েজ না ভেবে থামার সংকেত হিসেবে নিন। আরও বেশি সিগন্যাল দরকার হলে একই ডেটায় বাড়তি ইপক চালানোর চেয়ে আরও বৈচিত্র্যময় ডেটা যোগ করা বেশি কাজে দেয়।

## ১২. ডেটা: পরিমাণের চেয়ে মান বড়
{: id="12-data-quality-beats-quantity"}

সেকশন ২-এ আমরা দেখেছি, SFT মডেলকে ডেটার প্রতিটি আচরণ ঢেকে ফেলার দিকে ঠেলে দেয়। এই ফলাফলের বাস্তব রূপ হলো LIMA (Zhou et al., 2023): মাত্র **১,০০০টি** যত্ন করে বাছাই করা উদাহরণে ফাইন-টিউন করা একটি 65B মডেল এমন উত্তর দিয়েছিল, যেগুলোকে মানুষ প্রায়ই এর চেয়ে অনেক বেশি ডেটায় ট্রেইন করা মডেলের উত্তরের সমান ভালো বলে রেট করেছে। এক হাজার চমৎকার উদাহরণ শেখায় ফরম্যাট আর স্টাইল; এক লাখ গড়পড়তা উদাহরণ শেখায় গড়পড়তা হতে।

একটি ভালো SFT ডেটাসেট দেখতে যেমন হয়:

<ul class="checklist">
  <li><strong>সঠিক।</strong> প্রতিটি উত্তর এমন হতে হবে যা আপনি নিশ্চিন্তে ব্যবহারকারীর হাতে তুলে দিতে পারেন। মডেল ভালো উত্তরগুলো যতটা নিখুঁতভাবে নকল করবে, ভুলগুলোও ঠিক ততটাই নিখুঁতভাবে নকল করবে।</li>
  <li><strong>সামঞ্জস্যপূর্ণ।</strong> একটি ফরম্যাট, একটি টোন, না বলার একটিই ধরন। নানা রকম রীতি মিশে থাকলে মডেলও এলোমেলোভাবে এক রীতি থেকে আরেক রীতিতে লাফাবে।</li>
  <li><strong>প্রম্পটে বৈচিত্র্য।</strong> যত ধরনের অনুরোধ আসবে বলে আশা করেন, তার পুরো পরিসরটাই রাখুন, অস্পষ্ট, লম্বা আর ইচ্ছা করে বিপাকে ফেলতে চাওয়া (adversarial) অনুরোধও। একই প্রশ্নকে বিশ রকমভাবে ঘুরিয়ে লিখলে সেটা বৈচিত্র্য নয়।</li>
  <li><strong>ডুপ্লিকেটমুক্ত।</strong> হুবহু এক এবং প্রায় এক রকম, দুই ধরনের ডুপ্লিকেটই সরিয়ে ফেলুন (এ কাজে MinHash ভালো চলে)। একই উদাহরণ বারবার থাকলে তা ওই উদাহরণগুলোর ওপর বাড়তি ইপকের মতো কাজ করে।</li>
  <li><strong>কনটামিনেশনমুক্ত (decontaminated)।</strong> আপনার মূল্যায়নের সেটগুলোর সাথে মিলে যায় এমন সবকিছু সরিয়ে দিন, নইলে আপনার মূল্যায়ন আসলে মাপবে মডেলের মুখস্থশক্তি।</li>
  <li><strong>সাধারণ ডেটার সাথে মেশানো</strong>, যদি কাজটি সংকীর্ণ হয়। খানিকটা সাধারণ ইনস্ট্রাকশন ডেটা রাখলে যে দক্ষতাগুলো আপনি ট্রেইন করছেন না, মডেল সেগুলো ধরে রাখতে পারে।</li>
</ul>

## ১৩. ফাইন-টিউন করা মডেলের মূল্যায়ন
{: id="13-evaluating-a-fine-tuned-model"}

ভ্যালিডেশন লস বলে দেয় কখন থামতে হবে। মডেল আসলে আগের চেয়ে ভালো হয়েছে কি না, তা বলে না। সেটা জানতে দরকার এমন মূল্যায়ন, যা দেখতে আসল কাজের মতো:

- আলাদা করে রাখা সেটে **টাস্ক মেট্রিক**: এক্সট্র্যাকশনের জন্য exact match বা F1, JSON আউটপুটের জন্য স্কিমা ঠিক আছে কি না, কোডের জন্য ইউনিট টেস্ট (pass@k)।
- বেস মডেলের সাথে, এবং ভালো একটি প্রম্পট দেওয়া বেস মডেলের সাথে **পাশাপাশি তুলনা**। যত্ন করে প্রম্পট লিখে বা রিট্রিভাল দিয়েই যদি ৯৫% পথ পার হয়ে যায়, তাহলে ফাইন-টিউন করা মডেল রক্ষণাবেক্ষণের ঝামেলা হয়তো পোষাবে না।
- যেসব উত্তরের কোনো একক সঠিক রূপ নেই (open-ended), সেগুলোর জন্য লিখিত রুব্রিকসহ **LLM-as-judge**। এর দেওয়া সংখ্যায় ভরসা করার আগে কয়েক ডজন মানুষের দেওয়া লেবেলের সাথে বিচারক মডেলটিকে মিলিয়ে দেখুন।
- সাধারণ দক্ষতার ওপর **রিগ্রেশন চেক**: নির্দেশনা মেনে চলা, একটি সাধারণ জ্ঞানের বেঞ্চমার্ক আর আপনার সেফটি প্রম্পটগুলো। ভুলে যাওয়া (catastrophic forgetting) ধরা পড়ে ঠিক এখানেই।

ডিকোডিং সেটিং স্থির রাখুন (greedy, অথবা নির্দিষ্ট temperature আর seed), যাতে পার্থক্যগুলো আসে মডেল থেকে, স্যাম্পলিং থেকে নয়।

## ১৪. একটি পূর্ণাঙ্গ QLoRA SFT স্ক্রিপ্ট
{: id="14-a-complete-qlora-sft-script"}

এবার সব একসাথে: ৪-বিট বেস মডেল, প্রতিটি লিনিয়ার লেয়ারে LoRA, মাস্ক করা লেবেল, আর একটি ভ্যালিডেশন স্প্লিট, যা সেরা চেকপয়েন্টটি বেছে নেয়। ডেটা হলো একটি JSONL ফাইল, যার প্রতি লাইনে একটি করে কথোপকথন:

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

`DataCollatorForSeq2Seq` কজাল (causal) মডেলের জন্যও কাজ করে: এটি `input_ids`-কে প্যাড টোকেন দিয়ে আর `labels`-কে `-100` দিয়ে প্যাড করে, ফলে প্যাডিং কখনো লস পর্যন্ত পৌঁছায় না। ফলাফলটি সার্ভ করতে অ্যাডাপ্টারকে বেস মডেলের একটি bf16 কপিতে মার্জ করুন:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)   # torch_dtype= on transformers before 4.56
merged = PeftModel.from_pretrained(base, "sft-qlora/adapter").merge_and_unload()
merged.save_pretrained("sft-merged")
AutoTokenizer.from_pretrained(MODEL).save_pretrained("sft-merged")
```

## শেষ কথা
{: id="final-thoughts"}

একটি ল্যাঙ্গুয়েজ মডেল ট্রেইন করার ধাপগুলোর মধ্যে সুপারভাইজড ফাইন-টিউনিংই সবচেয়ে সাদামাটা। অবজেক্টিভটা সেটাই, যা দিয়ে মডেলের প্রিট্রেনিং হয়েছিল, আর কোডটা স্রেফ একটি মাস্ক করা ক্রস-এনট্রপি। ফলাফল বদলে দেওয়ার আসল জায়গাগুলো এর চারপাশে: কোন টোকেনগুলো লসে গোনা হবে, উদাহরণগুলোকে কতটা গুরুত্ব দেওয়া হবে, কোন প্যারামিটারগুলোকে বদলাতে দেওয়া হবে, আর সবচেয়ে বড় কথা, আপনি মডেলকে কোন উদাহরণগুলো দেখাচ্ছেন।

এই পোস্ট থেকে যদি তিনটি ধারণা মনে রাখতে হয়, তাহলে সেগুলো হবে এই। SFT ফরোয়ার্ড KL কমায়, তাই আপনার ডেটায় যা কিছু আছে মডেল তার সবই নকল করবে; ফলে ডেটা বাছাই করে গুছিয়ে তোলাই আসল কাজ। প্রম্পট মাস্ক করুন, এন্ড-অফ-টার্ন টোকেনটি রেখে দিন, আর ট্রেনিংয়ের আগে আপনার লেবেলগুলো ডিকোড করে দেখে নিন। আর শুরু করুন সব লিনিয়ার লেয়ারে QLoRA, একটি ভ্যালিডেশন স্প্লিট আর এক বা দুই ইপক দিয়ে; ফুল ফাইন-টিউনিংয়ের দিকে যান কেবল তখনই, যখন কোনো মূল্যায়ন দেখায় যে LoRA-ই আসল বাধা (bottleneck)।

## তথ্যসূত্র
{: id="references"}

- Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). InstructGPT-এর গবেষণাপত্র।
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
- Zhou et al. (2023). [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206).
- Biderman et al. (2024). [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673).
- Gekhman et al. (2024). [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904)
- Shi et al. (2024). [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394).
- Kalajdzievski (2023). [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732).
- Rajbhandari et al. (2020). [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
- Hugging Face. [PEFT documentation](https://huggingface.co/docs/peft) এবং [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)।
