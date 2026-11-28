---
title: "গ্রাফ নিউরাল নেটওয়ার্ক, সহজ করে: একটি নোড কীভাবে প্রতিবেশীদের কাছ থেকে শেখে"
category: "মেশিন লার্নিং"
tags: [GNN, "গ্রাফ নিউরাল নেটওয়ার্ক", GCN, GraphSAGE, GAT, PyTorch Geometric]
excerpt: "অণু, বন্ধুত্ব, রাস্তার মানচিত্র আর লেনদেন, সবই থাকে গ্রাফে, আর গ্রাফের কোনো প্রথম সারি নেই। তবু গ্রাফ নিউরাল নেটওয়ার্ক কীভাবে শেখে, অ্যানিমেশনে এক ধাপ এক ধাপ করে: মেসেজ পাসিং, GCN, GraphSAGE আর GAT, গভীরতা কেন ক্ষতি করে, আর PyTorch-এর ৩০ লাইনে একটি চালু মডেল।"
description: "গ্রাফ নিউরাল নেটওয়ার্কের একটি ছবিনির্ভর, ধাপে ধাপে পরিচিতি: অ্যাডজেসেন্সি ম্যাট্রিক্স আর পারমুটেশন ইনভেরিয়েন্স, মেসেজ পাসিং, রিসেপটিভ ফিল্ড আর কম্পিউটেশন ট্রি, ম্যাট্রিক্স রূপে GCN লেয়ার, GraphSAGE আর গ্রাফ অ্যাটেনশন, যোগফল বনাম গড় অ্যাগ্রিগেশন আর Weisfeiler-Lehman সীমা, ওভারস্মুদিং, টাস্ক হেড, আর Cora-তে চালানো যায় এমন PyTorch ও PyTorch Geometric কোড।"
image: https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1600&auto=format&fit=crop
image_alt: "কক্ষপথ থেকে দেখা রাতের পৃথিবী, মহাদেশজুড়ে শহরের আলো যেন একটি নেটওয়ার্কের নোড আর এজের মতো একে অপরের সঙ্গে জুড়ে আছে"
math: true
---

ডিপ লার্নিং যে ধরনের ডেটা নিয়ে বড় হয়েছে, তার বেশিরভাগেরই একটা নির্ভরযোগ্য আকার আছে। ছবি একটি গ্রিড: প্রতিটি পিক্সেলের ওপরে, নিচে আর দুই পাশে একটি করে পিক্সেল থাকে। বাক্য একটি সরলরেখা: প্রতিটি শব্দের আগে একটি শব্দ, পরে একটি। কনভোলিউশন আর ট্রান্সফরমার তৈরিই হয়েছে এই নিয়মিত গঠনকে ঘিরে।

কিন্তু গুরুত্বপূর্ণ অনেক ডেটার এমন কোনো আকার নেই। একটি অণু হলো বন্ধনে জোড়া পরমাণু। একটি সোশ্যাল নেটওয়ার্ক হলো বন্ধুত্বে জোড়া মানুষ। রাস্তার মানচিত্র হলো রাস্তায় জোড়া মোড়, আর একটি পেমেন্ট সিস্টেম হলো লেনদেনে জোড়া অ্যাকাউন্ট। প্রতিটি ক্ষেত্রেই কোনো কিছু সম্পর্কে আপনি যা জানেন, তা নির্ভর করে সেটি কার সঙ্গে যুক্ত তার ওপর, আর এই সংযোগগুলো অনিয়মিত: একটি পরমাণুর বন্ধন একটি, আরেকটির চারটি; একটি অ্যাকাউন্টের লেনদেনের সঙ্গী দুজন, আরেকটির দুই হাজার। এখানে প্রথম সারি নেই, বাঁ পাশের প্রতিবেশী নেই, নির্দিষ্ট আকারও নেই।

**গ্রাফ নিউরাল নেটওয়ার্ক** (GNN) দিয়েই ডিপ লার্নিং এই সমস্যা সামলায়। মূল ধারণাটা এক বাক্যে বলা যায়: *প্রতিটি নোডকে তার প্রতিবেশীদের দিয়ে বর্ণনা করুন, তারপর আবার করুন।* GCN থেকে গ্রাফ অ্যাটেনশন পর্যন্ত বাকি সবকিছুই এই বর্ণনা ঠিক কীভাবে করা হবে, তারই নানা রূপ।

এই পোস্টে ধারণাটা এক ধাপ এক ধাপ করে গড়ে তোলা হয়েছে, প্রতিটি ধাপের জন্য একটি করে চিত্রসহ: কম্পিউটার একটি গ্রাফ কীভাবে রাখে, সাধারণ নেটওয়ার্ক কেন গ্রাফ নিয়ে হিমশিম খায়, মেসেজ পাসিং, লেয়ার বাড়ালে একটি নোড কীভাবে আরও দূরে দেখতে পায়, GCN-এর পেছনের ম্যাট্রিক্স রূপ, সব জায়গায় দেখা মেলে এমন তিনটি লেয়ার, যোগফল আর গড়ের মধ্যে বেছে নেওয়া কেন জরুরি, গভীর মানেই কেন ভালো নয়, আর শেষে ল্যাপটপে চালানো যায় এমন একটি চালু মডেল।

ম্যাথের জন্য ম্যাট্রিক্স গুণ জানলেই যথেষ্ট। চিত্রগুলো স্ক্রল করে সামনে এলেই অ্যানিমেট হয়: সঠিক মান দেখতে কার্সর রাখুন, আবার দেখতে **আবার দেখুন** চাপুন। যেসব চিত্রের নিচে ধাপের বাটন আছে, সেগুলো একটি প্রক্রিয়া ধাপে ধাপে দেখায়; **চালান** চাপুন, অথবা নিজের গতিতে ধাপগুলোতে ক্লিক করুন। কোডে PyTorch আর PyTorch Geometric ব্যবহার করা হয়েছে, আর কোড থেকে উদ্ধৃত প্রতিটি সংখ্যা আসলে চালিয়ে পাওয়া।
{: .callout}

## ১. কম্পিউটারের চোখে একটি গ্রাফ
{: id="1-what-a-graph-looks-like-to-a-computer"}

গ্রাফ হলো কিছু **নোড** (যাদের ভার্টেক্সও বলা হয়) আর তাদের মধ্যকার কিছু **এজ**-এর সমষ্টি। একটি নিউরাল নেটওয়ার্ককে গ্রাফ দিতে হলে দুটি জিনিস লাগে:

- **নোড ফিচার:** একটি ম্যাট্রিক্স $$X$$, যেখানে প্রতিটি নোডের জন্য একটি সারি। একজন মানুষের ক্ষেত্রে সেই সারিতে থাকতে পারে বয়স, সক্রিয়তা আর আগ্রহ; একটি পরমাণুর ক্ষেত্রে মৌল আর তার চার্জ; একটি পেপারের ক্ষেত্রে কোন কোন শব্দ তাতে আছে।
- **গঠন:** সাধারণত একটি **অ্যাডজেসেন্সি ম্যাট্রিক্স** $$A$$। $$n$$টি নোড থাকলে এটি $$n \times n$$, আর $$u$$ ও $$v$$-এর মধ্যে এজ থাকলে $$A_{uv} = 1$$।

{::nomarkdown}
<figure class="viz" id="fig-graph">
<div class="viz-head"><span class="viz-label">চিত্র ১ // কম্পিউটারের চোখে একটি গ্রাফ</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 290" role="img" aria-labelledby="fig-graph-t fig-graph-d">
<title id="fig-graph-t">একটি ছোট গ্রাফ, তার অ্যাডজেসেন্সি ম্যাট্রিক্স আর ফিচার ম্যাট্রিক্স</title>
<desc id="fig-graph-d">A থেকে G পর্যন্ত সাতটি নোড আর আটটি এজের একটি গ্রাফ। দুটি নোডের মধ্যে এজ থাকলে অ্যাডজেসেন্সি ম্যাট্রিক্সের সেই ঘরে 1 থাকে: সারি A-তে B, C আর D-এর ঘরে 1। ফিচার ম্যাট্রিক্সে প্রতিটি নোডের জন্য তিনটি সংখ্যার একটি করে সারি।</desc>
<text class="tm ta" x="24" y="26">গ্রাফ // ৭টি নোড, ৮টি এজ</text>
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
<text class="tm ta" x="440" y="26">অ্যাডজেসেন্সি ম্যাট্রিক্স A</text>
<text class="tm" x="440" y="42">1 = এজ আছে, 0 = নেই</text>
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
<rect class="cell" x="440" y="70" width="20" height="20" data-tip="A[A, A] = 0" data-tip-sub="সেলফ-লুপ নেই (GCN পরে যোগ করে)"/>
<rect class="cell" x="460" y="70" width="20" height="20" data-tip="A[A, B] = 1" data-tip-sub="A আর B-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="463" y="73" width="14" height="14" rx="3" style="--d:0.70s;pointer-events:none"/>
<rect class="cell" x="480" y="70" width="20" height="20" data-tip="A[A, C] = 1" data-tip-sub="A আর C-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="483" y="73" width="14" height="14" rx="3" style="--d:0.87s;pointer-events:none"/>
<rect class="cell" x="500" y="70" width="20" height="20" data-tip="A[A, D] = 1" data-tip-sub="A আর D-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="503" y="73" width="14" height="14" rx="3" style="--d:1.04s;pointer-events:none"/>
<rect class="cell" x="520" y="70" width="20" height="20" data-tip="A[A, E] = 0" data-tip-sub="A আর E-এর মধ্যে এজ নেই"/>
<rect class="cell" x="540" y="70" width="20" height="20" data-tip="A[A, F] = 0" data-tip-sub="A আর F-এর মধ্যে এজ নেই"/>
<rect class="cell" x="560" y="70" width="20" height="20" data-tip="A[A, G] = 0" data-tip-sub="A আর G-এর মধ্যে এজ নেই"/>
<rect class="cell" x="440" y="90" width="20" height="20" data-tip="A[B, A] = 1" data-tip-sub="B আর A-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="443" y="93" width="14" height="14" rx="3" style="--d:0.70s;pointer-events:none"/>
<rect class="cell" x="460" y="90" width="20" height="20" data-tip="A[B, B] = 0" data-tip-sub="সেলফ-লুপ নেই (GCN পরে যোগ করে)"/>
<rect class="cell" x="480" y="90" width="20" height="20" data-tip="A[B, C] = 1" data-tip-sub="B আর C-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="483" y="93" width="14" height="14" rx="3" style="--d:1.21s;pointer-events:none"/>
<rect class="cell" x="500" y="90" width="20" height="20" data-tip="A[B, D] = 0" data-tip-sub="B আর D-এর মধ্যে এজ নেই"/>
<rect class="cell" x="520" y="90" width="20" height="20" data-tip="A[B, E] = 0" data-tip-sub="B আর E-এর মধ্যে এজ নেই"/>
<rect class="cell" x="540" y="90" width="20" height="20" data-tip="A[B, F] = 0" data-tip-sub="B আর F-এর মধ্যে এজ নেই"/>
<rect class="cell" x="560" y="90" width="20" height="20" data-tip="A[B, G] = 0" data-tip-sub="B আর G-এর মধ্যে এজ নেই"/>
<rect class="cell" x="440" y="110" width="20" height="20" data-tip="A[C, A] = 1" data-tip-sub="C আর A-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="443" y="113" width="14" height="14" rx="3" style="--d:0.87s;pointer-events:none"/>
<rect class="cell" x="460" y="110" width="20" height="20" data-tip="A[C, B] = 1" data-tip-sub="C আর B-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="463" y="113" width="14" height="14" rx="3" style="--d:1.21s;pointer-events:none"/>
<rect class="cell" x="480" y="110" width="20" height="20" data-tip="A[C, C] = 0" data-tip-sub="সেলফ-লুপ নেই (GCN পরে যোগ করে)"/>
<rect class="cell" x="500" y="110" width="20" height="20" data-tip="A[C, D] = 0" data-tip-sub="C আর D-এর মধ্যে এজ নেই"/>
<rect class="cell" x="520" y="110" width="20" height="20" data-tip="A[C, E] = 1" data-tip-sub="C আর E-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="523" y="113" width="14" height="14" rx="3" style="--d:1.38s;pointer-events:none"/>
<rect class="cell" x="540" y="110" width="20" height="20" data-tip="A[C, F] = 0" data-tip-sub="C আর F-এর মধ্যে এজ নেই"/>
<rect class="cell" x="560" y="110" width="20" height="20" data-tip="A[C, G] = 0" data-tip-sub="C আর G-এর মধ্যে এজ নেই"/>
<rect class="cell" x="440" y="130" width="20" height="20" data-tip="A[D, A] = 1" data-tip-sub="D আর A-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="443" y="133" width="14" height="14" rx="3" style="--d:1.04s;pointer-events:none"/>
<rect class="cell" x="460" y="130" width="20" height="20" data-tip="A[D, B] = 0" data-tip-sub="D আর B-এর মধ্যে এজ নেই"/>
<rect class="cell" x="480" y="130" width="20" height="20" data-tip="A[D, C] = 0" data-tip-sub="D আর C-এর মধ্যে এজ নেই"/>
<rect class="cell" x="500" y="130" width="20" height="20" data-tip="A[D, D] = 0" data-tip-sub="সেলফ-লুপ নেই (GCN পরে যোগ করে)"/>
<rect class="cell" x="520" y="130" width="20" height="20" data-tip="A[D, E] = 0" data-tip-sub="D আর E-এর মধ্যে এজ নেই"/>
<rect class="cell" x="540" y="130" width="20" height="20" data-tip="A[D, F] = 1" data-tip-sub="D আর F-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="543" y="133" width="14" height="14" rx="3" style="--d:1.55s;pointer-events:none"/>
<rect class="cell" x="560" y="130" width="20" height="20" data-tip="A[D, G] = 0" data-tip-sub="D আর G-এর মধ্যে এজ নেই"/>
<rect class="cell" x="440" y="150" width="20" height="20" data-tip="A[E, A] = 0" data-tip-sub="E আর A-এর মধ্যে এজ নেই"/>
<rect class="cell" x="460" y="150" width="20" height="20" data-tip="A[E, B] = 0" data-tip-sub="E আর B-এর মধ্যে এজ নেই"/>
<rect class="cell" x="480" y="150" width="20" height="20" data-tip="A[E, C] = 1" data-tip-sub="E আর C-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="483" y="153" width="14" height="14" rx="3" style="--d:1.38s;pointer-events:none"/>
<rect class="cell" x="500" y="150" width="20" height="20" data-tip="A[E, D] = 0" data-tip-sub="E আর D-এর মধ্যে এজ নেই"/>
<rect class="cell" x="520" y="150" width="20" height="20" data-tip="A[E, E] = 0" data-tip-sub="সেলফ-লুপ নেই (GCN পরে যোগ করে)"/>
<rect class="cell" x="540" y="150" width="20" height="20" data-tip="A[E, F] = 1" data-tip-sub="E আর F-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="543" y="153" width="14" height="14" rx="3" style="--d:1.72s;pointer-events:none"/>
<rect class="cell" x="560" y="150" width="20" height="20" data-tip="A[E, G] = 0" data-tip-sub="E আর G-এর মধ্যে এজ নেই"/>
<rect class="cell" x="440" y="170" width="20" height="20" data-tip="A[F, A] = 0" data-tip-sub="F আর A-এর মধ্যে এজ নেই"/>
<rect class="cell" x="460" y="170" width="20" height="20" data-tip="A[F, B] = 0" data-tip-sub="F আর B-এর মধ্যে এজ নেই"/>
<rect class="cell" x="480" y="170" width="20" height="20" data-tip="A[F, C] = 0" data-tip-sub="F আর C-এর মধ্যে এজ নেই"/>
<rect class="cell" x="500" y="170" width="20" height="20" data-tip="A[F, D] = 1" data-tip-sub="F আর D-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="503" y="173" width="14" height="14" rx="3" style="--d:1.55s;pointer-events:none"/>
<rect class="cell" x="520" y="170" width="20" height="20" data-tip="A[F, E] = 1" data-tip-sub="F আর E-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="523" y="173" width="14" height="14" rx="3" style="--d:1.72s;pointer-events:none"/>
<rect class="cell" x="540" y="170" width="20" height="20" data-tip="A[F, F] = 0" data-tip-sub="সেলফ-লুপ নেই (GCN পরে যোগ করে)"/>
<rect class="cell" x="560" y="170" width="20" height="20" data-tip="A[F, G] = 1" data-tip-sub="F আর G-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="563" y="173" width="14" height="14" rx="3" style="--d:1.89s;pointer-events:none"/>
<rect class="cell" x="440" y="190" width="20" height="20" data-tip="A[G, A] = 0" data-tip-sub="G আর A-এর মধ্যে এজ নেই"/>
<rect class="cell" x="460" y="190" width="20" height="20" data-tip="A[G, B] = 0" data-tip-sub="G আর B-এর মধ্যে এজ নেই"/>
<rect class="cell" x="480" y="190" width="20" height="20" data-tip="A[G, C] = 0" data-tip-sub="G আর C-এর মধ্যে এজ নেই"/>
<rect class="cell" x="500" y="190" width="20" height="20" data-tip="A[G, D] = 0" data-tip-sub="G আর D-এর মধ্যে এজ নেই"/>
<rect class="cell" x="520" y="190" width="20" height="20" data-tip="A[G, E] = 0" data-tip-sub="G আর E-এর মধ্যে এজ নেই"/>
<rect class="cell" x="540" y="190" width="20" height="20" data-tip="A[G, F] = 1" data-tip-sub="G আর F-এর মধ্যে এজ আছে"/>
<rect class="cell-on a-pop" x="543" y="193" width="14" height="14" rx="3" style="--d:1.89s;pointer-events:none"/>
<rect class="cell" x="560" y="190" width="20" height="20" data-tip="A[G, G] = 0" data-tip-sub="সেলফ-লুপ নেই (GCN পরে যোগ করে)"/>
<rect class="cell-hl a-in" x="438" y="68" width="144" height="24" rx="4" style="--d:2.3s"/>
<text class="tm ta" x="622" y="26">ফিচার X</text>
<text class="tm" x="622" y="42">প্রতি নোডে এক সারি</text>
<rect class="cell" x="622" y="70" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="73" width="14" height="14" rx="3" style="--d:1.95s;opacity:0.92" data-tip="xA = [0.9, 0.2, 0.4]" data-tip-sub="A-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="642" y="70" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="73" width="14" height="14" rx="3" style="--d:1.95s;opacity:0.32" data-tip="xA = [0.9, 0.2, 0.4]" data-tip-sub="A-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="662" y="70" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="73" width="14" height="14" rx="3" style="--d:1.95s;opacity:0.49" data-tip="xA = [0.9, 0.2, 0.4]" data-tip-sub="A-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<text class="tm" x="614" y="84" text-anchor="end">A</text>
<rect class="cell" x="622" y="90" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="93" width="14" height="14" rx="3" style="--d:2.00s;opacity:0.83" data-tip="xB = [0.8, 0.1, 0.5]" data-tip-sub="B-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="642" y="90" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="93" width="14" height="14" rx="3" style="--d:2.00s;opacity:0.23" data-tip="xB = [0.8, 0.1, 0.5]" data-tip-sub="B-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="662" y="90" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="93" width="14" height="14" rx="3" style="--d:2.00s;opacity:0.57" data-tip="xB = [0.8, 0.1, 0.5]" data-tip-sub="B-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<text class="tm" x="614" y="104" text-anchor="end">B</text>
<rect class="cell" x="622" y="110" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="113" width="14" height="14" rx="3" style="--d:2.05s;opacity:0.74" data-tip="xC = [0.7, 0.3, 0.6]" data-tip-sub="C-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="642" y="110" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="113" width="14" height="14" rx="3" style="--d:2.05s;opacity:0.41" data-tip="xC = [0.7, 0.3, 0.6]" data-tip-sub="C-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="662" y="110" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="113" width="14" height="14" rx="3" style="--d:2.05s;opacity:0.66" data-tip="xC = [0.7, 0.3, 0.6]" data-tip-sub="C-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<text class="tm" x="614" y="124" text-anchor="end">C</text>
<rect class="cell" x="622" y="130" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="133" width="14" height="14" rx="3" style="--d:2.10s;opacity:0.32" data-tip="xD = [0.2, 0.8, 0.3]" data-tip-sub="D-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="642" y="130" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="133" width="14" height="14" rx="3" style="--d:2.10s;opacity:0.83" data-tip="xD = [0.2, 0.8, 0.3]" data-tip-sub="D-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="662" y="130" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="133" width="14" height="14" rx="3" style="--d:2.10s;opacity:0.41" data-tip="xD = [0.2, 0.8, 0.3]" data-tip-sub="D-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<text class="tm" x="614" y="144" text-anchor="end">D</text>
<rect class="cell" x="622" y="150" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="153" width="14" height="14" rx="3" style="--d:2.15s;opacity:0.41" data-tip="xE = [0.3, 0.9, 0.1]" data-tip-sub="E-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="642" y="150" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="153" width="14" height="14" rx="3" style="--d:2.15s;opacity:0.92" data-tip="xE = [0.3, 0.9, 0.1]" data-tip-sub="E-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="662" y="150" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="153" width="14" height="14" rx="3" style="--d:2.15s;opacity:0.23" data-tip="xE = [0.3, 0.9, 0.1]" data-tip-sub="E-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<text class="tm" x="614" y="164" text-anchor="end">E</text>
<rect class="cell" x="622" y="170" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="173" width="14" height="14" rx="3" style="--d:2.20s;opacity:0.23" data-tip="xF = [0.1, 0.7, 0.2]" data-tip-sub="F-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="642" y="170" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="173" width="14" height="14" rx="3" style="--d:2.20s;opacity:0.74" data-tip="xF = [0.1, 0.7, 0.2]" data-tip-sub="F-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="662" y="170" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="173" width="14" height="14" rx="3" style="--d:2.20s;opacity:0.32" data-tip="xF = [0.1, 0.7, 0.2]" data-tip-sub="F-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<text class="tm" x="614" y="184" text-anchor="end">F</text>
<rect class="cell" x="622" y="190" width="20" height="20"/>
<rect class="cell-on a-in" x="625" y="193" width="14" height="14" rx="3" style="--d:2.25s;opacity:0.32" data-tip="xG = [0.2, 0.6, 0.9]" data-tip-sub="G-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="642" y="190" width="20" height="20"/>
<rect class="cell-on a-in" x="645" y="193" width="14" height="14" rx="3" style="--d:2.25s;opacity:0.66" data-tip="xG = [0.2, 0.6, 0.9]" data-tip-sub="G-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<rect class="cell" x="662" y="190" width="20" height="20"/>
<rect class="cell-on a-in" x="665" y="193" width="14" height="14" rx="3" style="--d:2.25s;opacity:0.92" data-tip="xG = [0.2, 0.6, 0.9]" data-tip-sub="G-কে বর্ণনা করে এমন তিনটি সংখ্যা"/>
<text class="tm" x="614" y="204" text-anchor="end">G</text>
<g class="a-in" style="--d:2.5s"><text class="t2" x="440" y="246">সারি A-তে B, C আর D-এর ঘরে 1: এরাই A-এর প্রতিবেশী।</text><text class="t2" x="440" y="266">এজ দুই দিকেই যায়, তাই ম্যাট্রিক্সটি প্রতিসম।</text></g>
</svg>
</div>
<figcaption>একটি গ্রাফ সম্পর্কে GNN যা জানে তার সবটুকু: কে কার সঙ্গে যুক্ত (অ্যাডজেসেন্সি ম্যাট্রিক্স) আর প্রতিটি নোড দেখতে কেমন (ফিচার ম্যাট্রিক্স)। মান দেখতে যেকোনো ঘরে কার্সর রাখুন। সাত নোডের এই গ্রাফটিই পুরো পোস্টজুড়ে আমাদের উদাহরণ।</figcaption>
</figure>
{:/nomarkdown}

এজের নিজস্ব ফিচারও থাকতে পারে (বন্ধনের ধরন, লেনদেনের পরিমাণ, সময়), আর এজ হতে পারে ডিরেক্টেড (সাইটেশন এক দিকেই যায়) বা আনডিরেক্টেড (বন্ধুত্ব দুই দিকেই)। নিচের সব ধারণাই এদের সবার জন্য খাটে। উদাহরণগুলোতে নোড ফিচারসহ আনডিরেক্টেড গ্রাফ রাখা হয়েছে, কারণ এগুলো আঁকা সবচেয়ে সহজ।

বাস্তবে কেউই $$A$$-কে ডেন্স ম্যাট্রিক্স হিসেবে রাখে না। আসল গ্রাফগুলো **স্পার্স**: শত কোটি ব্যবহারকারীর একটি সোশ্যাল নেটওয়ার্কে প্রত্যেকের বন্ধু কয়েকশো, শত কোটি নয়। তাই লাইব্রেরিগুলো রাখে একটি **এজ লিস্ট**, অর্থাৎ (সোর্স, টার্গেট) জোড়ার একটি $$2 \times E$$ অ্যারে, ফলে মেমরি বাড়ে এজের সংখ্যার সঙ্গে, $$n^2$$-এর সঙ্গে নয়। ম্যাথের জন্য ম্যাট্রিক্সের ছবিটা মাথায় রাখুন, আর কোডের জন্য এজ লিস্ট।

একটি গ্রাফ হাতে পেলে তাকে তিন ধরনের প্রশ্ন করা যায়।

{::nomarkdown}
<figure class="viz" id="fig-tasks">
<div class="viz-head"><span class="viz-label">চিত্র ২ // তিন ধরনের প্রশ্ন</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 246" role="img" aria-labelledby="fig-tasks-t fig-tasks-d">
<title id="fig-tasks-t">একটি গ্রাফকে যে তিন ধরনের প্রশ্ন করা যায়</title>
<desc id="fig-tasks-d">একই ছোট গ্রাফ তিনবার। নোড ক্লাসিফিকেশন হাইলাইট করা একটি নোডের লেবেল অনুমান করে। লিংক প্রেডিকশন জানতে চায় দুটি নোডের মধ্যে ড্যাশ দেওয়া সম্ভাব্য এজটি থাকা উচিত কি না। গ্রাফ ক্লাসিফিকেশন দাগ দিয়ে ঘেরা পুরো গ্রাফের জন্য একটি লেবেল অনুমান করে।</desc>
<g class="a-in" style="--d:0.05s"><rect class="n" x="12" y="12" width="228" height="222" rx="14"/><text class="tm ta" x="26" y="32">প্রতি নোডে</text><line class="ge" x1="64" y1="74" x2="120" y2="48"/><line class="ge" x1="120" y1="48" x2="182" y2="72"/><line class="ge" x1="182" y1="72" x2="190" y2="132"/><line class="ge" x1="190" y1="132" x2="126" y2="152"/><line class="ge" x1="126" y1="152" x2="66" y2="130"/><line class="ge" x1="66" y1="130" x2="64" y2="74"/><line class="ge" x1="120" y1="48" x2="126" y2="152"/><circle class="gn" cx="64" cy="74" r="10"/><circle class="gn" cx="120" cy="48" r="10"/><circle class="gn-acc" cx="182" cy="72" r="10"/><circle class="gn" cx="190" cy="132" r="10"/><circle class="gn" cx="126" cy="152" r="10"/><circle class="gn" cx="66" cy="130" r="10"/><text class="t big" x="28" y="196">নোড ক্লাসিফিকেশন</text><text class="t2" x="28" y="216">এই পেপারটি কোন বিষয়ের?</text></g>
<g class="a-in" style="--d:0.2s"><rect class="n" x="266" y="12" width="228" height="222" rx="14"/><text class="tm ta" x="280" y="32">প্রতি জোড়ায়</text><line class="ge" x1="318" y1="74" x2="374" y2="48"/><line class="ge" x1="374" y1="48" x2="436" y2="72"/><line class="ge" x1="436" y1="72" x2="444" y2="132"/><line class="ge" x1="444" y1="132" x2="380" y2="152"/><line class="ge" x1="380" y1="152" x2="320" y2="130"/><line class="ge" x1="320" y1="130" x2="318" y2="74"/><line class="ge" x1="374" y1="48" x2="380" y2="152"/><circle class="gn" cx="318" cy="74" r="10"/><circle class="gn" cx="374" cy="48" r="10"/><circle class="gn-acc" cx="436" cy="72" r="10"/><circle class="gn" cx="444" cy="132" r="10"/><circle class="gn-acc" cx="380" cy="152" r="10"/><circle class="gn" cx="320" cy="130" r="10"/><text class="t big" x="282" y="196">লিংক প্রেডিকশন</text><text class="t2" x="282" y="216">এই দুজন কি যুক্ত হবেন?</text></g>
<g class="a-in" style="--d:0.35s"><rect class="n" x="520" y="12" width="228" height="222" rx="14"/><text class="tm ta" x="534" y="32">প্রতি গ্রাফে</text><line class="ge" x1="572" y1="74" x2="628" y2="48"/><line class="ge" x1="628" y1="48" x2="690" y2="72"/><line class="ge" x1="690" y1="72" x2="698" y2="132"/><line class="ge" x1="698" y1="132" x2="634" y2="152"/><line class="ge" x1="634" y1="152" x2="574" y2="130"/><line class="ge" x1="574" y1="130" x2="572" y2="74"/><line class="ge" x1="628" y1="48" x2="634" y2="152"/><circle class="gn" cx="572" cy="74" r="10"/><circle class="gn" cx="628" cy="48" r="10"/><circle class="gn" cx="690" cy="72" r="10"/><circle class="gn" cx="698" cy="132" r="10"/><circle class="gn" cx="634" cy="152" r="10"/><circle class="gn" cx="574" cy="130" r="10"/><text class="t big" x="536" y="196">গ্রাফ ক্লাসিফিকেশন</text><text class="t2" x="536" y="216">এই অণুটি কি বিষাক্ত?</text></g>
<circle class="ring a-ping" cx="182" cy="72" r="10" style="--d:1.2s"/>
<g class="a-pop" style="--d:0.8s"><circle class="gn-acc" cx="182" cy="72" r="10"/><text class="gl" x="182" y="72" dy="0.35em">?</text></g>
<path class="drop a-draw" pathLength="1" d="M436 72L380 152" style="--d:1.0s;--t:.5s"/>
<g class="a-pop" style="--d:1.5s"><circle class="gn-acc" cx="408" cy="112" r="10"/><text class="gl" x="408" y="112" dy="0.35em">?</text></g>
<rect class="q-ghost a-in" x="548" y="50" width="174" height="120" rx="16" style="--d:1.6s;stroke-dasharray:4 4"/>
<g class="a-pop" style="--d:1.9s"><circle class="gn-acc" cx="722" cy="50" r="10"/><text class="gl" x="722" y="50" dy="0.35em">?</text></g>
</svg>
</div>
<figcaption>একই গ্রাফ, তিন ধরনের প্রশ্ন। তিনটির জন্যই GNN লেয়ারগুলো হুবহু এক; বদলায় শুধু শেষ ধাপটি, যা নোডের ভেক্টরকে উত্তরে পরিণত করে।</figcaption>
</figure>
{:/nomarkdown}

- **নোড ক্লাসিফিকেশন** প্রতিটি নোড সম্পর্কে কিছু অনুমান করে: একটি পেপার কোন বিষয়ের, একটি অ্যাকাউন্ট বট কি না, একটি প্রোটিন কী কাজ করে।
- **লিংক প্রেডিকশন** অনুমান করে দুটি নোডের মধ্যে এজ থাকা উচিত কি না: বন্ধুর সাজেশন, পণ্যের রেকমেন্ডেশন, কোন ওষুধ কোন প্রোটিনের সঙ্গে জোড়া লাগে।
- **গ্রাফ ক্লাসিফিকেশন** (বা রিগ্রেশন) পুরো একটি গ্রাফ সম্পর্কে কিছু অনুমান করে: একটি অণু বিষাক্ত কি না, সেটি কতটা দ্রবণীয়, একগুচ্ছ লেনদেন কোনো জালিয়াত চক্রের মতো দেখায় কি না।

তিনটির জন্যই একই GNN লেয়ার কাজ করে। বদলায় শুধু শেষ ধাপটি, অর্থাৎ যে "হেড" নোডের ভেক্টরকে উত্তরে পরিণত করে। কীভাবে, তা দেখানো হয়েছে ৯ নম্বর অংশে।

## ২. সাধারণ নিউরাল নেটওয়ার্ক কেন গ্রাফ নিয়ে হিমশিম খায়
{: id="2-why-ordinary-neural-networks-struggle-with-graphs"}

প্রথম যে চেষ্টাটা মাথায় আসে, তা হলো অ্যাডজেসেন্সি ম্যাট্রিক্স আর ফিচারগুলোকে চ্যাপ্টা করে একটি লম্বা ভেক্টর বানিয়ে একটি মাল্টিলেয়ার পারসেপট্রনে (MLP) দেওয়া। এটি দুটি কারণে ব্যর্থ হয়।

প্রথম কারণ আকার। MLP-র ইনপুটের সংখ্যা নির্দিষ্ট, তাই এটি কেবল ঠিক ততগুলো নোডের গ্রাফই পড়তে পারে, যতগুলোর জন্য এটি বানানো। অথচ একটি অণুতে পরমাণু থাকতে পারে ৯টি, কিংবা ৯০টি।

দ্বিতীয় কারণটি সূক্ষ্ম, আর বেশি গুরুত্বপূর্ণ। **নোডের নম্বরগুলো ইচ্ছেমতো দেওয়া।** গ্রাফের কোথাও লেখা থাকে না কোনটি ১ নম্বর নোড। নোডগুলোর নম্বর বদলে দিলে গ্রাফ একই থাকে, কিন্তু ম্যাট্রিক্স বদলে যায়:

{::nomarkdown}
<figure class="viz" id="fig-perm">
<div class="viz-head"><span class="viz-label">চিত্র ৩ // নম্বরিংয়ের সমস্যা</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 240" role="img" aria-labelledby="fig-perm-t fig-perm-d">
<title id="fig-perm-t">নোডের নম্বর বদলালে ম্যাট্রিক্স বদলায়, গ্রাফ বদলায় না</title>
<desc id="fig-perm-d">চার নোডের একই গ্রাফ, লেজওয়ালা একটি ত্রিভুজ, দুইভাবে নম্বর দেওয়া। গ্রাফ হুবহু এক হলেও দুটি অ্যাডজেসেন্সি ম্যাট্রিক্সের অনেক ঘর আলাদা, আর সাধারণ নেটওয়ার্ক যে চ্যাপ্টা ভেক্টর পড়বে সেগুলোও আলাদা।</desc>
<g class="a-in" style="--d:0.05s"><text class="tm ta" x="24" y="26">নম্বরিং ১</text><line class="ge" x1="84" y1="80" x2="174" y2="64"/><line class="ge" x1="174" y1="64" x2="154" y2="150"/><line class="ge" x1="84" y1="80" x2="154" y2="150"/><line class="ge" x1="154" y1="150" x2="70" y2="176"/><circle class="gn" cx="84" cy="80" r="15"/><circle class="gn" cx="174" cy="64" r="15"/><circle class="gn" cx="154" cy="150" r="15"/><circle class="gn" cx="70" cy="176" r="15"/></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="84" y="80" dy="0.35em">1</text></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="174" y="64" dy="0.35em">2</text></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="154" y="150" dy="0.35em">3</text></g>
<g class="a-in" style="--d:0.1s"><text class="gl" x="70" y="176" dy="0.35em">4</text></g>
<g class="a-in" style="--d:0.25s"><text class="tm" x="239" y="53" text-anchor="middle">1</text><text class="tm" x="220" y="75" text-anchor="end">1</text><text class="tm" x="261" y="53" text-anchor="middle">2</text><text class="tm" x="220" y="97" text-anchor="end">2</text><text class="tm" x="283" y="53" text-anchor="middle">3</text><text class="tm" x="220" y="119" text-anchor="end">3</text><text class="tm" x="305" y="53" text-anchor="middle">4</text><text class="tm" x="220" y="141" text-anchor="end">4</text><rect class="cell" x="228" y="60" width="22" height="22"/><rect class="cell" x="250" y="60" width="22" height="22"/><rect class="cell-on" x="253" y="63" width="16" height="16" rx="3"/><rect class="cell" x="272" y="60" width="22" height="22"/><rect class="cell-on" x="275" y="63" width="16" height="16" rx="3"/><rect class="cell" x="294" y="60" width="22" height="22"/><rect class="cell" x="228" y="82" width="22" height="22"/><rect class="cell-on" x="231" y="85" width="16" height="16" rx="3"/><rect class="cell" x="250" y="82" width="22" height="22"/><rect class="cell" x="272" y="82" width="22" height="22"/><rect class="cell-on" x="275" y="85" width="16" height="16" rx="3"/><rect class="cell" x="294" y="82" width="22" height="22"/><rect class="cell" x="228" y="104" width="22" height="22"/><rect class="cell-on" x="231" y="107" width="16" height="16" rx="3"/><rect class="cell" x="250" y="104" width="22" height="22"/><rect class="cell-on" x="253" y="107" width="16" height="16" rx="3"/><rect class="cell" x="272" y="104" width="22" height="22"/><rect class="cell" x="294" y="104" width="22" height="22"/><rect class="cell-on" x="297" y="107" width="16" height="16" rx="3"/><rect class="cell" x="228" y="126" width="22" height="22"/><rect class="cell" x="250" y="126" width="22" height="22"/><rect class="cell" x="272" y="126" width="22" height="22"/><rect class="cell-on" x="275" y="129" width="16" height="16" rx="3"/><rect class="cell" x="294" y="126" width="22" height="22"/><text class="tm" x="208" y="174">একটি ভেক্টর হিসেবে</text><text class="tk" x="208" y="192">0110 1010 1101 0010</text></g>
<g class="a-in" style="--d:1.0s"><text class="tm ta" x="414" y="26">নম্বরিং ২, একই গ্রাফ</text><line class="ge" x1="474" y1="80" x2="564" y2="64"/><line class="ge" x1="564" y1="64" x2="544" y2="150"/><line class="ge" x1="474" y1="80" x2="544" y2="150"/><line class="ge" x1="544" y1="150" x2="460" y2="176"/><circle class="gn" cx="474" cy="80" r="15"/><circle class="gn" cx="564" cy="64" r="15"/><circle class="gn" cx="544" cy="150" r="15"/><circle class="gn" cx="460" cy="176" r="15"/></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="474" y="80" dy="0.35em">3</text></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="564" y="64" dy="0.35em">1</text></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="544" y="150" dy="0.35em">4</text></g>
<g class="a-pop" style="--d:1.3s"><text class="gl" x="460" y="176" dy="0.35em">2</text></g>
<g class="a-in" style="--d:1.2s"><text class="tm" x="629" y="53" text-anchor="middle">1</text><text class="tm" x="610" y="75" text-anchor="end">1</text><text class="tm" x="651" y="53" text-anchor="middle">2</text><text class="tm" x="610" y="97" text-anchor="end">2</text><text class="tm" x="673" y="53" text-anchor="middle">3</text><text class="tm" x="610" y="119" text-anchor="end">3</text><text class="tm" x="695" y="53" text-anchor="middle">4</text><text class="tm" x="610" y="141" text-anchor="end">4</text><rect class="cell" x="618" y="60" width="22" height="22"/><rect class="cell" x="640" y="60" width="22" height="22"/><rect class="cell" x="662" y="60" width="22" height="22"/><rect class="cell-on" x="665" y="63" width="16" height="16" rx="3"/><rect class="cell" x="684" y="60" width="22" height="22"/><rect class="cell-on" x="687" y="63" width="16" height="16" rx="3"/><rect class="cell" x="618" y="82" width="22" height="22"/><rect class="cell" x="640" y="82" width="22" height="22"/><rect class="cell" x="662" y="82" width="22" height="22"/><rect class="cell" x="684" y="82" width="22" height="22"/><rect class="cell-on" x="687" y="85" width="16" height="16" rx="3"/><rect class="cell" x="618" y="104" width="22" height="22"/><rect class="cell-on" x="621" y="107" width="16" height="16" rx="3"/><rect class="cell" x="640" y="104" width="22" height="22"/><rect class="cell" x="662" y="104" width="22" height="22"/><rect class="cell" x="684" y="104" width="22" height="22"/><rect class="cell-on" x="687" y="107" width="16" height="16" rx="3"/><rect class="cell" x="618" y="126" width="22" height="22"/><rect class="cell-on" x="621" y="129" width="16" height="16" rx="3"/><rect class="cell" x="640" y="126" width="22" height="22"/><rect class="cell-on" x="643" y="129" width="16" height="16" rx="3"/><rect class="cell" x="662" y="126" width="22" height="22"/><rect class="cell-on" x="665" y="129" width="16" height="16" rx="3"/><rect class="cell" x="684" y="126" width="22" height="22"/><text class="tm" x="598" y="174">একটি ভেক্টর হিসেবে</text><text class="tk" x="598" y="192">0011 0001 1001 1110</text></g>
<rect class="a-in" x="641" y="61" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="685" y="61" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="619" y="83" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="663" y="83" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="685" y="83" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="641" y="105" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="619" y="127" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<rect class="a-in" x="641" y="127" width="20" height="20" rx="3" style="--d:1.9s;fill:none;stroke:#e0736d;stroke-width:1.5"/>
<line class="ln-faint" x1="380" y1="40" x2="380" y2="200"/>
<text class="t2 strong a-in" x="380" y="226" text-anchor="middle" style="--d:2.3s">একই গ্রাফ, ভিন্ন ইনপুট। গ্রাফ মডেলকে দুটির জন্যই একই উত্তর দিতে হবে।</text>
</svg>
</div>
<figcaption>নোডের নম্বর শুধু সুবিধার জন্য দেওয়া নাম। নম্বর বদলালে অ্যাডজেসেন্সি ম্যাট্রিক্সের ১৬টি ঘরের ৮টি বদলে যায়, ফলে একে চ্যাপ্টা ভেক্টর হিসেবে পড়া সাধারণ নেটওয়ার্ক ভিন্ন ইনপুট দেখে। লাল দাগ দেওয়া ঘরগুলোই বদলেছে।</figcaption>
</figure>
{:/nomarkdown}

MLP দুটি সম্পর্কহীন ইনপুট দেখবে, আর দুটি ভিন্ন উত্তরও দিতে পারে। $$n$$টি নোড থাকলে একই গ্রাফে নম্বর দেওয়ার $$n!$$টি উপায় আছে (মাত্র ১০টি নোডেই ৩৬ লাখের বেশি), আর সবগুলোতে ট্রেইন করা সম্ভব নয়।

তাই গ্রাফের মডেল এমনভাবে বানাতে হয় যাতে নম্বর দেওয়ার ধরন কোনো প্রভাবই ফেলতে না পারে। আনুষ্ঠানিকভাবে বললে, যদি $$P$$ হয় নোডের নম্বর বদলানোর একটি পারমুটেশন ম্যাট্রিক্স, তাহলে নোড-স্তরের মডেল $$f$$-কে হতে হবে **পারমুটেশন ইকুইভেরিয়েন্ট**: ইনপুটের নম্বর বদলালে আউটপুটের নম্বরও একইভাবে বদলায়, আর কিছুই বদলায় না।

$$
f\big(PX,\; PAP^\top\big) = P\, f(X, A)
$$

গ্রাফ-স্তরের মডেল $$g$$-কে হতে হবে **পারমুটেশন ইনভেরিয়েন্ট**: $$g(PX, PAP^\top) = g(X, A)$$।

শুনতে বিমূর্ত লাগলেও সমাধানটা আপনার জানা। যোগফল, গড় আর সর্বোচ্চ মান ক্রমের তোয়াক্কা করে না: $$3 + 5 + 1$$ আর $$1 + 3 + 5$$ একই। এ ধরনের অপারেশন দিয়ে মডেল বানালে নম্বর দেওয়ার ধরন আপনা থেকেই অপ্রাসঙ্গিক হয়ে যায়। মেসেজ পাসিং ঠিক এটাই করে।

## ৩. মেসেজ পাসিং: প্রতিটি GNN-এর পেছনের একটিমাত্র ধারণা
{: id="3-message-passing-the-one-idea-behind-every-gnn"}

ধরুন এমন একজন মানুষ সম্পর্কে কিছু আন্দাজ করতে হবে, যার সম্পর্কে আপনি কিছুই জানেন না, শুধু জানেন তার বন্ধু কারা। আপনি বন্ধুদের দিকেই তাকাবেন। যুক্ত জিনিসগুলো সাধারণত একে অপরের মতো হয়, এই বৈশিষ্ট্যের নাম **হোমোফিলি**: পেপার সাইট করে কাছাকাছি বিষয়ের পেপারকে, জালিয়াত অ্যাকাউন্ট ডিভাইস ভাগ করে অন্য জালিয়াত অ্যাকাউন্টের সঙ্গে, আর অক্সিজেনের পাশের একটি পরমাণু কার্বনের পাশের পরমাণুর চেয়ে ভিন্ন আচরণ করে।

একটি GNN লেয়ার এই অনুমানকে একটি হিসাবে পরিণত করে। প্রতিটি নোড একই সময়ে তিনটি ধাপ চালায়:

1. **পাঠানো।** প্রতিটি নোড তার বর্তমান ভেক্টর প্রতিবেশীদের কাছে পাঠায়। (সাধারণভাবে মেসেজ হতে পারে প্রেরক, প্রাপক আর তাদের মধ্যকার এজের একটি শেখা ফাংশন।)
2. **একত্র করা।** প্রতিটি নোড প্রাপ্ত মেসেজগুলোকে এমন একটি ফাংশন দিয়ে মেলায় যা তাদের ক্রমের তোয়াক্কা করে না: যোগফল, গড় বা সর্বোচ্চ।
3. **আপডেট।** প্রতিটি নোড শেখা ওয়েট আর একটি ননলিনিয়ারিটির মাধ্যমে এই একত্র করা মানকে নিজের বর্তমান ভেক্টরের সঙ্গে মিশিয়ে নতুন ভেক্টর পায়।
{: .steps}

উদাহরণের গ্রাফে ধাপগুলো দেখে নিন। এখানে প্রতিটি নোডে একটিমাত্র সংখ্যা, তাই হিসাবটা মিলিয়ে দেখা সহজ।

{::nomarkdown}
<figure class="viz viz-steps" id="fig-mp" data-dur="4200">
<div class="viz-head"><span class="viz-label">চিত্র ৪ // মেসেজ পাসিং, এক ধাপ এক ধাপ করে</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 290" role="img" aria-labelledby="fig-mp-t fig-mp-d">
<title id="fig-mp-t">সাত নোডের একটি গ্রাফে এক রাউন্ড মেসেজ পাসিং</title>
<desc id="fig-mp-d">A নোডের প্রতিবেশী B, C আর D, যাদের মান 0.8, 0.2 আর 0.5। তারা A-কে নিজেদের মান পাঠায়, A গড় নেয়, 0.5, আর নিজেকে আপডেট করে ReLU(0.5 গুণ 0.3 যোগ 1.0 গুণ 0.5) = 0.65-এ। তারপর সব নোড একসঙ্গে একই কাজ করে: A 0.30 থেকে 0.65, B 0.80 থেকে 0.65, C 0.20 থেকে 0.77, D 0.50 থেকে 0.45, E 0.90 থেকে 0.60, F 0.10 থেকে 0.72, G 0.60 থেকে 0.40।</desc>
<text class="tm ta" x="24" y="26">প্রতিটি নোডে একটি করে সংখ্যা</text>
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
<text class="tm ta" x="440" y="68" data-text="[&quot;ধাপ ১ // পাঠানো&quot;, &quot;ধাপ ২ // একত্র করা&quot;, &quot;ধাপ ৩ // আপডেট&quot;, &quot;সব নোড, একই নিয়ম, একসঙ্গে&quot;]">সব নোড, একই নিয়ম, একসঙ্গে</text>
<text class="t2 is-off" x="440" y="102" data-on="0-2">B-এর মেসেজ</text>
<text class="t is-off" x="716" y="102" text-anchor="end" data-on="0-2">0.80</text>
<text class="t2 is-off" x="440" y="126" data-on="0-2">C-এর মেসেজ</text>
<text class="t is-off" x="716" y="126" text-anchor="end" data-on="0-2">0.20</text>
<text class="t2 is-off" x="440" y="150" data-on="0-2">D-এর মেসেজ</text>
<text class="t is-off" x="716" y="150" text-anchor="end" data-on="0-2">0.50</text>
<line class="ln-faint is-off" x1="440" y1="164" x2="716" y2="164" data-on="1-2"/>
<text class="t2 is-off" x="440" y="184" data-on="1-2">তিনটির গড়</text>
<text class="t is-off" x="716" y="184" text-anchor="end" data-on="1-2">0.50</text>
<text class="tm is-off" x="440" y="220" data-on="2">ReLU(0.5 × 0.30 + 1.0 × 0.50)</text>
<text class="t2 is-off" x="440" y="248" data-on="2">A-এর নতুন মান</text>
<text class="t big is-off" x="716" y="248" text-anchor="end" style="fill:#e8b84e" data-on="2">0.65</text>
<text class="tm" x="440" y="96" data-on="3">নোড</text>
<text class="tm" x="550" y="96" text-anchor="end" data-on="3">আগে</text>
<text class="tm" x="716" y="96" text-anchor="end" data-on="3">১ লেয়ারের পরে</text>
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
<div class="viz-ctl" role="group" aria-label="মেসেজ পাসিংয়ের ধাপ"><button type="button" data-go="0" data-note="পাঠানো। A-এর প্রতিটি প্রতিবেশী এজ ধরে নিজের বর্তমান মান পাঠায়: B পাঠায় 0.80, C পাঠায় 0.20 আর D পাঠায় 0.50। (অন্য সব নোডও একই কাজ করছে, তবে আপাতত শুধু A-কে দেখুন।)" aria-pressed="false">১ · পাঠানো</button><button type="button" data-go="1" data-note="একত্র করা। যা এসেছে, A তা এমন একটি ফাংশন দিয়ে মেলায় যা ক্রমের তোয়াক্কা করে না। এখানে সেটি গড়: (0.80 + 0.20 + 0.50) / 3 = 0.50। মেসেজ যে ক্রমেই আসুক, উত্তর একই।" aria-pressed="false">২ · একত্র করা</button><button type="button" data-go="2" data-note="আপডেট। শেখা দুটি ওয়েট আর একটি ReLU দিয়ে A একত্র করা মানটিকে নিজের পুরোনো মানের সঙ্গে মেশায়: 0.5 × 0.30 + 1.0 × 0.50 = 0.65। এটাই A-এর নতুন মান।" aria-pressed="false">৩ · আপডেট</button><button type="button" data-go="3" data-note="প্রতিটি নোড একই সময়ে একই তিনটি ধাপ চালায়, একই দুটি ওয়েট দিয়ে। আটটি এজ দিয়ে যায় ষোলোটি মেসেজ, আর প্রতিটি নোড পায় নতুন মান। এটাই একটি GNN লেয়ার।" aria-pressed="true">৪ · সব নোড</button></div>
<p class="viz-note" aria-live="polite">প্রতিটি নোড একই সময়ে একই তিনটি ধাপ চালায়, একই দুটি ওয়েট দিয়ে। আটটি এজ দিয়ে যায় ষোলোটি মেসেজ, আর প্রতিটি নোড পায় নতুন মান। এটাই একটি GNN লেয়ার।</p>
<figcaption>ধীর গতিতে একটি লেয়ার। বাটনগুলো দিয়ে ধাপে ধাপে দেখুন, অথবা চালান চাপুন। আসল GNN-এ প্রতিটি মান একটি ভেক্টর আর প্রতিটি ওয়েট একটি ম্যাট্রিক্স, কিন্তু ধাপ তিনটি হুবহু এগুলোই।</figcaption>
</figure>
{:/nomarkdown}

প্রতিবেশী $$N(v)$$-সহ একটি নোড $$v$$-এর জন্য লিখলে চিত্রের লেয়ারটি দাঁড়ায়:

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( W_{\text{self}}\, \mathbf{h}_v^{(k)} \;+\; W_{\text{neigh}}\, \frac{1}{\lvert N(v) \rvert} \sum_{u \in N(v)} \mathbf{h}_u^{(k)} \Big)
$$

এখানে $$\mathbf{h}_v^{(k)}$$ হলো $$k$$টি লেয়ারের পরে নোড $$v$$-এর ভেক্টর, যার শুরু তার ইনপুট ফিচার থেকে, $$\mathbf{h}_v^{(0)} = \mathbf{x}_v$$। $$W_{\text{self}}$$ আর $$W_{\text{neigh}}$$ হলো গ্রেডিয়েন্ট ডিসেন্টে শেখা ওয়েট ম্যাট্রিক্স, আর $$\sigma$$ হলো ReLU-র মতো একটি ননলিনিয়ারিটি। চিত্রে ভেক্টরগুলো এক মাত্রার, তাই ম্যাট্রিক্স দুটি ছোট হয়ে দাঁড়িয়েছে দুটি সংখ্যায়, 0.5 আর 1.0।

Gilmer et al. (2017)-এর মেসেজ পাসিং কাঠামো থেকে আসা সাধারণ রূপটি প্রতিটি অংশকে যেকোনো শেখা ফাংশন দিয়ে বদলে দেয়:

$$
\mathbf{h}_v^{(k+1)} = \phi\Big( \mathbf{h}_v^{(k)},\; \bigoplus_{u \in N(v)} \psi\big(\mathbf{h}_u^{(k)}, \mathbf{h}_v^{(k)}, \mathbf{e}_{uv}\big) \Big)
$$

এখানে $$\psi$$ মেসেজ তৈরি করে (এবং এজের ফিচার $$\mathbf{e}_{uv}$$ ব্যবহার করতে পারে), $$\bigoplus$$ হলো ক্রম-নিরপেক্ষ অ্যাগ্রিগেটর, আর $$\phi$$ হলো আপডেট। যত GNN-এর কথা আপনি পড়বেন, তার প্রায় সবই এই তিনটি ফাংশনের কোনো একটি নির্দিষ্ট পছন্দ।

<div class="key-idea">
  <span class="key-label">এক বাক্যে</span>
  <p>একটি GNN লেয়ার প্রতিটি নোডকে তার প্রতিবেশীদের ভেক্টর সংগ্রহ করতে দেয়, ক্রম-নিরপেক্ষ একটি ফাংশন দিয়ে সেগুলো মেলায়, আর সব নোডের জন্য একই ওয়েট দিয়ে ফলটাকে নিজের ভেক্টরে মিশিয়ে নেয়।</p>
</div>

শেষ অংশটি, অর্থাৎ **শেয়ার করা ওয়েট**, এটাকেই কার্যকর করে। কনভোলিউশন একই ছোট ফিল্টার ছবির প্রতিটি জায়গায় চালায়; GNN লেয়ার একই $$W_{\text{self}}$$ আর $$W_{\text{neigh}}$$ প্রতিটি নোডে প্রয়োগ করে। প্যারামিটারের সংখ্যা গ্রাফের আকারের ওপর নির্ভর করে না, তাই ছোট অণুতে ট্রেইন করা মডেল বড় অণুতেও চলে, আর গত মাসের লেনদেনের গ্রাফে ট্রেইন করা মডেল চলে এই মাসের গ্রাফে।

## ৪. লেয়ারের ওপর লেয়ার: একটি নোড কত দূর দেখতে পায়
{: id="4-stacking-layers-how-far-a-node-can-see"}

একটি লেয়ার সরাসরি প্রতিবেশীদের তথ্য মেশায়। দ্বিতীয় লেয়ার যোগ করলে, যে প্রতিবেশীদের কাছ থেকে আপনি পড়ছেন তারা আগেই *তাদের* প্রতিবেশীদের তথ্য মিশিয়ে নিয়েছে, ফলে দুই হপ দূরের তথ্যও আপনার কাছে পৌঁছায়। $$k$$টি লেয়ার থাকলে একটি নোডের ভেক্টর নির্ভর করে $$k$$ হপের মধ্যের সবকিছুর ওপর। এই এলাকাটিই তার **রিসেপটিভ ফিল্ড**।

একটি নোডের জন্য GNN কী হিসাব করে, তা সবচেয়ে পরিষ্কার দেখা যায় একে একটি **কম্পিউটেশন ট্রি**-তে খুলে ফেললে: গোড়ায় নোডটি, তার নিচে প্রতিবেশীরা, তাদের নিচে তাদের প্রতিবেশীরা, প্রতিটি লেয়ারের জন্য একটি স্তর।

{::nomarkdown}
<figure class="viz viz-steps" id="fig-receptive" data-dur="3600">
<div class="viz-head"><span class="viz-label">চিত্র ৫ // প্রতিটি লেয়ারে দৃষ্টি আরও প্রশস্ত</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="fig-receptive-t fig-receptive-d">
<title id="fig-receptive-t">প্রতিটি লেয়ারে A নোডের রিসেপটিভ ফিল্ড কীভাবে বাড়ে</title>
<desc id="fig-receptive-d">বাঁয়ে সাত নোডের গ্রাফ, যেখানে A যাদের কথা শুনতে পায় তারা হাইলাইট করা: ০ লেয়ারে শুধু A, ১ লেয়ারে A, B, C আর D, ২ লেয়ারে E আর F-ও, ৩ লেয়ারে সাতটিই। ডানে A-এর কম্পিউটেশন ট্রি, প্রতি লেয়ারে একটি স্তর: ১টি নোড, তারপর ৩, তারপর ৭, তারপর ১৯।</desc>
<text class="tm ta" x="20" y="26">গ্রাফ</text>
<text class="tm ta" x="430" y="26">A-এর কম্পিউটেশন ট্রি</text>
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
<text class="tm ta" x="20" y="292" data-text="[&quot;A শুনেছে ৭টির মধ্যে ১টি নোডের কথা&quot;, &quot;A শুনেছে ৭টির মধ্যে ৪টি নোডের কথা&quot;, &quot;A শুনেছে ৭টির মধ্যে ৬টি নোডের কথা&quot;, &quot;A শুনেছে ৭টি নোডেরই কথা&quot;]">A শুনেছে ৭টি নোডেরই কথা</text>
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
<text class="tm" x="588" y="280" text-anchor="middle" data-text="[&quot;১টি নোড&quot;, &quot;১ + ৩টি নোড&quot;, &quot;১ + ৩ + ৭টি নোড&quot;, &quot;১ + ৩ + ৭ + ১৯টি নোড&quot;]">১ + ৩ + ৭ + ১৯টি নোড</text>
</svg>
</div>
<div class="viz-ctl" role="group" aria-label="লেয়ারের সংখ্যা"><button type="button" data-go="0" data-note="কোনো লেয়ার না থাকলে A শুধু নিজের ফিচার জানে। তার কম্পিউটেশন ট্রি একটিমাত্র নোড।" aria-pressed="false">০ লেয়ার</button><button type="button" data-go="1" data-note="এক লেয়ার: A মিশিয়ে নেয় B, C আর D-কে। ট্রিতে যোগ হয় একটি স্তর, প্রতিটি প্রতিবেশীর জন্য একটি শাখা।" aria-pressed="false">১ লেয়ার</button><button type="button" data-go="2" data-note="দুই লেয়ার: B, C আর D আগে নিজেদের প্রতিবেশীদের কাছ থেকে আপডেট হয়েছে, তাই E আর F এখন A পর্যন্ত পৌঁছায়। খেয়াল করুন, A নিজের ট্রিতেই তিনবার আছে: সে তার প্রতিবেশীদেরও প্রতিবেশী।" aria-pressed="false">২ লেয়ার</button><button type="button" data-go="3" data-note="তিন লেয়ার: তিন হপ দূরের G অবশেষে A-তে পৌঁছায়, আর A পুরো গ্রাফের কথা শুনে ফেলে। শুধু নিচের স্তরেই এখন ১৯টি নোড, আর প্রতিটি লেয়ারে এটি বাড়তেই থাকে।" aria-pressed="true">৩ লেয়ার</button></div>
<p class="viz-note" aria-live="polite">তিন লেয়ার: তিন হপ দূরের G অবশেষে A-তে পৌঁছায়, আর A পুরো গ্রাফের কথা শুনে ফেলে। শুধু নিচের স্তরেই এখন ১৯টি নোড, আর প্রতিটি লেয়ারে এটি বাড়তেই থাকে।</p>
<figcaption>k লেয়ারের একটি GNN প্রতিটি নোডের ভেক্টর হিসাব করে k গভীরতার একটি ট্রি থেকে। বাঁয়ের রিসেপটিভ ফিল্ড আর ডানের ট্রি একই জিনিস, দুইভাবে দেখা। ট্রিতে হাইলাইট করা নোডগুলো A নিজেই।</figcaption>
</figure>
{:/nomarkdown}

ট্রিটিতে দুটি জিনিস খেয়াল করার মতো। প্রথমত, **A নিজের ট্রির ভেতরেই আছে।** দুই লেয়ার নিচে A তার প্রতিবেশীদের প্রতিবেশী, তাই একটি নোড নিজের প্রতিধ্বনিও শোনে। এতে ক্ষতি নেই, আর ৫ নম্বর অংশের সেলফ-লুপ যে স্বাভাবিক, এটি তার একটি কারণ। দ্বিতীয়ত, **ট্রিটি দ্রুত বাড়ে**: এখানে প্রতি স্তরে ১, ৩, ৭ আর ১৯টি নোড। যে সোশ্যাল নেটওয়ার্কে মানুষের বন্ধু কয়েকশো, সেখানে তিন লেয়ারেই পুরো গ্রাফের বড় একটি অংশে পৌঁছানো যায়। এই বৃদ্ধিই ৮ নম্বর অংশের দুটি সমস্যার মূলে, আর এ কারণেই বড় গ্রাফের পদ্ধতিগুলো সব প্রতিবেশী না নিয়ে প্রতিবেশীদের নমুনা নেয়।

বাস্তবে বেশিরভাগ GNN-এ থাকে **দুই বা তিনটি লেয়ার**। বেশিরভাগ নোড-স্তরের কাজে দরকারি সংকেত স্থানীয়, আর তা ধরতে এটুকুই যথেষ্ট।

## ৫. একই ধারণা, ম্যাট্রিক্স রূপে: GCN
{: id="5-the-same-idea-in-matrix-form-gcn"}

নোড ধরে ধরে মেসেজ পাসিং বোঝা সহজ, কিন্তু লিখে রাখা ক্লান্তিকর। অ্যাডজেসেন্সি ম্যাট্রিক্স দিয়ে পুরো লেয়ারটি, সব নোডের জন্য একসঙ্গে, দাঁড়ায় দুটি ম্যাট্রিক্স গুণে। শুরু করা যাক সবচেয়ে সরল রূপ দিয়ে: প্রতিবেশীদের যোগ করুন, তারপর রূপান্তর করুন।

$$
H^{(k+1)} = \sigma\big(A\, H^{(k)}\, W^{(k)}\big)
$$

$$H^{(k)}$$-এ প্রতিটি নোডের ভেক্টর একটি করে সারি হিসেবে সাজানো। $$AH$$-এর সারি $$v$$ হলো $$H$$-এর সেই সারিগুলোর যোগফল যেগুলো $$v$$-এর প্রতিবেশীদের, কারণ ঠিক ওই $$u$$-গুলোর জন্যই $$A_{vu}$$ হলো 1। এরপর $$W$$ দিয়ে গুণ করলে প্রতিটি নোডে একই লিনিয়ার ম্যাপ প্রয়োগ হয়। এটাই যোগফল অ্যাগ্রিগেশনসহ মেসেজ পাসিং, লিনিয়ার অ্যালজেব্রার ভাষায় লেখা।

এর দুটি সমস্যা আছে, আর Kipf ও Welling (2017)-এর **গ্রাফ কনভোলিউশনাল নেটওয়ার্ক** (GCN) দুটিরই সমাধান করে:

- **নোড নিজেকে ভুলে যায়।** $$A_{vv} = 0$$, তাই একটি নোডের আপডেটে তার নিজের ভেক্টরই থাকে না। সমাধান হলো সেলফ-লুপ যোগ করে $$A + I$$ ব্যবহার করা।
- **ডিগ্রির সঙ্গে যোগফল বাড়ে।** ৫০০ প্রতিবেশীর একটি নোড ২ প্রতিবেশীর নোডের চেয়ে কয়েকশো গুণ বড় ভেক্টর পায়, আর কয়েক লেয়ার পরেই মানের মাপ আর সামলানো যায় না। সমাধান হলো ডিগ্রি দিয়ে নরমালাইজ করা।

GCN একটি প্রতিসম নরমালাইজেশন ব্যবহার করে, যেখানে $$\tilde{D}$$ হলো সেলফ-লুপসহ নোডের ডিগ্রির কর্ণ ম্যাট্রিক্স:

$$
H^{(k+1)} = \sigma\big(\hat{A}\, H^{(k)}\, W^{(k)}\big), \qquad \hat{A} = \tilde{D}^{-1/2}\, (A + I)\, \tilde{D}^{-1/2}
$$

নোড ধরে লিখলে $$v$$-এর প্রতিটি প্রতিবেশী $$u$$ পায় $$1/\sqrt{\tilde{d}_u \tilde{d}_v}$$ ওয়েট:

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( \sum_{u \in N(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_u\, \tilde{d}_v}}\; W^{(k)}\, \mathbf{h}_u^{(k)} \Big)
$$

{::nomarkdown}
<figure class="viz" id="fig-matrix">
<div class="viz-head"><span class="viz-label">চিত্র ৬ // ম্যাট্রিক্সে একটি GCN লেয়ার</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 270" role="img" aria-labelledby="fig-matrix-t fig-matrix-d">
<title id="fig-matrix-t">দুটি ম্যাট্রিক্স গুণ হিসেবে GCN লেয়ার</title>
<desc id="fig-matrix-d">নরমালাইজ করা অ্যাডজেসেন্সি ম্যাট্রিক্স A-hat (সাত বাই সাত) গুণ ফিচার ম্যাট্রিক্স H (সাত বাই তিন) গুণ সবার জন্য এক ওয়েট ম্যাট্রিক্স W (তিন বাই দুই) দেয় নতুন ফিচার ম্যাট্রিক্স H-prime (সাত বাই দুই)। A-hat-এর সারি A শুধু A, B, C আর D-এ শূন্য নয়, তাই A-এর নতুন সারি H-এর ঠিক ওই চারটি সারি মেশায়। A আর B-এর A-hat মান 1 ভাগ রুট 12, প্রায় 0.29।</desc>
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
<rect class="cell" x="52" y="56" width="18" height="18" data-tip="Â[A, A] = 0.25" data-tip-sub="1 / √(4 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="54" y="58" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="70" y="56" width="18" height="18" data-tip="Â[A, B] = 0.29" data-tip-sub="1 / √(4 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="72" y="58" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="88" y="56" width="18" height="18" data-tip="Â[A, C] = 0.25" data-tip-sub="1 / √(4 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="90" y="58" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="106" y="56" width="18" height="18" data-tip="Â[A, D] = 0.29" data-tip-sub="1 / √(4 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="108" y="58" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="124" y="56" width="18" height="18"/>
<rect class="cell" x="142" y="56" width="18" height="18"/>
<rect class="cell" x="160" y="56" width="18" height="18"/>
<rect class="cell" x="52" y="74" width="18" height="18" data-tip="Â[B, A] = 0.29" data-tip-sub="1 / √(3 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="54" y="76" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="70" y="74" width="18" height="18" data-tip="Â[B, B] = 0.33" data-tip-sub="1 / √(3 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="72" y="76" width="14" height="14" rx="2" style="opacity:0.75;pointer-events:none"/>
<rect class="cell" x="88" y="74" width="18" height="18" data-tip="Â[B, C] = 0.29" data-tip-sub="1 / √(3 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="90" y="76" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="106" y="74" width="18" height="18"/>
<rect class="cell" x="124" y="74" width="18" height="18"/>
<rect class="cell" x="142" y="74" width="18" height="18"/>
<rect class="cell" x="160" y="74" width="18" height="18"/>
<rect class="cell" x="52" y="92" width="18" height="18" data-tip="Â[C, A] = 0.25" data-tip-sub="1 / √(4 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="54" y="94" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="70" y="92" width="18" height="18" data-tip="Â[C, B] = 0.29" data-tip-sub="1 / √(4 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="72" y="94" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="88" y="92" width="18" height="18" data-tip="Â[C, C] = 0.25" data-tip-sub="1 / √(4 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="90" y="94" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="106" y="92" width="18" height="18"/>
<rect class="cell" x="124" y="92" width="18" height="18" data-tip="Â[C, E] = 0.29" data-tip-sub="1 / √(4 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="126" y="94" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="142" y="92" width="18" height="18"/>
<rect class="cell" x="160" y="92" width="18" height="18"/>
<rect class="cell" x="52" y="110" width="18" height="18" data-tip="Â[D, A] = 0.29" data-tip-sub="1 / √(3 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="54" y="112" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="70" y="110" width="18" height="18"/>
<rect class="cell" x="88" y="110" width="18" height="18"/>
<rect class="cell" x="106" y="110" width="18" height="18" data-tip="Â[D, D] = 0.33" data-tip-sub="1 / √(3 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="108" y="112" width="14" height="14" rx="2" style="opacity:0.75;pointer-events:none"/>
<rect class="cell" x="124" y="110" width="18" height="18"/>
<rect class="cell" x="142" y="110" width="18" height="18" data-tip="Â[D, F] = 0.29" data-tip-sub="1 / √(3 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="144" y="112" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="160" y="110" width="18" height="18"/>
<rect class="cell" x="52" y="128" width="18" height="18"/>
<rect class="cell" x="70" y="128" width="18" height="18"/>
<rect class="cell" x="88" y="128" width="18" height="18" data-tip="Â[E, C] = 0.29" data-tip-sub="1 / √(3 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="90" y="130" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="106" y="128" width="18" height="18"/>
<rect class="cell" x="124" y="128" width="18" height="18" data-tip="Â[E, E] = 0.33" data-tip-sub="1 / √(3 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="126" y="130" width="14" height="14" rx="2" style="opacity:0.75;pointer-events:none"/>
<rect class="cell" x="142" y="128" width="18" height="18" data-tip="Â[E, F] = 0.29" data-tip-sub="1 / √(3 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="144" y="130" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="160" y="128" width="18" height="18"/>
<rect class="cell" x="52" y="146" width="18" height="18"/>
<rect class="cell" x="70" y="146" width="18" height="18"/>
<rect class="cell" x="88" y="146" width="18" height="18"/>
<rect class="cell" x="106" y="146" width="18" height="18" data-tip="Â[F, D] = 0.29" data-tip-sub="1 / √(4 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="108" y="148" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="124" y="146" width="18" height="18" data-tip="Â[F, E] = 0.29" data-tip-sub="1 / √(4 × 3), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="126" y="148" width="14" height="14" rx="2" style="opacity:0.68;pointer-events:none"/>
<rect class="cell" x="142" y="146" width="18" height="18" data-tip="Â[F, F] = 0.25" data-tip-sub="1 / √(4 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="144" y="148" width="14" height="14" rx="2" style="opacity:0.62;pointer-events:none"/>
<rect class="cell" x="160" y="146" width="18" height="18" data-tip="Â[F, G] = 0.35" data-tip-sub="1 / √(4 × 2), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="162" y="148" width="14" height="14" rx="2" style="opacity:0.78;pointer-events:none"/>
<rect class="cell" x="52" y="164" width="18" height="18"/>
<rect class="cell" x="70" y="164" width="18" height="18"/>
<rect class="cell" x="88" y="164" width="18" height="18"/>
<rect class="cell" x="106" y="164" width="18" height="18"/>
<rect class="cell" x="124" y="164" width="18" height="18"/>
<rect class="cell" x="142" y="164" width="18" height="18" data-tip="Â[G, F] = 0.35" data-tip-sub="1 / √(2 × 4), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="144" y="166" width="14" height="14" rx="2" style="opacity:0.78;pointer-events:none"/>
<rect class="cell" x="160" y="164" width="18" height="18" data-tip="Â[G, G] = 0.50" data-tip-sub="1 / √(2 × 2), সেলফ-লুপসহ ডিগ্রি"/>
<rect class="cell-on" x="162" y="166" width="14" height="14" rx="2" style="opacity:1.00;pointer-events:none"/>
<rect class="cell-hl a-in" x="50" y="54" width="130" height="22" rx="3" style="--d:0.6s"/>
<text class="t big" x="115" y="208" text-anchor="middle">Â</text>
<text class="tm" x="115" y="226" text-anchor="middle">7 × 7, কে কার কথা শোনে</text>
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
<text class="tm ta a-in" x="460" y="64" style="--d:0.7s">১ // Â-এর সারি A পড়ুন</text>
<text class="t2 a-in" x="460" y="84" style="--d:0.8s">A, B, C আর D-এ শূন্য নয়: A নিজে</text>
<text class="t2 a-in" x="460" y="101" style="--d:0.8s">আর তার তিন প্রতিবেশী।</text>
<text class="tm ta a-in" x="460" y="132" style="--d:1.4s">২ // ওয়েট = 1 / √(ডিগ্রি × ডিগ্রি)</text>
<text class="t2 a-in" x="460" y="152" style="--d:1.5s">সেলফ-লুপসহ A-এর ডিগ্রি 4, B-এর 3,</text>
<text class="t2 a-in" x="460" y="169" style="--d:1.5s">তাই Â[A, B] = 1/√12 = 0.29। ঘরে কার্সর রাখুন।</text>
<text class="tm ta a-in" x="460" y="200" style="--d:2.0s">৩ // একটিই W, সব নোডের জন্য</text>
<text class="t2 a-in" x="460" y="220" style="--d:2.1s">এটি প্রতিটি নোডের ৩টি ফিচারকে ২টিতে নেয়।</text>
<text class="t2 strong a-in" x="460" y="251" style="--d:2.6s">H′-এর সারি A হলো A-এর নতুন ভেক্টর।</text>
</svg>
</div>
<figcaption>সব নোডের জন্য পুরো লেয়ার একসঙ্গে। Â-এর সারি A বেছে নেয় H-এর সেই সারিগুলো যাদের কথা A শোনে, আর সেগুলোকে ওয়েট দেয়; তারপর W একই ওয়েট দিয়ে প্রতিটি নোডকে রূপান্তর করে। Â-এর ঘরের উজ্জ্বলতাই ওয়েট; সঠিক মান দেখতে ঘরে কার্সর রাখুন।</figcaption>
</figure>
{:/nomarkdown}

দুই দিকের বর্গমূলটি একটি আপস। শুধু $$\tilde{d}_v$$ দিয়ে ভাগ করলে পাওয়া যেত সাধারণ গড়। সঙ্গে $$\sqrt{\tilde{d}_u}$$ দিয়েও ভাগ করলে ব্যস্ত হাবগুলোর মেসেজের গুরুত্বও কমে, যারা নিজেদের সব সংযোগকে একই মেসেজ পাঠায়। "কনভোলিউশন" নামটি এসেছে স্পেকট্রাল গ্রাফ থিওরি থেকে, যেখানে এই লেয়ারটি গ্রাফের ফ্রিকোয়েন্সির ওপর একটি ফিল্টারের সস্তা, প্রথম-ক্রমের আনুমানিক রূপ, তবে এটি ব্যবহার করতে এসবের কিছুই লাগে না। একে পড়ুন নিজের আর প্রতিবেশীদের একটি ওয়েটেড গড় হিসেবে, যার পরে আছে সব নোডের জন্য একই একটি লিনিয়ার লেয়ার।

এখানে একটি ডেন্স $$\hat{A}$$ দিয়ে PyTorch-এ শূন্য থেকে লেখা একটি GCN। কয়েক হাজার নোডের গ্রাফের জন্য এটি একেবারে ঠিক আছে:

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

$$\hat{A}$$ দিয়ে গুণ করার আগে $$HW$$ হিসাব করা একই ম্যাথ, শুধু সস্তা ক্রমে: $$n \times n$$ গুণের আগেই এটি ফিচারকে ১,৪৩৩টি কলাম থেকে ১৬টিতে নামিয়ে আনে।

বড় গ্রাফে ডেন্স $$n \times n$$ ম্যাট্রিক্সের প্রশ্নই ওঠে না, তাই লাইব্রেরিগুলো একই জিনিস সরাসরি এজ লিস্ট থেকে হিসাব করে: একটি **গ্যাদার** (প্রতিটি এজের সোর্স ভেক্টর পড়া), তারপর একটি **স্ক্যাটার** (তা এজের টার্গেটে যোগ করা)। $$2 \times E$$ এজ লিস্টের ওপর গড় অ্যাগ্রিগেশন মাত্র চার লাইনের, আর এর খরচ বাড়ে এজের সংখ্যার সঙ্গে:

```python
def mean_neighbours(h, edge_index):
    src, dst = edge_index                                    # one column per directed edge src -> dst
    total = torch.zeros_like(h).index_add_(0, dst, h[src])   # gather from src, scatter-add into dst
    count = torch.zeros(h.size(0), dtype=h.dtype).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    return total / count.clamp(min=1).unsqueeze(-1)          # sum -> mean; isolated nodes stay zero
```

Cora-তে (১০ নম্বর অংশ) এটি ডেন্স সংস্করণের হুবহু একই ফল দেয়, অথচ মেমরি লাগে ৭৩ লাখ ম্যাট্রিক্স-ঘরের বদলে ১০,৫৫৬টি এজের জন্য। PyTorch Geometric, DGL আর অন্যান্য গ্রাফ লাইব্রেরি মূলত এই ফাংশনেরই দ্রুত আর যত্নশীল সংস্করণ।

## ৬. তিনটি লেয়ার, যাদের দেখা মিলবে সব জায়গায়: GCN, GraphSAGE আর GAT
{: id="6-three-layers-you-will-meet-everywhere-gcn-graphsage-and-gat"}

বেশিরভাগ GNN লেয়ারের পার্থক্য একটি জায়গায়: **প্রতিটি প্রতিবেশীর মেসেজের গুরুত্ব কতটা।** একই প্রতিবেশে তিনটি ক্লাসিক উত্তর বদলে বদলে দেখুন।

{::nomarkdown}
<figure class="viz viz-steps" id="fig-layers" data-dur="4600">
<div class="viz-head"><span class="viz-label">চিত্র ৭ // একই প্রতিবেশে GCN, GRAPHSAGE আর GAT</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 290" role="img" aria-labelledby="fig-layers-t fig-layers-d">
<title id="fig-layers-t">একই চার প্রতিবেশীকে GCN, GraphSAGE আর GAT কীভাবে ওয়েট দেয়</title>
<desc id="fig-layers-d">একটি নোড v, যার চার প্রতিবেশীর ডিগ্রি 1, 2, 9 আর 4। এজের পুরুত্ব দেখায় প্রতিটি প্রতিবেশীর ওয়েট। GCN ব্যবহার করে 1 ভাগ ডিগ্রির গুণফলের রুট: 0.32, 0.26, 0.14, 0.20, আর সেলফ-লুপ 0.20। GraphSAGE প্রতিটি প্রতিবেশীকে দেয় 0.25 আর v-এর নিজের ভেক্টর আলাদা রাখে। GAT অ্যাটেনশন ওয়েট শেখে, এখানে 0.45, 0.08, 0.05, 0.22, আর নিজের জন্য 0.20।</desc>
<text class="tm ta" x="24" y="26">একটি নোড v, চারটি প্রতিবেশী</text>
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
<text class="tm" x="190" y="84" text-anchor="middle" data-text="[&quot;নিজে 0.20&quot;, &quot;নিজে: আলাদা রাখা&quot;, &quot;নিজে 0.20&quot;]">নিজে 0.20</text>
<circle class="gn-acc" cx="190" cy="150" r="22"/>
<text class="gl" x="190" y="150" dy="0.35em" style="font-size:15px">v</text>
<circle class="gn" cx="66" cy="70" r="17"/>
<text class="gl" x="66" y="70" dy="0.35em" style="font-size:12px">u1</text>
<text class="tm" x="66" y="44" text-anchor="middle">ডিগ্রি 1</text>
<text class="t" x="130.5" y="96.6" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.32&quot;, &quot;0.25&quot;, &quot;0.45&quot;]">0.45</text>
<circle class="gn" cx="318" cy="60" r="17"/>
<text class="gl" x="318" y="60" dy="0.35em" style="font-size:12px">u2</text>
<text class="tm" x="318" y="34" text-anchor="middle">ডিগ্রি 2</text>
<text class="t" x="269.6" y="117.6" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.26&quot;, &quot;0.25&quot;, &quot;0.08&quot;]">0.08</text>
<circle class="gn" cx="322" cy="236" r="17"/>
<text class="gl" x="322" y="236" dy="0.35em" style="font-size:12px">u3</text>
<text class="tm" x="322" y="267" text-anchor="middle">ডিগ্রি 9</text>
<text class="t" x="271.3" y="187.9" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.14&quot;, &quot;0.25&quot;, &quot;0.05&quot;]">0.05</text>
<circle class="gn" cx="60" cy="232" r="17"/>
<text class="gl" x="60" y="232" dy="0.35em" style="font-size:12px">u4</text>
<text class="tm" x="60" y="263" text-anchor="middle">ডিগ্রি 4</text>
<text class="t" x="127" y="212.6" text-anchor="middle" style="font-size:12px;font-family:var(--font-mono)" data-text="[&quot;0.20&quot;, &quot;0.25&quot;, &quot;0.22&quot;]">0.22</text>
<rect class="n" x="420" y="34" width="322" height="240" rx="12"/>
<text class="tm ta" x="440" y="64" data-text="[&quot;KIPF &amp; WELLING, 2017&quot;, &quot;HAMILTON, YING &amp; LESKOVEC, 2017&quot;, &quot;VELIČKOVIĆ ET AL., 2018&quot;]">VELIČKOVIĆ ET AL., 2018</text>
<text class="t big" x="440" y="96" style="font-size:22px" data-text="[&quot;GCN&quot;, &quot;GraphSAGE&quot;, &quot;GAT&quot;]">GAT</text>
<text class="tm" x="440" y="126" style="fill:#e8b84e" data-text="[&quot;ওয়েট = 1 / √((ডিগ্রি u + 1)(ডিগ্রি v + 1))&quot;, &quot;ওয়েট = 1 / (প্রতিবেশীর সংখ্যা)&quot;, &quot;ওয়েট = শেখা স্কোরের সফটম্যাক্স&quot;]">ওয়েট = শেখা স্কোরের সফটম্যাক্স</text>
<text class="t2" x="440" y="160" data-text="[&quot;ওয়েট আসে শুধু গ্রাফের গঠন থেকে।&quot;, &quot;প্রতিটি প্রতিবেশীর গুরুত্ব সমান।&quot;, &quot;ওয়েট শেখা হয় ফিচার থেকে।&quot;]">ওয়েট শেখা হয় ফিচার থেকে।</text>
<text class="t2" x="440" y="182" data-text="[&quot;৯টি লিংকের হাব u3-এর গুরুত্ব কমানো&quot;, &quot;v-এর নিজের ভেক্টর আলাদা রেখে&quot;, &quot;এখানে মডেল ঠিক করেছে u1-ই সবচেয়ে&quot;]">এখানে মডেল ঠিক করেছে u1-ই সবচেয়ে</text>
<text class="t2" x="440" y="204" data-text="[&quot;হয়, তাই সে একা প্রভাব খাটাতে পারে না।&quot;, &quot;জুড়ে দেওয়া হয়: [ নিজে ‖ গড় ]।&quot;, &quot;জরুরি, আর হাব u3 প্রায় গুরুত্বহীন।&quot;]">জরুরি, আর হাব u3 প্রায় গুরুত্বহীন।</text>
<text class="t2" x="440" y="226" data-text="[&quot;বাড়তি প্যারামিটার নেই, খুব দ্রুত।&quot;, &quot;প্রতিবেশীর নমুনা নেয়, তাই স্কেল করে।&quot;, &quot;প্রতি এজে বেশি হিসাব, শিখতে হয় বেশি।&quot;]">প্রতি এজে বেশি হিসাব, শিখতে হয় বেশি।</text>
<text class="tm" x="440" y="256" data-text="[&quot;ওয়েটে সেলফ-লুপও আছে&quot;, &quot;প্রতিবেশীদের ওয়েটের যোগফল 1&quot;, &quot;উদাহরণের ওয়েট, পাঁচটির যোগফল 1&quot;]">উদাহরণের ওয়েট, পাঁচটির যোগফল 1</text>
</svg>
</div>
<div class="viz-ctl" role="group" aria-label="একটি GNN লেয়ার বেছে নিন"><button type="button" data-go="0" data-note="GCN: ডিগ্রি থেকে স্থির ওয়েট। u1-এর লিংক মাত্র একটি, তাই তার মেসেজের গুরুত্ব সবচেয়ে বেশি (0.32); হাব u3 সবার সঙ্গে কথা বলে, তাই তার গুরুত্ব সবচেয়ে কম (0.14)।" aria-pressed="false">GCN</button><button type="button" data-go="1" data-note="GraphSAGE: সাধারণ গড়, তাই প্রতিটি প্রতিবেশী পায় 0.25, আর v-এর নিজের ভেক্টর গড়ে মেশানোর বদলে পাশে জুড়ে (কনক্যাটেনেট করে) দেওয়া হয়।" aria-pressed="false">GraphSAGE</button><button type="button" data-go="2" data-note="GAT: অ্যাটেনশন। ওয়েট হিসাব হয় নোডগুলোর ফিচার থেকে, তাই প্রতিটি নোড আর প্রতিটি এজে তা আলাদা হতে পারে। এই মানগুলো ট্রেইন করা একটি মডেল কী শিখতে পারে তার উদাহরণমাত্র।" aria-pressed="true">GAT</button></div>
<p class="viz-note" aria-live="polite">GAT: অ্যাটেনশন। ওয়েট হিসাব হয় নোডগুলোর ফিচার থেকে, তাই প্রতিটি নোড আর প্রতিটি এজে তা আলাদা হতে পারে। এই মানগুলো ট্রেইন করা একটি মডেল কী শিখতে পারে তার উদাহরণমাত্র।</p>
<figcaption>একই প্রতিবেশ, প্রতিটি মেসেজের গুরুত্ব ঠিক করার তিনটি নিয়ম। এজের পুরুত্বই ওয়েট। GCN আর GraphSAGE-এর সংখ্যাগুলো ডিগ্রি থেকেই আসে; GAT-এর সংখ্যাগুলো ধারণাটি বোঝাতে বানানো, কারণ আসল অ্যাটেনশন ওয়েট নির্ভর করে ফিচারের ওপর।</figcaption>
</figure>
{:/nomarkdown}

**GCN** (Kipf ও Welling, 2017) আগের অংশের স্থির ওয়েট ব্যবহার করে। এগুলো নির্ভর করে শুধু গ্রাফের গঠনের ওপর, তাই শিখতে কোনো খরচ নেই। সীমাবদ্ধতা হলো, একই ডিগ্রির দুটি প্রতিবেশী সবসময় সমান গুরুত্ব পায়, তারা আসলে যা-ই বলুক।

**GraphSAGE** (Hamilton, Ying ও Leskovec, 2017) দুটি পরিবর্তন আনে। এটি নোডের নিজের ভেক্টরকে প্রতিবেশীদের গড় থেকে আলাদা রাখে, দুটিকে একসঙ্গে গড় না করে কনক্যাটেনেশন ($$\Vert$$) দিয়ে জুড়ে দেয়। আর সব প্রতিবেশী না নিয়ে নির্দিষ্ট সংখ্যক প্রতিবেশীর **নমুনা** নেয়:

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( W^{(k)} \big[\, \mathbf{h}_v^{(k)} \;\Vert\; \operatorname{mean}_{u \in S(v)} \mathbf{h}_u^{(k)} \,\big] \Big)
$$

$$S(v)$$ হলো $$v$$-এর প্রতিবেশীদের একটি র‍্যান্ডম নমুনা; পেপারে প্রথম হপে ২৫টি আর দ্বিতীয় হপে ১০টি নেওয়া হয়েছিল। নমুনা নেওয়ায় কম্পিউটেশন ট্রির আকার সীমিত থাকে, আর এ কারণেই GraphSAGE লাখ লাখ নোডের গ্রাফে মিনি-ব্যাচে ট্রেইন করতে পারে। আর যেহেতু এটি প্রতিটি নোডের জন্য একটি ভেক্টর নয়, বরং প্রতিবেশের একটি *ফাংশন* শেখে, তাই এটি **ইনডাকটিভ**: ট্রেনিংয়ের সময় ছিল না এমন নোডকেও এটি এমবেড করতে পারে, যেমন নতুন একজন ব্যবহারকারী বা নতুন একটি লেনদেন।

**GAT**, অর্থাৎ গ্রাফ অ্যাটেনশন নেটওয়ার্ক (Veličković et al., 2018), ওয়েটগুলো মডেলকেই শিখতে দেয়। এটি দুই নোডের ফিচারের একটি ছোট অ্যাটেনশন ফাংশন দিয়ে প্রতিটি প্রতিবেশীকে স্কোর দেয়, আর একটি সফটম্যাক্স সেই স্কোরগুলোকে ওয়েটে পরিণত করে:

$$
e_{vu} = \operatorname{LeakyReLU}\big( \mathbf{a}^\top [\, W\mathbf{h}_v \,\Vert\, W\mathbf{h}_u \,] \big), \qquad
\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{w \in N(v) \cup \{v\}} \exp(e_{vw})}
$$

$$
\mathbf{h}_v^{(k+1)} = \sigma\Big( \sum_{u \in N(v) \cup \{v\}} \alpha_{vu}\, W \mathbf{h}_u^{(k)} \Big)
$$

ট্রান্সফরমারের মতো GAT-ও কয়েকটি অ্যাটেনশন হেড সমান্তরালে চালায় আর তাদের আউটপুট জুড়ে দেয়; পেপারে Cora-তে ৮টি হেড ব্যবহার করা হয়েছিল। ওয়েট যেহেতু বিষয়বস্তুর ওপর নির্ভর করে, একটি নোড শিখে নিতে পারে কোন একটি প্রতিবেশীর কথা জরুরি আর বাকিদের উপেক্ষা করা যায়। এর মূল্য হলো প্রতি এজে বেশি হিসাব আর শেখার জন্য বেশি প্যারামিটার।

{::nomarkdown}
<div class="table-wrap">
  <table>
    <thead>
      <tr><th></th><th>GCN</th><th>GraphSAGE</th><th>GAT</th></tr>
    </thead>
    <tbody>
      <tr><td>প্রতিবেশীর ওয়েট</td><td>স্থির, ডিগ্রি থেকে</td><td>সমান (গড়)</td><td>শেখা অ্যাটেনশন</td></tr>
      <tr><td>নিজের ভেক্টর</td><td>সেলফ-লুপ দিয়ে গড়ে মেশানো</td><td>আলাদা রেখে জুড়ে দেওয়া</td><td>সেলফ-লুপ দিয়ে অ্যাটেনশনে</td></tr>
      <tr><td>পেপারে ট্রেনিং</td><td>পুরো গ্রাফ, ট্রান্সডাকটিভ</td><td>নমুনা নেওয়া মিনি-ব্যাচ, ইনডাকটিভ</td><td>পুরো গ্রাফ, দুই ধরনেই</td></tr>
      <tr><td>প্রতি এজে খরচ</td><td>সবচেয়ে কম</td><td>কম</td><td>সবচেয়ে বেশি</td></tr>
      <tr><td>কখন বেছে নেবেন</td><td>দ্রুত, শক্ত একটি বেসলাইন চাইলে</td><td>গ্রাফ বড় হলে বা বাড়তে থাকলে</td><td>কিছু প্রতিবেশী অন্যদের চেয়ে অনেক বেশি জরুরি হলে</td></tr>
    </tbody>
  </table>
</div>
{:/nomarkdown}

## ৭. যোগফল, গড় নাকি সর্বোচ্চ: অ্যাগ্রিগেটর কেন জরুরি
{: id="7-sum-mean-or-max-why-the-aggregator-matters"}

অ্যাগ্রিগেটরকে ছোট খুঁটিনাটি মনে হতে পারে, কিন্তু GNN কী আলাদা করতে পারবে আর কী পারবে না, তা এটিই ঠিক করে। Xu et al. "How Powerful are Graph Neural Networks?" (2019) পেপারে প্রশ্নটিকে নিখুঁত রূপ দিয়েছেন: কখন একটি GNN দুটি ভিন্ন প্রতিবেশকে দুটি ভিন্ন ভেক্টর দেয়?

ওয়ান-হট ফিচারসহ তিনটি প্রতিবেশ নিন, যেখানে সোনালি হলো $$[1, 0]$$ আর নীল $$[0, 1]$$:

{::nomarkdown}
<figure class="viz" id="fig-agg">
<div class="viz-head"><span class="viz-label">চিত্র ৮ // যোগফল, গড় আর সর্বোচ্চ কী আলাদা করতে পারে</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 268" role="img" aria-labelledby="fig-agg-t fig-agg-d">
<title id="fig-agg-t">তিনটি প্রতিবেশে যোগফল, গড় আর সর্বোচ্চ দিয়ে অ্যাগ্রিগেশন</title>
<desc id="fig-agg-d">ওয়ান-হট ফিচারসহ তিনটি প্রতিবেশ, সোনালি [1, 0] আর নীল [0, 1]: একটি সোনালি ও একটি নীল; প্রতিটি দুটি করে; দুটি সোনালি ও একটি নীল। যোগফল দেয় [1, 1], [2, 2] আর [2, 1], সবগুলো আলাদা। গড় দুবার দেয় [0.5, 0.5] আর একবার [0.67, 0.33], তাই প্রথম দুটি এক হয়ে যায়। সর্বোচ্চ তিনটির জন্যই দেয় [1, 1]।</desc>
<g class="a-in" style="--d:0.05s"><text class="tm ta" x="170" y="24" text-anchor="middle">প্রতিবেশ ১</text><line class="ge" x1="170" y1="108" x2="111.7" y2="86.8"/><line class="ge" x1="170" y1="108" x2="228.3" y2="86.8"/><circle class="gn" cx="170" cy="108" r="13"/><text class="gl" x="170" y="108" dy="0.35em">?</text><circle class="s1" cx="111.7" cy="86.8" r="11"/><circle class="s2" cx="228.3" cy="86.8" r="11"/></g>
<g class="a-in" style="--d:0.17s"><text class="tm ta" x="350" y="24" text-anchor="middle">প্রতিবেশ ২</text><line class="ge" x1="350" y1="108" x2="291.7" y2="86.8"/><line class="ge" x1="350" y1="108" x2="325.4" y2="51.1"/><line class="ge" x1="350" y1="108" x2="374.6" y2="51.1"/><line class="ge" x1="350" y1="108" x2="408.3" y2="86.8"/><circle class="gn" cx="350" cy="108" r="13"/><text class="gl" x="350" y="108" dy="0.35em">?</text><circle class="s1" cx="291.7" cy="86.8" r="11"/><circle class="s1" cx="325.4" cy="51.1" r="11"/><circle class="s2" cx="374.6" cy="51.1" r="11"/><circle class="s2" cx="408.3" cy="86.8" r="11"/></g>
<g class="a-in" style="--d:0.29s"><text class="tm ta" x="530" y="24" text-anchor="middle">প্রতিবেশ ৩</text><line class="ge" x1="530" y1="108" x2="471.7" y2="86.8"/><line class="ge" x1="530" y1="108" x2="530" y2="46"/><line class="ge" x1="530" y1="108" x2="588.3" y2="86.8"/><circle class="gn" cx="530" cy="108" r="13"/><text class="gl" x="530" y="108" dy="0.35em">?</text><circle class="s1" cx="471.7" cy="86.8" r="11"/><circle class="s1" cx="530" cy="46" r="11"/><circle class="s2" cx="588.3" cy="86.8" r="11"/></g>
<g class="a-in" style="--d:0.8s"><text class="tm ta" x="24" y="168">যোগফল</text><text class="t ok-v" x="170" y="168" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text><text class="t ok-v" x="350" y="168" text-anchor="middle" style="font-family:var(--font-mono)">[2, 2]</text><text class="t ok-v" x="530" y="168" text-anchor="middle" style="font-family:var(--font-mono)">[2, 1]</text></g>
<g class="a-pop" style="--d:1.1s"><rect class="ok-badge" x="628" y="154" width="118" height="20" rx="10"/><text class="tm ok-t" x="687" y="168" text-anchor="middle" style="font-size:9.5px">তিনটিই আলাদা</text></g>
<g class="a-in" style="--d:1.35s"><text class="tm ta" x="24" y="198">গড়</text><text class="t bad-t" x="170" y="198" text-anchor="middle" style="font-family:var(--font-mono)">[0.5, 0.5]</text><text class="t bad-t" x="350" y="198" text-anchor="middle" style="font-family:var(--font-mono)">[0.5, 0.5]</text><text class="t ok-v" x="530" y="198" text-anchor="middle" style="font-family:var(--font-mono)">[0.67, 0.33]</text></g>
<g class="a-pop" style="--d:1.65s"><rect class="evil-bg" x="628" y="184" width="118" height="20" rx="10" style="stroke:#e0736d;stroke-opacity:.6"/><text class="tm bad-t" x="687" y="198" text-anchor="middle" style="font-size:9.5px">১ আর ২ এক হয়ে যায়</text></g>
<g class="a-in" style="--d:1.9s"><text class="tm ta" x="24" y="228">সর্বোচ্চ</text><text class="t bad-t" x="170" y="228" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text><text class="t bad-t" x="350" y="228" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text><text class="t bad-t" x="530" y="228" text-anchor="middle" style="font-family:var(--font-mono)">[1, 1]</text></g>
<g class="a-pop" style="--d:2.2s"><rect class="evil-bg" x="628" y="214" width="118" height="20" rx="10" style="stroke:#e0736d;stroke-opacity:.6"/><text class="tm bad-t" x="687" y="228" text-anchor="middle" style="font-size:9.5px">তিনটিই এক হয়ে যায়</text></g>
<line class="ln-faint" x1="24" y1="146" x2="746" y2="146"/>
<text class="tm" x="24" y="256">ভেক্টর = [সোনালি, নীল]: সোনালি = [1, 0], নীল = [0, 1]</text>
</svg>
</div>
<figcaption>কোন অ্যাগ্রিগেটর কী দেখতে পায়। সর্বোচ্চ রাখে শুধু কোন ফিচার আছে, গড় রাখে অনুপাত, যোগফল রাখে গণনা। শুধু যোগফলই তিনটি ভিন্ন প্রতিবেশকে তিনটি ভিন্ন ফল দেয়।</figcaption>
</figure>
{:/nomarkdown}

- **সর্বোচ্চ** দেখে শুধু *কোন* ফিচারগুলো আছে। তিনটি প্রতিবেশেই সোনালি আর নীল আছে, তাই সর্বোচ্চ এদের আলাদা করতে পারে না।
- **গড়** দেখে *অনুপাত*। এটি তৃতীয় প্রতিবেশটিকে আলাদা করে, যার দুই-তৃতীয়াংশ সোনালি, কিন্তু প্রথম দুটিকে নয়, কারণ দুটিই অর্ধেক অর্ধেক।
- **যোগফল** দেখে *গণনা*, তাই এটি তিনটিকেই আলাদা করে।

গড় আর সর্বোচ্চ ভুল নয়। প্রতিবেশীর সংখ্যা যখন সংকেত নয়, নয়েজ, তখন এরাই ভালো, আর ডিগ্রি খুব বেশি ওঠানামা করলে গড় স্থির থাকে। কিন্তু গণনা যখন জরুরি, যেমন রসায়নে, যেখানে তিনটি হাইড্রোজেনওয়ালা কার্বন আর একটি হাইড্রোজেনওয়ালা কার্বন এক নয়, তখন দরকার যোগফল। Xu et al. প্রমাণ করেছেন যে যোগফল অ্যাগ্রিগেশন আর তার পরে একটি MLP মেসেজ পাসিং যতটা শক্তিশালী হতে পারে ততটাই শক্তিশালী, আর ফলটির নাম দিয়েছেন **গ্রাফ আইসোমরফিজম নেটওয়ার্ক** (GIN):

$$
\mathbf{h}_v^{(k+1)} = \operatorname{MLP}^{(k)}\Big( \big(1 + \epsilon^{(k)}\big)\, \mathbf{h}_v^{(k)} + \sum_{u \in N(v)} \mathbf{h}_u^{(k)} \Big)
$$

"মেসেজ পাসিং যতটা শক্তিশালী হতে পারে" কথাটির একটি সীমা আছে: **Weisfeiler-Lehman টেস্ট** (1-WL), দুটি গ্রাফ একই হতে পারে কি না যাচাইয়ের একটি ক্লাসিক অ্যালগরিদম। 1-WL যে জোড়া গ্রাফ আলাদা করতে পারে না, মেসেজ পাসিংও তা পারে না। পাঠ্যবইয়ের উদাহরণ হলো ছয় নোডের একটি বৃত্ত বনাম দুটি আলাদা ত্রিভুজ। দুটি গ্রাফেই প্রতিটি নোডের ঠিক দুটি করে প্রতিবেশী, যারা হুবহু তার মতোই দেখতে, তাই মেসেজ পাসিংয়ের প্রতিটি রাউন্ডে প্রতিটি নোড একই ভেক্টর পায়। ফলে একটিতে ত্রিভুজ থাকলেও আর অন্যটিতে না থাকলেও, দুটি গ্রাফ হুবহু একই দেখায়। যখন এটা জরুরি, যেমন অণুতে বলয় গোনার সময়, তখন সাধারণ সমাধান হলো সাইকেল কাউন্ট বা পজিশনাল এনকোডিংয়ের মতো বাড়তি ইনপুট ফিচার, অথবা আরও শক্তিশালী আর্কিটেকচার।

## ৮. গভীর মানেই ভালো নয়: ওভারস্মুদিং
{: id="8-why-deeper-is-not-better-oversmoothing"}

কম্পিউটার ভিশনে গভীর নেটওয়ার্ক সাধারণত ভালো করে। GNN সাধারণত দুই বা তিন লেয়ারে সেরা ফল দেয়, তারপর খারাপ হতে থাকে। মূল কারণটা দেখা যায় শেখা ওয়েট সরিয়ে শুধু গড় নেওয়ার ধাপটা বারবার করলে। রাউন্ডগুলো ধাপে ধাপে দেখুন:

{::nomarkdown}
<figure class="viz viz-steps" id="fig-smooth" data-dur="3400">
<div class="viz-head"><span class="viz-label">চিত্র ৯ // ওভারস্মুদিং, রাউন্ড ধরে ধরে</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="fig-smooth-t fig-smooth-d">
<title id="fig-smooth-t">বারবার প্রতিবেশের গড় নিলে প্রতিটি নোড একই মানে পৌঁছায়</title>
<desc id="fig-smooth-d">সাত নোডের গ্রাফ, প্রতিটি নোড তার মান অনুযায়ী নীল (0) থেকে সোনালি (1) রঙে, আর একই মানগুলো আসল ক্লাস অনুযায়ী ভাগ করা একটি সংখ্যারেখায়। প্রতিবেশীদের সঙ্গে 0, 1, 2, 4, 8 আর 32 রাউন্ড গড় নেওয়ার পরের মান: 0 রাউন্ড পরে: A 0.90, B 0.35, C 0.80, D 0.60, E 0.10, F 0.20, G 0.15; 2 রাউন্ড পরে: A 0.61, B 0.63, C 0.56, D 0.50, E 0.39, F 0.34, G 0.22; 32 রাউন্ড পরে: A 0.48, B 0.48, C 0.48, D 0.48, E 0.48, F 0.48, G 0.48। দুই ক্লাসের গড়ের ব্যবধান 0.42 থেকে নেমে আসে 0.00-এ।</desc>
<text class="tm ta" x="20" y="26">গ্রাফ, মান অনুযায়ী রঙ করা</text>
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
<text class="tm" x="34" y="290">আসল ক্লাস সোনালি</text>
<rect x="150" y="282" width="9" height="9" rx="2" style="fill:#3987e5"/>
<text class="tm" x="164" y="290">আসল ক্লাস নীল</text>
<text class="tm ta" x="430" y="26">আসল ক্লাস অনুযায়ী প্রতিটি নোডের মান</text>
<text class="tm" x="430" y="62" style="fill:#c98500">সোনালি</text>
<text class="tm" x="430" y="156" style="fill:#3987e5">নীল</text>
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
<text class="tm ta" x="430" y="270" data-text="[&quot;দুই ক্লাসের গড়ের ব্যবধান: 0.42&quot;, &quot;দুই ক্লাসের গড়ের ব্যবধান: 0.29&quot;, &quot;দুই ক্লাসের গড়ের ব্যবধান: 0.24&quot;, &quot;দুই ক্লাসের গড়ের ব্যবধান: 0.15&quot;, &quot;দুই ক্লাসের গড়ের ব্যবধান: 0.06&quot;, &quot;দুই ক্লাসের গড়ের ব্যবধান: 0.00&quot;]">দুই ক্লাসের গড়ের ব্যবধান: 0.00</text>
<text class="tm" x="430" y="289" data-text="[&quot;একটি থ্রেশহোল্ডে ক্লাস দুটি আলাদা হয়: না&quot;, &quot;একটি থ্রেশহোল্ডে ক্লাস দুটি আলাদা হয়: না&quot;, &quot;একটি থ্রেশহোল্ডে ক্লাস দুটি আলাদা হয়: হ্যাঁ&quot;, &quot;একটি থ্রেশহোল্ডে ক্লাস দুটি আলাদা হয়: হ্যাঁ&quot;, &quot;একটি থ্রেশহোল্ডে ক্লাস দুটি আলাদা হয়: হ্যাঁ&quot;, &quot;একটি থ্রেশহোল্ডে ক্লাস দুটি আলাদা হয়: না&quot;]">একটি থ্রেশহোল্ডে ক্লাস দুটি আলাদা হয়: না</text>
</svg>
</div>
<div class="viz-ctl" role="group" aria-label="গড় নেওয়ার রাউন্ড"><button type="button" data-go="0" data-note="লেয়ার ০: কাঁচা, নয়েজি ফিচার। B সোনালি নোড হলেও তার মান কম, আর D নীল নোড হলেও তার মান বেশি, তাই কোনো একটি থ্রেশহোল্ড সব নোডকে ঠিক দিকে রাখতে পারে না।" aria-pressed="false">০</button><button type="button" data-go="1" data-note="এক রাউন্ড পরে প্রতিটি নোড তার প্রতিবেশের গড়ের দিকে সরে যায়। B-কে টেনে তোলে তার সোনালি প্রতিবেশী A আর C। D-কে তখনও A ওপরে ধরে রেখেছে, তাই ক্লাস দুটি এখনও মিশে আছে।" aria-pressed="false">১</button><button type="button" data-go="2" data-note="দুই রাউন্ড পরে ক্লাস দুটি পরিষ্কারভাবে আলাদা: এখন একটি থ্রেশহোল্ডই প্রতিটি নোডকে ঠিকভাবে ক্লাসিফাই করবে। স্মুদিংয়ের এটাই কাজের দিক, আর এ কারণেই সাধারণত দুই বা তিনটি লেয়ার নেওয়া হয়।" aria-pressed="false">২</button><button type="button" data-go="3" data-note="চার রাউন্ড পরে A-D আর C-E এজ দিয়ে সোনালি চুইয়ে ঢুকছে নীলে। দুই ক্লাসের গড়ের ব্যবধান 0.42 থেকে কমে হয়েছে 0.15।" aria-pressed="false">৪</button><button type="button" data-go="4" data-note="আট রাউন্ড পরে ক্লাস দুটি প্রায় মিশে গেছে। কাগজে-কলমে এখনও আলাদা, কিন্তু সবচেয়ে কাছের সোনালি আর নীল নোডের পার্থক্য মাত্র 0.03, যা বাস্তবের যেকোনো নয়েজে মুছে যাবে।" aria-pressed="false">৮</button><button type="button" data-go="5" data-note="৩২ রাউন্ড পরে দুই দশমিক ঘর পর্যন্ত প্রতিটি নোডের মান এক। কোনো ক্লাসিফায়ার আর ক্লাস দুটি আলাদা করতে পারবে না। এটাই ওভারস্মুদিং।" aria-pressed="true">৩২ লেয়ার</button></div>
<p class="viz-note" aria-live="polite">৩২ রাউন্ড পরে দুই দশমিক ঘর পর্যন্ত প্রতিটি নোডের মান এক। কোনো ক্লাসিফায়ার আর ক্লাস দুটি আলাদা করতে পারবে না। এটাই ওভারস্মুদিং।</p>
<figcaption>প্রতিটি রাউন্ডে নোডের মান বদলে হয় নিজের আর প্রতিবেশীদের গড়, অর্থাৎ শেখা ওয়েট বাদ দেওয়া একটি GNN লেয়ার। ড্যাশ দেওয়া রেখা দুটি দুই ক্লাসের গড়। অল্প স্মুদিং নয়েজ পরিষ্কার করে; বেশি স্মুদিং ক্লাসের পার্থক্যই মুছে দেয়।</figcaption>
</figure>
{:/nomarkdown}

প্রতিটি লেয়ার একটি নোডের ভেক্টরকে তার প্রতিবেশের গড় দিয়ে বদলে দেয়, যা একটি স্মুদিং অপারেশন, অনেকটা ছবি ঝাপসা করার মতো। অল্প স্মুদিং উপকারী: একই ক্লাসের প্রতিবেশীরা নয়েজি একটি নোডকে তার ক্লাসের দিকে টেনে আনে, আর GNN ঠিক এ কারণেই কাজ করে। বেশি হলে, যুক্ত একটি গ্রাফে গড়ের গড় সব জায়গায় একই ভেক্টরে গিয়ে ঠেকে। Li, Han ও Wu (2018) দেখিয়েছেন যে GCN লেয়ার একধরনের ল্যাপ্লাসিয়ান স্মুদিং, আর অনেকগুলো লেয়ার পরপর বসালে নোডের ভেক্টরগুলো আর আলাদা করা যায় না। এটাই **ওভারস্মুদিং**। (GCN-এর প্রতিসম নরমালাইজেশনে সীমার মান নোডভেদে আলাদা হয় শুধু ডিগ্রির ওপর নির্ভরশীল একটি গুণকে, যা ক্লাস আলাদা করার কাজে একটুও বেশি সাহায্য করে না।)

দ্বিতীয় সম্পর্কিত সমস্যাটি **ওভারস্কোয়াশিং** (Alon ও Yahav, 2021)। কম্পিউটেশন ট্রির কথা মনে করুন: $$k$$ হপের মধ্যের নোডের সংখ্যা সূচকীয় হারে বাড়তে পারে, কিন্তু তাদের সব তথ্যকে গোড়ার নির্দিষ্ট আকারের একটিমাত্র ভেক্টরে চেপে ঢুকতে হয়। দূরের সংকেত এত সংকুচিত হয়ে পৌঁছায় যে কাজে লাগানো যায় না। যেসব কাজে সত্যিই দূরের তথ্য লাগে, সেখানেই এটি বেশি ক্ষতি করে।

যা সাহায্য করে:

- **অগভীর থাকুন।** দুই বা তিন লেয়ারই সঠিক শুরু। বেশি সক্ষমতা লাগলে লেয়ারগুলো চওড়া করুন, স্ট্যাক আরও গভীর নয়।
- **রেসিডুয়াল কানেকশন**, $$\mathbf{h}^{(k+1)} = \mathbf{h}^{(k)} + \operatorname{layer}\big(\mathbf{h}^{(k)}\big)$$, যাতে মূল সংকেত টিকে থাকে।
- **জাম্পিং নলেজ** (Xu et al., 2018): আউটপুট লেয়ারকে প্রতিটি মধ্যবর্তী রিপ্রেজেন্টেশন দিন, যাতে মডেল প্রতিটি নোডের জন্য সঠিক গভীরতা বেছে নিতে পারে।
- **গ্রাফের জন্য বানানো নরমালাইজেশন**, যেমন PairNorm (Zhao ও Akoglu, 2020), যা নোডের ভেক্টরগুলোকে একে অপরের দিকে ধসে পড়তে দেয় না।
- **DropEdge** এবং এ ধরনের পদ্ধতি, যা ট্রেনিংয়ের সময় র‍্যান্ডম এজ সরিয়ে দিয়ে স্মুদিংকে ধীর করে।

## ৯. নোডের ভেক্টর থেকে উত্তর: তিনটি হেড
{: id="9-from-node-vectors-to-answers-the-three-heads"}

GNN লেয়ারগুলোর পর প্রতিটি নোডের কাছে থাকে তার প্রতিবেশের সারসংক্ষেপ করা একটি ভেক্টর। একটি ছোট **হেড** সেই ভেক্টরগুলোকে ১ নম্বর অংশের প্রতিটি প্রশ্নের উত্তরে পরিণত করে।

{::nomarkdown}
<figure class="viz" id="fig-heads">
<div class="viz-head"><span class="viz-label">চিত্র ১০ // নোডের ভেক্টর থেকে উত্তর</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 268" role="img" aria-labelledby="fig-heads-t fig-heads-d">
<title id="fig-heads-t">একটি GNN, তিনটি হেড</title>
<desc id="fig-heads-d">ইনপুট গ্রাফ দুই বা তিনটি GNN লেয়ারের ভেতর দিয়ে যায়, আর প্রতিটি নোডের জন্য তৈরি হয় একটি ভেক্টর। নোড হেড প্রতিটি নোডে একটি লিনিয়ার লেয়ার আর সফটম্যাক্স চালায়, এজ হেড ডট প্রোডাক্ট দিয়ে এক জোড়া নোডকে স্কোর দেয়, আর গ্রাফ হেড সব নোডের ভেক্টর পুল করে একটি MLP চালায়।</desc>
<defs><marker id="fig-heads-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9z" fill="#d4a03c"/></marker></defs>
<g class="a-in" style="--d:0.05s"><text class="tm ta" x="24" y="40">ইনপুট গ্রাফ</text><line class="ge" x1="40" y1="112" x2="88" y2="84"/><line class="ge" x1="88" y1="84" x2="110" y2="146"/><line class="ge" x1="40" y1="112" x2="110" y2="146"/><line class="ge" x1="110" y1="146" x2="58" y2="178"/><line class="ge" x1="110" y1="146" x2="132" y2="196"/><line class="ge" x1="58" y1="178" x2="132" y2="196"/><circle class="gn" cx="40" cy="112" r="9"/><circle class="gn" cx="88" cy="84" r="9"/><circle class="gn" cx="110" cy="146" r="9"/><circle class="gn" cx="58" cy="178" r="9"/><circle class="gn" cx="132" cy="196" r="9"/></g>
<path class="flow" d="M150 142H186" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:0.2s"><rect class="n-acc" x="194" y="98" width="124" height="88" rx="12"/><rect class="glow" x="194" y="98" width="124" height="88" rx="12" style="--d:1.0s;--p:5s"/><text class="t" x="256" y="132" text-anchor="middle">GNN লেয়ার</text><text class="t2" x="256" y="150" text-anchor="middle">মেসেজ পাসিং</text><text class="tm" x="256" y="170" text-anchor="middle">× ২ বা ৩</text></g>
<path class="flow" d="M318 142H344" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:0.45s"><text class="tm ta" x="352" y="78">প্রতি নোডে</text><text class="tm ta" x="352" y="92">একটি ভেক্টর</text><rect class="cell-on" x="354" y="106" width="12" height="12" rx="2" style="opacity:0.92"/><rect class="cell-on" x="368" y="106" width="12" height="12" rx="2" style="opacity:0.44"/><rect class="cell-on" x="382" y="106" width="12" height="12" rx="2" style="opacity:0.68"/><rect class="cell-on" x="396" y="106" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="354" y="122" width="12" height="12" rx="2" style="opacity:0.52"/><rect class="cell-on" x="368" y="122" width="12" height="12" rx="2" style="opacity:0.84"/><rect class="cell-on" x="382" y="122" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="396" y="122" width="12" height="12" rx="2" style="opacity:0.60"/><rect class="cell-on" x="354" y="138" width="12" height="12" rx="2" style="opacity:0.76"/><rect class="cell-on" x="368" y="138" width="12" height="12" rx="2" style="opacity:0.68"/><rect class="cell-on" x="382" y="138" width="12" height="12" rx="2" style="opacity:0.92"/><rect class="cell-on" x="396" y="138" width="12" height="12" rx="2" style="opacity:0.44"/><rect class="cell-on" x="354" y="154" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="368" y="154" width="12" height="12" rx="2" style="opacity:0.52"/><rect class="cell-on" x="382" y="154" width="12" height="12" rx="2" style="opacity:0.44"/><rect class="cell-on" x="396" y="154" width="12" height="12" rx="2" style="opacity:0.92"/><rect class="cell-on" x="354" y="170" width="12" height="12" rx="2" style="opacity:0.60"/><rect class="cell-on" x="368" y="170" width="12" height="12" rx="2" style="opacity:0.36"/><rect class="cell-on" x="382" y="170" width="12" height="12" rx="2" style="opacity:0.76"/><rect class="cell-on" x="396" y="170" width="12" height="12" rx="2" style="opacity:0.68"/></g>
<path class="flow" d="M418 142L464 58" marker-end="url(#fig-heads-ah)"/>
<circle class="packet a-travel" cx="418" cy="142" r="3.5" style="--dx:42px;--dy:-84px;--t:1.7s;--d:1.40s"/>
<g class="a-in" style="--d:0.8s"><rect class="n" x="474" y="30" width="140" height="56" rx="10"/><text class="t" x="490" y="54">নোড হেড</text><text class="t2" x="490" y="72">লিনিয়ার + সফটম্যাক্স</text></g>
<path class="flow" d="M614 58H636" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:1.1s"><text class="tm ta" x="644" y="52">প্রতি নোডে</text><text class="t2 strong" x="644" y="70">পেপারের বিষয়</text></g>
<path class="flow" d="M418 142L464 142" marker-end="url(#fig-heads-ah)"/>
<circle class="packet a-travel" cx="418" cy="142" r="3.5" style="--dx:42px;--dy:0px;--t:1.7s;--d:1.75s"/>
<g class="a-in" style="--d:1.0s"><rect class="n" x="474" y="114" width="140" height="56" rx="10"/><text class="t" x="490" y="138">এজ হেড</text><text class="t2" x="490" y="156">স্কোর = hᵤ · hᵥ</text></g>
<path class="flow" d="M614 142H636" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:1.3s"><text class="tm ta" x="644" y="136">প্রতি জোড়ায়</text><text class="t2 strong" x="644" y="154">লিংক স্কোর</text></g>
<path class="flow" d="M418 142L464 226" marker-end="url(#fig-heads-ah)"/>
<circle class="packet a-travel" cx="418" cy="142" r="3.5" style="--dx:42px;--dy:84px;--t:1.7s;--d:2.10s"/>
<g class="a-in" style="--d:1.2s"><rect class="n" x="474" y="198" width="140" height="56" rx="10"/><text class="t" x="490" y="222">গ্রাফ হেড</text><text class="t2" x="490" y="240">পুল, তারপর MLP</text></g>
<path class="flow" d="M614 226H636" marker-end="url(#fig-heads-ah)"/>
<g class="a-in" style="--d:1.5s"><text class="tm ta" x="644" y="220">প্রতি গ্রাফে</text><text class="t2 strong" x="644" y="238">বিষাক্ত কি না</text></g>
</svg>
</div>
<figcaption>নোডের ভেক্টর হিসাব করা লেয়ারগুলো সব কাজেই এক। উত্তর প্রতি নোডে, প্রতি জোড়ায় নাকি প্রতি গ্রাফে হবে, তা ঠিক করে শুধু শেষের ছোট হেডটি।</figcaption>
</figure>
{:/nomarkdown}

- **নোড হেড।** প্রতিটি নোডের ভেক্টরে একটি লিনিয়ার লেয়ার আর সফটম্যাক্স, যেসব নোডের লেবেল আছে সেগুলোর ওপর ক্রস-এনট্রপি দিয়ে ট্রেইন করা।
- **এজ হেড।** এক জোড়া নোডের জন্য একটি স্কোর, সবচেয়ে সহজ রূপে ডট প্রোডাক্ট $$\mathbf{h}_u^\top \mathbf{h}_v$$, একটি সিগময়েডের ভেতর দিয়ে। এটি ট্রেইন করা হয় বিদ্যমান এজগুলোকে পজিটিভ আর র‍্যান্ডমভাবে বেছে নেওয়া অ-এজগুলোকে নেগেটিভ ধরে।
- **গ্রাফ হেড।** একটি **রিডআউট** সব নোডের ভেক্টরকে যোগফল, গড় বা সর্বোচ্চ দিয়ে একটিতে পুল করে (আবারও ক্রম-নিরপেক্ষ, ২ নম্বর অংশের কারণেই), আর একটি MLP তা থেকে লেবেল অনুমান করে।

## ১০. Cora-তে ৩০ লাইনে একটি চালু GNN
{: id="10-a-working-gnn-on-cora-in-30-lines"}

Cora হলো গ্রাফ লার্নিংয়ের MNIST: মেশিন লার্নিংয়ের ২,৭০৮টি পেপারের একটি সাইটেশন নেটওয়ার্ক। প্রতিটি পেপারকে বর্ণনা করে ১,৪৩৩ মাত্রার একটি ব্যাগ-অফ-ওয়ার্ডস ভেক্টর, আর প্রতিটি পেপার ৭টি বিষয়ের একটির অন্তর্ভুক্ত। এদের মধ্যে সাইটেশন লিংক ৫,২৭৮টি (প্রতিটি দুই দিকে একবার করে, মোট ১০,৫৫৬টি ডিরেক্টেড এজ হিসেবে রাখা)। Yang, Cohen ও Salakhutdinov (2016)-এর স্ট্যান্ডার্ড স্প্লিটটি ইচ্ছে করেই কঠিন: **ট্রেনিংয়ের জন্য লেবেল আছে মাত্র ১৪০টি পেপারের**, প্রতি বিষয়ে ২০টি, সঙ্গে ভ্যালিডেশনের জন্য ৫০০টি আর টেস্টের জন্য ১,০০০টি। বাকিদের বিষয় মডেলকে অনুমান করতে হয় শব্দ আর সাইটেশন থেকে।

GCN পেপারের হাইপারপ্যারামিটার দিয়ে PyTorch Geometric-এ দুই লেয়ারের একটি GCN:

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

ল্যাপটপের CPU-তে এটি এক মিনিটেরও অনেক কম সময়ে ট্রেইন হয়। দশটি র‍্যান্ডম সিডে এর গড় টেস্ট অ্যাকুরেসি ছিল ৮১.৭% (৮০.৪% থেকে ৮২.৬%-এর মধ্যে), যা পেপারের ৮১.৫%-এর সঙ্গে মেলে। ৫ নম্বর অংশের শূন্য থেকে লেখা সংস্করণটি পায় ৮১.৪%: একই মডেল, একই ফল।

লসের লাইনটি খেয়াল করুন। তাতে অবদান রাখে মাত্র ১৪০টি নোড, কিন্তু ফরোয়ার্ড পাস চলে ২,৭০৮টির ওপরই, আর লেবেলহীন পেপারগুলোও তাদের শব্দ আর সাইটেশনের মাধ্যমে প্রতিটি অনুমানে প্রভাব রাখে। টেস্টের নোডগুলো ট্রেনিংয়ের সময়ও গ্রাফে থাকে, শুধু তাদের লেবেল লুকানো থাকে; এই ধরনকে বলে **ট্রান্সডাকটিভ** লার্নিং।

গ্রাফের মূল্য কতটা? এখানে একই ১৪০টি লেবেল চারভাবে ব্যবহার করা হয়েছে:

{::nomarkdown}
<figure class="viz" id="fig-cora">
<div class="viz-head"><span class="viz-label">চিত্র ১১ // Cora-তে গ্রাফের মূল্য কতটা</span></div>
<div class="viz-scroll">
<svg viewBox="0 0 760 262" role="img" aria-labelledby="fig-cora-t fig-cora-d">
<title id="fig-cora-t">১৪০টি লেবেল করা পেপার দিয়ে Cora-তে টেস্ট অ্যাকুরেসি</title>
<desc id="fig-cora-d">Cora সাইটেশন গ্রাফে, স্ট্যান্ডার্ড স্প্লিটে টেস্ট অ্যাকুরেসি। শুধু শব্দের ওপর MLP: 55.1%। শুধু সাইটেশনের ওপর লেবেল প্রোপাগেশন: 68.0%। দুটো মিলিয়ে GCN: 81.5%। দুটো মিলিয়ে GAT: 83.0%, ± 0.7। এই পোস্টের GCN কোড: 81.7%, 10টি সিডের গড়।</desc>
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
<g class="row" tabindex="0" data-tip="টেস্ট অ্যাকুরেসি 55.1%" data-tip-sub="MLP: GAT পেপার, টেবিল ২" aria-label="MLP, টেস্ট অ্যাকুরেসি 55.1%"><rect class="hit" x="12" y="26" width="736" height="36"/><text class="t" x="20" y="42">MLP</text><text class="t2" x="20" y="58">শুধু শব্দ, গ্রাফ নেই</text><rect class="hn a-grow" x="300" y="34" width="233.6" height="20" rx="3" style="--d:0.15s"/><text class="axis" x="541.6" y="49" style="fill:var(--text-primary)">55.1%</text></g>
<g class="row" tabindex="0" data-tip="টেস্ট অ্যাকুরেসি 68.0%" data-tip-sub="লেবেল প্রোপাগেশন: GCN পেপার, টেবিল ২" aria-label="লেবেল প্রোপাগেশন, টেস্ট অ্যাকুরেসি 68.0%"><rect class="hit" x="12" y="66" width="736" height="36"/><text class="t" x="20" y="82">লেবেল প্রোপাগেশন</text><text class="t2" x="20" y="98">শুধু সাইটেশন, শব্দ নেই</text><rect class="hn a-grow" x="300" y="74" width="288.3" height="20" rx="3" style="--d:0.33s"/><text class="axis" x="596.3" y="89" style="fill:var(--text-primary)">68.0%</text></g>
<g class="row" tabindex="0" data-tip="টেস্ট অ্যাকুরেসি 81.5%" data-tip-sub="GCN: Kipf &amp; Welling, 2017" aria-label="GCN, টেস্ট অ্যাকুরেসি 81.5%"><rect class="hit" x="12" y="106" width="736" height="36"/><text class="t" x="20" y="122">GCN</text><text class="t2" x="20" y="138">শব্দ + সাইটেশন</text><rect class="s1 a-grow" x="300" y="114" width="345.6" height="20" rx="3" style="--d:0.51s"/><text class="axis" x="653.6" y="129" style="fill:var(--text-primary)">81.5%</text></g>
<g class="row" tabindex="0" data-tip="টেস্ট অ্যাকুরেসি 83.0% ± 0.7" data-tip-sub="GAT: Veličković et al., 2018, ১০০টি রানের ± std" aria-label="GAT, টেস্ট অ্যাকুরেসি 83.0% ± 0.7"><rect class="hit" x="12" y="146" width="736" height="36"/><text class="t" x="20" y="162">GAT</text><text class="t2" x="20" y="178">শব্দ + সাইটেশন, শেখা ওয়েট</text><rect class="s1 a-grow" x="300" y="154" width="351.9" height="20" rx="3" style="--d:0.69s"/><text class="axis" x="659.9" y="169" style="fill:var(--text-primary)">83.0%</text></g>
<g class="row" tabindex="0" data-tip="টেস্ট অ্যাকুরেসি 81.7% ± 0.7" data-tip-sub="GCN, এই পোস্টের কোড: ১০টি সিডে std 0.7" aria-label="GCN, এই পোস্টের কোড, টেস্ট অ্যাকুরেসি 81.7% ± 0.7"><rect class="hit" x="12" y="186" width="736" height="36"/><text class="t" x="20" y="202">GCN, এই পোস্টের কোড</text><text class="t2" x="20" y="218">CPU-তে ১০টি সিডের গড়</text><rect class="s2 a-grow" x="300" y="194" width="346.3" height="20" rx="3" style="--d:0.87s"/><text class="axis" x="654.3" y="209" style="fill:var(--text-primary)">81.7%</text></g>
</svg>
</div>
<figcaption>একই ১৪০টি লেবেল, ব্যবহারের চারটি উপায়। প্রকাশিত সংখ্যাগুলো GCN আর GAT পেপার থেকে নেওয়া; শেষ বারটি ওপরের কোড, দশটি ভিন্ন র‍্যান্ডম সিডে চালানো। উৎস দেখতে বারের ওপর কার্সর রাখুন।</figcaption>
</figure>
{:/nomarkdown}

শুধু শব্দ দিয়ে পাওয়া যায় প্রায় ৫৫%। শুধু সাইটেশন দিয়ে, অর্থাৎ লেবেল প্রোপাগেশনে জানা লেবেলগুলো এজ ধরে ছড়িয়ে দিয়ে, পাওয়া যায় ৬৮%। দুটিকে মেলানো GCN পৌঁছায় প্রায় ৮১%-এ। Cora প্রবলভাবে হোমোফিলিক: এর ৮১% এজ একই বিষয়ের দুটি পেপারকে যুক্ত করে, আর GNN ঠিক এই পরিস্থিতির জন্যই তৈরি।

অন্য লেয়ারগুলো চেষ্টা করতে কনভোলিউশনটা বদলে দিন। GraphSAGE সরাসরি বসিয়ে দেওয়া যায়; GAT-এর লাগে তার হেডগুলো, বেশি ড্রপআউট আর একটি ELU অ্যাক্টিভেশন:

```python
from torch_geometric.nn import GATConv, SAGEConv

# GraphSAGE: the node's own vector alongside the mean of its neighbours
self.conv1, self.conv2 = SAGEConv(in_dim, 16), SAGEConv(16, n_classes)

# GAT: 8 heads of 8 features, concatenated, then one averaged head at the output
self.conv1 = GATConv(in_dim, 8, heads=8, dropout=0.6)
self.conv2 = GATConv(8 * 8, n_classes, heads=1, concat=False, dropout=0.6)
# in forward(): dropout p=0.6 and F.elu instead of p=0.5 and F.relu; train with lr=0.005
```

একই ২০০ ইপকের লুপ আর প্রতিটির জন্য দশটি সিডে GraphSAGE পেয়েছে ৮০.৯% আর GAT ৮১.৮%। (GAT পেপারের ৮৩.০% আসে আরও লম্বা একটি রান থেকে, যা ভ্যালিডেশন সেটের ভিত্তিতে আগেভাগে থামে।) এত ছোট আর এত হোমোফিলিক একটি গ্রাফে তিনটি লেয়ারই একে অপরের প্রায় এক পয়েন্টের মধ্যে থাকে, আর এটি নিজেই একটি কাজের শিক্ষা: আসল কাজটা করে গ্রাফ, আর লেয়ার বাছাই মানুষ যতটা ভাবে ততটা জরুরি নয়।

## ১১. GNN বেছে নেওয়ার আগে
{: id="11-before-you-reach-for-a-gnn"}

সংযোগগুলো যখন এমন তথ্য বহন করে যা ফিচারে নেই, তখনই GNN শক্তিশালী। এটি আপনা থেকেই ভালো নয়, আর একই কয়েকটি ভুল বারবার ঘটে।

- **আগে হোমোফিলি যাচাই করুন।** GCN-ধরনের গড় নেওয়া ধরে নেয় প্রতিবেশীরা একই রকম। **হেটেরোফিলিক** গ্রাফে, যেখানে যুক্ত নোডগুলো সাধারণত ভিন্ন হয়, যেমন জালিয়াতরা যখন ইচ্ছে করে সাধারণ গ্রাহকদের সঙ্গে লেনদেন করে, কিংবা বিশ্ববিদ্যালয়ের ওয়েব পেজের WebKB গ্রাফে, একটি সাধারণ MLP-ও GCN-কে হারাতে পারে (Zhu et al., 2020)। মডেল ডিজাইন করার আগে মেপে নিন ট্রেনিং এজের কত অংশ একই লেবেলের দুটি নোডকে যুক্ত করে।
- **গ্রাফ উপেক্ষা করা একটি শক্ত বেসলাইনকে হারান।** ট্যাবুলার সমস্যায় হাতে বানানো কয়েকটি প্রতিবেশ-ফিচারসহ (কতগুলো লেনদেনে এই কার্ড আছে, আর তার কতগুলো জালিয়াতি ছিল) গ্রেডিয়েন্ট-বুস্টেড ট্রি হারানো কঠিন। বাড়তি জটিলতার যোগ্যতা GNN-কে অর্জন করে নিতে হয়।
- **লিংক প্রেডিকশনে এজ দিয়ে লেবেল ফাঁস করবেন না।** যেসব এজে আপনি টেস্ট করছেন, সেগুলো যদি তখনও সেই গ্রাফে থাকে যার ওপর GNN মেসেজ পাঠায়, তাহলে মডেল সরাসরি উত্তরটাই দেখে ফেলতে পারে। আগে এজগুলো ভাগ করুন, আর মেসেজ পাঠান শুধু ট্রেনিং এজের ওপর। PyTorch Geometric-এর `RandomLinkSplit` কাজটা আপনার হয়ে করে দেয়।
- **সময়ের নিয়ম মানুন।** যে গ্রাফ বাড়তে থাকে (লেনদেন, সাইটেশন, বন্ধুত্ব), সেখানে ট্রেনিংয়ের সময় একটি নোডের দেখা উচিত শুধু তার আগে থেকে থাকা এজগুলো, ঠিক যে নিয়ম আপনি যেকোনো টাইম সিরিজে মানতেন।
- **গ্রাফ মেমরিতে না ধরলে নমুনা নিন।** লাখ লাখ নোডের কোথাও গিয়ে পুরো গ্রাফে ট্রেনিং আর বাস্তবসম্মত থাকে না। `NeighborLoader` নমুনা নেওয়া কম্পিউটেশন ট্রির ওপর GraphSAGE-এর ধাঁচে ট্রেইন করে, একবারে টার্গেট নোডের একটি মিনি-ব্যাচ। এর জন্য PyG-এর দুটি ঐচ্ছিক কম্পাইল করা প্যাকেজ, `pyg-lib` বা `torch-sparse`, এর যেকোনো একটি আপনার PyTorch সংস্করণ মিলিয়ে ইনস্টল থাকতে হয়:

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

## শেষ কথা
{: id="final-thoughts"}

নোটেশন সরিয়ে দিলে গ্রাফ নিউরাল নেটওয়ার্ক একটি সাদামাটা যন্ত্র। প্রতিটি নোড তার প্রতিবেশীদের দিকে তাকায়, যা দেখে তা ক্রম-নিরপেক্ষ একটি ফাংশন দিয়ে সংক্ষেপ করে, আর সব নোডের জন্য একই ওয়েট দিয়ে নিজেকে আপডেট করে। দুই বা তিনবার এটা করলেই প্রতিটি নোডের কাছে থাকে তার প্রতিবেশের এমন একটি বর্ণনা, যাকে একটি ছোট হেড পরিণত করতে পারে একটি লেবেলে, একটি লিংক স্কোরে, কিংবা পুরো গ্রাফ সম্পর্কে একটি রায়ে।

ডিজাইনের বেশিরভাগ সিদ্ধান্তই সেই খুঁটিনাটিতে, যেগুলো এই পোস্টে ধাপে ধাপে দেখানো হলো: প্রতিটি প্রতিবেশীর গুরুত্ব কতটা (GCN, GraphSAGE নাকি GAT), যোগফল নেবেন নাকি গড় (GIN), আর প্রতিটি নোড একই রকম দেখাতে শুরু করার আগে কত রাউন্ড চালাবেন। একটিমাত্র ব্যবহারিক নিয়ম মনে রাখতে হলে সেটা হোক এই: GCN বা GraphSAGE-এর দুটি লেয়ার দিয়ে শুরু করুন, পাশে রাখুন গ্রাফ উপেক্ষা করা একটি শক্ত বেসলাইন, আর দুটির ব্যবধানই আপনাকে বলে দেবে সংযোগগুলোর মূল্য কতটা।

## তথ্যসূত্র
{: id="references"}

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
- PyTorch Geometric ডকুমেন্টেশন: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
