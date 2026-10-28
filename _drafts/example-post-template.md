---
# DRAFT TEMPLATE: files in _drafts/ are never published.
# To publish a post: copy this file to _posts/YYYY-MM-DD-your-title.md,
# replace the text with your own, then commit and push.
title: "Building a multilingual RAG system: what actually mattered"
category: LLMs                 # one per post; becomes a filter on /blog/
tags: [RAG, Embeddings, Evaluation]
excerpt: "Chunking, embeddings and retrieval choices that moved answer quality the most, and the ones that didn't."
image: https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=1600&auto=format&fit=crop
image_alt: "Abstract 3D render of the letters AI"
---

The first paragraph is the hook. **Bold text** stands out in white, and [links](https://github.com/MasrukHabib) are underlined in amber.

## Each H2 becomes a table-of-contents entry

Write normal Markdown. Every `##` heading shows up in the "On this page" sidebar on wide screens.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# fenced code blocks get syntax highlighting and a copy button
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)
```

## Quotes, lists and callouts

> A blockquote renders with an amber bar.

- Bullet lists use the same arrow markers as the rest of the site
- Numbered lists work too

Put `{: .callout}` on the line under a paragraph to turn it into a tip box.
{: .callout}

## Images

Put post images in `assets/blog/` and add them like this:

```markdown
![Describe the image](/assets/blog/your-image.webp)
```
