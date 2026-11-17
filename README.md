<img width="2064" height="512" alt="jim cover" src="https://github.com/user-attachments/assets/bef7dba7-81de-4b58-9c15-1467713be4ef" />

Personal site of Masruk Habib, built with Jekyll and published by GitHub Pages at https://masrukhabib.github.io.

## Design

The site follows the Dark Luxury system described in `DESIGN.md` (tokens, typography, components, motion).
The matching Claude Code skill lives in `.claude/skills/dark-luxury-design/`.

- Styles: `css/style.css` (whole site) and `css/blog.css` (blog listing and articles)
- Icons: inline SVG from Lucide. Use `{% include icon.html name="brain" %}` (optional `px=16`, default 18; don't name it `size`, which Liquid reads as the parameter count); add new icons to `_data/icons.yml`
- Layout: `_layouts/default.html` with `_includes/nav.html` and `_includes/footer.html`

## Writing a blog post

1. Copy `_drafts/example-post-template.md` to `_posts/YYYY-MM-DD-your-title.md`.
2. Fill in the front matter (`title`, `category`, `tags`, `excerpt`, `image`) and write the post in Markdown. For equations, add `math: true` and write LaTeX between `$$` signs (inline or on their own lines). Use `\mid` instead of a bare `|`, which kramdown reads as a table.
3. Commit and push. GitHub Pages builds it, adds it to `/blog/`, the homepage "Notes from the lab" section and the RSS feed.

Files in `_drafts/` are never published.

## English / বাংলা

The Writing section has two languages. A switch in the blog nav moves between them, and the reader's choice is remembered for the listing.

- English posts live in `_posts/` and publish at `/blog/<slug>/`.
- Bengali posts live in `_posts/bn/` and publish at `/bn/blog/<slug>/`. The listing is `/bn/blog/`.
- A translation **uses the same filename** as its English post (for example `_posts/2026-10-28-your-journey-into-machine-learning.md` and `_posts/bn/2026-10-28-your-journey-into-machine-learning.md`). That pairing is what links the two in the switch, so don't rename one without the other. A post with no twin shows the other language greyed out.
- Don't set `lang` or `permalink` in a Bengali post. `_config.yml` sets both for everything in `_posts/bn/`.
- Give each Bengali heading the English heading's id, on the line directly under it:

  ```markdown
  ## ১. মেশিন লার্নিং আসলে কী
  {: id="1-understand-what-machine-learning-really-is"}
  ```

  Then a link to a section works in both languages, and switching language keeps the reader's place. Use exactly this `{: id="..."}` form: kramdown silently ignores `{#id}` and `{: #id}` when the id starts with a digit. The English id is the heading in lowercase, punctuation dropped, spaces turned into hyphens (or copy it from the English page's URL when you click a section in its table of contents).
- UI text (nav, dates, "min read", buttons, footer) comes from `_data/i18n.yml`. The home page shows English posts only.
