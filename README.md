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
2. Fill in the front matter (`title`, `category`, `tags`, `excerpt`, `image`) and write the post in Markdown.
3. Commit and push. GitHub Pages builds it, adds it to `/blog/`, the homepage "Notes from the lab" section and the RSS feed.

Files in `_drafts/` are never published.
