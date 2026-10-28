/* ============================================================
   Masruk Habib | Blog Scripts
   ============================================================ */
(() => {
  "use strict";

  /* ---------- Listing: category filter + search ---------- */
  const grid = document.getElementById("post-grid");
  if (grid) {
    const cards = [...grid.querySelectorAll(".post-card")];
    const buttons = [...document.querySelectorAll("#filters button")];
    const search = document.getElementById("post-search");
    const featured = document.getElementById("featured");
    const noMatch = document.getElementById("no-match");
    let filter = "all";

    const apply = () => {
      const q = search.value.trim().toLowerCase();
      const filtering = filter !== "all" || q !== "";
      grid.classList.toggle("filtering", filtering);
      featured?.classList.toggle("is-hidden", filtering);
      let shown = 0;
      cards.forEach((c) => {
        const ok = (filter === "all" || c.dataset.cat === filter) && c.textContent.toLowerCase().includes(q);
        c.hidden = !ok;
        // reveal-on-scroll may not have fired for cards that were hidden
        if (ok) c.classList.add("visible");
        if (ok && (filtering || !c.classList.contains("is-featured"))) shown++;
      });
      if (noMatch) noMatch.hidden = shown > 0 || !filtering;
    };

    buttons.forEach((b) =>
      b.addEventListener("click", () => {
        buttons.forEach((x) => x.classList.toggle("active", x === b));
        filter = b.dataset.f;
        apply();
      })
    );
    search?.addEventListener("input", apply);
  }

  /* ---------- Article: code block header + copy ---------- */
  document.querySelectorAll(".prose div.highlighter-rouge").forEach((block) => {
    const lang = [...block.classList].find((c) => c.startsWith("language-"))?.slice(9) || "code";
    const head = document.createElement("div");
    head.className = "code-head";
    head.innerHTML = '<div class="dots"><span></span><span></span><span></span></div><button type="button" class="code-copy"></button>';
    const btn = head.querySelector("button");
    btn.textContent = lang === "plaintext" ? "copy" : lang + " · copy";
    btn.addEventListener("click", () => {
      navigator.clipboard?.writeText(block.querySelector("code").innerText).then(() => {
        const prev = btn.textContent;
        btn.textContent = "copied";
        setTimeout(() => (btn.textContent = prev), 1500);
      });
    });
    block.prepend(head);
  });

  /* ---------- Article: copy link ---------- */
  document.querySelector(".copy-link")?.addEventListener("click", (e) => {
    const btn = e.currentTarget;
    navigator.clipboard?.writeText(location.href).then(() => {
      btn.classList.add("copied");
      setTimeout(() => btn.classList.remove("copied"), 1500);
    });
  });

  /* ---------- Article: table of contents ---------- */
  const toc = document.getElementById("toc");
  const headings = [...document.querySelectorAll(".prose h2[id]")];
  if (toc && headings.length >= 2) {
    const links = headings.map((h) => {
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      toc.appendChild(a);
      return a;
    });
    toc.hidden = false;
    const spy = () => {
      let i = 0;
      headings.forEach((h, n) => { if (h.getBoundingClientRect().top < 140) i = n; });
      links.forEach((a, n) => a.classList.toggle("on", n === i));
    };
    window.addEventListener("scroll", spy, { passive: true });
    spy();
  }
})();
