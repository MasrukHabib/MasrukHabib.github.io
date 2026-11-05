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

  /* ---------- Article: animated figures (play on scroll, replay, tooltips) ---------- */
  const figs = [...document.querySelectorAll(".prose .viz")];
  if (figs.length) {
    const motionOK = window.matchMedia("(prefers-reduced-motion: no-preference)").matches && "IntersectionObserver" in window;
    const play = (f) => {
      f.classList.remove("is-playing");
      void f.getBoundingClientRect(); // restart the CSS animations
      f.classList.add("is-playing");
    };
    const io = motionOK
      ? new IntersectionObserver((entries) => entries.forEach((en) => {
          if (en.isIntersecting) { en.target.classList.add("is-playing"); io.unobserve(en.target); }
        }), { threshold: 0.3 })
      : null;

    figs.forEach((f) => {
      if (io && f.querySelector(".a-in, .a-draw, .a-grow, .a-pop, .a-travel")) {
        f.classList.add("armed");
        io.observe(f);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "viz-replay";
        btn.textContent = "↻ replay";
        btn.setAttribute("aria-label", "Replay this animation");
        btn.addEventListener("click", () => play(f));
        f.querySelector(".viz-head")?.appendChild(btn);
      }

      // on narrow screens the figure scrolls sideways inside its card; say so
      const sc = f.querySelector(".viz-scroll");
      if (sc) {
        const hint = document.createElement("p");
        hint.className = "viz-hint";
        hint.textContent = "Swipe sideways to see the whole figure →";
        sc.before(hint);
        const check = () => f.classList.toggle("is-scrollable", sc.scrollWidth > sc.clientWidth + 4);
        check();
        window.addEventListener("resize", check, { passive: true });
      }

      const marks = f.querySelectorAll("[data-tip]");
      if (!marks.length) return;
      const tip = document.createElement("div");
      tip.className = "viz-tip";
      tip.setAttribute("aria-hidden", "true"); // marks carry their own aria-label
      f.appendChild(tip);
      const show = (m, cx, cy) => {
        const strong = document.createElement("strong");
        strong.textContent = m.dataset.tip;
        const sub = document.createElement("span");
        sub.textContent = m.dataset.tipSub || "";
        tip.replaceChildren(strong, sub);
        const fr = f.getBoundingClientRect();
        const r = m.getBoundingClientRect();
        const x = (cx ?? r.left + r.width / 2) - fr.left;
        const y = (cy ?? r.top) - fr.top;
        let left = x + 14;
        if (left + tip.offsetWidth > fr.width - 8) left = x - tip.offsetWidth - 14;
        let top = y - tip.offsetHeight - 12;
        if (top < 8) top = y + 18;
        tip.style.left = Math.max(8, left) + "px";
        tip.style.top = top + "px";
        tip.classList.add("on");
      };
      const hide = () => tip.classList.remove("on");
      marks.forEach((m) => {
        m.addEventListener("pointermove", (ev) => show(m, ev.clientX, ev.clientY));
        m.addEventListener("pointerleave", hide);
        m.addEventListener("focus", () => show(m));
        m.addEventListener("blur", hide);
      });
    });
  }
})();
