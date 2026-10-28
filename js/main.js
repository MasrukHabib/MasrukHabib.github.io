/* ============================================================
   Masruk Habib | Portfolio Scripts
   ============================================================ */
(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasIO = "IntersectionObserver" in window;

  /* ---------- Nav: blur on scroll + mobile menu ---------- */
  const nav = document.getElementById("nav");
  const toggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");

  const onScroll = () => nav?.classList.toggle("scrolled", window.scrollY > 24);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const setMenu = (open) => {
    nav.classList.toggle("menu-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    document.body.style.overflow = open ? "hidden" : "";
  };
  toggle?.addEventListener("click", () => setMenu(!nav.classList.contains("menu-open")));
  navLinks?.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && nav?.classList.contains("menu-open")) { setMenu(false); toggle.focus(); }
  });

  /* ---------- Nav: highlight the section in view ---------- */
  const sectionLinks = [...document.querySelectorAll(".nav-links a")].filter(
    (a) => a.hash && a.pathname === location.pathname && document.querySelector(a.hash)
  );
  if (sectionLinks.length && hasIO) {
    const byId = new Map(sectionLinks.map((a) => [a.hash.slice(1), a]));
    const spy = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (!e.isIntersecting) return;
        sectionLinks.forEach((a) => a.classList.remove("active"));
        byId.get(e.target.id)?.classList.add("active");
      }),
      { rootMargin: "-45% 0px -50% 0px" }
    );
    byId.forEach((_, id) => spy.observe(document.getElementById(id)));
  }

  /* ---------- Scroll reveal, staggered 90ms per sibling ---------- */
  const revealEls = [...document.querySelectorAll(".reveal")];
  if (hasIO && !reducedMotion) {
    revealEls.forEach((el) => {
      const siblings = [...el.parentElement.children].filter((c) => c.classList.contains("reveal"));
      el.dataset.delay = siblings.indexOf(el) * 90;
    });
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const el = e.target;
        setTimeout(() => {
          el.classList.add("visible");
          setTimeout(() => el.classList.remove("reveal"), 650);
        }, +el.dataset.delay || 0);
        io.unobserve(e.target);
      }),
      { threshold: 0.08 }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("visible"));
  }

  /* ---------- Number count-up ---------- */
  const counters = document.querySelectorAll("[data-countup]");
  if (hasIO && !reducedMotion) {
    const cio = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (!e.isIntersecting) return;
        cio.unobserve(e.target);
        const el = e.target;
        const target = parseFloat(el.dataset.countup);
        const t0 = performance.now();
        const tick = (now) => {
          const p = Math.min((now - t0) / 1800, 1);
          el.textContent = Math.floor((1 - Math.pow(1 - p, 3)) * target);
          if (p < 1) requestAnimationFrame(tick);
          else el.textContent = target;
        };
        el.textContent = "0";
        requestAnimationFrame(tick);
      }),
      { threshold: 0.4 }
    );
    counters.forEach((c) => cio.observe(c));
  }

  /* ---------- Hero word cycling ---------- */
  const cw = document.querySelector(".cycling-word");
  if (cw && !reducedMotion) {
    const words = cw.dataset.words.split("|");
    let idx = 0;
    setInterval(() => {
      cw.style.cssText = "opacity:0;transform:translateY(-8px);transition:opacity 300ms ease,transform 300ms ease";
      setTimeout(() => {
        idx = (idx + 1) % words.length;
        cw.textContent = words[idx];
        cw.style.cssText = "opacity:0;transform:translateY(10px);transition:none";
        requestAnimationFrame(() => requestAnimationFrame(() => {
          cw.style.cssText = "opacity:1;transform:translateY(0);transition:opacity 350ms ease,transform 350ms ease";
        }));
      }, 320);
    }, 2600);
  }

  /* ---------- Marquees: duplicate the group for a seamless loop ---------- */
  document.querySelectorAll(".marquee").forEach((m) => {
    const track = m.querySelector(".marquee-track");
    const group = track?.querySelector(".marquee-group");
    if (!group || reducedMotion) return;
    const copy = group.cloneNode(true);
    copy.setAttribute("aria-hidden", "true");
    track.appendChild(copy);
    m.classList.add("is-ready");
  });

  /* ---------- Binary texture behind the stats ---------- */
  const binary = document.querySelector(".binary-bg");
  if (binary) {
    let rows = "";
    for (let r = 0; r < 40; r++) {
      let line = "";
      for (let c = 0; c < 120; c++) line += Math.random() < 0.5 ? "0" : "1";
      rows += line + "\n";
    }
    binary.textContent = rows;
  }

  /* ---------- Contact form -> mailto ---------- */
  document.getElementById("contact-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = document.getElementById("cf-name").value.trim();
    const email = document.getElementById("cf-email").value.trim();
    const msg = document.getElementById("cf-msg").value.trim();
    const subject = encodeURIComponent("Portfolio inquiry from " + name);
    const body = encodeURIComponent(msg + "\n\n" + name + "\n" + email);
    window.location.href = "mailto:masrukjim@gmail.com?subject=" + subject + "&body=" + body;
  });

  /* ---------- Footer year ---------- */
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();
})();
