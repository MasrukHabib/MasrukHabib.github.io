/* ============================================================
   Masruk Habib — Portfolio Scripts
   ============================================================ */
(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Preloader ---------- */
  const preloader = document.getElementById("preloader");
  window.addEventListener("load", () => {
    setTimeout(() => preloader && preloader.classList.add("done"), reducedMotion ? 0 : 450);
  });
  // Safety: never trap the user behind the loader
  setTimeout(() => preloader && preloader.classList.add("done"), 3500);

  /* ---------- Scroll progress bar ---------- */
  const progress = document.getElementById("scroll-progress");
  const navbar = document.getElementById("navbar");
  const toTop = document.getElementById("to-top");

  const onScroll = () => {
    const doc = document.documentElement;
    const max = doc.scrollHeight - doc.clientHeight;
    const pct = max > 0 ? (window.scrollY / max) * 100 : 0;
    if (progress) progress.style.width = pct + "%";
    if (navbar) navbar.classList.toggle("scrolled", window.scrollY > 30);
    if (toTop) toTop.classList.toggle("show", window.scrollY > 600);
    updateCursorGlow();
  };
  window.addEventListener("scroll", onScroll, { passive: true });

  /* ---------- Mobile menu ---------- */
  const hamburger = document.getElementById("hamburger");
  const navLinks = document.getElementById("nav-links");

  const closeMenu = () => {
    hamburger?.classList.remove("open");
    navLinks?.classList.remove("open");
    hamburger?.setAttribute("aria-expanded", "false");
    document.body.style.overflow = "";
  };

  hamburger?.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    hamburger.classList.toggle("open", open);
    hamburger.setAttribute("aria-expanded", String(open));
    document.body.style.overflow = open ? "hidden" : "";
  });
  navLinks?.querySelectorAll("a").forEach((a) => a.addEventListener("click", closeMenu));

  /* ---------- Active nav link highlighting ---------- */
  const sections = [...document.querySelectorAll("section[id]")];
  const links = [...document.querySelectorAll(".nav-link")];
  const setActiveLink = () => {
    let currentId = "";
    for (const sec of sections) {
      if (window.scrollY >= sec.offsetTop - 140) currentId = sec.id;
    }
    links.forEach((l) =>
      l.classList.toggle("active", l.getAttribute("href") === "#" + currentId)
    );
  };
  window.addEventListener("scroll", setActiveLink, { passive: true });
  setActiveLink();

  /* ---------- Reveal on scroll ---------- */
  const revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reducedMotion) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("visible");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("visible"));
  }

  /* ---------- Animated counters ---------- */
  const counters = document.querySelectorAll(".stat-num[data-count]");
  const animateCount = (el) => {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || "";
    if (reducedMotion) { el.textContent = target + suffix; return; }
    const dur = 1600;
    const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min((t - t0) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      el.textContent = Math.round(target * eased) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  if ("IntersectionObserver" in window) {
    const cio = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) { animateCount(e.target); cio.unobserve(e.target); }
        });
      },
      { threshold: 0.5 }
    );
    counters.forEach((c) => cio.observe(c));
  } else {
    counters.forEach(animateCount);
  }

  /* ---------- Skill bars fill ---------- */
  const bars = document.querySelectorAll(".bar-fill[data-w]");
  if ("IntersectionObserver" in window) {
    const bio = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.style.width = e.target.dataset.w + "%";
            bio.unobserve(e.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    bars.forEach((b) => bio.observe(b));
  } else {
    bars.forEach((b) => (b.style.width = b.dataset.w + "%"));
  }

  /* ---------- Typing effect ---------- */
  const typedEl = document.getElementById("typed");
  if (typedEl) {
    const roles = [
      "AI Engineer",
      "Machine Learning Researcher",
      "R&D Engineer @ GoScurry.ai",
      "LLM & RAG Specialist",
      "Data Analyst",
    ];
    if (reducedMotion) {
      typedEl.textContent = roles[0];
    } else {
      let ri = 0, ci = 0, deleting = false;
      const type = () => {
        const word = roles[ri];
        ci += deleting ? -1 : 1;
        typedEl.textContent = word.slice(0, ci);
        let delay = deleting ? 38 : 78;
        if (!deleting && ci === word.length) { delay = 1900; deleting = true; }
        else if (deleting && ci === 0) { deleting = false; ri = (ri + 1) % roles.length; delay = 420; }
        setTimeout(type, delay);
      };
      type();
    }
  }

  /* ---------- Cursor glow follower ---------- */
  const glow = document.getElementById("cursor-glow");
  let gx = window.innerWidth / 2, gy = window.innerHeight / 2;
  let tx = gx, ty = gy;
  window.addEventListener("pointermove", (e) => { tx = e.clientX; ty = e.clientY; }, { passive: true });
  function updateCursorGlow() {
    if (!glow || reducedMotion) return;
    gx += (tx - gx) * 0.08;
    gy += (ty - gy) * 0.08;
    glow.style.left = gx + "px";
    glow.style.top = gy + "px";
  }
  // rAF loop only when pointer is fine
  if (window.matchMedia("(pointer: fine)").matches && !reducedMotion) {
    (function loopGlow() { updateCursorGlow(); requestAnimationFrame(loopGlow); })();
  }

  /* ---------- Card tilt (projects / avatar) ---------- */
  const canHover = window.matchMedia("(hover: hover) and (pointer: fine)").matches && !reducedMotion;
  if (canHover) {
    document.querySelectorAll("[data-tilt]").forEach((card) => {
      const strength = 9;
      card.addEventListener("mousemove", (e) => {
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - 0.5;
        const py = (e.clientY - r.top) / r.height - 0.5;
        card.style.transform =
          "perspective(900px) rotateY(" + px * strength + "deg) rotateX(" + (-py * strength) + "deg)";
      });
      card.addEventListener("mouseleave", () => {
        card.style.transform = "perspective(900px) rotateY(0deg) rotateX(0deg)";
      });
    });
  }

  /* ---------- Back to top ---------- */
  toTop?.addEventListener("click", () => window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" }));

  /* ---------- Contact form -> mailto ---------- */
  const form = document.getElementById("contact-form");
  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = document.getElementById("cf-name").value.trim();
    const email = document.getElementById("cf-email").value.trim();
    const msg = document.getElementById("cf-msg").value.trim();
    const subject = encodeURIComponent("Portfolio inquiry from " + name);
    const body = encodeURIComponent(msg + "\n\n— " + name + "\n" + email);
    window.location.href = "mailto:masrukjim@gmail.com?subject=" + subject + "&body=" + body;
  });

  /* ---------- Footer year ---------- */
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* ============================================================
     Hero particle network ("neural net")
     ============================================================ */
  const canvas = document.getElementById("particle-canvas");
  if (canvas && !reducedMotion) {
    const ctx = canvas.getContext("2d");
    let w, h, dpr, particles = [], rafId;
    const mouse = { x: null, y: null };
    const LINK_DIST = 130;

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.offsetWidth; h = canvas.offsetHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.min(Math.floor((w * h) / 16000), 90);
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.6 + 0.6,
        hue: [187, 235, 293][Math.floor(Math.random() * 3)], // cyan/indigo/fuchsia
      }));
    };

    const step = () => {
      ctx.clearRect(0, 0, w, h);

      for (const p of particles) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < -20) p.x = w + 20; if (p.x > w + 20) p.x = -20;
        if (p.y < -20) p.y = h + 20; if (p.y > h + 20) p.y = -20;

        // gentle mouse attraction
        if (mouse.x !== null) {
          const dx = mouse.x - p.x, dy = mouse.y - p.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 120 * 120) { p.x += dx * 0.0012; p.y += dy * 0.0012; }
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "hsla(" + p.hue + ", 85%, 70%, 0.75)";
        ctx.fill();
      }

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const a = particles[i], b = particles[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist = Math.hypot(dx, dy);
          if (dist < LINK_DIST) {
            const alpha = (1 - dist / LINK_DIST) * 0.28;
            ctx.strokeStyle = "hsla(230, 85%, 74%, " + alpha + ")";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      rafId = requestAnimationFrame(step);
    };

    const hero = document.getElementById("home");
    hero?.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    });
    hero?.addEventListener("pointerleave", () => { mouse.x = mouse.y = null; });

    let resizeT;
    window.addEventListener("resize", () => {
      clearTimeout(resizeT);
      resizeT = setTimeout(resize, 180);
    });

    // Pause when tab hidden or hero off-screen
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(
        (entries) => {
          entries.forEach((en) => {
            if (en.isIntersecting) { cancelAnimationFrame(rafId); rafId = requestAnimationFrame(step); }
            else cancelAnimationFrame(rafId);
          });
        },
        { threshold: 0.02 }
      ).observe(hero);
    }
    if (document.hidden) {
      document.addEventListener("visibilitychange", () => {
        if (!document.hidden) { cancelAnimationFrame(rafId); rafId = requestAnimationFrame(step); }
        else cancelAnimationFrame(rafId);
      }, { once: false });
    }

    resize();
    rafId = requestAnimationFrame(step);
  }
})();
