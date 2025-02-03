document.addEventListener("DOMContentLoaded", () => {
  const sections = document.querySelectorAll(".section");
  const navLinks = document.querySelectorAll("nav a");
  const homeSection = document.getElementById("home");

  // Smooth scrolling for navigation
  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetSectionId = link.getAttribute("data-section");
      const targetSection = document.getElementById(targetSectionId);

      // Remove active class from all sections and nav links
      sections.forEach((section) => section.classList.remove("active"));
      navLinks.forEach((navLink) => navLink.classList.remove("active"));

      // Add active class to target section and nav link
      targetSection.classList.add("active");
      link.classList.add("active");

      // Smooth scroll to section
      targetSection.scrollIntoView({ behavior: "smooth" });
    });
  });

  // Skills progress bar animation
  const skillBars = document.querySelectorAll(".skill-bar");
  const observerOptions = {
    threshold: 0.5,
  };

  const skillObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const progressBar = entry.target;
        const percentage = progressBar.dataset.percentage;
        progressBar.style.width = `${percentage}%`;
      }
    });
  }, observerOptions);

  skillBars.forEach((bar) => skillObserver.observe(bar));

  // Project filter functionality
  const projectFilter = document.querySelector(".project-filter");
  const projects = document.querySelectorAll(".project-card");

  if (projectFilter) {
    projectFilter.addEventListener("change", (e) => {
      const filterValue = e.target.value;

      projects.forEach((project) => {
        if (filterValue === "all" || project.classList.contains(filterValue)) {
          project.style.display = "block";
        } else {
          project.style.display = "none";
        }
      });
    });
  }

  // Dark mode toggle (optional enhancement)
  const darkModeToggle = document.getElementById("dark-mode-toggle");
  if (darkModeToggle) {
    darkModeToggle.addEventListener("click", () => {
      document.body.classList.toggle("dark-mode");
    });
  }
});

// Optional: Add scroll-based animations
const animateOnScroll = () => {
  const scrollElements = document.querySelectorAll(".scroll-animate");

  const elementInView = (el, dividend = 1) => {
    const elementTop = el.getBoundingClientRect().top;
    return (
      elementTop <=
      (window.innerHeight || document.documentElement.clientHeight) / dividend
    );
  };

  const displayScrollElement = (element) => {
    element.classList.add("scrolled");
  };

  const hideScrollElement = (element) => {
    element.classList.remove("scrolled");
  };

  const handleScrollAnimation = () => {
    scrollElements.forEach((el) => {
      if (elementInView(el, 1.25)) {
        displayScrollElement(el);
      } else {
        hideScrollElement(el);
      }
    });
  };

  window.addEventListener("scroll", () => {
    handleScrollAnimation();
  });
};

animateOnScroll();
