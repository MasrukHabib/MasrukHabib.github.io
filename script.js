document.addEventListener("DOMContentLoaded", () => {
  // Show all sections by default
  const sections = document.querySelectorAll(".content-section");
  sections.forEach((section) => {
    section.style.display = "block";
    section.style.opacity = "1";
  });

  // Add active state to the first navigation button
  updateNavigationState("about");

  // Add event listeners to navigation buttons
  const buttons = document.querySelectorAll("nav button");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const sectionId = button.getAttribute("data-section-id");
      showSection(sectionId);
    });
  });
});

// Function to show selected section
function showSection(sectionId) {
  const selectedSection = document.getElementById(sectionId);
  if (selectedSection) {
    selectedSection.scrollIntoView({ behavior: "smooth" });
    updateNavigationState(sectionId);
  }
}

// Function to update navigation button states
function updateNavigationState(sectionId) {
  const buttons = document.querySelectorAll("nav button");
  buttons.forEach((button) => {
    // Remove active class from all buttons
    button.classList.remove("active");

    // Add active class to current section button
    if (button.getAttribute("data-section-id") === sectionId) {
      button.classList.add("active");
    }
  });
}

// Add CSS styles for animations
const style = document.createElement("style");
style.textContent = `
     .content-section {
         opacity: 1;
         transition: opacity 0.3s ease-in-out;
     }
     
     nav button.active {
         background: #2c3e50 !important;
         color: white !important;
     }
 `;
document.head.appendChild(style);

// Optional: Add keyboard navigation
document.addEventListener("keydown", (e) => {
  const sections = [
    "about",
    "education",
    "skills",
    "experience",
    "projects",
    "publications",
    "contact",
  ];
  const currentSection = document.querySelector(
    '.content-section[style*="display: block"]'
  );
  const currentIndex = sections.indexOf(currentSection.id);

  if (e.key === "ArrowRight" || e.key === "ArrowDown") {
    const nextIndex = (currentIndex + 1) % sections.length;
    showSection(sections[nextIndex]);
  } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
    const prevIndex = (currentIndex - 1 + sections.length) % sections.length;
    showSection(sections[prevIndex]);
  }
});

// Optional: Add scroll-based animations
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.transform = "translateY(0)";
        entry.target.style.opacity = "1";
      }
    });
  },
  {
    threshold: 0.1,
  }
);

// Apply animations to section elements
document
  .querySelectorAll(".education-entry, .project-card, .skill-category")
  .forEach((el) => {
    el.style.transition = "transform 0.5s ease-out, opacity 0.5s ease-out";
    el.style.transform = "translateY(20px)";
    el.style.opacity = "0";
    observer.observe(el);
  });
