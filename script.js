// DOM Content Loaded Event Listener
document.addEventListener("DOMContentLoaded", () => {
  // Show about section by default
  showSection("about");

  // Add active state to navigation buttons
  updateNavigationState("about");
});

// Function to show selected section and hide others
function showSection(sectionId) {
  // Hide all content sections with a fade out effect
  const sections = document.querySelectorAll(".content-section");
  sections.forEach((section) => {
    section.style.opacity = "0";
    setTimeout(() => {
      section.style.display = "none";
    }, 300);
  });

  // Show the selected section with a fade in effect
  const selectedSection = document.getElementById(sectionId);
  if (selectedSection) {
    setTimeout(() => {
      selectedSection.style.display = "block";
      setTimeout(() => {
        selectedSection.style.opacity = "1";
      }, 50);
    }, 300);
  }

  // Update navigation state
  updateNavigationState(sectionId);

  // Scroll to section smoothly
  scrollToSection(sectionId);
}

// Function to update navigation button states
function updateNavigationState(sectionId) {
  const buttons = document.querySelectorAll("nav button");
  buttons.forEach((button) => {
    // Remove active class from all buttons
    button.classList.remove("active");

    // Add active class to current section button
    if (button.getAttribute("onclick").includes(sectionId)) {
      button.classList.add("active");
    }
  });
}

// Function to scroll to section
function scrollToSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    window.scrollTo({
      top: section.offsetTop - 20,
      behavior: "smooth",
    });
  }
}

// Add CSS styles for animations
const style = document.createElement("style");
style.textContent = `
     .content-section {
         opacity: 0;
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

// Delay for fade-out
section.style.opacity = "0";
setTimeout(() => {
  section.style.display = "none";
}, 300); // Adjust this delay to control fade-out speed

// Delay for fade-in
setTimeout(() => {
  selectedSection.style.display = "block";
  setTimeout(() => {
    selectedSection.style.opacity = "1";
  }, 50); // Adjust this delay to control when the opacity starts
}, 300); // Adjust this delay to control fade-in speed
