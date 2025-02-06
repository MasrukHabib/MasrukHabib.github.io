document.addEventListener("DOMContentLoaded", () => {
  const sections = document.querySelectorAll(".section");
  const navLinks = document.querySelectorAll("nav a");
  const header = document.querySelector("header");
  const offset = header.offsetHeight;
  const upToNavBtn = document.getElementById("up-to-nav");
  // Show the button when the user scrolls down a bit
  window.addEventListener("scroll", () => {
    if (window.scrollY > 300) {
      // You can adjust the scroll threshold
      upToNavBtn.style.display = "block"; // Show button
    } else {
      upToNavBtn.style.display = "none"; // Hide button
    }
  });
  // Scroll to the top when the button is clicked
  upToNavBtn.addEventListener("click", () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });
  // Adjust content padding to avoid overlap
  // document.body.style.paddingTop = `${offset}px`;
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
      // Smooth scroll to section with offset
      window.scrollTo({
        top: targetSection.offsetTop - offset,
        behavior: "smooth",
      });
    });
  });
  // Ensure all sections are visible when scrolled
  sections.forEach((section) => {
    section.style.display = "block";
    section.style.opacity = "1";
    section.style.transform = "translateY(0)";
  });
});
