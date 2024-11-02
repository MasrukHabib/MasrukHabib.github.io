function showSection(sectionId) {
  // Hide all content sections
  const sections = document.querySelectorAll(".content-section");
  sections.forEach((section) => (section.style.display = "none"));

  // Show the selected section
  document.getElementById(sectionId).style.display = "block";
}
