/* Shell behaviour: off-canvas navigation on small screens and confirm prompts. */
(function () {
  "use strict";
  var app = document.querySelector(".app");
  var toggle = document.getElementById("nav-toggle");
  var sidebar = document.getElementById("sidebar");
  var scrim = document.querySelector(".scrim");

  if (app && toggle && sidebar && scrim) {
    var setOpen = function (open) {
      app.classList.toggle("nav-open", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      scrim.hidden = !open;
      if (open) {
        var first = sidebar.querySelector("a, button");
        if (first) first.focus();
      } else if (document.activeElement && sidebar.contains(document.activeElement)) {
        toggle.focus();
      }
    };
    toggle.addEventListener("click", function () {
      setOpen(!app.classList.contains("nav-open"));
    });
    document.querySelectorAll("[data-nav-close]").forEach(function (el) {
      el.addEventListener("click", function () { setOpen(false); });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && app.classList.contains("nav-open")) setOpen(false);
    });
    window.matchMedia("(min-width: 1024px)").addEventListener("change", function (m) {
      if (m.matches) setOpen(false);
    });
  }

  // Forms carrying data-confirm ask before a destructive action.
  document.addEventListener("submit", function (e) {
    var msg = e.target.getAttribute && e.target.getAttribute("data-confirm");
    if (msg && !window.confirm(msg)) e.preventDefault();
  });
})();
