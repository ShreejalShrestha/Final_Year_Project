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

  // Forms carrying data-confirm ask before a destructive action, via the
  // shared <dialog> modal rather than the browser's native confirm().
  var confirmDialog = document.getElementById("confirm-dialog");
  var confirmSupported = !!(confirmDialog && confirmDialog.showModal);
  var confirmText = document.getElementById("confirm-dialog-text");
  var confirmOk = document.getElementById("confirm-dialog-ok");
  var pendingForm = null;

  document.addEventListener("submit", function (e) {
    var msg = e.target.getAttribute && e.target.getAttribute("data-confirm");
    if (!msg) return;
    if (!confirmSupported) {
      if (!window.confirm(msg)) e.preventDefault();
      return;
    }
    e.preventDefault();
    pendingForm = e.target;
    confirmText.textContent = msg;
    confirmDialog.showModal();
  });
  if (confirmSupported) {
    confirmOk.addEventListener("click", function () {
      var form = pendingForm;
      pendingForm = null;
      confirmDialog.close();
      if (form) {
        form.removeAttribute("data-confirm");
        form.requestSubmit();
      }
    });
    confirmDialog.addEventListener("close", function () { pendingForm = null; });
  }

  // Toasts: auto-dismiss informational ones, let any of them be closed by hand.
  document.querySelectorAll("[data-toast]").forEach(function (toast) {
    var remove = function () {
      toast.classList.add("is-leaving");
      setTimeout(function () { toast.remove(); }, 160);
    };
    var closeBtn = toast.querySelector("[data-toast-close]");
    if (closeBtn) closeBtn.addEventListener("click", remove);
    if (toast.hasAttribute("data-toast-autodismiss")) {
      var timer = setTimeout(remove, 5000);
      toast.addEventListener("mouseenter", function () { clearTimeout(timer); });
      toast.addEventListener("mouseleave", function () { timer = setTimeout(remove, 2500); });
    }
  });
})();
