(function () {
  const SID = window.SESSION_ID;
  const stage = document.getElementById("stage");
  const feed = document.getElementById("feed");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const fpsEl = document.getElementById("fps");
  const stateEl = document.getElementById("pipeline-state");
  const errEl = document.getElementById("pipeline-error");
  const panelBody = document.getElementById("panel-body");
  const countView = document.getElementById("count-view");
  const countKnown = document.getElementById("count-known");
  const countUnknown = document.getElementById("count-unknown");
  const FONT = '600 12.5px Inter, system-ui, sans-serif';

  let tracks = [];
  let selectedStudent = null;
  let startingSince = 0;

  // User-supplied text (names, programs) must never reach innerHTML unescaped.
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  function csrf() {
    const m = document.cookie.match(/csrftoken=([^;]+)/);
    return m ? m[1] : "";
  }
  const post = (url) =>
    fetch(url, { method: "POST", headers: { "X-CSRFToken": csrf() } }).then((r) => r.json());

  function setState(state, label) {
    stateEl.dataset.state = state;
    stateEl.textContent = label;
  }
  function setError(message) {
    errEl.textContent = message || "";
    errEl.hidden = !message;
  }
  function setCounts(list) {
    const known = list.filter((t) => t.student_name).length;
    countView.textContent = list.length;
    countKnown.textContent = known;
    countUnknown.textContent = list.length - known;
  }

  document.getElementById("btn-start").onclick = () => {
    setState("starting", "Starting camera…");
    startingSince = Date.now();
    post(`/dashboard/session/${SID}/start/`)
      .then((d) => {
        setError(d.error);
        startFeed();
      })
      .catch(() => {
        setState("stopped", "Camera stopped");
        setError("Could not reach the server to start the camera.");
      });
  };
  document.getElementById("btn-stop").onclick = () => {
    post(`/dashboard/session/${SID}/stop/`)
      .then(() => {
        setState("stopped", "Camera stopped");
        feed.removeAttribute("src");
        stage.classList.remove("is-live");
        tracks = [];
        setCounts(tracks);
        draw();
      })
      .catch(() => setError("Could not reach the server to stop the camera."));
  };

  function startFeed() {
    feed.src = `/dashboard/session/${SID}/feed/?t=` + Date.now();
  }

  function resizeCanvas() {
    const w = feed.clientWidth, h = feed.clientHeight;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    // Pin the canvas to the video's exact box so 1 canvas px = 1 screen px.
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
  }
  window.addEventListener("resize", resizeCanvas);
  feed.addEventListener("load", resizeCanvas);

  function scale() {
    const nw = feed.naturalWidth || feed.clientWidth;
    const nh = feed.naturalHeight || feed.clientHeight;
    return { sx: feed.clientWidth / nw, sy: feed.clientHeight / nh };
  }

  const INDICATOR_LABELS = {
    possible_drowsiness: "prolonged eye closure",
    left_seat: "prolonged absence",
  };

  // Text on a translucent chip so labels stay readable on any video background.
  function tag(text, x, y, fg, bg) {
    ctx.font = FONT;
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = bg;
    ctx.fillRect(x, y - 15, w, 20);
    ctx.fillStyle = fg;
    ctx.fillText(text, x + 5, y);
  }

  function draw() {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const { sx, sy } = scale();
    tracks.forEach((t) => {
      const [x1, y1, x2, y2] = t.bbox;
      const known = !!t.student_name;
      const colour = known ? "#22c55e" : "#f59e0b";
      ctx.lineWidth = t.track_id === (selectedStudent && selectedStudent.track_id) ? 3 : 2;
      ctx.strokeStyle = colour;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      const label =
        `#${t.track_id} ` + (known ? t.student_name : "Unknown") +
        (t.attendance_status === "present" ? " ✓" : "");
      tag(label, x1 * sx, Math.max(y1 * sy - 4, 16), "#0b0d1f", colour);
      tag(
        (t.indicator_is_event ? "! " : "") +
          (INDICATOR_LABELS[t.indicator] || (t.indicator || "uncertain").replace(/_/g, " ")),
        x1 * sx,
        y2 * sy + 18,
        "#fff",
        "rgba(11, 13, 31, 0.75)"
      );
    });
  }

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const { sx, sy } = scale();
    for (const t of tracks) {
      const [x1, y1, x2, y2] = t.bbox;
      if (px >= x1 * sx && px <= x2 * sx && py >= y1 * sy && py <= y2 * sy) {
        if (t.student_id) {
          selectedStudent = { track_id: t.track_id, id: t.student_id };
          loadProfile(t.student_id);
        }
        return;
      }
    }
  });

  function loadProfile(studentId) {
    panelBody.innerHTML = "<p class='fine'>Loading student details…</p>";
    fetch(`/dashboard/session/${SID}/student/${studentId}/`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then(renderProfile)
      .catch(() => {
        panelBody.innerHTML =
          "<div class='alert alert--error'><p>Student details could not be loaded. Try selecting the student again.</p></div>";
      });
  }

  function row(k, v) {
    return `<div class="kv__row"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`;
  }
  const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

  function renderProfile(p) {
    const hist = p.attendance.history
      .map(
        (h) =>
          `<li><span>${esc(h.date)} · ${esc(h.class)}</span><span class="badge badge--plain">${esc(cap(h.status))}</span></li>`
      )
      .join("");
    const events =
      p.monitoring.recent_events
        .map(
          (ev) =>
            `<li><span>${esc(ev.type)} (${esc(ev.duration_s)}s)${ev.open ? " · ongoing" : ""}</span><span class="time">${esc(ev.start)}</span></li>`
        )
        .join("") || "<li class='fine'>None this session</li>";
    const initial = esc((p.identity.name || "?").charAt(0).toUpperCase());
    const today =
      cap(p.attendance.today_status) +
      (p.attendance.recognition_time ? " · " + p.attendance.recognition_time : "");
    panelBody.innerHTML = `
      <div class="sidepanel__who">
        <span class="avatar avatar--lg">${p.identity.photo ? `<img src="${esc(p.identity.photo)}" alt="">` : initial}</span>
        <div><h3 class="sidepanel__name">${esc(p.identity.name)}</h3><p class="fine num">${esc(p.identity.student_id)}</p></div>
      </div>
      <dl class="kv">
        ${row("Program", p.academic.program || "—")}
        ${row("Semester / section", p.academic.semester_section || "—")}
        ${row("This session", today)}
        ${row("Attendance rate", p.attendance.percentage + "%")}
      </dl>
      <div><h3>Recent attendance</h3><ul class="mini-list">${hist || "<li class='fine'>No history</li>"}</ul></div>
      <div><h3>Recent monitoring events</h3><ul class="mini-list">${events}</ul></div>
      <p class="fine">Excluded by default: ${esc(p.excluded_by_default.join(", "))}.</p>
    `;
  }

  function poll() {
    if (document.hidden) return Promise.resolve();   // nothing to draw in a background tab
    return fetch(`/dashboard/session/${SID}/tracks/`)
      .then((r) => r.json())
      .then((d) => {
        tracks = d.tracks || [];
        fpsEl.textContent = (d.fps || 0).toFixed(1) + " fps";
        if (d.running) setState("running", "Camera running");
        else if (stateEl.dataset.state !== "starting" || Date.now() - startingSince > 8000) {
          setState("stopped", "Camera stopped");
        }
        setError(d.error);
        stage.classList.toggle("is-live", !!(d.running && feed.getAttribute("src")));
        setCounts(tracks);
        draw();
      })
      .catch(() => {});
  }

  // Chain polls: the next request starts only after the previous one finished, so slow
  // responses can't pile up. At ~10 Hz the overlay stays close to the 30 fps video; the old
  // 600 ms interval made the boxes visibly lag behind the picture.
  const POLL_MS = 100;
  (function loop() {
    poll().finally(() => setTimeout(loop, POLL_MS));
  })();
  if (window.WORKER_RUNNING) startFeed();
})();
