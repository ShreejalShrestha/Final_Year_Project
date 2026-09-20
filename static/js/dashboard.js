(function () {
  const SID = window.SESSION_ID;
  const feed = document.getElementById("feed");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const fpsEl = document.getElementById("fps");
  const stateEl = document.getElementById("pipeline-state");
  const errEl = document.getElementById("pipeline-error");
  const panelBody = document.getElementById("panel-body");

  let tracks = [];
  let selectedStudent = null;

  function csrf() {
    const m = document.cookie.match(/csrftoken=([^;]+)/);
    return m ? m[1] : "";
  }
  const post = (url) =>
    fetch(url, { method: "POST", headers: { "X-CSRFToken": csrf() } }).then((r) => r.json());

  document.getElementById("btn-start").onclick = () => {
    stateEl.textContent = "pipeline: starting…";
    post(`/dashboard/session/${SID}/start/`).then((d) => {
      if (d.error) errEl.textContent = d.error;
      startFeed();
    });
  };
  document.getElementById("btn-stop").onclick = () => {
    post(`/dashboard/session/${SID}/stop/`).then(() => {
      stateEl.textContent = "pipeline: stopped";
      feed.src = "";
    });
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

  function draw() {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const { sx, sy } = scale();
    tracks.forEach((t) => {
      const [x1, y1, x2, y2] = t.bbox;
      const known = !!t.student_name;
      ctx.lineWidth = t.track_id === (selectedStudent && selectedStudent.track_id) ? 3 : 2;
      ctx.strokeStyle = known ? "#1a9c5b" : "#d98324";
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      const label =
        `#${t.track_id} ` + (known ? t.student_name : "Unknown") +
        (t.attendance_status === "present" ? " ✓" : "");
      ctx.font = "13px system-ui";
      ctx.fillStyle = known ? "#1a9c5b" : "#d98324";
      ctx.fillText(label, x1 * sx, y1 * sy - 4);
      ctx.fillStyle = "#fff";
      ctx.fillText(
        (t.indicator_is_event ? "! " : "") +
          (t.indicator === "possible_drowsiness" ? "prolonged eye closure" :
            (t.indicator || "uncertain").replace(/_/g, " ")),
        x1 * sx,
        y2 * sy + 14
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
    panelBody.innerHTML = "<p class='hint'>Loading…</p>";
    fetch(`/dashboard/session/${SID}/student/${studentId}/`)
      .then((r) => r.json())
      .then(renderProfile);
  }

  function row(k, v) {
    return `<div class="row"><span>${k}</span><strong>${v}</strong></div>`;
  }

  function renderProfile(p) {
    const hist = p.attendance.history
      .map((h) => `<li>${h.date} — ${h.class}: <span class="pill">${h.status}</span></li>`)
      .join("");
    const events = p.monitoring.recent_events
      .map(
        (ev) =>
          `<li>${ev.start} — ${ev.type} (${ev.duration_s}s)${ev.open ? " …ongoing" : ""}</li>`
      )
      .join("") || "<li class='hint'>None this session</li>";
    panelBody.innerHTML = `
      <h2>${p.identity.name}</h2>
      ${p.identity.photo ? `<img src="${p.identity.photo}" style="width:100%;border-radius:8px;margin-bottom:10px">` : ""}
      ${row("Student ID", p.identity.student_id)}
      ${row("Program", p.academic.program || "—")}
      ${row("Semester / section", p.academic.semester_section || "—")}
      ${row("Today", p.attendance.today_status + (p.attendance.recognition_time ? " @ " + p.attendance.recognition_time : ""))}
      ${row("Attendance %", p.attendance.percentage + "%")}
      <h2 style="margin-top:14px">Recent attendance</h2>
      <ul>${hist || "<li class='hint'>No history</li>"}</ul>
      <h2>Recent monitoring events</h2>
      <ul>${events}</ul>
      <p class="hint">Excluded by default: ${p.excluded_by_default.join(", ")}.</p>
    `;
  }

  function poll() {
    fetch(`/dashboard/session/${SID}/tracks/`)
      .then((r) => r.json())
      .then((d) => {
        tracks = d.tracks || [];
        fpsEl.textContent = "fps: " + (d.fps || 0).toFixed(1);
        stateEl.textContent = "pipeline: " + (d.running ? "running" : "stopped");
        errEl.textContent = d.error || "";
        draw();
        if (selectedStudent) {
          const still = tracks.find((t) => t.student_id === selectedStudent.id);
          if (!still) { /* keep last profile visible */ }
        }
      })
      .catch(() => {});
  }

  setInterval(poll, 600);
  poll();
  if (window.WORKER_RUNNING) startFeed();
})();
