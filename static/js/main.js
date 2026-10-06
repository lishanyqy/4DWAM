(function () {
  'use strict';

  /* ---------- Interactive hotspots on the robot platform ---------- */
  var wrap = document.getElementById('hotspotWrap');
  var hotspots = wrap ? Array.prototype.slice.call(wrap.querySelectorAll('.hotspot')) : [];
  var legendBtns = Array.prototype.slice.call(document.querySelectorAll('#legend button'));
  var taskCards = Array.prototype.slice.call(document.querySelectorAll('.task[data-task]'));

  function clearActive() {
    hotspots.forEach(function (h) { h.classList.remove('active'); });
  }
  function highlightTask(task) {
    hotspots.forEach(function (h) {
      var mine = h.getAttribute('data-task') === task;
      h.classList.toggle('dim', !!task && !mine);
      h.classList.toggle('hl', !!task && mine);
    });
    legendBtns.forEach(function (b) { b.classList.toggle('active', b.getAttribute('data-task') === task); });
  }

  // Compact label shown when a whole task is highlighted from the legend.
  hotspots.forEach(function (h) {
    var obj = h.querySelector('.tip .obj');
    if (!obj) return;
    var l = document.createElement('span'); l.className = 'lbl'; l.textContent = obj.textContent;
    h.appendChild(l);
  });

  // Tap / click toggles a tooltip (touch devices have no hover).
  hotspots.forEach(function (h) {
    var dot = h.querySelector('.dot');
    dot.addEventListener('click', function (e) {
      e.stopPropagation();
      var on = h.classList.contains('active');
      clearActive();
      highlightTask(null);
      if (!on) h.classList.add('active');
    });
    dot.addEventListener('focus', function () { h.classList.add('active'); });
    dot.addEventListener('blur', function () { h.classList.remove('active'); });
  });
  document.addEventListener('click', function (e) {
    if (!e.target.closest || !e.target.closest('.hotspot')) { clearActive(); highlightTask(null); }
  });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { clearActive(); highlightTask(null); } });

  // Legend buttons highlight all markers belonging to a task.
  legendBtns.forEach(function (b) {
    var t = b.getAttribute('data-task');
    b.addEventListener('mouseenter', function () { highlightTask(t); });
    b.addEventListener('mouseleave', function () { highlightTask(null); });
    b.addEventListener('focus', function () { highlightTask(t); });
    b.addEventListener('blur', function () { highlightTask(null); });
    b.addEventListener('click', function (e) {
      e.stopPropagation();
      var card = document.querySelector('.task[data-task="' + t + '"]');
      if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });

  // Keep tooltips inside the figure: flip left/right near the edges on narrow screens.
  function placeTips() {
    if (!wrap) return;
    var rect = wrap.getBoundingClientRect();
    hotspots.forEach(function (h) {
      var x = parseFloat(h.style.left) / 100 * rect.width;
      h.classList.remove('left', 'right');
      if (x < 140) h.classList.add('left');
      else if (rect.width - x < 140) h.classList.add('right');
    });
  }
  placeTips();
  window.addEventListener('resize', placeTips);

  /* ---------- Lazy-load and auto-play task videos when visible ---------- */
  var lazyVideos = Array.prototype.slice.call(document.querySelectorAll('video[data-src]'));
  function loadVideo(v) {
    if (v.getAttribute('src')) return;
    v.setAttribute('src', v.getAttribute('data-src'));
    v.load();
  }
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target;
        if (en.isIntersecting) {
          loadVideo(v);
          var p = v.play(); if (p && p.catch) p.catch(function () {});
        } else if (!v.paused) {
          v.pause();
        }
      });
    }, { rootMargin: '200px 0px', threshold: 0.1 });
    lazyVideos.forEach(function (v) { io.observe(v); });
  } else {
    lazyVideos.forEach(function (v) { loadVideo(v); v.play(); });
  }

  // Hovering a task card highlights its objects on the platform figure.
  taskCards.forEach(function (c) {
    var t = c.getAttribute('data-task');
    c.addEventListener('mouseenter', function () { highlightTask(t); });
    c.addEventListener('mouseleave', function () { highlightTask(null); });
  });

  /* ---------- Copy BibTeX ---------- */
  var copyBtn = document.getElementById('copyBib');
  var bib = document.getElementById('bibText');
  if (copyBtn && bib) {
    copyBtn.addEventListener('click', function () {
      var text = bib.textContent;
      var done = function () { copyBtn.textContent = 'Copied!'; setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(text); done(); });
      } else { fallback(text); done(); }
    });
  }
  function fallback(text) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }
})();
