(function () {
  var header = document.querySelector(".site-header");
  function updateHeader() {
    if (!header) return;
    header.classList.toggle("is-scrolled", window.scrollY > 24);
  }

  window.addEventListener("scroll", updateHeader, { passive: true });
  updateHeader();

  var demo = document.querySelector(".embedded-demo");
  if (demo) {
    var playDemo = function () {
      demo.play().catch(function () {});
    };
    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(function (entries) {
        if (entries.some(function (entry) { return entry.isIntersecting; })) {
          playDemo();
        }
      }, { threshold: 0.25 });
      observer.observe(demo);
    }
    demo.addEventListener("canplay", playDemo, { once: true });
  }
})();
