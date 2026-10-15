// DGX Spark Metrics dashboard screenshot tool (runs inside the omp eval JS runtime;
// uses the global `browser` object. No login required - dashboard is anonymous).
//
// Usage (eval, language: js):
//   const { run } = await import("/home/kambiz/omp/dgx/dgx-dashboard-shot.mjs");
//   await run([])                    // full page (all 55 panels incl. GPU HARDWARE row)
//   await run(["--gpu-stats"])       // full page, ensures GPU HARDWARE row rendered
//   await run(["--gpu-stats-only"])  // only the "DGX SPARK GB10 GPU HARDWARE" section
//   await run(["--gpu-stats-only", "--out=/tmp/gpu.png"])  // custom output path
//
// Outputs:
//   grafana-dashboard-dgx.png          (default)
//   grafana-dashboard-dgx-gpu.png      (--gpu-stats)
//   grafana-dashboard-dgx-gpu-only.png (--gpu-stats-only)

const DASH = "http://grafana.localdomain/d/dgx-spark-metrics/dgx-spark-metrics";

export async function shot({ gpuStats = false, gpuStatsOnly = false, range = "now-15m", width = 1706, outPath } = {}) {
  outPath = outPath || (gpuStatsOnly
    ? "/home/kambiz/omp/dgx/grafana-dashboard-dgx-gpu-only.png"
    : gpuStats
      ? "/home/kambiz/omp/dgx/grafana-dashboard-dgx-gpu.png"
      : "/home/kambiz/omp/dgx/grafana-dashboard-dgx.png");

  const url = `${DASH}?from=${range}&to=now&refresh=30s`;
  const tab = await browser.open({ name: "dgx-shot", url, wait_until: "networkidle2", timeout: 90 });
  try {
    await new Promise(r => setTimeout(r, 12000)); // let panels fetch + render

    const info = await tab.run(async ({ page }, { gpuStats, gpuStatsOnly, width, outPath }) => {
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      const MARK = "DGX SPARK GB10";
      await page.setViewport({ width, height: 960, deviceScaleFactor: 1 });
      await sleep(2000);

      // Grafana scrolls inside an inner div; the document never grows, so Puppeteer
      // fullPage is useless until we expand that scroller.
      const expanded = await page.evaluate(() => {
        const sc = [...document.querySelectorAll('*')].find(
          e => e.scrollHeight > e.clientHeight + 200 && e.clientHeight > 200);
        if (!sc) return false;
        const names = ["overflow", "height", "maxHeight"]; // kludge: set via style
        sc.style.overflow = "visible";
        sc.style.height = "auto";
        sc.style.maxHeight = "none";
        document.body.style.overflow = "visible";
        document.documentElement.style.overflow = "visible";
        return { tag: sc.tagName, cls: String(sc.className).slice(0, 40), sh: sc.scrollHeight };
      });

      // GPU section = the .dashboard-row-wrapper whose header says "DGX SPARK GB10".
      // (Grafana renders each labelled row as a wrapper containing header + child panels;
      //  no data-panelid attributes exist in this DOM.)
      const section = async () => page.evaluate(mark => {
        const wrappers = [...document.querySelectorAll('.dashboard-row-wrapper')];
        const w = wrappers.find(e => (e.textContent || "").includes(mark));
        if (!w) return null;
        const r = w.getBoundingClientRect();
        return { x: 0, y: r.top + window.scrollY,
                 width: document.documentElement.scrollWidth, height: r.height };
      }, "DGX SPARK GB10");

      const gpuState = async () => {
        const sec = await section();
        return { rowFound: !!sec, rowVisible: !!sec && sec.height > 5 };
      };

      await sleep(1500); // reflow after expanding scroller

      if (gpuStatsOnly) {
        const clip = await section();
        if (!clip) throw new Error("GPU HARDWARE row not found in DOM");
        const buf = await page.screenshot({ type: "png", clip, captureBeyondViewport: true });
        await Bun.write(outPath, buf);
        return { mode: "gpu-only", bytes: buf.length, expanded, clip: { w: Math.round(clip.width), h: Math.round(clip.height) }, ...(await gpuState()) };
      }

      if (gpuStats) {
        // make sure every panel (incl. GPU row) got a render pass
        await page.evaluate(() => window.scrollTo(0, 0));
        await sleep(1000);
      }
      const buf = await page.screenshot({ fullPage: true, type: "png" });
      await Bun.write(outPath, buf);
      const dims = await page.evaluate(() => ({ w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight }));
      return { mode: gpuStats ? "full+gpu" : "full", bytes: buf.length, expanded, dims, ...(await gpuState()) };
    }, { args: [{ gpuStats, gpuStatsOnly, width, outPath }], timeout: 120 });

    console.log("captured:", JSON.stringify(info));
    return { outPath, ...info };
  } finally {
    await tab.close();
  }
}

export async function run(argv = []) {
  const gpuStats = argv.includes("--gpu-stats");
  const gpuStatsOnly = argv.includes("--gpu-stats-only");
  const out = argv.find(a => a.startsWith("--out="));
  return shot({ gpuStats, gpuStatsOnly, outPath: out ? out.slice("--out=".length) : undefined });
}
