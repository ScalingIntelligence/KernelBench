function initializeMermaid() {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: true,
    securityLevel: "loose",
    theme: "neutral",
    themeVariables: {
      fontFamily: "system-ui, sans-serif",
      fontSize: "11px",
      primaryColor: "#f8fafc",
      primaryTextColor: "#111827",
      primaryBorderColor: "#2563eb",
      lineColor: "#475569",
      clusterBkg: "#ffffff",
      clusterBorder: "#cbd5e1",
    },
    flowchart: {
      curve: "linear",
      htmlLabels: true,
      nodeSpacing: 50,
      rankSpacing: 70,
      useMaxWidth: true,
    },
  });
}

async function resolveVersionSwitcherLinks() {
  const versionLinks = document.querySelectorAll(
    ".docs-version-switcher__link[data-target-path]",
  );

  await Promise.all(
    Array.from(versionLinks, async (link) => {
      if (link.classList.contains("is-active")) {
        return;
      }

      const targetPath = link.dataset.targetPath;
      const fallbackPath = link.dataset.fallbackPath;
      if (!targetPath || !fallbackPath || targetPath === fallbackPath) {
        return;
      }

      try {
        const response = await fetch(targetPath, { method: "HEAD" });
        if (response.ok) {
          link.href = targetPath;
        }
      } catch (_error) {
        // Keep the safer version-home fallback when same-page probing fails.
      }
    }),
  );
}

window.addEventListener("DOMContentLoaded", () => {
  initializeMermaid();
  void resolveVersionSwitcherLinks();
});
