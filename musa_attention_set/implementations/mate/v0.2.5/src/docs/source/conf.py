# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Allow API docs to build on hosts without visible MUSA devices.
os.environ.setdefault("MATE_MUSA_ARCH_LIST", "3.1")

import mate  # noqa: E402,F401
from packaging.version import Version  # noqa: E402

project = "MATE"
copyright = "2020-2026, MooreThreads GPU Computing Team"
author = "MATE Contributors"
CURRENT_DOCS_RELEASE = (ROOT / "version.txt").read_text(encoding="utf-8").strip()


def normalize_docs_version_entries(
    raw_versions: list, latest_release: str
) -> list[dict[str, str]]:
    normalized_entries = []
    seen_versions = set()

    for raw_entry in raw_versions:
        if isinstance(raw_entry, str):
            version_name = raw_entry.strip()
        elif isinstance(raw_entry, dict):
            version_name = str(raw_entry.get("version", "")).strip()
        else:
            continue

        if not version_name or version_name in seen_versions:
            continue

        seen_versions.add(version_name)
        normalized_entries.append({"version": version_name})

    if latest_release and latest_release not in seen_versions:
        normalized_entries.insert(0, {"version": latest_release})

    return normalized_entries


def load_docs_versions() -> dict:
    versions_path = ROOT / "docs" / "versions.json"
    if not versions_path.exists():
        return {
            "latest": CURRENT_DOCS_RELEASE,
            "versions": [{"version": CURRENT_DOCS_RELEASE}],
        }

    with versions_path.open(encoding="utf-8") as versions_file:
        versions_data = json.load(versions_file)

    latest_release = versions_data.get("latest", CURRENT_DOCS_RELEASE)
    known_versions = versions_data.get("versions", [CURRENT_DOCS_RELEASE])
    normalized_versions = normalize_docs_version_entries(known_versions, latest_release)

    return {"latest": latest_release, "versions": normalized_versions}


DOCS_VERSIONS = load_docs_versions()
LATEST_DOCS_RELEASE = os.getenv("DOCS_LATEST_VERSION", DOCS_VERSIONS["latest"])
PINNED_DOCS_RELEASES = [entry["version"] for entry in DOCS_VERSIONS["versions"]]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]
autodoc_mock_imports = [
    "mate.jit",
    "mate.mate_runtime",
    "mate.sparse_mla",
    "torch",
    "torch_musa",
    "tvm",
    "tvm_ffi",
    "tilelang",
]

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


def resolve_docs_release() -> str:
    env_release = os.getenv("DOCS_RELEASE")
    if env_release:
        return env_release

    if CURRENT_DOCS_RELEASE:
        return CURRENT_DOCS_RELEASE

    try:
        raw_version = getattr(mate, "__version__", None)
        if raw_version and raw_version != "unknown":
            return Version(raw_version).base_version
    except Exception:
        pass

    return "dev"


def resolve_docs_current_version(release_name: str) -> str:
    env_current_version = os.getenv("DOCS_CURRENT_VERSION")
    if env_current_version:
        return env_current_version

    return "latest" if release_name == "latest" else release_name


release = resolve_docs_release()
version = ".".join(release.split(".")[:2]) if release != "dev" else "dev"
html_title = f"{project} {release}"
html_short_title = html_title
docs_current_version = resolve_docs_current_version(release)
html_context = {
    "docs_current_version": docs_current_version,
    "docs_version_targets": [
        {
            "label": f"Latest ({LATEST_DOCS_RELEASE})",
            "value": "latest",
            "url_prefix": "/latest/",
        },
        *[
            {
                "label": docs_version,
                "value": docs_version,
                "url_prefix": f"/{docs_version}/",
            }
            for docs_version in PINNED_DOCS_RELEASES
            if docs_version != LATEST_DOCS_RELEASE
        ],
    ],
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"] if (Path(__file__).parent / "_static").exists() else []
html_css_files = ["custom.css"] if html_static_path else []
html_js_files = (
    [
        "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js",
        "mermaid-init.js",
    ]
    if html_static_path
    else []
)
