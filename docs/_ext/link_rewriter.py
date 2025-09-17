from __future__ import annotations
from pathlib import PurePosixPath
from docutils import nodes
from sphinx.util import logging

from sphinx.transforms.post_transforms import SphinxPostTransform


logger = logging.getLogger(__name__)


def _is_external(href: str) -> bool:
    return href.startswith(("http://", "https://", "mailto:"))


def _strip_prefixes(path_str: str, prefixes: tuple[str, ...]) -> str:
    # Normalize to posix form so prefixes like "docs/" match on all platforms
    p = str(PurePosixPath(path_str))
    for pref in prefixes:
        if p.startswith(pref):
            return p[len(pref) :]
    return p


class RstLinkRewriter(SphinxPostTransform):
    default_priority = 999

    def run(self) -> None:
        builder = self.app.builder
        changed = 0
        prefixes = tuple(self.app.config.link_rewriter_strip_prefixes or ("docs/", "./", "source/"))
        # Extensions to rewrite
        exts = tuple(self.app.config.link_rewriter_extensions or (".rst", ".md", ".txt"))

        for ref in list(self.document.traverse(nodes.reference)):
            uri = ref.get("refuri")
            if not uri or _is_external(uri):
                continue

            if "#" in uri:
                path_str, frag = uri.split("#", 1)
                frag = frag.strip()
            else:
                path_str, frag = uri, ""

            path = PurePosixPath(path_str)
            if path.suffix not in exts:
                continue

            # Remove leading folder prefixes like "docs/"
            path_str = _strip_prefixes(path_str, prefixes)

            target_doc = str(PurePosixPath(path_str).with_suffix("")).lstrip("./")
            html_uri = builder.get_target_uri(target_doc)
            if frag:
                html_uri = f"{html_uri}#{frag}"

            ref["refuri"] = html_uri
            changed += 1

        if changed:
            logger.info("[link_rewriter] %s: rewrote %d link(s)", self.env.docname, changed)


def setup(app):
    logger.info("[link_rewriter] extension loaded")
    app.add_config_value("link_rewriter_strip_prefixes", (), "env")
    app.add_config_value("link_rewriter_extensions", (), "env")  # override in conf.py if needed
    app.add_post_transform(RstLinkRewriter)
    return {"version": "1.0", "parallel_read_safe": True}
