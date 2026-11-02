#!/usr/bin/env python3
import os
import re
from pathlib import Path

IGNORED_DIRS = {"stylesheets", "assets", "images", "img", "javascripts", "__pycache__"}

def extract_md_title(md_path: Path, fallback: str) -> str:
    """Extract topic_title/title from YAML frontmatter or first H1 heading if available."""
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return fallback

    # Check YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm_match:
        fm = fm_match.group(1)
        for key in ("topic_title", "title"):
            m = re.search(rf"^{key}:\s*['\"]?([^'\"\n]+)['\"]?", fm, re.MULTILINE)
            if m:
                return m.group(1).strip()

    # Check first H1
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if h1_match:
        title_text = h1_match.group(1).strip()
        if not title_text.startswith("--8<--"):
            return title_text

    return fallback

def generate_nav(docs_dir="docs"):
    docs_path = Path(docs_dir)
    nav_items = []

    def process_dir(current_path, level=1):
        items = []
        try:
            entries = sorted(os.listdir(current_path))
        except FileNotFoundError:
            return []

        for entry in entries:
            if entry.startswith(".") or entry == "index.md" or entry in IGNORED_DIRS:
                continue

            full_path = current_path / entry
            rel_path = full_path.relative_to(docs_path).as_posix()

            raw_name = entry.rsplit(".", 1)[0] if full_path.is_file() else entry
            clean_title = re.sub(r"^\d+[-_]", "", raw_name).replace("-", " ").replace("_", " ").title()
            if clean_title.lower() == "adrs":
                clean_title = "ADRs"

            if full_path.is_file():
                if not (entry.endswith(".md") or entry.endswith(".ipynb")):
                    continue
                if entry.endswith(".md"):
                    clean_title = extract_md_title(full_path, clean_title)
                # Escape colons in YAML titles
                safe_title = f'"{clean_title}"' if ":" in clean_title else clean_title
                items.append(f"{'  ' * level}- {safe_title}: {rel_path}")

            elif full_path.is_dir():
                sub_items = process_dir(full_path, level + 1)
                has_index = (full_path / "index.md").exists()
                if sub_items or has_index:
                    items.append(f"{'  ' * level}- {clean_title}:")
                    if has_index:
                        items.append(f"{'  ' * (level + 1)}- Overview: {rel_path}/index.md")
                    items.extend(sub_items)

        return items

    if (docs_path / "index.md").exists():
        nav_items.append("  - Home: index.md")

    nav_items.extend(process_dir(docs_path))
    return nav_items

def update_mkdocs_yml():
    yml_path = Path("mkdocs.yml")
    if not yml_path.exists():
        print("mkdocs.yml not found.")
        return False

    content = yml_path.read_text(encoding="utf-8")
    start_marker = "# >>> auto-nav (managed by documentation-setup-mkdocs:update_nav.py) >>>"
    legacy_marker = "# >>> auto-nav (managed by setup-mkdocs:update_nav.py) >>>"
    end_marker = "# <<< auto-nav <<<"

    active_start = None
    if start_marker in content:
        active_start = start_marker
    elif legacy_marker in content:
        active_start = legacy_marker

    if not active_start or end_marker not in content:
        print("Auto-nav markers not found in mkdocs.yml.")
        return False

    nav_lines = generate_nav()
    nav_content = start_marker + "\nnav:\n" + "\n".join(nav_lines) + "\n" + end_marker

    pattern = re.compile(rf"{re.escape(active_start)}.*?{re.escape(end_marker)}", re.DOTALL)
    new_content = pattern.sub(nav_content, content)

    yml_path.write_text(new_content, encoding="utf-8")
    print("Successfully updated nav in mkdocs.yml")
    return True

if __name__ == "__main__":
    import sys
    success = update_mkdocs_yml()
    sys.exit(0 if success else 1)
