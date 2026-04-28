#!/usr/bin/env python3
"""
onboarding.py -- MemPalace first-run setup.

Asks the user:
  1. How they're using MemPalace (work / personal / combo)
  2. Who the people in their life are (names, nicknames, relationships)
  3. What their projects are

Seeds the entity_registry with confirmed data so MemPalace knows your world
from minute one -- before a single session is indexed.

Usage:
    python3 -m mempalace.onboarding
    or: mempalace init
"""

from pathlib import Path
from mempalace.entity_registry import EntityRegistry
from mempalace.entity_detector import detect_entities, scan_for_detection


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _hr():
    print(f"\n{'─' * 58}")


def _header(text):
    print(f"\n{'=' * 58}")
    print(f"  {text}")
    print(f"{'=' * 58}")


def _ask(prompt, default=None):
    if default:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt}: ").strip()


def _yn(prompt, default="y"):
    val = input(f"  {prompt} [{'Y/n' if default == 'y' else 'y/N'}]: ").strip().lower()
    if not val:
        return default == "y"
    return val.startswith("y")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Mode selection
# ─────────────────────────────────────────────────────────────────────────────


def _ask_mode() -> str:
    _header("Welcome to MemPalace")
    print("""
  MemPalace is a personal memory system. To work well, it needs to know
  a little about your world -- who the people are, what the projects
  are, and how you want your memory organized.

  This takes about 2 minutes. You can always update it later.
""")
    print("  How are you using MemPalace?")
    print()
    print("    [1]  Work     -- notes, projects, clients, colleagues, decisions")
    print("    [2]  Personal -- diary, family, health, relationships, reflections")
    print("    [3]  Both     -- personal and professional mixed")
    print()

    while True:
        choice = input("  Your choice [1/2/3]: ").strip()
        if choice == "1":
            return "work"
        elif choice == "2":
            return "personal"
        elif choice == "3":
            return "combo"
        print("  Please enter 1, 2, or 3.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: People
# ─────────────────────────────────────────────────────────────────────────────


def _ask_people(mode: str) -> tuple[list, dict]:
    """Returns (people_list, aliases_dict)."""
    people = []
    aliases = {}  # nickname → full name

    if mode in ("personal", "combo"):
        _hr()
        print("""
  Personal world -- who are the important people in your life?

  Format: name, relationship (e.g. "Riley, daughter" or just "Devon")
  For nicknames, you'll be asked separately.
  Type 'done' when finished.
""")
        while True:
            entry = input("  Person: ").strip()
            if entry.lower() in ("done", ""):
                break
            parts = [p.strip() for p in entry.split(",", 1)]
            name = parts[0]
            relationship = parts[1] if len(parts) > 1 else ""
            if name:
                # Ask about nicknames
                nick = input(f"  Nickname for {name}? (or enter to skip): ").strip()
                if nick:
                    aliases[nick] = name
                people.append({"name": name, "relationship": relationship, "context": "personal"})

    if mode in ("work", "combo"):
        _hr()
        print("""
  Work world -- who are the colleagues, clients, or collaborators
  you'd want to find in your notes?

  Format: name, role (e.g. "Ben, co-founder" or just "Sarah")
  Type 'done' when finished.
""")
        while True:
            entry = input("  Person: ").strip()
            if entry.lower() in ("done", ""):
                break
            parts = [p.strip() for p in entry.split(",", 1)]
            name = parts[0]
            role = parts[1] if len(parts) > 1 else ""
            if name:
                people.append({"name": name, "relationship": role, "context": "work"})

    return people, aliases


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Projects
# ─────────────────────────────────────────────────────────────────────────────


def _ask_projects(mode: str) -> list:
    if mode == "personal":
        return []

    _hr()
    print("""
  What are your main projects? (These help MemPalace distinguish project
  names from person names -- e.g. "Lantern" the project vs. "Lantern" the word.)

  Type 'done' when finished.
""")
    projects = []
    while True:
        proj = input("  Project: ").strip()
        if proj.lower() in ("done", ""):
            break
        if proj:
            projects.append(proj)
    return projects


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Auto-detect from files
# ─────────────────────────────────────────────────────────────────────────────


def _auto_detect(directory: str, known_people: list) -> list:
    """Scan directory for additional entity candidates."""
    known_names = {p["name"].lower() for p in known_people}

    try:
        files = scan_for_detection(directory)
        if not files:
            return []
        detected = detect_entities(files)
        new_people = [
            e
            for e in detected["people"]
            if e["name"].lower() not in known_names and e["confidence"] >= 0.7
        ]
        return new_people
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Ambiguity warnings
# ─────────────────────────────────────────────────────────────────────────────


def _warn_ambiguous(people: list) -> list:
    """
    Flag names that are also common English words.
    Returns list of ambiguous names for user awareness.
    """
    from mempalace.entity_registry import COMMON_ENGLISH_WORDS

    ambiguous = []
    for p in people:
        if p["name"].lower() in COMMON_ENGLISH_WORDS:
            ambiguous.append(p["name"])
    return ambiguous


# ─────────────────────────────────────────────────────────────────────────────
# Main onboarding flow
# ─────────────────────────────────────────────────────────────────────────────


def run_onboarding(
    directory: str = ".",
    config_dir: Path = None,
    auto_detect: bool = True,
) -> EntityRegistry:
    """
    Run the full onboarding flow.
    Returns the seeded EntityRegistry.
    """
    # Step 1: Mode
    mode = _ask_mode()

    # Step 2: People
    people, aliases = _ask_people(mode)

    # Step 3: Projects
    projects = _ask_projects(mode)

    # Step 4: Auto-detect additional people from files
    if auto_detect and _yn("\nScan your files for additional names we might have missed?"):
        directory = _ask("Directory to scan", default=directory)
        detected = _auto_detect(directory, people)
        if detected:
            _hr()
            print(f"\n  Found {len(detected)} additional name candidates:\n")
            for e in detected:
                print(
                    f"    {e['name']:20} confidence={e['confidence']:.0%}  "
                    f"({', '.join(e['signals'][:1])})"
                )
            print()
            if _yn("  Add any of these to your registry?"):
                for e in detected:
                    ans = input(f"    {e['name']} -- (p)erson, (s)kip? ").strip().lower()
                    if ans == "p":
                        rel = input(f"    Relationship/role for {e['name']}? ").strip()
                        ctx = (
                            "personal"
                            if mode == "personal"
                            else (
                                "work"
                                if mode == "work"
                                else input("    Context -- (p)ersonal or (w)ork? ")
                                .strip()
                                .lower()
                                .replace("w", "work")
                                .replace("p", "personal")
                            )
                        )
                        people.append({"name": e["name"], "relationship": rel, "context": ctx})

    # Step 5: Warn about ambiguous names
    ambiguous = _warn_ambiguous(people)
    if ambiguous:
        _hr()
        print(f"""
  Heads up -- these names are also common English words:
    {", ".join(ambiguous)}

  MemPalace will check the context before treating them as person names.
  For example: "I picked up Riley" → person.
               "Have you ever tried" → adverb.
""")

    # Build and save registry
    registry = EntityRegistry.load(config_dir)
    registry.seed(mode=mode, people=people, projects=projects, aliases=aliases)

    # Summary
    _header("Setup Complete")
    print()
    print(f"  {registry.summary()}")
    print(f"\n  Registry saved to: {registry._path}")
    print("\n  Your AI will know your world from the first session.")
    print()

    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Quick setup (non-interactive, for testing)
# ─────────────────────────────────────────────────────────────────────────────


def quick_setup(
    mode: str,
    people: list,
    projects: list = None,
    aliases: dict = None,
    config_dir: Path = None,
) -> EntityRegistry:
    """
    Programmatic setup without interactive prompts.
    Used in tests and benchmark scripts.

    people: list of dicts {"name": str, "relationship": str, "context": str}
    """
    registry = EntityRegistry.load(config_dir)
    registry.seed(
        mode=mode,
        people=people,
        projects=projects or [],
        aliases=aliases or {},
    )
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    run_onboarding(directory=directory)
