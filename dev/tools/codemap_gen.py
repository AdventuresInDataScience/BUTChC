"""Regenerate the function/class reference tables in ``docs/codemap.md``.

    python dev/tools/codemap_gen.py > /tmp/codemap_tables.md

Static analysis via the standard-library ``ast`` module: for every function,
method and class in ``butchc/``, ``butchc/interop/``, ``benchmarks/`` and
``dev/eval/``, it lists where it's defined, its docstring's first line, what
it calls, and what calls it.

Call resolution is import-aware (it follows ``from x import y``, module
aliases, ``self.method()``, and one hop of ``var = ClassName(...)`` typing)
rather than matching bare names globally, which is what makes it safe to
trust: a generic parameter name like ``objective`` in one file is never
confused with an unrelated top-level function of the same name in another.

Known blind spots, by design rather than oversight:

- **Module-level calls are invisible.** A call made directly in a module's
  top-level code (e.g. ``benchmarks/problems.py`` building its ``TUNING``/
  ``HELDOUT`` dicts from problem-constructor calls at import time) is not
  inside any function body, so it never appears as a "call". The functions
  built that way legitimately show no callers here even though they are used.
- **Dispatch through data is invisible.** A function referenced only as a
  value (stored in a dict, passed as a callback, handed to an executor) is
  not a ``Call`` node at its point of reference and will not show up as
  "called by" there.
- **Tests are out of scope.** ``tests/`` is deliberately excluded — its
  functions are entry points for pytest, not part of the call graph, and
  including them would mostly just show "every library function is called by
  some test", which is not useful signal for understanding how the library
  itself is wired together.
- **This goes stale.** It reflects the source tree at the moment it's run.
  Re-run it after any change to public signatures, module structure, or
  import layout, and paste the output back into ``docs/codemap.md`` under
  "Detailed reference".
"""

import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

TARGET_DIRS = ["butchc", "butchc/interop", "benchmarks", "dev/eval"]


def all_files():
    out = []
    for d in TARGET_DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            if fn.endswith(".py"):
                # Forward slashes, not os.path.join: every path check below
                # ("butchc/interop/", split("/"), replace("/", ".")) treats "/"
                # as the separator, so a backslash from a Windows join makes
                # all of them silently miss and the import graph comes out
                # empty of cross-module calls.
                out.append(f"{d}/{fn}")
    return out


FILES = all_files()
FILESET = set(FILES)

# bare importable name -> modpath, e.g. "problems" -> "benchmarks/problems.py"
# (valid because benchmarks/ and dev/eval/ import siblings via sys.path, not
# a package-relative import)
BARE_NAME_TO_MODPATH = {os.path.basename(f)[:-3]: f for f in FILES}

# dotted package path -> modpath, for the butchc package itself
DOTTED_TO_MODPATH = {
    "butchc": "butchc/__init__.py",
    "butchc.interop": "butchc/interop/__init__.py",
}
for f in FILES:
    if f.startswith("butchc/interop/"):
        DOTTED_TO_MODPATH["butchc.interop." + os.path.basename(f)[:-3]] = f
    elif f.startswith("butchc/"):
        DOTTED_TO_MODPATH["butchc." + os.path.basename(f)[:-3]] = f


def resolve_module(modpath, node):
    """Resolve an ast.Import / ast.ImportFrom's module reference to a modpath
    we know about, or None."""
    if isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            pkg_dir = os.path.dirname(modpath)
            parts = pkg_dir.split("/") if pkg_dir else []
            up = node.level - 1
            if up:
                parts = parts[: len(parts) - up] if up <= len(parts) else []
            base = "/".join(parts)
            if node.module:
                candidate = f"{base}/{node.module}.py" if base else f"{node.module}.py"
            else:
                candidate = f"{base}/__init__.py"
            return candidate if candidate in FILESET else None
        mod = node.module or ""
        return DOTTED_TO_MODPATH.get(mod) or BARE_NAME_TO_MODPATH.get(mod)
    return None  # ast.Import handled per-alias by the caller


class Def:
    def __init__(self, name, kind, lineno, doc, modpath, cls=None):
        self.name, self.kind, self.lineno = name, kind, lineno
        self.doc, self.modpath, self.cls = doc, modpath, cls
        self.calls = []

    @property
    def qual(self):
        return f"{self.cls}.{self.name}" if self.cls else self.name

    @property
    def key(self):
        return f"{self.modpath}::{self.qual}"


def first_line(doc):
    return doc.strip().splitlines()[0].strip() if doc else ""


def parse_file(modpath):
    # encoding is explicit: the sources carry non-ASCII in docstrings (the
    # update rule's proportional sign, for one), and on Windows the default
    # locale encoding is cp1252, which cannot decode them.
    with open(os.path.join(ROOT, modpath), encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=modpath)
    module_doc = ast.get_docstring(tree)
    name_binding = {}  # local_name -> ("name"|"module", target_modpath, target_name_or_None)

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            target_mod = resolve_module(modpath, node)
            if target_mod:
                for alias in node.names:
                    name_binding[alias.asname or alias.name] = ("name", target_mod, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                target_mod = DOTTED_TO_MODPATH.get(alias.name) or BARE_NAME_TO_MODPATH.get(alias.name)
                if target_mod:
                    name_binding[local] = ("module", target_mod, None)

    local_class_names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    defs = []

    class CallCollector(ast.NodeVisitor):
        """Collects call targets, plus a one-hop type trace: a local
        ``var = ClassName(...)`` lets a later ``var.method()`` resolve to
        ``ClassName.method`` instead of being silently dropped."""
        def __init__(self):
            self.raw, self.var_class = [], {}

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id in local_class_names):
                self.var_class[node.targets[0].id] = node.value.func.id
            self.generic_visit(node)

        def visit_Call(self, node):
            f = node.func
            if isinstance(f, ast.Name):
                self.raw.append(("name", f.id))
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                qualifier = self.var_class.get(f.value.id, f.value.id)
                self.raw.append(("attr", qualifier, f.attr))
            self.generic_visit(node)

    def handle_func(node, cls):
        cc = CallCollector()
        cc.visit(node)
        d = Def(node.name, "method" if cls else "function", node.lineno,
                ast.get_docstring(node), modpath, cls)
        d.calls = cc.raw
        defs.append(d)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            handle_func(node, None)
        elif isinstance(node, ast.ClassDef):
            defs.append(Def(node.name, "class", node.lineno, ast.get_docstring(node), modpath))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    handle_func(item, node.name)

    return module_doc, defs, name_binding


def main():
    per_file, same_file_defs, same_file_methods, all_defs = {}, {}, {}, {}

    for modpath in FILES:
        module_doc, defs, name_binding = parse_file(modpath)
        per_file[modpath] = (module_doc, defs, name_binding)
        bare = {}
        for d in defs:
            all_defs[d.key] = d
            if d.cls is None:
                bare.setdefault(d.name, []).append(d)
            else:
                same_file_methods.setdefault((modpath, d.cls), {})[d.name] = d
        same_file_defs[modpath] = bare

    called_by = {k: set() for k in all_defs}

    def resolve_in_module(mod, name, depth=0):
        if depth > 5:
            return []
        direct = same_file_defs.get(mod, {}).get(name)
        if direct:
            return direct
        _, _, binding = per_file.get(mod, (None, None, {}))
        b = binding.get(name)
        return resolve_in_module(b[1], b[2], depth + 1) if b and b[0] == "name" else []

    resolved = {}
    for modpath in FILES:
        _, defs, name_binding = per_file[modpath]
        for d in defs:
            targets, recursive = [], False
            for call in d.calls:
                if call[0] == "name":
                    name = call[1]
                    hit = same_file_defs.get(modpath, {}).get(name)
                    if hit:
                        for h in hit:
                            recursive = recursive or h.key == d.key
                            if h.key != d.key:
                                targets.append(h.key)
                        continue
                    binding = name_binding.get(name)
                    if binding and binding[0] == "name":
                        targets += [h.key for h in resolve_in_module(binding[1], binding[2])]
                else:
                    qualifier, attr = call[1], call[2]
                    if qualifier == "self" and d.cls:
                        h = same_file_methods.get((modpath, d.cls), {}).get(attr)
                        if h:
                            if h.key == d.key:
                                recursive = True
                            else:
                                targets.append(h.key)
                        continue
                    binding = name_binding.get(qualifier)
                    if binding and binding[0] == "module":
                        targets += [h.key for h in resolve_in_module(binding[1], attr)]
                    else:
                        hit = same_file_methods.get((modpath, qualifier), {})
                        if attr in hit:
                            targets.append(hit[attr].key)
            targets = sorted(set(targets))
            resolved[d.key] = (targets, recursive)
            for t in targets:
                called_by.setdefault(t, set()).add(d.key)

    def label(key, current_modpath):
        mod, qual = key.split("::")
        if mod == current_modpath:
            return f"`{qual}`"
        return f"`{mod[:-3].replace('/', '.')}.{qual}`"

    for modpath in FILES:
        module_doc, defs, _ = per_file[modpath]
        print(f"\n### `{modpath}`\n\n{first_line(module_doc)}\n")
        print("| Name | Line | Kind | Purpose | Calls | Called by |")
        print("|---|---|---|---|---|---|")
        for d in defs:
            calls, recursive = resolved.get(d.key, ([], False))
            calls_list = [label(c, modpath) for c in calls]
            if recursive:
                calls_list.append("*(itself, recursively)*")
            callers = sorted(called_by.get(d.key, []))
            doc = first_line(d.doc).replace("|", "\\|")
            print(f"| `{d.qual}` | {d.lineno} | {d.kind} | {doc} | "
                  f"{', '.join(calls_list) or '—'} | "
                  f"{', '.join(label(c, modpath) for c in callers) or '—'} |")


if __name__ == "__main__":
    # The tables contain em-dashes, so stdout has to be UTF-8. On Windows it
    # defaults to the console codepage (cp1252), which mangles them into "?"
    # on redirect — exactly what the documented `> codemap_tables.md` does.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
