# dev/

Development-only material. None of it ships: `pyproject.toml` builds from
`include = ["butchc*"]`, so nothing under `dev/` reaches the wheel, and nothing
here is importable from `butchc`.

The separation is deliberate. The library's selling point is that it has no
dependencies, and the moment a benchmark that needs `optuna`, `yahpo-gym` and
`ConfigSpace` lives next to the package, someone eventually imports across the
boundary and the claim quietly stops being true.

| Directory | Needs | Purpose |
|---|---|---|
| `../benchmarks/` | nothing | Fast synthetic suite. The regression loop you run on every change. |
| `eval/` | `optuna`, `yahpo-gym`, `ConfigSpace` | Real-world hierarchical benchmarks. Run before a release, not on every commit. |
| `archive/` | nothing | Retired process documents — past code reviews, planning notes, anything that narrates a *decision* rather than the *current* state of the code. Nothing in it is linked from the README or `docs/`. |
| `tools/` | nothing | One-off scripts that help maintain the repo itself (e.g. regenerating `docs/codemap.md`). Not a runtime dependency of anything. |

**What goes in `archive/` vs. `docs/`.** `docs/` describes the library as it
is today and is expected to stay accurate — if it goes stale, fix it.
`archive/` describes a past decision or a one-time review — if it goes stale,
that's fine, it's a historical record and is dated implicitly by its content.
When in doubt: would a new contributor reading this file need to know it was
written *about* an old version, to make sense of it? If deleting the version
history would make the document confusing, it belongs in `archive/`, not
`docs/`.

`../benchmarks/` stays at the top level because the README points at it and it
runs anywhere with a bare Python. If you would rather have one development
root, move it to `dev/bench/` and update the two README references — nothing
else depends on its location.
