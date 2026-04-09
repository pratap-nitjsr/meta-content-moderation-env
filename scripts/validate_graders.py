import importlib
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
OPENENV = ROOT / "openenv.yaml"

# Ensure repo root is on sys.path so imports like `server.graders` work when run from scripts/
sys.path.insert(0, str(ROOT))

if not OPENENV.exists():
    print(f"openenv.yaml not found at {OPENENV}")
    sys.exit(2)

with OPENENV.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

tasks = cfg.get("tasks", [])
if not tasks:
    print("No tasks found in openenv.yaml")
    sys.exit(2)

print(f"Found {len(tasks)} tasks. Validating grader importability...")

failed = []
for t in tasks:
    tid = t.get("id") or t.get("name") or "<unknown>"
    grader_spec = t.get("grader")
    grader_dot = t.get("grader_import")
    print(f"\n- Task: {tid}")

    tried = []
    ok = False

    for spec in (grader_spec, grader_dot):
        if not spec:
            continue
        tried.append(spec)
        # Support both "module:callable" and "module.callable"
        module = None
        attr = None
        if ":" in spec:
            module, attr = spec.split(":", 1)
        elif "." in spec:
            parts = spec.rsplit(".", 1)
            if len(parts) == 2:
                module, attr = parts
            else:
                module = spec
                attr = None
        else:
            module = spec
            attr = None

        try:
            m = importlib.import_module(module)
            if attr:
                if hasattr(m, attr):
                    print(f"  OK: imported {module} and found attribute '{attr}'")
                    ok = True
                    break
                else:
                    print(f"  ERROR: imported {module} but attribute '{attr}' not found")
            else:
                print(f"  OK: imported module {module}")
                ok = True
                break
        except Exception as e:
            print(f"  ERROR importing {module}: {e}")

    if not ok:
        failed.append((tid, tried))

print("\nSummary:")
if not failed:
    print("All graders importable ✅")
    sys.exit(0)
else:
    print(f"{len(failed)} task(s) failed to import graders:")
    for tid, tried in failed:
        print(f" - {tid}: tried {tried}")
    sys.exit(1)
