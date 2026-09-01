"""One-off: 3 known cases through the LLM lane (local Ollama, auto chain)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)
os.chdir(_ROOT)                                  # absolute output paths regardless of CWD
sys.path.insert(0, _ROOT)                        # repo root
sys.path.insert(0, _here)                        # scripts/ (agent3_batch lives here)
sys.argv = [
    "agent3_batch.py", "--sample", "3", "--render", "--backend", "auto",
    "--model", "glm-5.3-flash:cloud",
    "--out", os.path.join(_ROOT, "runs", "app_rex_adapted", "agent3_batch_llm"),
]
import agent3_batch

try:
    agent3_batch.main()
except SystemExit:
    raise
except BaseException:
    import traceback
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    with open(os.path.join(_ROOT, "runs", "app_rex_adapted",
                           "agent3_batch_llm_sidecar_error.log"), "w") as f:
        f.write(tb)
    raise