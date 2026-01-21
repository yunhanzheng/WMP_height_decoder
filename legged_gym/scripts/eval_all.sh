#!/bin/bash
# Evaluate all registered tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All registered go2 tasks
TASKS=(
    "go2_blind"
    "go2_baseline_blind"
    "go2_him_blind"
    "go2_himloco"
    "go2_xiao"
    "go2_baseline"
)

echo "========================================"
echo "EVAL ALL TASKS"
echo "========================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating: $task"
    echo "========================================"
    python "$SCRIPT_DIR/eval.py" --task "$task" --headless
done

echo ""
echo "========================================"
echo "All evaluations complete"
echo "========================================"
