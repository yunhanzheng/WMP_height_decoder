#!/bin/bash
# Run tasks consecutively.
# Comment out any line below to skip it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$SCRIPT_DIR/legged_gym/scripts"

set -e  # Stop on first error. Remove this line to continue even if a task fails.

echo "========================================"
echo "  RUN ALL"
echo "========================================"

# ── TRAIN ─────────────────────────────────────────────────────────────────────
python "$SCRIPTS/train.py" --task go2_blind          --headless
python "$SCRIPTS/train.py" --task go2_baseline_blind --headless
# python "$SCRIPTS/train.py" --task go2_baseline       --headless
python "$SCRIPTS/train.py" --task go2_himloco        --headless
python "$SCRIPTS/train.py" --task go2_xiao           --headless

# # ── EVAL ──────────────────────────────────────────────────────────────────────
# python "$SCRIPTS/eval.py"  --task go2_blind          --headless
# python "$SCRIPTS/eval.py"  --task go2_baseline_blind --headless
# python "$SCRIPTS/eval.py"  --task go2_himloco        --headless
# python "$SCRIPTS/eval.py"  --task go2_xiao           --headless
# python "$SCRIPTS/eval.py"  --task go2_baseline       --headless

# # ── PLAY ──────────────────────────────────────────────────────────────────────
# python "$SCRIPTS/play.py"  --task go2_blind          --headless
# python "$SCRIPTS/play.py"  --task go2_baseline_blind --headless
# python "$SCRIPTS/play.py"  --task go2_himloco        --headless
# python "$SCRIPTS/play.py"  --task go2_xiao           --headless
# python "$SCRIPTS/play.py"  --task go2_baseline       --headless

echo ""
echo "========================================"
echo "  ALL DONE"
echo "========================================"
