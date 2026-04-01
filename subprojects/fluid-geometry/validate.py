#!/usr/bin/env python3
"""
Validate fluid_geometry.py syntax and structure locally.

This script checks that the processor code is syntactically correct
and has the expected class structure without requiring vLLM or PyTorch.
"""

import ast
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROCESSOR_FILE = SCRIPT_DIR / "fluid_geometry.py"

EXPECTED_CLASSES = [
    "PhaseState",
    "CalibrationSnapshot",
    "Accumulator",
    "StructuralLaws",
    "StabilityMonitor",
    "Calibrator",
    "GeometricRequestProcessor",
    "FluidGeometryLogitsProcessor",
]

EXPECTED_METHODS = {
    "Accumulator": ["__init__", "update"],
    "StructuralLaws": ["temperature", "think_token_bias"],
    "StabilityMonitor": ["__init__", "record"],
    "Calibrator": ["__init__", "confidence", "update", "report_quality", "save_state"],
    "GeometricRequestProcessor": ["__init__", "_is_thinking", "__call__"],
    "FluidGeometryLogitsProcessor": ["__init__", "is_argmax_invariant", "new_req_logits_processor"],
}


def validate():
    """Run validation checks."""
    print(f"Validating {PROCESSOR_FILE}...")

    # Check file exists
    if not PROCESSOR_FILE.exists():
        print(f"  ERROR: File not found: {PROCESSOR_FILE}")
        return False

    # Parse AST
    try:
        source = PROCESSOR_FILE.read_text()
        tree = ast.parse(source)
        print("  [OK] Syntax valid")
    except SyntaxError as e:
        print(f"  ERROR: Syntax error at line {e.lineno}: {e.msg}")
        return False

    # Find classes
    found_classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            found_classes[node.name] = methods

    # Check expected classes
    all_ok = True
    for cls_name in EXPECTED_CLASSES:
        if cls_name in found_classes:
            print(f"  [OK] Class {cls_name} found")
        else:
            print(f"  ERROR: Class {cls_name} NOT found")
            all_ok = False

    # Check expected methods
    for cls_name, expected_methods in EXPECTED_METHODS.items():
        if cls_name not in found_classes:
            continue
        actual_methods = found_classes[cls_name]
        for method in expected_methods:
            if method in actual_methods:
                print(f"  [OK] {cls_name}.{method}() found")
            else:
                print(f"  ERROR: {cls_name}.{method}() NOT found")
                all_ok = False

    # Summary
    print()
    if all_ok:
        print("Validation PASSED")
        return True
    else:
        print("Validation FAILED")
        return False


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
