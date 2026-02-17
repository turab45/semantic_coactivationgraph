"""
Run All Steps
==============
Execute the full SCAG pipeline:
  Step 1: Build base coactivation graph (neuron↔neuron)
  Step 2: Compute neuron→concept associations (spatial IoU)
  Step 3: Extend graph with concept nodes
  Step 4: Analysis & figures
  Step 5: Interactive visualisation

Usage:
    python run_all.py                # Full pipeline from scratch
    python run_all.py --skip 1 2     # Skip steps 1 and 2 (load existing)
    python run_all.py --only 4 5     # Run only steps 4 and 5
"""

import argparse
import time


def run_step(step_num, skip_set, only_set):
    """Check if step should run based on --skip and --only flags."""
    if only_set and step_num not in only_set:
        return False
    if step_num in skip_set:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run SCAG pipeline')
    parser.add_argument('--skip', nargs='+', type=int, default=[],
                        help='Steps to skip (e.g., --skip 1 2)')
    parser.add_argument('--only', nargs='+', type=int, default=[],
                        help='Run only these steps (e.g., --only 4 5)')
    args = parser.parse_args()

    skip = set(args.skip)
    only = set(args.only)

    start = time.time()
    print("=" * 60)
    print("  SCAG: Semantic Coactivation Graph Pipeline")
    print("=" * 60)

    # Step 1
    if run_step(1, skip, only):
        print("\n>>> Step 1: Base Coactivation Graph")
        from step1_base_graph import build
        build(argparse.Namespace(load=False))
    else:
        print("\n>>> Step 1: SKIPPED")

    # Step 2
    if run_step(2, skip, only):
        print("\n>>> Step 2: Concept Associations")
        from step2_concept_associations import build_associations
        build_associations()
    else:
        print("\n>>> Step 2: SKIPPED")

    # Step 3
    if run_step(3, skip, only):
        print("\n>>> Step 3: Extend Graph")
        from step3_extend_graph import main as step3_main
        step3_main()
    else:
        print("\n>>> Step 3: SKIPPED")

    # Step 4
    if run_step(4, skip, only):
        print("\n>>> Step 4: Analysis & Figures")
        from step4_analysis import main as step4_main
        import sys
        sys.argv = ['step4_analysis.py']  # reset argv for argparse
        step4_main()
    else:
        print("\n>>> Step 4: SKIPPED")

    # Step 5
    if run_step(5, skip, only):
        print("\n>>> Step 5: Interactive Visualisation")
        from step5_visualize import main as step5_main
        step5_main()
    else:
        print("\n>>> Step 5: SKIPPED")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()