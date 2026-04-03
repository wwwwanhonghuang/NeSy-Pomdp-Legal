import os
from encoder import NeuralEncoder
from logic_engine import SymbolicGrounding
from pomdp_core import ParticleFilter
from policy import SimplePolicy
from utils import fuse_signals, print_belief_summary


def run_evaluation_demo(data_dir: str = "data"):
    encoder = NeuralEncoder()
    grounder = SymbolicGrounding()
    pf = ParticleFilter(num_particles=300)
    policy = SimplePolicy()

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])

    print("--- Starting NeSy-POMDP Evaluation Demo ---")
    print(f"Found {len(files)} thesis components.\n")

    step_idx = 0

    while True:
        belief_summary = pf.summary()
        remaining_files = len(files) - step_idx

        action_info = policy.choose_action(
            belief_summary=belief_summary,
            step_idx=step_idx,
            remaining_files=remaining_files
        )

        print(f"[Policy] Action: {action_info['action']}")
        if action_info["focus_aq"] is not None:
            print(f"[Policy] Focus AQ: {action_info['focus_aq']}")
        print(f"[Policy] Reason: {action_info['reason']}")

        if action_info["action"] == "grade_now":
            break

        if action_info["action"] == "check_criterion":
            print("\n--- Criterion check requested ---")
            print_belief_summary(belief_summary)
            break

        filename = files[step_idx]
        filepath = os.path.join(data_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"\n[Step {step_idx + 1}] Reading: {filename}")

        neural_signals = encoder.get_aq_signals(text)
        symbolic_signals = grounder.ground_component(text)

        aq_targets, aq_strengths, fused_unit = fuse_signals(neural_signals, symbolic_signals)

        print("Neural/Symbolic fused AQ unit scores [0,1]:")
        for aq, val in fused_unit.items():
            print(f"  {aq}: {val:.3f}")

        print("AQ target scores [1,6]:")
        for aq, val in aq_targets.items():
            print(f"  {aq}: {val:.2f}")

        pf.update(aq_targets=aq_targets, aq_strengths=aq_strengths)

        print()
        print_belief_summary(pf.summary())
        print("-" * 60)

        step_idx += 1

        if step_idx >= len(files):
            print("\nNo more sections remain.")
            break

    print("\n--- Final Assessment ---")
    final_summary = pf.summary()
    print_belief_summary(final_summary)

    print("\nFinal recommended AQ means:")
    for aq, stats in final_summary.items():
        print(f"  {aq}: {stats['mean']:.2f}")


if __name__ == "__main__":
    run_evaluation_demo()