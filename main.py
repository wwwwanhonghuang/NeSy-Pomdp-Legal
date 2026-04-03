import os

from encoder import NeuralEncoder
from claim_extractor import ClaimExtractor
from evidence_extractor import EvidenceExtractor
from claim_linker import ClaimEvidenceLinker
from evidence_scorer import EvidenceScorer
from support_aggregator import SupportAggregator
from logic_engine import SymbolicGroundingEngine
from brief import EvaluationBrief
from pomdp_core import ParticleFilter
from policy import ClaimDrivenPolicy
from utils import fuse_signals


def run_demo(data_dir="data"):
    encoder = NeuralEncoder()
    claim_extractor = ClaimExtractor()
    evidence_extractor = EvidenceExtractor()
    linker = ClaimEvidenceLinker(encoder)
    evidence_scorer = EvidenceScorer()
    aggregator = SupportAggregator()
    logic_engine = SymbolicGroundingEngine()
    pf = ParticleFilter(num_particles=300)
    policy = ClaimDrivenPolicy()
    brief = EvaluationBrief()

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
    all_claims, all_evidences = [], []

    for step_idx, filename in enumerate(files):
        remaining_sections = files[step_idx:]
        action = policy.choose_action(pf.summary(), brief, step_idx, remaining_sections)
        brief.set_policy_focus(action)
        print(f"Step {step_idx+1}: {action['action']} | {action['reason']}")
        if action["action"] != "read_next_section":
            break

        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()

        all_claims.extend(claim_extractor.extract_claims(text, filename))
        all_evidences.extend(evidence_extractor.extract_evidence(text, filename))
        links = linker.link(all_claims, all_evidences)
        evidence_lookup = {e['id']: e for e in all_evidences}
        claim_lookup = {c['id']: c for c in all_claims}
        scored_links = [evidence_scorer.score_evidence(claim_lookup[l['claim_id']], evidence_lookup[l['evidence_id']], l) for l in links]
        claim_states = aggregator.aggregate(all_claims, scored_links, evidence_lookup)
        symbolic_aq = logic_engine.claims_to_aq_scores(all_claims, claim_states)
        neural_signals = encoder.get_section_neural_signals(text)
        aq_targets, aq_strengths, fused_unit = fuse_signals(neural_signals, symbolic_aq)
        pf.update_claim_memory(claim_states)
        pf.update(aq_targets, aq_strengths)
        brief.sync(all_claims, all_evidences, links, scored_links, claim_states, symbolic_aq)

        print("Top open questions:")
        for q in brief.top_open_questions(3):
            print(f" - {q['aq_type']} | {q['support_status']} | missing={q['missing_evidence_types']} | {q['claim_text'][:80]}")
        print("AQ means:", {k: round(v['mean'], 2) for k, v in pf.summary().items()})


if __name__ == "__main__":
    run_demo()
