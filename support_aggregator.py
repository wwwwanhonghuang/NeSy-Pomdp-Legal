from collections import defaultdict
from typing import List, Dict


class SupportAggregator:
    """
    Aggregate evidence into richer claim-level states.

    We distinguish:
    - support existence
    - reasoning quality
    - method fit
    - clarity
    - counterargument handling
    - overclaim risk

    So final AQ is not dominated only by evidence integrity.
    """

    def aggregate(self, claims: List[Dict], scored_links: List[Dict]) -> Dict[str, Dict]:
        by_claim = defaultdict(list)
        for item in scored_links:
            by_claim[item["claim_id"]].append(item)

        results = {}

        for claim in claims:
            cid = claim["id"]
            items = by_claim.get(cid, [])

            supports = [x for x in items if x["relation"] == "supports"]
            attacks = [x for x in items if x["relation"] == "attacks"]

            support_score = min(1.0, sum(x["overall_strength"] for x in supports))
            attack_score = min(1.0, sum(x["overall_strength"] for x in attacks))

            found_types = set()
            for s in supports:
                found_types.update(s.get("evidence_types", []))

            required_types = set(claim.get("required_evidence", []))
            missing_types = sorted(list(required_types - found_types))

            # ---- support status ----
            net = support_score - 0.7 * attack_score

            if support_score < 0.20 and len(found_types) == 0:
                support_status = "unsupported"
            elif net < 0.20:
                support_status = "weakly_supported"
            elif attack_score > support_score and attack_score > 0.35:
                support_status = "contested"
            elif net < 0.55:
                support_status = "supported"
            else:
                support_status = "strongly_supported"

            # ---- reasoning quality ----
            # proxy: good support + low attack + some diversity of evidence
            evidence_diversity = min(len(found_types) / 3.0, 1.0)
            reasoning_quality = (
                0.45 * support_score +
                0.30 * evidence_diversity +
                0.25 * (1.0 - attack_score)
            )

            # ---- method fit ----
            # methodological claims require method evidence more strongly
            if claim.get("claim_type") == "methodological":
                method_bonus = 0.20 if "method_statement" in found_types else 0.0
                method_fit = min(1.0, 0.60 * support_score + 0.20 * evidence_diversity + method_bonus)
            else:
                method_fit = min(1.0, 0.50 * support_score + 0.20 * evidence_diversity + 0.30)

            # ---- clarity ----
            # currently mild proxy; should later be improved from discourse features
            clarity = min(1.0, 0.40 + 0.35 * support_score + 0.25 * (1.0 - attack_score))

            # ---- counterargument handling ----
            if claim.get("aq_type") == "DR" or claim.get("claim_type") == "rebuttal":
                counterargument_handling = min(1.0, 0.55 * support_score + 0.45 * (1.0 - attack_score))
            else:
                counterargument_handling = min(1.0, 0.35 + 0.30 * support_score)

            # ---- overclaim risk ----
            # high if support weak but claim likely assertive
            assertive_claim = claim.get("claim_type") in {"doctrinal", "conclusion", "general"}
            overclaim_risk = 0.0
            if assertive_claim:
                overclaim_risk = max(0.0, 0.8 - support_score)
                overclaim_risk += 0.4 * attack_score
                overclaim_risk += 0.15 * min(len(missing_types) / 3.0, 1.0)
                overclaim_risk = min(1.0, overclaim_risk)

            results[cid] = {
                "claim_id": cid,
                "support_score": float(support_score),
                "attack_score": float(attack_score),
                "support_status": support_status,
                "missing_evidence_types": missing_types,
                "num_links": len(items),

                "reasoning_quality": float(reasoning_quality),
                "method_fit": float(method_fit),
                "clarity": float(clarity),
                "counterargument_handling": float(counterargument_handling),
                "overclaim_risk": float(overclaim_risk),

                "found_evidence_types": sorted(list(found_types)),
            }

        return results