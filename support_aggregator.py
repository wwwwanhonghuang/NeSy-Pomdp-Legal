from collections import defaultdict
from typing import List, Dict


class SupportAggregator:
    """Aggregate evidence into claim-level support state and missing evidence."""

    def aggregate(self, claims: List[Dict], scored_links: List[Dict], evidence_lookup: Dict[str, Dict]) -> Dict[str, Dict]:
        by_claim = defaultdict(list)
        for item in scored_links:
            by_claim[item["claim_id"]].append(item)

        results = {}

        for claim in claims:
            items = by_claim.get(claim["id"], [])
            supports = [x for x in items if x["relation"] == "supports"]
            attacks = [x for x in items if x["relation"] == "attacks"]

            found_types = set()
            for x in supports:
                evidence = evidence_lookup.get(x["evidence_id"], {})
                for ev_type in evidence.get("evidence_types", []):
                    found_types.add(ev_type)

            required = list(claim.get("required_evidence", []))
            missing = [ev for ev in required if ev not in found_types]

            if not items:
                results[claim["id"]] = {
                    "claim_id": claim["id"],
                    "support_score": 0.10,
                    "attack_score": 0.0,
                    "support_status": "unsupported",
                    "best_support_id": None,
                    "num_links": 0,
                    "found_evidence_types": [],
                    "missing_evidence_types": missing,
                }
                continue

            support_score = min(1.0, sum(x["overall_strength"] for x in supports))
            attack_score = min(1.0, sum(x["overall_strength"] for x in attacks))
            net = support_score - 0.7 * attack_score

            if len(missing) >= 2 and support_score < 0.55:
                status = "weakly_supported"
            elif net < 0.20:
                status = "weakly_supported"
            elif net < 0.45:
                status = "supported"
            else:
                status = "strongly_supported"

            if attack_score > support_score and attack_score > 0.35:
                status = "contested"

            best_support_id = None
            if supports:
                best = max(supports, key=lambda x: x["overall_strength"])
                best_support_id = best["evidence_id"]

            results[claim["id"]] = {
                "claim_id": claim["id"],
                "support_score": float(support_score),
                "attack_score": float(attack_score),
                "support_status": status,
                "best_support_id": best_support_id,
                "num_links": len(items),
                "found_evidence_types": sorted(found_types),
                "missing_evidence_types": missing,
            }

        return results
