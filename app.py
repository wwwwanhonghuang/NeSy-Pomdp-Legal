import time
from io import StringIO
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from encoder import NeuralEncoder
from claim_extractor import ClaimExtractor
from evidence_extractor import EvidenceExtractor
from claim_linker import ClaimEvidenceLinker
from evidence_scorer import EvidenceScorer
from support_aggregator import SupportAggregator
from logic_engine import SymbolicGroundingEngine
from brief import EvaluationBrief
from observation_model import ObservationBuilder
from belief_utils import build_latent_state_hypothesis
from pomdp_core import ParticleFilter
from policy import ClaimDrivenPolicy
from utils import fuse_signals
from formal_validator import FormalValidator


AQ_LABELS = ["LC", "RE", "DR", "DN", "OQ"]


# ============================================================
# Page
# ============================================================

st.set_page_config(
    page_title="NeSy-POMDP Thesis Evaluation Demo",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Neuro-Symbolic POMDP Thesis Evaluation Demo")
st.caption(
    "Observation graph, latent brief hypothesis, formal validation, uncertainty-aware belief update, and claim-driven action policy"
)


# ============================================================
# Init / state
# ============================================================

def init_components():
    encoder = NeuralEncoder()
    return {
        "encoder": encoder,
        "claim_extractor": ClaimExtractor(),
        "evidence_extractor": EvidenceExtractor(),
        "linker": ClaimEvidenceLinker(encoder),
        "evidence_scorer": EvidenceScorer(),
        "aggregator": SupportAggregator(),
        "logic_engine": SymbolicGroundingEngine(),
        "formal_validator": FormalValidator(),
        "observation_builder": ObservationBuilder(),
        "pf": ParticleFilter(num_particles=300),
        "policy": ClaimDrivenPolicy(),
        "brief": EvaluationBrief(),
    }


def reset_run_state():
    st.session_state.run_initialized = False
    st.session_state.step_idx = 0
    st.session_state.section_order = []
    st.session_state.revealed_sections = {}
    st.session_state.all_claims = []
    st.session_state.all_evidences = []

    st.session_state.last_links = []
    st.session_state.last_scored_links = []
    st.session_state.last_claim_states = {}
    st.session_state.last_symbolic_aq = {}
    st.session_state.last_neural_signals = {}
    st.session_state.last_fused_unit = {}
    st.session_state.last_aq_targets = {}
    st.session_state.last_action_info = {}
    st.session_state.last_formal_validations = {}
    st.session_state.last_formal_validation_summary = {}
    st.session_state.last_brief_diff = {}
    st.session_state.last_observation = None
    st.session_state.last_latent_state = None
    st.session_state.history = []

    st.session_state.components = init_components()


def ensure_state():
    if "run_initialized" not in st.session_state:
        reset_run_state()


# ============================================================
# Parsing
# ============================================================

def parse_uploaded_files(uploaded_files) -> List[Tuple[str, str]]:
    sections = []
    for f in uploaded_files:
        try:
            content = StringIO(f.getvalue().decode("utf-8")).read()
        except UnicodeDecodeError:
            content = StringIO(f.getvalue().decode("utf-8", errors="ignore")).read()
        sections.append((f.name, content))
    sections.sort(key=lambda x: x[0])
    return sections


# ============================================================
# Dataframe helpers
# ============================================================

def belief_summary_to_df(summary: Dict) -> pd.DataFrame:
    rows = []
    for aq, stats in summary.items():
        rows.append({
            "AQ": aq,
            "Mean": stats["mean"],
            "Variance": stats["var"],
            "Std": stats["std"],
            "CI Low": stats["ci_low"],
            "CI High": stats["ci_high"],
        })
    return pd.DataFrame(rows)


def latent_meta_summary_to_df(meta: Dict) -> pd.DataFrame:
    return pd.DataFrame([meta])


def claim_states_to_df(claims: List[Dict], claim_states: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    claim_map = {c["id"]: c for c in claims}

    for cid, state in claim_states.items():
        c = claim_map.get(cid, {})
        rows.append({
            "Claim ID": cid,
            "Section": c.get("section", ""),
            "AQ": c.get("aq_type", ""),
            "Claim Type": c.get("claim_type", ""),
            "Text": c.get("text", ""),
            "Support Status": state.get("support_status", ""),
            "Support Score": state.get("support_score", 0.0),
            "Attack Score": state.get("attack_score", 0.0),
            "Missing Evidence": ", ".join(state.get("missing_evidence_types", [])),
            "Found Evidence Types": ", ".join(state.get("found_evidence_types", [])),
            "Reasoning Quality": state.get("reasoning_quality", 0.0),
            "Method Fit": state.get("method_fit", 0.0),
            "Clarity": state.get("clarity", 0.0),
            "Counterargument Handling": state.get("counterargument_handling", 0.0),
            "Overclaim Risk": state.get("overclaim_risk", 0.0),
            "Num Links": state.get("num_links", 0),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["AQ", "Support Score"], ascending=[True, False])


def scored_links_to_df(scored_links: List[Dict], all_claims: List[Dict], all_evidences: List[Dict]) -> pd.DataFrame:
    claim_map = {c["id"]: c for c in all_claims}
    ev_map = {e["id"]: e for e in all_evidences}

    rows = []
    for item in scored_links:
        c = claim_map.get(item["claim_id"], {})
        e = ev_map.get(item["evidence_id"], {})
        rows.append({
            "Claim AQ": c.get("aq_type", ""),
            "Claim Text": c.get("text", ""),
            "Evidence Types": ", ".join(e.get("evidence_types", [])),
            "Evidence Text": e.get("text", ""),
            "Relation": item.get("relation", ""),
            "Authority": item.get("authority", 0.0),
            "Relevance": item.get("relevance", 0.0),
            "Specificity": item.get("specificity", 0.0),
            "Consistency": item.get("consistency", 0.0),
            "Overall Strength": item.get("overall_strength", 0.0),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("Overall Strength", ascending=False)


def observed_claims_to_df(observed_claims) -> pd.DataFrame:
    rows = []
    for c in observed_claims:
        rows.append({
            "Claim ID": c.id,
            "Section": c.section,
            "AQ": c.aq_type,
            "Claim Type": c.claim_type,
            "Sentence Idx": c.sentence_idx,
            "Required Evidence": ", ".join(c.required_evidence),
            "Text": c.text,
        })
    return pd.DataFrame(rows)


def observed_evidences_to_df(observed_evidences) -> pd.DataFrame:
    rows = []
    for e in observed_evidences:
        rows.append({
            "Evidence ID": e.id,
            "Section": e.section,
            "Sentence Idx": e.sentence_idx,
            "Evidence Types": ", ".join(e.evidence_types),
            "Text": e.text,
        })
    return pd.DataFrame(rows)


def observed_relations_to_df(observed_relations) -> pd.DataFrame:
    rows = []
    for r in observed_relations:
        rows.append({
            "Claim ID": r.claim_id,
            "Evidence ID": r.evidence_id,
            "Relation Type": r.relation_type,
            "Candidate Score": r.candidate_score,
            "Semantic Similarity": r.semantic_similarity,
            "Type Score": r.type_score,
            "Section Score": r.section_score,
        })
    return pd.DataFrame(rows).sort_values("Candidate Score", ascending=False) if observed_relations else pd.DataFrame()


def open_questions_to_df(latent_state) -> pd.DataFrame:
    if latent_state is None:
        return pd.DataFrame()

    rows = []
    for cid in latent_state.unresolved_claim_ids:
        hyp = latent_state.claim_hypotheses[cid]
        rows.append({
            "Claim ID": cid,
            "Support Status": hyp.support_status,
            "Support Score": hyp.support_score,
            "Attack Score": hyp.attack_score,
            "Missing Evidence": ", ".join(hyp.missing_evidence_types),
            "Reasoning Complete": hyp.reasoning_complete,
            "Method Consistent": hyp.method_consistent,
            "Contradiction": hyp.contradiction_flag,
            "Formal Validity": hyp.formal_validity_score,
            "Reasoning Quality": hyp.reasoning_quality,
            "Method Fit": hyp.method_fit,
            "Clarity": hyp.clarity,
            "Counterargument Handling": hyp.counterargument_handling,
            "Overclaim Risk": hyp.overclaim_risk,
        })

    return pd.DataFrame(rows)


def formal_validations_to_df(formal_validations: Dict[str, Dict], claims: List[Dict]) -> pd.DataFrame:
    claim_map = {c["id"]: c for c in claims}
    rows = []

    for cid, item in formal_validations.items():
        claim = claim_map.get(cid, {})
        rows.append({
            "Claim ID": cid,
            "AQ": claim.get("aq_type", ""),
            "Claim Type": claim.get("claim_type", ""),
            "Claim Text": claim.get("text", ""),
            "Schema OK": item.get("schema_ok", False),
            "Missing Required": ", ".join(item.get("missing_required_any", [])),
            "Missing Preferred": ", ".join(item.get("missing_preferred", [])),
            "Reasoning Complete": item.get("reasoning_complete", False),
            "Reasoning Notes": "; ".join(item.get("reasoning_notes", [])),
            "Contradiction Flag": item.get("contradiction_flag", False),
            "Contradiction Notes": "; ".join(item.get("contradiction_notes", [])),
            "Formal Validity Score": item.get("formal_validity_score", 0.0),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("Formal Validity Score", ascending=True)


def brief_diff_to_df(brief_diff: Dict) -> pd.DataFrame:
    rows = []

    for item in brief_diff.get("changed_claims", []):
        rows.append({
            "Claim ID": item.get("claim_id", ""),
            "Change Type": item.get("change_type", ""),
            "Old Status": item.get("old_status", ""),
            "New Status": item.get("new_status", ""),
            "Old Missing": ", ".join(item.get("old_missing", []) or []),
            "New Missing": ", ".join(item.get("new_missing", []) or []),
        })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Charts
# ============================================================

def render_belief_errorbar_chart(summary: Dict):
    df = belief_summary_to_df(summary)

    base = alt.Chart(df).encode(
        x=alt.X("AQ:N", sort=AQ_LABELS, title="AQ Dimension"),
    )

    bars = base.mark_bar().encode(
        y=alt.Y("Mean:Q", title="Belief Mean", scale=alt.Scale(domain=[0, 6])),
        tooltip=["AQ", "Mean", "Variance", "Std", "CI Low", "CI High"],
    )

    error_bars = base.mark_errorbar().encode(
        y="CI Low:Q",
        y2="CI High:Q",
    )

    points = base.mark_point(filled=True, size=70).encode(
        y="Mean:Q",
    )

    chart = (bars + error_bars + points).properties(
        width=500,
        height=320,
        title="Belief state with uncertainty",
    )

    st.altair_chart(chart, use_container_width=True)


def animate_belief_transition(
    old_summary: Dict,
    new_summary: Dict,
    chart_placeholder,
    steps: int = 12,
    delay: float = 0.05,
):
    old_means = np.array([old_summary[aq]["mean"] for aq in AQ_LABELS], dtype=float)
    new_means = np.array([new_summary[aq]["mean"] for aq in AQ_LABELS], dtype=float)

    old_ci_low = np.array([old_summary[aq]["ci_low"] for aq in AQ_LABELS], dtype=float)
    new_ci_low = np.array([new_summary[aq]["ci_low"] for aq in AQ_LABELS], dtype=float)

    old_ci_high = np.array([old_summary[aq]["ci_high"] for aq in AQ_LABELS], dtype=float)
    new_ci_high = np.array([new_summary[aq]["ci_high"] for aq in AQ_LABELS], dtype=float)

    for alpha in np.linspace(0.0, 1.0, steps):
        means = (1 - alpha) * old_means + alpha * new_means
        ci_low = (1 - alpha) * old_ci_low + alpha * new_ci_low
        ci_high = (1 - alpha) * old_ci_high + alpha * new_ci_high

        df = pd.DataFrame({
            "AQ": AQ_LABELS,
            "Mean": means,
            "CI Low": ci_low,
            "CI High": ci_high,
        })

        base = alt.Chart(df).encode(
            x=alt.X("AQ:N", sort=AQ_LABELS, title="AQ Dimension"),
        )

        bars = base.mark_bar().encode(
            y=alt.Y("Mean:Q", title="Belief Mean", scale=alt.Scale(domain=[0, 6])),
            tooltip=["AQ", "Mean", "CI Low", "CI High"],
        )

        error_bars = base.mark_errorbar().encode(
            y="CI Low:Q",
            y2="CI High:Q",
        )

        points = base.mark_point(filled=True, size=70).encode(
            y="Mean:Q",
        )

        chart = (bars + error_bars + points).properties(
            width=500,
            height=320,
            title="Belief state with uncertainty",
        )

        chart_placeholder.altair_chart(chart, use_container_width=True)
        time.sleep(delay)


def render_action_box(action_info: Dict):
    action = action_info.get("action", "N/A")
    focus_aq = action_info.get("focus_aq")
    focus_claim_id = action_info.get("focus_claim_id")
    missing = action_info.get("missing_evidence", [])
    suggested = action_info.get("suggested_section")
    reason = action_info.get("reason", "")

    if action == "grade_now":
        st.success(f"**Action:** {action}")
    elif action == "read_next_section":
        st.info(f"**Action:** {action}")
    else:
        st.warning(f"**Action:** {action}")

    if focus_aq:
        st.write(f"**Focus AQ:** {focus_aq}")
    if focus_claim_id:
        st.write(f"**Target claim ID:** {focus_claim_id}")
    if missing:
        st.write(f"**Missing evidence:** {missing}")
    if suggested:
        st.write(f"**Suggested section:** {suggested}")
    st.write(f"**Reason:** {reason}")


# ============================================================
# Core step
# ============================================================

def choose_next_section_by_suggestion(
    sections: List[Tuple[str, str]],
    current_idx: int,
    suggested_section: Optional[str],
) -> Tuple[int, str, str]:
    remaining = sections[current_idx:]

    if len(remaining) == 0:
        raise IndexError("No remaining sections.")

    if suggested_section:
        for i, (name, text) in enumerate(remaining):
            if name == suggested_section:
                actual_index = current_idx + i
                return actual_index, name, text

    name, text = remaining[0]
    return current_idx, name, text


def validate_all_claims(
    claims: List[Dict],
    claim_states: Dict[str, Dict],
    validator: FormalValidator,
) -> Dict[str, Dict]:
    validations = {}

    for claim in claims:
        cid = claim["id"]
        state = claim_states[cid]
        found_types = state.get("found_evidence_types", [])
        validations[cid] = validator.validate_claim_hypothesis(
            claim=claim,
            found_evidence_types=found_types,
            support_score=state.get("support_score", 0.0),
            attack_score=state.get("attack_score", 0.0),
            method_fit=state.get("method_fit", 0.5),
            overclaim_risk=state.get("overclaim_risk", 0.0),
        )

    return validations


def one_step(sections: List[Tuple[str, str]]):
    comps = st.session_state.components
    brief = comps["brief"]
    pf = comps["pf"]
    policy = comps["policy"]

    if st.session_state.step_idx >= len(sections):
        belief_now = pf.summary()
        action_info = policy.choose_action(
            brief=brief,
            belief_summary=belief_now,
            step_idx=st.session_state.step_idx,
            remaining_sections=[],
        )
        st.session_state.last_action_info = action_info
        brief.set_policy_focus(action_info)
        return False

    belief_before = pf.summary()
    remaining_section_names = [name for name, _ in sections[st.session_state.step_idx:]]

    action_info = policy.choose_action(
        brief=brief,
        belief_summary=belief_before,
        step_idx=st.session_state.step_idx,
        remaining_sections=remaining_section_names,
    )

    st.session_state.last_action_info = action_info

    if action_info["action"] != "read_next_section":
        brief.set_policy_focus(action_info)
        return False

    actual_index, filename, text = choose_next_section_by_suggestion(
        sections,
        st.session_state.step_idx,
        action_info.get("suggested_section"),
    )

    if actual_index != st.session_state.step_idx:
        sections[st.session_state.step_idx], sections[actual_index] = (
            sections[actual_index],
            sections[st.session_state.step_idx],
        )

    old_snapshot = brief.snapshot()

    # reveal section
    st.session_state.revealed_sections[filename] = text

    # extract observed claims / evidences
    section_claims = comps["claim_extractor"].extract_claims(text, filename)
    section_evidences = comps["evidence_extractor"].extract_evidence(text, filename)

    st.session_state.all_claims.extend(section_claims)
    st.session_state.all_evidences.extend(section_evidences)

    # candidate relations
    links = comps["linker"].link(
        st.session_state.all_claims,
        st.session_state.all_evidences,
    )

    evidence_by_id = {e["id"]: e for e in st.session_state.all_evidences}
    claim_by_id = {c["id"]: c for c in st.session_state.all_claims}

    scored_links = []
    for link in links:
        claim = claim_by_id[link["claim_id"]]
        evidence = evidence_by_id[link["evidence_id"]]
        scored_links.append(
            comps["evidence_scorer"].score_evidence(claim, evidence, link)
        )

    # aggregate observable evidence into inferred claim states
    claim_states = comps["aggregator"].aggregate(
        st.session_state.all_claims,
        scored_links,
    )

    # formal validation over inferred claim hypotheses
    formal_validations = validate_all_claims(
        claims=st.session_state.all_claims,
        claim_states=claim_states,
        validator=comps["formal_validator"],
    )
    formal_validation_summary = comps["formal_validator"].summarize(formal_validations)

    # symbolic AQ
    symbolic_aq = comps["logic_engine"].claims_to_aq_scores(
        st.session_state.all_claims,
        claim_states,
    )

    # neural AQ
    neural_signals = comps["encoder"].get_section_neural_signals(text)
    neural_aq = {
        "LC": neural_signals["LC_shape"],
        "RE": neural_signals["RE"],
        "DR": neural_signals["DR_neural"],
        "DN": neural_signals["DN_neural"],
        "OQ": neural_signals["OQ_neural"],
    }

    # fused AQ
    aq_targets, aq_strengths, fused_unit = fuse_signals(neural_signals, symbolic_aq)

    # explicit observation graph
    observation_graph = comps["observation_builder"].build(
        timestep=st.session_state.step_idx + 1,
        raw_claims=st.session_state.all_claims,
        raw_evidences=st.session_state.all_evidences,
        raw_links=links,
    )

    # latent brief state hypothesis
    latent_state = build_latent_state_hypothesis(
        claims=st.session_state.all_claims,
        claim_states=claim_states,
        validations=formal_validations,
        symbolic_aq=symbolic_aq,
    )

    # explicit POMDP observation
    observation = pf.build_observation(
        observation_graph={
            "claims": observation_graph.claims,
            "evidences": observation_graph.evidences,
            "relations": observation_graph.candidate_relations,
        },
        latent_state_mean={
            "aq_scores": latent_state.aq_scores,
            "global_formal_validity": latent_state.global_formal_validity,
            "unresolved_claim_ids": latent_state.unresolved_claim_ids,
        },
        fused_aq=fused_unit,
        formal_validation_summary=formal_validation_summary,
    )

    pf.update_from_observation(
        observation=observation,
        action=action_info["action"],
        aq_strengths=aq_strengths,
    )

    # update brief
    brief.set_observed_claims(observation_graph.claims)
    brief.set_observed_evidences(observation_graph.evidences)
    brief.set_observed_relations(observation_graph.candidate_relations)
    brief.set_latent_state_mean(latent_state)
    brief.set_policy_focus(action_info)

    # diff
    old_latent = old_snapshot.get("latent_state_mean")
    changed_claims = []
    if old_latent is not None and latent_state is not None:
        for cid, new_hyp in latent_state.claim_hypotheses.items():
            old_hyp = old_latent.claim_hypotheses.get(cid) if cid in old_latent.claim_hypotheses else None
            if old_hyp is None:
                changed_claims.append({
                    "claim_id": cid,
                    "change_type": "new_claim_hypothesis",
                    "old_status": None,
                    "new_status": new_hyp.support_status,
                    "old_missing": [],
                    "new_missing": new_hyp.missing_evidence_types,
                })
            else:
                if (
                    old_hyp.support_status != new_hyp.support_status
                    or old_hyp.missing_evidence_types != new_hyp.missing_evidence_types
                ):
                    changed_claims.append({
                        "claim_id": cid,
                        "change_type": "updated_claim_hypothesis",
                        "old_status": old_hyp.support_status,
                        "new_status": new_hyp.support_status,
                        "old_missing": old_hyp.missing_evidence_types,
                        "new_missing": new_hyp.missing_evidence_types,
                    })

    st.session_state.last_brief_diff = {"changed_claims": changed_claims}

    # cache for UI
    st.session_state.last_links = links
    st.session_state.last_scored_links = scored_links
    st.session_state.last_claim_states = claim_states
    st.session_state.last_symbolic_aq = symbolic_aq
    st.session_state.last_neural_signals = neural_signals
    st.session_state.last_fused_unit = fused_unit
    st.session_state.last_aq_targets = aq_targets
    st.session_state.last_formal_validations = formal_validations
    st.session_state.last_formal_validation_summary = formal_validation_summary
    st.session_state.last_observation = observation_graph
    st.session_state.last_latent_state = latent_state

    belief_after = pf.summary()
    latent_meta = pf.latent_meta_summary()

    # history
    history_item = {
        "step": st.session_state.step_idx + 1,
        "file": filename,
        "action": action_info["action"],
        "focus_aq": action_info.get("focus_aq"),
        "focus_claim_id": action_info.get("focus_claim_id"),
        "suggested_section": action_info.get("suggested_section"),
        "reason": action_info.get("reason"),
        "claims_added": len(section_claims),
        "evidence_added": len(section_evidences),
        "num_links": len(links),
        "mean_formal_validity": formal_validation_summary.get("mean_formal_validity", None),
        "latent_formal_validity": latent_meta.get("mean_formal_validity_latent", None),
        "latent_unresolved_claims": latent_meta.get("mean_unresolved_claims_latent", None),
        "belief_before": belief_before,
        "belief_after": belief_after,
    }
    st.session_state.history.append(history_item)
    brief.add_action_record(history_item)

    st.session_state.step_idx += 1
    return True


# ============================================================
# View
# ============================================================

def render_current_view(show_full_tables: bool):
    comps = st.session_state.components
    brief = comps["brief"]
    pf = comps["pf"]

    current_summary = pf.summary()
    latent_meta = pf.latent_meta_summary()

    left_col, right_col = st.columns([1.1, 1.0])

    with left_col:
        st.subheader("Policy action")
        if st.session_state.last_action_info:
            render_action_box(st.session_state.last_action_info)
        else:
            st.write("No action taken yet.")

        st.subheader("Belief state / uncertainty")
        render_belief_errorbar_chart(current_summary)
        st.dataframe(belief_summary_to_df(current_summary), use_container_width=True)

        st.subheader("Latent meta summary")
        st.dataframe(latent_meta_summary_to_df(latent_meta), use_container_width=True)

        st.subheader("Latest fused AQ targets")
        if st.session_state.last_aq_targets:
            df_targets = pd.DataFrame({
                "AQ": list(st.session_state.last_aq_targets.keys()),
                "Target Score (1-6)": list(st.session_state.last_aq_targets.values()),
            })
            st.dataframe(df_targets, use_container_width=True)
        else:
            st.write("No target scores yet.")

        st.subheader("Formal validation summary")
        if st.session_state.last_formal_validation_summary:
            st.dataframe(
                pd.DataFrame([st.session_state.last_formal_validation_summary]),
                use_container_width=True,
            )
        else:
            st.write("No formal validation summary yet.")

        st.subheader("Top unresolved latent claims")
        open_df = open_questions_to_df(brief.latent_state_mean)
        if len(open_df) > 0:
            st.dataframe(open_df if show_full_tables else open_df.head(10), use_container_width=True)
        else:
            st.write("No unresolved claims.")

        st.subheader("Brief diff from last step")
        diff_df = brief_diff_to_df(st.session_state.last_brief_diff)
        if len(diff_df) > 0:
            st.dataframe(diff_df, use_container_width=True)
        else:
            st.write("No latent claim-status changes to show yet.")

    with right_col:
        st.subheader("Revealed sections")
        if st.session_state.revealed_sections:
            for name, text in st.session_state.revealed_sections.items():
                with st.expander(name, expanded=False):
                    st.write(text)
        else:
            st.write("No sections revealed yet.")

        st.subheader("Observed claims")
        if brief.observed_claims:
            df_obs_claims = observed_claims_to_df(brief.observed_claims)
            st.dataframe(df_obs_claims if show_full_tables else df_obs_claims.head(12), use_container_width=True)
        else:
            st.write("No observed claims yet.")

        st.subheader("Observed evidences")
        if brief.observed_evidences:
            df_obs_evidences = observed_evidences_to_df(brief.observed_evidences)
            st.dataframe(df_obs_evidences if show_full_tables else df_obs_evidences.head(12), use_container_width=True)
        else:
            st.write("No observed evidences yet.")

        st.subheader("Observed candidate relations")
        if brief.observed_relations:
            df_obs_rel = observed_relations_to_df(brief.observed_relations)
            st.dataframe(df_obs_rel if show_full_tables else df_obs_rel.head(12), use_container_width=True)
        else:
            st.write("No observed relations yet.")

        st.subheader("Inferred claim states")
        df_claims = claim_states_to_df(st.session_state.all_claims, st.session_state.last_claim_states)
        if len(df_claims) > 0:
            st.dataframe(df_claims if show_full_tables else df_claims.head(12), use_container_width=True)
        else:
            st.write("No inferred claim states yet.")

        st.subheader("Scored claim–evidence links")
        df_links = scored_links_to_df(
            st.session_state.last_scored_links,
            st.session_state.all_claims,
            st.session_state.all_evidences,
        )
        if len(df_links) > 0:
            st.dataframe(df_links if show_full_tables else df_links.head(12), use_container_width=True)
        else:
            st.write("No scored links yet.")

        st.subheader("Formal validation by claim")
        df_val = formal_validations_to_df(
            st.session_state.last_formal_validations,
            st.session_state.all_claims,
        )
        if len(df_val) > 0:
            st.dataframe(df_val if show_full_tables else df_val.head(12), use_container_width=True)
        else:
            st.write("No formal validation results yet.")

    st.subheader("Action history")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        display_cols = [
            "step",
            "file",
            "action",
            "focus_aq",
            "focus_claim_id",
            "suggested_section",
            "claims_added",
            "evidence_added",
            "num_links",
            "mean_formal_validity",
            "latent_formal_validity",
            "latent_unresolved_claims",
            "reason",
        ]
        st.dataframe(hist_df[display_cols], use_container_width=True)
    else:
        st.write("No history yet.")


# ============================================================
# App flow
# ============================================================

ensure_state()

with st.sidebar:
    st.header("Controls")

    uploaded_files = st.file_uploader(
        "Upload thesis sections (.txt)",
        type=["txt"],
        accept_multiple_files=True,
    )

    auto_delay = st.slider("Animation delay (seconds)", 0.00, 0.50, 0.08, 0.01)
    show_full_tables = st.checkbox("Show full tables", value=False)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset"):
            reset_run_state()
            st.rerun()
    with col_b:
        init_clicked = st.button("Initialize")

sections = parse_uploaded_files(uploaded_files) if uploaded_files else []

if init_clicked:
    reset_run_state()
    st.session_state.section_order = [name for name, _ in sections]
    st.session_state.run_initialized = True

if not st.session_state.run_initialized:
    st.info("Upload `.txt` section files, then click **Initialize**.")
    st.stop()

if len(sections) == 0:
    st.warning("No files uploaded.")
    st.stop()

top_left, top_right = st.columns([1.2, 1])

with top_left:
    st.subheader("Section order")
    st.write([name for name, _ in sections])

with top_right:
    st.subheader("Run status")
    st.write(f"Current step: **{st.session_state.step_idx} / {len(sections)}**")
    st.write(f"Raw claims in memory: **{len(st.session_state.all_claims)}**")
    st.write(f"Raw evidence items in memory: **{len(st.session_state.all_evidences)}**")

ctrl1, ctrl2 = st.columns([1, 1])

with ctrl1:
    step_clicked = st.button("Run one step")

with ctrl2:
    run_all_clicked = st.button("Run to end")

if step_clicked:
    old_summary = st.session_state.components["pf"].summary()
    progressed = one_step(sections)
    new_summary = st.session_state.components["pf"].summary()

    if progressed:
        anim_placeholder = st.empty()
        animate_belief_transition(
            old_summary,
            new_summary,
            anim_placeholder,
            steps=14,
            delay=auto_delay,
        )

if run_all_clicked:
    while True:
        old_summary = st.session_state.components["pf"].summary()
        progressed = one_step(sections)
        new_summary = st.session_state.components["pf"].summary()

        if progressed:
            anim_placeholder = st.empty()
            animate_belief_transition(
                old_summary,
                new_summary,
                anim_placeholder,
                steps=10,
                delay=auto_delay,
            )

        if not progressed:
            break

render_current_view(show_full_tables=show_full_tables)