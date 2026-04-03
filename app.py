import time
from io import StringIO
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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


AQ_LABELS = ["LC", "RE", "DR", "DN", "OQ"]


st.set_page_config(page_title="NeSy-POMDP Thesis Evaluation Demo", page_icon="⚖️", layout="wide")
st.title("⚖️ Neuro-Symbolic POMDP Thesis Evaluation Demo")
st.caption("Sequential claim–evidence grounding, evolving evaluation brief, and claim-driven action policy")


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
    st.session_state.history = []
    st.session_state.components = init_components()


def ensure_state():
    if "run_initialized" not in st.session_state:
        reset_run_state()


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


def belief_summary_to_df(summary: Dict) -> pd.DataFrame:
    rows = []
    for aq, stats in summary.items():
        rows.append({"AQ": aq, "Mean": stats["mean"], "Variance": stats["var"]})
    return pd.DataFrame(rows)


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
            "Num Links": state.get("num_links", 0),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["Support Status", "Support Score"], ascending=[True, True])


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


def brief_open_questions_df(brief: EvaluationBrief) -> pd.DataFrame:
    rows = []
    for q in brief.top_open_questions(20):
        rows.append({
            "AQ": q.get("aq_type"),
            "Claim": q.get("claim_text"),
            "Support Status": q.get("support_status"),
            "Support Score": q.get("support_score"),
            "Attack Score": q.get("attack_score"),
            "Missing Evidence": ", ".join(q.get("missing_evidence_types", [])),
        })
    return pd.DataFrame(rows)


def animate_belief_transition(old_summary: Dict, new_summary: Dict, chart_placeholder, steps: int = 12, delay: float = 0.05):
    old_means = np.array([old_summary[aq]["mean"] for aq in AQ_LABELS], dtype=float)
    new_means = np.array([new_summary[aq]["mean"] for aq in AQ_LABELS], dtype=float)
    for alpha in np.linspace(0.0, 1.0, steps):
        means = (1 - alpha) * old_means + alpha * new_means
        df = pd.DataFrame({"AQ": AQ_LABELS, "Mean": means}).set_index("AQ")
        chart_placeholder.bar_chart(df[["Mean"]])
        time.sleep(delay)


def render_action_box(action_info: Dict):
    action = action_info.get("action", "N/A")
    focus = action_info.get("focus_aq")
    claim_text = action_info.get("claim_text")
    reason = action_info.get("reason", "")
    suggested = action_info.get("suggested_section")
    missing = action_info.get("missing_evidence_types", [])

    if action == "grade_now":
        st.success(f"**Action:** {action}")
    elif action == "read_next_section":
        st.info(f"**Action:** {action}")
    else:
        st.warning(f"**Action:** {action}")

    if focus is not None:
        st.write(f"**Focus AQ:** {focus}")
    if claim_text:
        st.write(f"**Target claim:** {claim_text}")
    if missing:
        st.write(f"**Missing evidence:** {missing}")
    if suggested:
        st.write(f"**Suggested section:** {suggested}")
    st.write(f"**Reason:** {reason}")


def compute_brief_diff(old_states: Dict[str, Dict], new_states: Dict[str, Dict], claims: List[Dict]) -> pd.DataFrame:
    claim_map = {c['id']: c for c in claims}
    rows = []
    for cid, new_state in new_states.items():
        old_state = old_states.get(cid)
        if old_state is None or old_state.get('support_status') != new_state.get('support_status'):
            c = claim_map.get(cid, {})
            rows.append({
                'AQ': c.get('aq_type', ''),
                'Claim': c.get('text', ''),
                'Old Status': old_state.get('support_status') if old_state else '(new)',
                'New Status': new_state.get('support_status'),
                'Missing Evidence': ', '.join(new_state.get('missing_evidence_types', [])),
            })
    return pd.DataFrame(rows)


def one_step(sections: List[Tuple[str, str]]):
    comps = st.session_state.components
    pf = comps["pf"]
    brief = comps["brief"]

    remaining_sections = [name for name, _ in sections[st.session_state.step_idx:]]
    belief_before = pf.summary()
    old_claim_states = dict(brief.claim_states)

    action_info = comps["policy"].choose_action(
        belief_summary=belief_before,
        brief=brief,
        step_idx=st.session_state.step_idx,
        remaining_sections=remaining_sections,
    )
    brief.set_policy_focus(action_info)
    st.session_state.last_action_info = action_info

    if action_info["action"] != "read_next_section":
        return False, pd.DataFrame()

    suggested = action_info.get("suggested_section")
    current_remaining = sections[st.session_state.step_idx:]
    offset = 0
    if suggested:
        for i, (name, _) in enumerate(current_remaining):
            if name == suggested:
                offset = i
                break
    filename, text = current_remaining[offset]

    actual_index = st.session_state.step_idx + offset
    sections[st.session_state.step_idx], sections[actual_index] = sections[actual_index], sections[st.session_state.step_idx]

    st.session_state.revealed_sections[filename] = text
    section_claims = comps["claim_extractor"].extract_claims(text, filename)
    section_evidences = comps["evidence_extractor"].extract_evidence(text, filename)
    st.session_state.all_claims.extend(section_claims)
    st.session_state.all_evidences.extend(section_evidences)

    links = comps["linker"].link(st.session_state.all_claims, st.session_state.all_evidences)
    evidence_by_id = {e["id"]: e for e in st.session_state.all_evidences}
    claim_by_id = {c["id"]: c for c in st.session_state.all_claims}
    scored_links = []
    for link in links:
        claim = claim_by_id[link["claim_id"]]
        evidence = evidence_by_id[link["evidence_id"]]
        scored_links.append(comps["evidence_scorer"].score_evidence(claim, evidence, link))

    claim_states = comps["aggregator"].aggregate(st.session_state.all_claims, scored_links, evidence_by_id)
    pf.update_claim_memory(claim_states)

    symbolic_aq = comps["logic_engine"].claims_to_aq_scores(st.session_state.all_claims, claim_states)
    neural_signals = comps["encoder"].get_section_neural_signals(text)
    aq_targets, aq_strengths, fused_unit = fuse_signals(neural_signals, symbolic_aq)
    pf.update(aq_targets=aq_targets, aq_strengths=aq_strengths)

    brief.sync(st.session_state.all_claims, st.session_state.all_evidences, links, scored_links, claim_states, symbolic_aq)
    brief.set_policy_focus(action_info)

    belief_after = pf.summary()
    diff_df = compute_brief_diff(old_claim_states, claim_states, st.session_state.all_claims)

    st.session_state.last_links = links
    st.session_state.last_scored_links = scored_links
    st.session_state.last_claim_states = claim_states
    st.session_state.last_symbolic_aq = symbolic_aq
    st.session_state.last_neural_signals = neural_signals
    st.session_state.last_fused_unit = fused_unit
    st.session_state.last_aq_targets = aq_targets

    st.session_state.history.append({
        "step": st.session_state.step_idx + 1,
        "file": filename,
        "action": action_info["action"],
        "focus_aq": action_info.get("focus_aq"),
        "focus_claim_id": action_info.get("focus_claim_id"),
        "suggested_section": action_info.get("suggested_section"),
        "reason": action_info.get("reason"),
        "belief_before": belief_before,
        "belief_after": belief_after,
        "claims_added": len(section_claims),
        "evidence_added": len(section_evidences),
        "num_links": len(links),
    })

    st.session_state.step_idx += 1
    return True, diff_df


def render_current_view(show_full_tables: bool, diff_df: pd.DataFrame = None):
    comps = st.session_state.components
    brief = comps["brief"]
    current_summary = comps["pf"].summary()

    left_col, right_col = st.columns([1.1, 1.0])

    with left_col:
        st.subheader("Policy action")
        if st.session_state.last_action_info:
            render_action_box(st.session_state.last_action_info)
        else:
            st.write("No action taken yet.")

        st.subheader("Belief state / brief evolution")
        df_belief = belief_summary_to_df(current_summary).set_index("AQ")
        st.bar_chart(df_belief[["Mean"]])
        st.dataframe(belief_summary_to_df(current_summary), use_container_width=True)

        st.subheader("Latest fused AQ targets")
        if st.session_state.last_aq_targets:
            df_targets = pd.DataFrame({
                "AQ": list(st.session_state.last_aq_targets.keys()),
                "Target Score (1-6)": list(st.session_state.last_aq_targets.values())
            })
            st.dataframe(df_targets, use_container_width=True)
        else:
            st.write("No target scores yet.")

        st.subheader("Top open questions in brief")
        oq_df = brief_open_questions_df(brief)
        if len(oq_df) > 0:
            st.dataframe(oq_df if show_full_tables else oq_df.head(10), use_container_width=True)
        else:
            st.write("No open questions.")

        st.subheader("Brief diff from last step")
        if diff_df is not None and len(diff_df) > 0:
            st.dataframe(diff_df, use_container_width=True)
        else:
            st.write("No claim-status changes to show yet.")

    with right_col:
        st.subheader("Revealed sections")
        if st.session_state.revealed_sections:
            for name, text in st.session_state.revealed_sections.items():
                with st.expander(name, expanded=False):
                    st.write(text)
        else:
            st.write("No sections revealed yet.")

        st.subheader("Claim states")
        df_claims = claim_states_to_df(st.session_state.all_claims, st.session_state.last_claim_states)
        if len(df_claims) > 0:
            st.dataframe(df_claims if show_full_tables else df_claims.head(12), use_container_width=True)
        else:
            st.write("No claim states yet.")

        st.subheader("Claim–evidence links")
        df_links = scored_links_to_df(st.session_state.last_scored_links, st.session_state.all_claims, st.session_state.all_evidences)
        if len(df_links) > 0:
            st.dataframe(df_links if show_full_tables else df_links.head(12), use_container_width=True)
        else:
            st.write("No scored links yet.")

    st.subheader("Action history")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        display_cols = ["step", "file", "action", "focus_aq", "focus_claim_id", "suggested_section", "claims_added", "evidence_added", "num_links", "reason"]
        st.dataframe(hist_df[display_cols], use_container_width=True)
    else:
        st.write("No history yet.")


ensure_state()

with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader("Upload thesis sections (.txt)", type=["txt"], accept_multiple_files=True)
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
    st.write(f"Claims in memory: **{len(st.session_state.all_claims)}**")
    st.write(f"Evidence items in memory: **{len(st.session_state.all_evidences)}**")

ctrl1, ctrl2 = st.columns([1, 1])
with ctrl1:
    step_clicked = st.button("Run one step")
with ctrl2:
    run_all_clicked = st.button("Run to end")

brief_diff_df = None
if step_clicked:
    old_summary = st.session_state.components["pf"].summary()
    progressed, brief_diff_df = one_step(sections)
    new_summary = st.session_state.components["pf"].summary()
    if progressed:
        anim_placeholder = st.empty()
        animate_belief_transition(old_summary, new_summary, anim_placeholder, steps=14, delay=auto_delay)

if run_all_clicked:
    while True:
        old_summary = st.session_state.components["pf"].summary()
        progressed, brief_diff_df = one_step(sections)
        new_summary = st.session_state.components["pf"].summary()
        if progressed:
            anim_placeholder = st.empty()
            animate_belief_transition(old_summary, new_summary, anim_placeholder, steps=10, delay=auto_delay)
        if not progressed:
            break

render_current_view(show_full_tables=show_full_tables, diff_df=brief_diff_df)
