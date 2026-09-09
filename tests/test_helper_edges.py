from __future__ import annotations

from typing import Any

import pytest

from agent import agent_helpers as helpers


def candidate(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "title": candidate_id,
        "mechanism": "compression mechanism",
        "hypothesis": "compression improves performance",
        "supporting_paper_ids": ["p1"],
    }


def critique(candidate_id: str, score: int = 3) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "grounding": score,
        "novelty": score,
        "falsifiability": score,
        "feasibility": score,
        "clarity": score,
    }


def test_reference_normalization_ignores_invalid_allowlist_and_non_strings() -> None:
    value = {"paper_id": 42, "supporting_paper_ids": "not-list", "nested": [1]}
    normalized, corrections, unknown = helpers.normalize_paper_references(
        value, [{"paper_id": None}, {"paper_id": " "}]
    )
    assert normalized == value
    assert corrections == []
    assert unknown == set()


def test_prompt_formatter_empty_invalid_and_unlimited_budget() -> None:
    assert helpers.format_papers_for_prompt([]) == "No relevant papers found or provided."
    for kwargs in ({"max_length": 0}, {"max_total_length": -1}, {"max_authors": 0}):
        with pytest.raises(ValueError, match="limits must be positive"):
            helpers.format_papers_for_prompt([{"title": "paper"}], **kwargs)
    rendered = helpers.format_papers_for_prompt(
        [{"title": "paper", "authors": "A"}], max_total_length=0
    )
    assert "paper-1" in rendered


def test_sentence_and_author_compaction_edges() -> None:
    assert helpers._sentence_excerpt("  short  text ", 0) == "short text"
    assert helpers._sentence_excerpt("First sentence. Second sentence is long", 20).endswith("...")
    assert helpers._compact_authors(["A", "B"], 3) == "A, B"
    assert helpers._compact_authors(None, 3) == "N/A"


def test_compact_search_query_validation_deduplication_and_limit() -> None:
    with pytest.raises(ValueError, match="at least three"):
        helpers.compact_search_query("one two", max_terms=2)
    with pytest.raises(ValueError, match="too few"):
        helpers.compact_search_query("the and of")
    assert helpers.compact_search_query("alpha alpha beta gamma", max_terms=3) == (
        "alpha beta gamma"
    )


def test_search_builders_skip_blanks_duplicates_and_build_novelty_queries() -> None:
    plan = {
        "mechanism_query": "compression algorithm",
        "approaches_query": "compression algorithm",
        "prior_art_query": "",
    }
    queries = helpers.build_search_query(plan, "compression algorithm")
    assert len(queries) == 1
    novelty = helpers.build_novelty_queries(candidate("a"), candidate("b"))
    assert [item["candidate_id"] for item in novelty] == ["a", "b"]


def test_relevance_filter_rejects_duplicate_and_unknown_assessments() -> None:
    papers = [{"paper_id": "p1"}]
    decision = {"paper_id": "p1", "score": 5, "role": "direct_evidence"}
    with pytest.raises(ValueError, match="duplicate"):
        helpers.filter_relevant_papers(papers, {"assessments": [decision, decision]})
    with pytest.raises(ValueError, match="unknown"):
        helpers.filter_relevant_papers(
            papers,
            {"assessments": [{"paper_id": "unknown", "score": 5, "role": "background"}]},
        )


def test_paper_batch_size_and_relevance_merge_conflict() -> None:
    for size in (0, 16):
        with pytest.raises(ValueError, match="between 1 and 15"):
            helpers.prepare_paper_batches([], batch_size=size)
    first = {"paper_id": "p1", "score": 3, "role": "background"}
    assert helpers.merge_relevance_assessments(
        [{"assessments": [first]}, {"assessments": [dict(first)]}]
    )["assessments"] == [first]
    with pytest.raises(ValueError, match="conflict"):
        helpers.merge_relevance_assessments(
            [
                {"assessments": [first]},
                {"assessments": [{**first, "score": 4}]},
            ]
        )


def test_evidence_recovery_rejects_duplicates_unknown_conflicts_and_gaps() -> None:
    papers = [{"paper_id": "p1"}, {"paper_id": "p2"}]
    duplicate = {"cards": [{"paper_id": "p1"}, {"paper_id": "p1"}]}
    with pytest.raises(ValueError, match="duplicate"):
        helpers.prepare_missing_evidence_cards(papers, duplicate)
    with pytest.raises(ValueError, match="unknown"):
        helpers.prepare_missing_evidence_cards(papers, {"cards": [{"paper_id": "x"}]})
    with pytest.raises(ValueError, match="conflict"):
        helpers.merge_evidence_ledgers(
            papers,
            {"cards": [{"paper_id": "p1", "findings": []}]},
            [{"cards": [{"paper_id": "p1", "findings": ["different"]}]}],
        )
    with pytest.raises(ValueError, match="coverage mismatch"):
        helpers.merge_evidence_ledgers(papers, {"cards": [{"paper_id": "p1"}]}, [])


def test_candidate_selection_rejects_bad_weights_ids_and_critic_coverage() -> None:
    candidates = [candidate("a"), candidate("b")]
    valid = {"critiques": [critique("a"), critique("b")]}
    with pytest.raises(ValueError, match="sum to 1.0"):
        helpers.select_candidate(candidates, valid, valid, grounding_weight=-1)
    with pytest.raises(ValueError, match="unique"):
        helpers.select_candidate([candidate("a"), candidate("a")], valid, valid)
    with pytest.raises(ValueError, match="every hypothesis"):
        helpers.select_candidate(candidates, {"critiques": [critique("a")]}, valid)


def test_prepare_candidate_critiques_rejects_invalid_identity_sets() -> None:
    candidates = [candidate("a"), candidate("b")]
    with pytest.raises(ValueError, match="unique"):
        helpers.prepare_missing_candidate_critiques([candidate("a"), candidate("a")], {})
    with pytest.raises(ValueError, match="duplicate"):
        helpers.prepare_missing_candidate_critiques(
            candidates, {"critiques": [critique("a"), critique("a")]}
        )
    with pytest.raises(ValueError, match="unknown"):
        helpers.prepare_missing_candidate_critiques(candidates, {"critiques": [critique("x")]})
    with pytest.raises(ValueError, match="unique"):
        helpers.merge_candidate_critiques([candidate("a"), candidate("a")], {}, [])


def test_novelty_recovery_rejects_duplicates_unknown_conflict_and_coverage() -> None:
    first, second = candidate("a"), candidate("b")
    with pytest.raises(ValueError, match="duplicate"):
        helpers.prepare_missing_novelty_verifications(
            first,
            second,
            {"verifications": [{"candidate_id": "a"}, {"candidate_id": "a"}]},
        )
    with pytest.raises(ValueError, match="unknown"):
        helpers.prepare_missing_novelty_verifications(
            first, second, {"verifications": [{"candidate_id": "x"}]}
        )
    with pytest.raises(ValueError, match="conflict"):
        helpers.merge_novelty_verifications(
            first,
            second,
            {"verifications": [{"candidate_id": "a", "status": "uncertain"}]},
            [{"verifications": [{"candidate_id": "a", "status": "novelty_supported"}]}],
        )
    with pytest.raises(ValueError, match="coverage mismatch"):
        helpers.merge_novelty_verifications(
            first, second, {"verifications": [{"candidate_id": "a"}]}, []
        )


def test_choose_verified_candidate_rejects_incomplete_or_duplicate_assessments() -> None:
    first, second = candidate("a"), candidate("b")
    with pytest.raises(ValueError, match="duplicate"):
        helpers.choose_verified_candidate(
            first,
            second,
            {"verifications": [{"candidate_id": "a"}, {"candidate_id": "a"}]},
        )
    with pytest.raises(ValueError, match="winner and alternate"):
        helpers.choose_verified_candidate(first, second, {"verifications": [{"candidate_id": "a"}]})


def test_merge_evidence_papers_keeps_best_metadata() -> None:
    merged = helpers.merge_evidence_papers(
        [{"paper_id": "p1", "composite_score": 2, "title": "best"}],
        [
            {"paper_id": "p1", "composite_score": 1, "title": "worse"},
            {"paper_id": "p2", "composite_score": 1},
        ],
    )
    assert [paper.get("title") for paper in merged] == ["best", None]


def test_proposal_validation_empty_claim_status_and_success_paths() -> None:
    papers = [{"paper_id": "p1", "relevance_judgement": {"role": "direct_evidence"}}]
    with pytest.raises(ValueError, match="must cite"):
        helpers.validate_research_proposal(
            {
                "proposal_status": "novelty_supported",
                "claim_evidence": [{"relationship": "supports", "paper_ids": []}],
            },
            papers,
            {"status": "novelty_supported"},
        )
    with pytest.raises(ValueError, match="proposal_status"):
        helpers.validate_research_proposal(
            {"proposal_status": "novelty_supported", "references": [], "claim_evidence": []},
            papers,
            {"status": "uncertain"},
        )
    proposal = {
        "proposal_status": "novelty_supported",
        "references": ["p1"],
        "claim_evidence": [
            {"relationship": "supports", "paper_ids": ["p1"]},
            {"relationship": "proposed_inference", "paper_ids": []},
        ],
    }
    _, audit = helpers.validate_research_proposal(proposal, papers, {"status": "novelty_supported"})
    assert audit["evidence_coverage"] == 0.5


def test_format_citations_rich_empty_and_populated() -> None:
    assert helpers.format_citations_rich([]) == "No relevant papers found."
    rendered = helpers.format_citations_rich(
        [{"paper_id": "p1", "title": "Paper", "composite_score": 0.5}]
    )
    assert "Paper" in rendered and "0.50" in rendered
