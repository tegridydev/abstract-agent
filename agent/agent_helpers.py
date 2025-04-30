# agent_helpers.py
# Helper functions for the abstract-agent pipeline
# Author: tegridydev
# Repo: https://github.com/tegridydev/abstract-agent
# License: MIT
# Year: 2025

from typing import List, Dict, Any
import re

def format_papers_for_prompt(top_papers: List[Dict[str, Any]], max_length: int = 500) -> str:
    """
    Formats a list of paper dictionaries into a string suitable for LLM prompts.
    Accepts 'top_papers' as the keyword argument for the list.

    Args:
        top_papers: A list of dictionaries, where each dict represents a paper.
        max_length: The maximum character length for each paper's summary in the output string.
                    Set to 0 or negative for no truncation.

    Returns:
        A formatted string listing the papers and their summaries.
    """
    if not top_papers:
        return "No relevant papers found or provided."

    formatted_string = ""
    for i, p in enumerate(top_papers):
        title = p.get('title', 'N/A')
        year = p.get('year', 'N/A')
        source = p.get('source', 'N/A')
        summary = str(p.get('summary', '')).strip()

        if max_length > 0 and len(summary) > max_length:
            summary = summary[:max_length] + "..."

        formatted_string += f"Paper {i+1}:\n"
        formatted_string += f"  Title: {title}\n"
        formatted_string += f"  Source: {source} ({year})\n"
        # formatted_string += f"  Authors: {authors}\n" # Uncomment if authors are needed
        formatted_string += f"  Summary: {summary}\n\n"

    return formatted_string.strip()


def format_citations_rich(citations: list) -> str:
    """
    Formats citations list into a simple string summary suitable for console display (using Rich).
    This is NOT intended for feeding back into an LLM prompt.

    Args:
        citations: A list of paper dictionaries.

    Returns:
        A formatted string summarizing the citations for display.
    """
    if not citations:
        return "No relevant papers found."

    summary_lines = ["Top Relevant papers found:"]
    for paper in citations:
        source = paper.get('source', 'N/A')
        title = paper.get('title', 'N/A')
        year = paper.get('year', 'N/A')
        score = paper.get('composite_score', 0)
        url = paper.get('url', 'N/A')

        summary_lines.append(f"- [{source}] {title} ({year})")
        summary_lines.append(f"  Score: {score:.2f} | URL: {url}")

    return "\n".join(summary_lines)

# You can add more helper functions here as needed for the pipeline
# For example, a function to parse the novelty score from text:
def calculate_novelty_score_from_text(assessment_text: str) -> float:
     """
     Parses Ollama novelty assessment text to extract a numerical score (e.g., X/10).
     Returns a float between 0.0 and 1.0, or 0.5 if parsing fails.

     Args:
         assessment_text: The text output from the novelty assessment LLM call.

     Returns:
         A float score between 0.0 and 1.0 (normalized from /10 scale) or 0.5 default.
     """
     if not isinstance(assessment_text, str):
         return 0.5

     match = re.search(r'(?:score|rating)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:/|out\s+of\s*)10', assessment_text, re.IGNORECASE)

     if match:
         try:
             score = float(match.group(1))
             return max(0.0, min(1.0, score / 10.0))
         except ValueError:
             print(f"[Warning] Could not parse score value '{match.group(1)}' as float.")
             return 0.5
     else:
         print("[Warning] Novelty score pattern (e.g., 'Score: X/10') not found in assessment text.")
         return 0.5