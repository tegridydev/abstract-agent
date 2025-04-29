# abstract-agent
# Author: tegridydev
# Repo: https://github.com/tegridydev/abstract-agent
# License: MIT
# Year: 2025

import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List

import yaml
import arxiv
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.panel import Panel
from ollama import Client

console = Console()

# --- Load Agent Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'agents_config.yaml')
def load_agent_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

AGENT_CONFIG = load_agent_config()

# --- Core Agent Class ---
import arxiv
from multi_source import aggregate_sources

class AbstractAgent:
    def __init__(self, name: str, persona: str, prompt_template: str, model: str = 'qwen3:0.6b', host: str = 'http://localhost:11434'):
        self.name = name
        self.persona = persona
        self.prompt_template = prompt_template
        self.model = model
        self.client = Client(host=host)

    def ask(self, prompt: str) -> str:
        response = self.client.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']

    def run(self, context: dict) -> str:
        prompt = self.prompt_template.format(**context)
        return self.ask(prompt)

    def search_multisource(self, topic: str, max_results: int = 5) -> list:
        return aggregate_sources(topic, max_results)

    def format_citations(self, citations: list) -> str:
        if not citations:
            return "No relevant papers found."
        summary = "Relevant papers from multiple sources:\n"
        for paper in citations:
            summary += f"- [{paper['source']}] {paper['title']} ({paper.get('year','')})\n  {paper.get('url','')}\n  Authors: {paper.get('authors','')}\n  Summary: {str(paper.get('summary',''))[:300].strip()}...\n"
        return summary

# --- Multi-Agent Manager ---
class MultiAgentManager:
    def __init__(self):
        self.config = AGENT_CONFIG['agents']
        self.agents = [
            AbstractAgent(
                name=agent_cfg['name'],
                persona=agent_cfg['persona'],
                prompt_template=agent_cfg['prompt'],
                model=agent_cfg.get('model', 'qwen3:0.6b'),
            ) for agent_cfg in self.config
        ]

    def collaborate(self, topic: str) -> Dict[str, Any]:
        import time
        start_time = time.time()
        outputs = {}
        context = {'topic': topic}
        total_refs_searched = 0

        console.rule('[bold blue]Step 1: Aggregating Multi-Source Literature')
        console.print('[cyan]Querying all public sources for relevant papers and scoring by relevance, recency, and citations...[/cyan]')
        from multi_source import aggregate_sources, summarize_papers_via_ollama
        try:
            top_papers, lit_summary = aggregate_sources(topic, max_results=7, summarize=True, ollama_model=self.agents[1].model)
            # Show top papers with scores
            table = Table(title="Top Papers by Composite Score", show_lines=True)
            table.add_column("Title", style="bold", width=40)
            table.add_column("Score", justify="center")
            table.add_column("Year")
            table.add_column("Citations")
            table.add_column("Source")
            for p in top_papers:
                table.add_row(
                    p['title'][:40],
                    f"{p['composite_score']:.2f}",
                    str(p.get('year','')),
                    str(p.get('citations','')),
                    p.get('source','')
                )
            console.print(table)
            # Show summary
            console.print(Panel(lit_summary, title="Literature Summary", style="cyan"))
            citations = top_papers
            citations_str = self.agents[1].format_citations(citations)
            outputs['Literature Citations'] = citations_str
            outputs['Citations List'] = citations
            outputs['Literature Summary'] = lit_summary
            context['literature'] = citations_str
            context['references'] = citations_str
            context['literature_summary'] = lit_summary
            total_refs_searched = len(top_papers)
            console.print(f'[green]Found {len(citations)} top papers from all sources.[/green]')
        except Exception as e:
            console.print(f'[red]Error during literature aggregation: {e}[/red]')
            citations = []
            citations_str = ''
            outputs['Literature Citations'] = ''
            outputs['Citations List'] = []
            outputs['Literature Summary'] = ''
            context['literature'] = ''
            context['references'] = ''
            context['literature_summary'] = ''
            total_refs_searched = 0

        agent_steps = [
            ('Agent A', 'Breakdown'),
            ('Agent B', 'Critical Review'),
            ('Agent C', 'Synthesis'),
            ('Agent D', 'Novelty Generation'),
            ('Agent E', 'Academic Structuring'),
        ]
        agent_context_keys = [
            {},
            {'breakdown': 'breakdown', 'literature': 'literature'},
            {'breakdown': 'breakdown', 'critical_review': 'critical_review'},
            {'synthesis': 'synthesis'},
            {'novel_hypothesis': 'novel_hypothesis', 'references': 'references'},
        ]
        for idx, (agent_name, task) in enumerate(agent_steps):
            agent = self.agents[idx]
            console.rule(f'[bold yellow]Step {idx+2}: {agent_name} - {task}')
            console.print(f"[bold magenta]Current Agent:[/bold magenta] {agent_name}")
            console.print(f"[bold magenta]Persona:[/bold magenta] {agent.persona}")
            console.print(f"[bold magenta]Model:[/bold magenta] {getattr(agent, 'model', 'qwen3:0.6b')}")
            console.print(f"[bold cyan]Task:[/bold cyan] {task}\n")

            # Build context for this agent
            agent_context = {'topic': topic, 'name': agent.name, 'persona': agent.persona}
            # Add previous outputs as needed
            for k, v in agent_context_keys[idx].items():
                if v in context:
                    agent_context[k] = context[v]
            try:
                out = agent.run(agent_context)
                outputs[agent_name] = out.strip()
                # Update context for next agent
                if task == 'Breakdown':
                    context['breakdown'] = out.strip()
                elif task == 'Critical Review':
                    context['critical_review'] = out.strip()
                elif task == 'Synthesis':
                    context['synthesis'] = out.strip()
                elif task == 'Novelty Generation':
                    context['novel_hypothesis'] = out.strip()
                elif task == 'Academic Structuring':
                    pass
                console.print(f"[green]{agent_name} completed: {task}[/green]\n")
            except Exception as e:
                outputs[agent_name] = f"[ERROR] {e}"
                console.print(f"[red]{agent_name} failed: {e}[/red]\n")

        total_refs_used = len(outputs.get('Citations List', []))

        # --- Advanced Novelty Detection (semantic similarity via Ollama) ---
        console.rule('[bold blue]Step 7: Novelty Detection')
        try:
            hypothesis = outputs.get('Agent D', '') or outputs.get('Agent E', '')
            if hypothesis and citations:
                # Build prompt for Ollama
                abstracts = '\n'.join([f"- {p['title']}: {p.get('summary','')[:500]}" for p in citations if p.get('summary')])
                novelty_prompt = (
                    f"You are an expert research evaluator. Given the following hypothesis and a set of related paper abstracts, assess how novel the hypothesis is. "
                    f"If the hypothesis is similar to any existing work, cite the closest ones. Otherwise, explain why it is likely novel. "
                    f"\n\n[Hypothesis]\n{hypothesis}\n\n[Related Papers]\n{abstracts}"
                )
                novelty_result = self.agents[4].ask(novelty_prompt)
                outputs['Novelty Assessment'] = novelty_result.strip()
                novelty_score = novelty_result.strip()
                console.print(Panel(novelty_result.strip(), title="Novelty Assessment", style="magenta"))
            else:
                novelty_score = '[No hypothesis or papers to compare.]'
                outputs['Novelty Assessment'] = novelty_score
                console.print('[yellow]No hypothesis or papers for novelty detection.[/yellow]')
        except Exception as e:
            novelty_score = f'[Novelty detection failed: {e}]'
            outputs['Novelty Assessment'] = novelty_score
            console.print(f'[red]Novelty detection failed: {e}[/red]')

        # --- Final Human-Readable Summary Panel ---
        end_time = time.time()
        elapsed = end_time - start_time
        final_hypothesis = outputs.get('Agent E', '').strip()
        novelty_panel = outputs.get('Novelty Assessment', '').strip()
        references = outputs.get('Citations List', [])
        ref_table = Table(title="References Used", show_lines=True)
        ref_table.add_column("Title", style="bold", width=40)
        ref_table.add_column("Year")
        ref_table.add_column("Source")
        ref_table.add_column("URL", style="blue")
        for p in references:
            ref_table.add_row(
                p['title'][:40],
                str(p.get('year','')),
                p.get('source',''),
                p.get('url','')
            )
        console.rule("[bold green]Final Result: Novel Hypothesis & Stats")
        console.print(Panel(final_hypothesis or '[No hypothesis generated]', title="[bold yellow]Novel Hypothesis", style="bold green"))
        console.print(Panel(novelty_panel or '[No novelty assessment]', title="[bold magenta]Novelty Assessment", style="magenta"))
        console.print(ref_table)
        stats_table = Table(title="Run Stats", show_header=False)
        stats_table.add_row("Total References Searched", str(total_refs_searched))
        stats_table.add_row("Total References Used", str(total_refs_used))
        stats_table.add_row("Total Time Taken (s)", f"{elapsed:.2f}")
        console.print(stats_table)

        return {
            'topic': topic,
            'outputs': outputs,
            'citations': citations,
            'novelty_score': novelty_score,
            'final_hypothesis': final_hypothesis
        }

# --- Output Saving ---
import json

def slugify(text):
    # Simple slugify: keep alphanum and underscores, replace spaces with underscores
    return ''.join(c if c.isalnum() else '_' for c in text.lower()).strip('_')

def save_hypothesis(result: Dict[str, Any]):
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    topic_slug = slugify(result['topic'])[:40] or 'untitled'
    txt_filename = f"hypothesis_{topic_slug}_{timestamp}_{unique_id}.txt"
    json_filename = f"hypothesis_{topic_slug}_{timestamp}_{unique_id}.json"
    txt_path = os.path.join(output_dir, txt_filename)
    json_path = os.path.join(output_dir, json_filename)
    # Write human-readable txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Topic: {result['topic']}\n\n")
        for agent, output in result['outputs'].items():
            if agent == 'Citations List':
                continue
            f.write(f"{agent}:\n{output}\n\n")
        if 'Citations List' in result['outputs']:
            f.write("References:\n")
            for paper in result['outputs']['Citations List']:
                f.write(f"- [{paper['source']}] {paper['title']} ({paper.get('year','')})\n  {paper.get('url','')}\n  Authors: {paper.get('authors','')}\n\n")
        f.write(f"Final Hypothesis (Agent E):\n{result['final_hypothesis']}\n")
    # Write structured JSON
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)
    return txt_path, json_path

# --- Interactive CLI Menu ---
def main_menu(manager: MultiAgentManager):
    while True:
        console.clear()
        console.rule('[bold green]Abstract-Agent: Multi-Agent Research Generator')
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("No.", style="dim", width=4)
        table.add_column("Action", style="bold")
        table.add_row("1", "Multi-Agent: Generate Collaborative Research Hypothesis")
        table.add_row("2", "Exit")
        console.print(table)
        choice = IntPrompt.ask("[yellow]Choose an option[/yellow]", choices=["1","2"])
        if choice == 1:
            handle_collaborative_generation(manager)
        elif choice == 2:
            console.print("[bold green]Goodbye!")
            sys.exit(0)

def handle_collaborative_generation(manager: MultiAgentManager):
    topic = Prompt.ask("Enter your research topic")
    console.print(Panel.fit(f"[bold]Multi-agent collaboration for topic:[/bold] [cyan]{topic}[/cyan]", title="Processing..."))
    try:
        result = manager.collaborate(topic)
        txt_path, json_path = save_hypothesis(result)
        console.print(Panel.fit(
            f"[bold]Final Hypothesis (Agent E):[/bold]\n[green]{result['final_hypothesis']}[/green]\n\n[bold]Agent Outputs:[/bold]\n" + '\n'.join(f"[bold]{k}:[/bold] {v}" for k, v in result['outputs'].items()) + f"\n\n[dim]Saved to:[/dim] {txt_path}\n[dim]JSON:[/dim] {json_path}",
            title="Collaboration Result", subtitle="Press Enter to return to menu"
        ))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
    input()

if __name__ == "__main__":
    manager = MultiAgentManager()
    main_menu(manager)
