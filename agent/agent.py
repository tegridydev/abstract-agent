# agent.py
# abstract-agent v1.1
# Author: tegridydev
# Repo: https://github.com/tegridydev/abstract-agent
# License: MIT
# Year: 2025

import os
import sys
import uuid
import importlib
import time
import json
import traceback
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import yaml
import requests 
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.box import ROUNDED
from ollama import Client, ResponseError

try:
    import multi_source
    import agent_helpers
except ImportError as e:
    console = Console()
    console.print(f"[bold red]Error: Failed to import required modules (multi_source, agent_helpers): {e}[/bold red]")
    console.print("Please ensure multi_source.py and agent_helpers.py exist and are in the same directory as agent.py.")
    sys.exit(1)

console = Console()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'agents_config.yaml')

def load_pipeline_config() -> Dict[str, Any]:
    """Loads and validates the pipeline configuration from YAML."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Basic validation
        if not isinstance(config, dict):
            raise ValueError("Config Error: Configuration file is not a valid YAML dictionary.")
        if 'steps' not in config or not isinstance(config['steps'], list):
            raise ValueError("Config Error: 'steps' list is missing or invalid in config.")
        if 'context_init' not in config or not isinstance(config['context_init'], list):
            config['context_init'] = []
            console.print("[Config] Note: 'context_init' key not found or invalid, assuming no initial context required beyond user input.", style="yellow")
        if 'final_outputs' not in config or not isinstance(config['final_outputs'], list):
             config['final_outputs'] = []
             console.print("[Config] Note: 'final_outputs' key not found or invalid, default display/save behaviour will be used.", style="yellow")

        for i, step in enumerate(config['steps']):
             if not isinstance(step, dict) or not step.get('name') or not step.get('type'):
                  raise ValueError(f"Config Error: Invalid step configuration at index {i}. Each step must be a dictionary with at least 'name' and 'type' keys.")

        return config
    except FileNotFoundError:
        console.print(f"[bold red]Error: Configuration file not found at {CONFIG_PATH}[/bold red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error: Failed to parse configuration file {CONFIG_PATH}: {e}[/bold red]")
        sys.exit(1)
    except ValueError as e:
         console.print(f"[bold red]{e}[/bold red]")
         sys.exit(1)
    except Exception as e:
         console.print(f"[bold red]An unexpected error occurred while loading config: {e}[/bold red]")
         traceback.print_exc()
         sys.exit(1)


# --- Core Agent Class  ---
class AbstractAgent:
    """Represents an LLM agent interaction point using Ollama."""
    def __init__(self, model: str = 'qwen3:0.6b', host: str = 'http://localhost:11434'):
        """
        Initializes the agent.

        Args:
            model: The name of the Ollama model to use (e.g., 'qwen3:0.6b').
            host: The URL of the Ollama host.
        """
        if not model:
            raise ValueError("Ollama model name cannot be empty.")
        self.model = model
        self.host = host
        try:
            self.client = Client(host=self.host)
        except Exception as e:
             raise ConnectionError(f"Failed to initialize Ollama client for host '{self.host}'. Is Ollama running? Error: {e}") from e

    def ask(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, timeout: int = 180) -> str:
        """
        Sends a prompt to the Ollama model and returns the response content.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: An optional system prompt to guide the model's behavior.
            temperature: The temperature setting for the Ollama model.
            timeout: Timeout in seconds for the Ollama API call (Note: library support varies).

        Returns:
            The content of the model's response message.

        Raises:
            ConnectionError: If there's an issue communicating with the Ollama API or host.
            ValueError: If prompt is empty or response is malformed.
        """
        if not prompt:
             raise ValueError("User prompt cannot be empty.")

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={'temperature': temperature}
            )

            message = response.get('message', {})
            content = message.get('content')

            if content is not None:
                 return content
            else:
                 console.print(f"[yellow]Warning: Ollama response for model '{self.model}' missing expected content.[/yellow]")
                 console.print(f"[dim]Full Response: {response}[/dim]")
                 raise ValueError(f"Ollama response for model '{self.model}' did not contain expected message content.")

        except ResponseError as e:
            if e.status_code == 404:
                 error_message = f"Ollama API error: Model '{self.model}' not found on host '{self.host}'. Please ensure the model is pulled (e.g., `ollama pull {self.model}`)."
            else:
                 error_message = f"Ollama API error (Model: {self.model}): Status {e.status_code} - {e.error}"
            raise ConnectionError(error_message) from e
        except (requests.exceptions.ConnectionError, ConnectionRefusedError) as e:
             raise ConnectionError(f"Failed to connect to Ollama host '{self.host}'. Is Ollama running and accessible at this address? Error: {e}") from e
        except Exception as e:
            console.print(f"[red]Unexpected error during Ollama API call:[/red] {type(e).__name__} - {e}")
            raise ConnectionError(f"An unexpected error occurred communicating with Ollama host '{self.host}' for model '{self.model}'. Error: {e}") from e


# --- Pipeline Manager ---
class PipelineManager:
    """Orchestrates the execution of steps defined in the pipeline config."""
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PipelineManager.

        Args:
            config: The loaded pipeline configuration dictionary.
        """
        self.config = config
        self.ollama_host = config.get('ollama_host', os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
        console.print(f"[info]Using Ollama Host:[/info] {self.ollama_host}")
        self._agent_cache: Dict[str, AbstractAgent] = {}
        self.current_step_outputs: List[str] = []

    def _get_agent(self, model_name: str) -> AbstractAgent:
        """Gets or creates an AbstractAgent instance for a given model, caching it."""
        if model_name not in self._agent_cache:
            console.print(f"[info]Initializing agent for model:[/info] {model_name}")
            try:
                self._agent_cache[model_name] = AbstractAgent(model=model_name, host=self.ollama_host)
            except (ValueError, ConnectionError) as e:
                 console.print(f"[bold red]Fatal Error: Cannot initialize agent for model '{model_name}'. Halting.[/bold red]")
                 console.print(f"[red]Details: {e}[/red]")
                 raise RuntimeError(f"Failed to initialize agent '{model_name}'") from e
        return self._agent_cache[model_name]

    def _format_prompt(self, template: str, context: Dict[str, Any], inputs: List[str]) -> str:
        """
        Safely formats a prompt template string using specified input keys from the context.
        (Note: Primarily for potential non-LLM steps using templates, LLM steps format directly now).
        """
        format_dict = {}
        missing_keys = []
        for key in inputs:
            if key not in context:
                missing_keys.append(key)
        if missing_keys:
             raise ValueError(f"Missing required context inputs for prompt formatting: {', '.join(missing_keys)}")

        for key in inputs:
             format_dict[key] = context[key]

        try:
            return template.format(**format_dict)
        except KeyError as e:
            raise ValueError(f"Prompt template expects key '{e}' which is not listed in the step's 'inputs' list.") from e
        except Exception as e:
            raise ValueError(f"Error formatting prompt template: {e}") from e


    def _call_tool(self, function_path: str, context: Dict[str, Any], inputs: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically calls a Python function specified in the config. Maps results to output keys.
        """
        try:
            if '.' not in function_path:
                 raise ValueError(f"Invalid function path '{function_path}'. Must include module (e.g., 'module.function').")
            module_path, function_name = function_path.rsplit('.', 1)

            try:
                 module = importlib.import_module(module_path)
            except ImportError as e:
                 raise ImportError(f"Could not import module '{module_path}' specified in function path '{function_path}': {e}") from e

            if not hasattr(module, function_name):
                 raise AttributeError(f"Function '{function_name}' not found in module '{module_path}'.")
            func = getattr(module, function_name)
            if not callable(func):
                  raise TypeError(f"'{function_path}' is not a callable function.")

            func_args = {}
            for key in inputs:
                if key in context:
                    func_args[key] = context[key]
                else:
                    raise ValueError(f"Internal Error: Missing required context input '{key}' for function '{function_path}' during tool call (should have been caught earlier).")
            func_args.update(params)

            result = func(**func_args)

            output_keys = self.current_step_outputs
            if not output_keys: # If step defines no outputs
                 if result is not None:
                      console.print(f"[yellow]Warning: Tool '{function_path}' returned a result ({type(result)}) but no output keys were defined for the step. Result ignored.[/yellow]")
                 return {} # Return empty dict if no outputs expected

            if isinstance(result, tuple):
                 if len(result) != len(output_keys):
                      raise ValueError(f"Tool Error: Function '{function_path}' returned a tuple with {len(result)} elements, but step expected {len(output_keys)} outputs ({output_keys}).")
                 return dict(zip(output_keys, result))
            elif len(output_keys) == 1:
                 # If expecting a single output, wrap the result in a dict
                 return {output_keys[0]: result}
            else:
                 # If expecting multiple outputs, but got a single non-tuple value (likely an error)
                 if not isinstance(result, dict):
                      raise TypeError(f"Tool Error: Function '{function_path}' returned a single value of type '{type(result)}', but step expected multiple outputs ({output_keys}). Did you mean to return a tuple or dict?")
                 # If result is a dict, check if it contains all expected keys
                 missing_keys_in_result = [key for key in output_keys if key not in result]
                 if missing_keys_in_result:
                      raise ValueError(f"Tool Error: Function '{function_path}' returned a dictionary, but it's missing expected output key(s): {', '.join(missing_keys_in_result)}. Expected: {output_keys}.")
                 # Return only the expected keys from the result dict
                 return {key: result[key] for key in output_keys}

        except (AttributeError, TypeError, ImportError, ValueError) as e:
            # Catch errors related to finding/calling the function or validating results
            raise ValueError(f"Error preparing or calling function '{function_path}': {e}") from e
        except Exception as e:
            # Catch errors raised *during* the execution of the tool function
            # Provide more context in the error message
            console.print(f"[red]Error details during tool execution '{function_path}':[/red] {traceback.format_exc()}", highlight=False)
            raise RuntimeError(f"Error executing tool '{function_path}'. Check logs above for details. Original error: {e}") from e

    def _check_condition(self, condition: Optional[str], context: Dict[str, Any]) -> bool:
        """Evaluates a simple condition string against the context."""
        if not condition:
            return True # No condition means always run

        try:
            # Simple check for key existence and truthiness (safer than eval)
            # Handles "context.get('key')" or "context.get('key', default_value)" patterns more safely.
            match = re.match(r"^\s*context\.get\(['\"]([^'\"]+)['\"](?:,\s*(.+))?\)\s*$", condition)
            if match:
                key = match.group(1)
                return bool(context.get(key))
            else:
                 console.print(f"[yellow]Warning: Using restricted 'eval' for condition '{condition}'. Ensure config file is trusted.[/yellow]", highlight=False)
                 safe_builtins = {'True': True, 'False': False, 'None': None, 'str': str, 'int': int, 'float': float, 'len': len, 'list': list, 'dict': dict}
                 return eval(condition, {"__builtins__": safe_builtins}, {"context": context})

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to evaluate condition '{condition}': {e}. Step will be skipped.[/yellow]", highlight=False)
            return False

    def run_pipeline(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the pipeline defined in the configuration."""
        start_time = time.time()

        context = initial_context.copy()
        pipeline_name = self.config.get('pipeline_name', 'Untitled Pipeline')
        pipeline_version = self.config.get('version', 'N/A')
        console.print(Panel(f"[bold cyan]Starting Pipeline: {pipeline_name}[/bold cyan]", subtitle=f"Version {pipeline_version}"))

        if not self.config.get('steps'):
             console.print("[yellow]Warning: No steps found in the pipeline configuration.[/yellow]")
             return context

        for i, step in enumerate(self.config['steps']):
            step_name = step['name']
            step_type = step['type']
            inputs = step.get('inputs', [])
            outputs = step.get('outputs', [])
            self.current_step_outputs = outputs

            console.rule(f"[bold yellow]Step {i+1}: {step_name} (Type: {step_type})[/bold yellow]")

            # 1. Check Condition
            condition = step.get('run_if')
            if not self._check_condition(condition, context):
                console.print(f"[yellow]Skipped due to condition: {condition or 'N/A'}[/yellow]\n")
                continue

            missing_inputs = [key for key in inputs if key not in context]
            if missing_inputs:
                error_msg = f"Missing required inputs for step '{step_name}': {', '.join(missing_inputs)}. Halting pipeline."
                console.print(f"[bold red]Error: {error_msg}[/bold red]")
                raise ValueError(error_msg)

            step_start_time = time.time()
            try:
                result_data = {}
                if step_type in ['llm_agent', 'ollama_call', 'summary_ollama', 'novelty_check_ollama']:
                    # --- Handle LLM/Ollama Calls ---
                    model = step.get('model')
                    if not model: raise ValueError(f"Config Error: Missing 'model' definition for LLM step '{step_name}'.")
                    agent = self._get_agent(model)
                    prompt_template = step.get('prompt_template')
                    if not prompt_template: raise ValueError(f"Config Error: Missing 'prompt_template' for LLM step '{step_name}'.")

                    system_prompt = step.get('persona') or step.get('system_prompt')

                    prompt_format_dict = {key: context[key] for key in inputs}
                    if 'name' not in prompt_format_dict:
                         prompt_format_dict['name'] = step_name
                    if 'persona' not in prompt_format_dict and 'persona' in step:
                         prompt_format_dict['persona'] = step.get('persona')

                    try:
                         user_prompt = prompt_template.format(**prompt_format_dict)
                    except KeyError as e:
                         raise ValueError(f"Prompt Template Error: Template for step '{step_name}' expects key '{e}' which is missing from context inputs ({inputs}) and step metadata (name, persona).") from e
                    except Exception as e:
                         raise ValueError(f"Prompt Template Error: Failed to format prompt for step '{step_name}': {e}") from e

                    console.print(f"[magenta]Model:[/magenta] {model}")
                    if system_prompt:
                         console.print(Panel(system_prompt, title="System Prompt", style="dim", border_style="dim", width=100))
                    console.print(Panel(user_prompt, title="User Prompt Sent", style="cyan", border_style="cyan", width=100))
                    temperature = float(step.get('temperature', 0.7))

                    response_content = agent.ask(user_prompt, system_prompt=system_prompt, temperature=temperature)
                    console.print(Panel(response_content, title="Response Received", style="green", border_style="green", width=100))
                    if len(outputs) != 1:
                         raise ValueError(f"Config Error: LLM/Ollama steps currently support exactly one output key. Step '{step_name}' expects {len(outputs)} ({outputs}).")
                    result_data = {outputs[0]: response_content.strip()}

                elif step_type in ['tool_call', 'data_fetch']:
                    function_path = step.get('function')
                    if not function_path: raise ValueError(f"Config Error: Missing 'function' path for tool step '{step_name}'.")
                    params = step.get('params', {})

                    console.print(f"[magenta]Function:[/magenta] {function_path}")
                    console.print(f"[magenta]Inputs (from context):[/magenta] {inputs}")
                    console.print(f"[magenta]Params (fixed):[/magenta] {params}")

                    result_data = self._call_tool(function_path, context, inputs, params)

                    display_result = {}
                    for k, v in result_data.items():
                        if isinstance(v, list): display_result[k] = f"[List with {len(v)} items]"
                        elif isinstance(v, dict): display_result[k] = f"[Dict with {len(v)} keys]"
                        elif isinstance(v, str) and len(v) > 150: display_result[k] = v[:150] + "..."
                        else: display_result[k] = v
                    console.print(Panel(str(display_result), title="Tool Result Stored", style="blue", border_style="blue"))

                else:
                    error_msg = f"Config Error: Unknown step type '{step_type}' for step '{step_name}'. Halting."
                    console.print(f"[bold red]Error: {error_msg}[/bold red]")
                    raise ValueError(error_msg)

                for key in outputs:
                    if key not in result_data:
                         console.print(f"[yellow]Warning: Step '{step_name}' of type '{step_type}' did not produce expected output key '{key}'. Setting to None.[/yellow]")
                         result_data[key] = None

                context.update(result_data)

                step_end_time = time.time()
                console.print(f"[green]Step '{step_name}' completed successfully in {step_end_time - step_start_time:.2f}s.[/green]\n")

            except (ValueError, ConnectionError, RuntimeError, TypeError, AttributeError, ImportError) as e:
                console.print(f"[bold red]---> Pipeline Error during step '{step_name}' <---[/bold red]")
                console.print(f"[red]Error Type:[/red] {type(e).__name__}")
                console.print(f"[red]Details:[/red] {e}")
                raise
            except Exception as e:
                 console.print(f"[bold red]---> Unexpected Error during step '{step_name}' <---[/bold red]")
                 console.print(f"[red]Error Type:[/red] {type(e).__name__} - {e}")
                 console.print(f"[dim]{traceback.format_exc()}[/dim]") # Print full traceback
                 raise # Re-raise to halt

        end_time = time.time()
        total_time = end_time - start_time
        console.rule(f"[bold green]Pipeline '{pipeline_name}' Finished in {total_time:.2f}s[/bold green]")

        return context


# --- Output Saving & Display ---

def slugify(text: str) -> str:
    """Simple slugify: keep alphanumeric, replace spaces/hyphens with underscores."""
    import re
    if not isinstance(text, str):
         text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'-+', '_', text)
    text = re.sub(r'[^\w_]+', '', text)
    text = text.strip('_')
    return text[:60] if text else "untitled"


def save_results(result_context: Dict[str, Any], config: Dict[str, Any], topic: str, ollama_host: str):
    """Saves the pipeline results to text and JSON files."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            console.print(f"[info]Created output directory:[/info] {output_dir}")
        except OSError as e:
            console.print(f"[red]Error creating output directory '{output_dir}': {e}. Results will not be saved.[/red]")
            return None, None # Indicate failure to save

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    topic_slug = slugify(topic)

    base_filename = f"result_{topic_slug}_{timestamp}_{unique_id}"
    txt_filename = f"{base_filename}.txt"
    json_filename = f"{base_filename}.json"
    txt_path = os.path.join(output_dir, txt_filename)
    json_path = os.path.join(output_dir, json_filename)

    # --- Output TXT file ---
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline: {config.get('pipeline_name', 'Untitled')} v{config.get('version', 'N/A')}\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            f.write("[Final Outputs]\n")
            f.write("-" * 15 + "\n")
            final_output_keys = config.get('final_outputs', [])
            if not final_output_keys:
                 default_final_keys = ['final_hypothesis_structured', 'novelty_assessment', 'top_papers']
                 f.write(f"(Note: 'final_outputs' not defined in config, showing defaults: {', '.join(default_final_keys)})\n\n")
                 final_output_keys = default_final_keys

            for key in final_output_keys:
                 f.write(f"--- {key.replace('_', ' ').title()} ---\n")
                 if key in result_context and result_context[key] is not None:
                     value = result_context[key]
                     if key == 'top_papers' and isinstance(value, list):
                         if value:
                              f.write(agent_helpers.format_citations_rich(value))
                         else:
                              f.write("  (No papers found or used)\n")
                     elif isinstance(value, (dict, list)):
                          try:
                               f.write(json.dumps(value, indent=2, default=str))
                          except Exception:
                               f.write(str(value))
                     elif isinstance(value, str):
                          f.write(value)
                     else:
                          f.write(str(value))
                     f.write("\n\n")
                 else:
                      f.write("  [Not Generated or Not Found in Context]\n\n")

            # Optionally include selected context keys for debugging
            # debug_keys = ['literature_summary', 'breakdown', 'critical_review', 'synthesis']
            # f.write("\n" + "="*50 + "\nDebugging Context:\n")
            # for key in debug_keys:
            #     if key in result_context:
            #         f.write(f"\n--- {key.replace('_', ' ').title()} ---\n{result_context[key]}\n")

        console.print(f"[green]Text results saved to:[/green] {txt_path}")
    except IOError as e:
        console.print(f"[red]Error writing text file '{txt_path}': {e}[/red]")
        txt_path = None
    except Exception as e:
         console.print(f"[red]Unexpected error writing text file '{txt_path}': {e}[/red]")
         txt_path = None

    # --- Output structured JSON file ---
    try:
        output_data = {
             'pipeline_config_snapshot': config,
             'run_info': {
                  'topic': topic,
                  'timestamp_iso': datetime.now().isoformat(),
                  'ollama_host_used': result_context.get('_pipeline_ollama_host', ollama_host)
             },
             'final_context': {} 
        }

        serializable_context = {}
        for k, v in result_context.items():
            try:
                json.dumps(v)
                serializable_context[k] = v
            except TypeError:
                # If direct serialization fails, convert known non-serializable types
                if isinstance(v, (datetime, uuid.UUID)):
                     serializable_context[k] = str(v)
                else:
                     try:
                          serializable_context[k] = f"[Unserializable Type: {type(v).__name__}] {str(v)}"
                     except Exception: # Catch errors during string conversion itself
                          serializable_context[k] = f"[Unserializable Type: {type(v).__name__}] <Error converting to string>"
            except Exception as json_err:
                 serializable_context[k] = f"[Serialization Check Error for key '{k}': {json_err}]"


        output_data['final_context'] = serializable_context

        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(output_data, jf, ensure_ascii=False, indent=2)
        console.print(f"[green]JSON results saved to:[/green] {json_path}")
    except IOError as e:
        console.print(f"[red]Error writing JSON file '{json_path}': {e}[/red]")
        json_path = None
    except TypeError as e:
         console.print(f"[red]Error serializing results to JSON: {e}. Check context for complex objects.[/red]")
         json_path = None
    except Exception as e:
         console.print(f"[red]Unexpected error writing JSON file '{json_path}': {e}[/red]")
         json_path = None

    return txt_path, json_path


def display_final_results(result_context: Dict[str, Any], config: Dict[str, Any]):
    """Displays key final results to the console using Rich."""
    console.rule("[bold green]Final Results Summary[/bold green]")

    final_output_keys = config.get('final_outputs', [])
    if not final_output_keys:
        console.print("[yellow]Note: 'final_outputs' not defined in config. Displaying default summary keys.[/yellow]")
        final_output_keys = ['final_hypothesis_structured', 'novelty_assessment', 'top_papers', 'all_fetched_papers_count']

    displayed_keys = set()

    # --- Display Prioritized Keys ---
    
    priority_display_order = ['final_hypothesis_structured', 'novelty_assessment', 'top_papers']
    for key in priority_display_order:
         if key in final_output_keys and key in result_context:
             value = result_context[key]
             title = key.replace('_', ' ').title()
             if key == 'top_papers' and isinstance(value, list):
                  if value:
                     ref_table = Table(title="Top Papers Used (Max 10 Shown)", show_lines=True, expand=False, box=ROUNDED)
                     ref_table.add_column("Title", style="bold", min_width=30, max_width=50, overflow="fold")
                     ref_table.add_column("Score", justify="center", width=6)
                     ref_table.add_column("Year", width=5)
                     ref_table.add_column("Source", width=15)
                     ref_table.add_column("URL", style="blue", no_wrap=True, overflow="ellipsis", max_width=40)
                     for p in value[:10]: # Limit display to top 10
                         ref_table.add_row(
                             p.get('title','N/A'), f"{p.get('composite_score',0):.2f}",
                             str(p.get('year','N/A')), p.get('source','N/A'), p.get('url','N/A')
                         )
                     console.print(ref_table)
                     if len(value) > 10:
                          console.print(f"[dim](Showing top 10 of {len(value)} references used)[/dim]")
                  else:
                     console.print(Panel("[dim]No relevant papers found or used.[/dim]", title=title, border_style="dim"))
             elif value is not None:
                  console.print(Panel(str(value), title=f"[bold yellow]{title}[/bold yellow]", border_style="yellow", expand=False))
             displayed_keys.add(key) # Mark as displayed

    # --- Display Other Final Outputs ---
    other_keys_to_display = [
         k for k in final_output_keys
         if k not in displayed_keys and k in result_context and result_context[k] is not None
    ]
    if other_keys_to_display:
         console.print("\n[bold]Other Final Outputs:[/bold]")
         for key in other_keys_to_display:
              title = key.replace('_', ' ').title()
              value = result_context[key]
              console.print(Panel(str(value), title=title, border_style="dim", expand=False))
              displayed_keys.add(key)

    # --- Indicate if Expected Final Outputs Were Not Generated ---
    missing_final_keys = [k for k in final_output_keys if k not in result_context or result_context[k] is None]
    if missing_final_keys:
        console.print("\n[yellow]Note: The following expected final outputs were not generated or found:[/yellow]")
        for key in missing_final_keys:
            status = "[Not Found in Context]" if key not in result_context else "[Value is None]"
            console.print(f"- {key.replace('_', ' ').title()} {status}")


# --- Interactive CLI Menu ---
def main_menu(manager: PipelineManager, config: Dict[str, Any]):
    while True:
        console.clear(home=True) # Clear screen and move cursor home
        console.rule(f"[bold green]Abstract-Agent v{config.get('version', 'N/A')}: {config.get('pipeline_name', 'Config-Driven')}[/bold green]")
        table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
        table.add_column("No.", style="dim", width=4)
        table.add_column("Action", style="bold")
        table.add_row("1", "Run Research Generation Pipeline")
        table.add_row("2", "Exit")
        console.print(table)

        try:
             choice = IntPrompt.ask("[yellow]Choose an option[/yellow]", choices=["1", "2"], default=1)

             if choice == 1:
                 handle_pipeline_execution(manager, config)
             elif choice == 2:
                 console.print("\n[bold green]Exiting Abstract-Agent. Goodbye![/bold green]")
                 sys.exit(0)

        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[yellow]Operation cancelled by user. Exiting.[/yellow]")
            sys.exit(1)


def handle_pipeline_execution(manager: PipelineManager, config: Dict[str, Any]):
    """Handles the process of getting user input and running the pipeline."""
    console.rule("Start New Pipeline Run")
    default_topic = "Novel New LLM Compression Method" # Example default topic
    try:
         topic = Prompt.ask(
              f"Enter your research topic",
              default=default_topic,
              show_default=True
         )
    except (KeyboardInterrupt, EOFError):
         console.print("\n[yellow]Input cancelled. Returning to menu.[/yellow]")
         return # Go back to main menu


    initial_context = {}
    required_init_keys = config.get('context_init', [])
    if 'topic' in required_init_keys:
         initial_context['topic'] = topic
    else:
         initial_context['topic'] = topic

    initial_context['_pipeline_ollama_host'] = manager.ollama_host
    initial_context['_pipeline_start_time_iso'] = datetime.now().isoformat()

    console.print(Panel.fit(f"[bold]Starting pipeline for topic:[/bold] [cyan]{topic}[/cyan]", title="Processing...", border_style="blue"))

    try:
        final_context = manager.run_pipeline(initial_context)
        display_final_results(final_context, config)
        txt_file, json_file = save_results(final_context, config, topic, manager.ollama_host)
        if not txt_file and not json_file:
             console.print("[yellow]Warning: Results could not be saved to file(s).[/yellow]")


    except (ValueError, ConnectionError, RuntimeError, TypeError, AttributeError, ImportError) as e:
        console.print(f"\n[bold red]---> Pipeline Execution Failed <---[/bold red]")
        console.print("[yellow]Pipeline halted due to error in a step. Check logs above.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]---> An Unexpected Error Occurred Post-Pipeline <---[/bold red]")
        console.print(f"[red]Error Type:[/red] {type(e).__name__}")
        console.print(f"[red]Details:[/red] {e}")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")

    # --- Wait for user before returning to menu ---
    try:
        Prompt.ask("\n[cyan]Press Enter to return to the main menu[/cyan]", default="")
    except (KeyboardInterrupt, EOFError):
         console.print("\n[yellow]Aborted. Returning to menu.[/yellow]")


if __name__ == "__main__":
    try:
        pipeline_config = load_pipeline_config()
        manager = PipelineManager(config=pipeline_config)
        main_menu(manager, pipeline_config)
    except (KeyboardInterrupt, EOFError):
         print("\n\n[yellow]Application terminated by user.[/yellow]")
         sys.exit(1)
    except Exception as e:
        error_console = Console(stderr=True)
        error_console.print("\n[bold red]---> Fatal Error During Application Startup <---[/bold red]")
        error_console.print(f"[red]Error Type:[/red] {type(e).__name__}")
        error_console.print(f"[red]Details:[/red] {e}")
        error_console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)