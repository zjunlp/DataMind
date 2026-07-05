#!/usr/bin/env python3
"""
UI components for the autonomous data analysis agent
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.align import Align
from typing import Any, Dict

console = Console()

def print_session_start(session, llm_provider, mcp_sessions, auto_finish):
    """Print session start information"""
    console.print("\n")
    console.rule("[bold blue]🤖 Autonomous Data Analysis Agent", style="blue")
    
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Field", style="bold cyan", width=20)
    table.add_column("Value", style="white")
    
    table.add_row("Session ID", f"[green]{session.session_id}[/green]")
    table.add_row("Task", f"[yellow]{session.task}[/yellow]")
    table.add_row("LLM Provider", f"[blue]{llm_provider.get_provider_name()}[/blue]")
    table.add_row("MCP Servers", f"[magenta]{len(mcp_sessions)}[/magenta] connected")
    table.add_row("Auto Finish", f"[{'green' if auto_finish else 'red'}]{auto_finish}[/{'green' if auto_finish else 'red'}]")
    
    # Show available tools
    total_tools = sum(len(tools) for tools in session.available_tools.values())
    table.add_row("Available Tools", f"[cyan]{total_tools}[/cyan] tools")
    
    console.print(Align.center(table))
    console.print()
    
    # Show server details
    if mcp_sessions:
        console.print("[bold]Connected MCP Servers:[/bold]")
        for server_name, tools in session.available_tools.items():
            console.print(f"  • [cyan]{server_name}[/cyan]: {len(tools)} tools")
    console.print()

def print_exploration_start(task, max_turns):
    """Print exploration start information"""
    console.print(f"\n[bold blue]🤖 Starting autonomous exploration...[/bold blue]")
    console.print(f"[dim]Task: {task}[/dim]")
    console.print(f"[dim]Max turns: {max_turns}[/dim]\n")

def print_turn_header(turn):
    """Print turn header"""
    console.print(f"\n[bold cyan]--- Turn {turn} ---[/bold cyan]")

def print_agent_turn(response, tool_call, thinking=""):
    """Display agent's reasoning and tool call"""
    # For Gemini, reasoning is thinking - show the main content
    if response.strip():
        console.print(f"[bold magenta]🧠 Agent Thinking:[/bold magenta]")
        console.print(f"[dim]{response}[/dim]")
    elif thinking.strip():
        # Fallback to thinking parameter if response is empty
        console.print(f"[bold magenta]🧠 Agent Thinking:[/bold magenta]")
        console.print(f"[dim]{thinking}[/dim]")
    
    # Only show tool call if it exists and has the required fields
    if tool_call and isinstance(tool_call, dict) and "tool" in tool_call:
        console.print(f"\n[bold green]🔧 Tool Call:[/bold green]")
        console.print(f"[cyan]{tool_call['tool']}[/cyan] with args: [dim]{tool_call.get('arguments', {})}[/dim]")
    elif tool_call and isinstance(tool_call, dict) and tool_call.get("error"):
        console.print(f"\n[bold red]❌ Tool Call Error:[/bold red]")
        console.print(f"[red]{tool_call['error']}[/red]")
    else:
        # No tool call or empty tool_call dict (e.g., for FINISH signal)
        console.print(f"\n[bold green]🔧 No Tool Call[/bold green]")
        console.print(f"[dim]Agent provided reasoning without tool execution[/dim]")

def print_environment_turn(result):
    """Display environment response"""
    console.print(f"\n[bold blue]🌍 Environment Response:[/bold blue]")
    
    # Format result for display
    if isinstance(result, dict):
        if result.get("error"):
            console.print(f"[red]❌ {result['error']}[/red]")
        else:
            # Truncate long results for display
            import json
            result_str = json.dumps(result, indent=2, ensure_ascii=False)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "... (truncated)"
            console.print(f"[dim]{result_str}[/dim]")
    else:
        result_str = str(result)
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "... (truncated)"
        console.print(f"[dim]{result_str}[/dim]")

def print_error(message):
    """Print error message"""
    console.print(f"[red]❌ {message}[/red]")

def print_warning(message):
    """Print warning message"""
    console.print(f"[yellow]⚠️ {message}[/yellow]")

def print_success(message):
    """Print success message"""
    console.print(f"[green]✅ {message}[/green]")

def print_info(message):
    """Print info message"""
    console.print(f"[blue]ℹ️ {message}[/blue]")

def print_completion(turn_count, completion_message):
    """Print completion information"""
    console.print(f"\n[bold green]✅ Exploration completed in {turn_count} turns[/bold green]")
    if completion_message:
        console.print(f"\n[bold magenta]📋 {completion_message}[/bold magenta]")

def print_log_info(timestamp, log_dir="./logs"):
    """Print log file information"""
    console.print(f"\n[bold green]✅ Exploration completed![/bold green]")
    console.print(f"[dim]Generated files in {log_dir}/:[/dim]")
    console.print(f"[dim]  • insights_{timestamp}.csv - Tool execution insights[/dim]")
    console.print(f"[dim]  • chat_messages_{timestamp}.csv - Complete conversation[/dim]")
    console.print(f"[dim]  • message_stats_{timestamp}.csv - Message statistics[/dim]")
    console.print(f"[dim]  • session_stats_{timestamp}.json - Overall session stats[/dim]")
