"""Demo module.
Non-authoritative and not part of canonical governed runtime.
"""

import sys
from pathlib import Path
import asyncio
import httpx

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.panel import Panel
except ImportError:
    print("❌ Error: 'rich' library is not installed. Run: pip install rich")
    sys.exit(1)

console = Console()
API_URL = "http://127.0.0.1:8005/api/v2/chat"

async def chat_loop():
    console.print(Panel("[bold magenta]AGENTIC TERMINAL CHAT REPL[/bold magenta]\n[dim]Type 'quit' or 'exit' to end.[/dim]", width=60))
    
    history = []
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]Bạn[/bold cyan]")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Exiting chat...[/yellow]")
                break
            
            if not user_input.strip():
                continue
                
            with console.status("[bold green]Agentic AI đang suy nghĩ...[/bold green]", spinner="dots"):
                payload = {
                    "message": user_input,
                    "history": history,
                    "ticker": None 
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(API_URL, json=payload)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("response", "No response")
                        
                        history.append({"role": "user", "content": user_input})
                        history.append({"role": "assistant", "content": answer})
                        
                        # Print Markdown
                        console.print()
                        console.print(Panel(Markdown(answer), title="[bold green]AI Agent[/bold green]", border_style="green"))
                    else:
                        console.print(f"[bold red]Lỗi API:[/bold red] {resp.status_code} - {resp.text}")
                        
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting chat...[/yellow]")
            break
        except httpx.ConnectError:
            console.print("[bold red]Lỗi: Không thể kết nối tới FastAPI Server ở localhost:8000.[/bold red]")
            console.print("[dim]Hãy chắc chắn bạn đã chạy lệnh `uvicorn src.api.main:app --reload`.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Lưu ý có lỗi:[/bold red] {str(e)}")

if __name__ == "__main__":
    asyncio.run(chat_loop())
