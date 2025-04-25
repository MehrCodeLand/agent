# agent.py
import os
from google.adk.agents import Agent

def create_bank_agent() -> Agent:
    """
    A sub-agent that handles banking questions, encourages saving,
    and replies in Farsi if asked.
    """
    return Agent(
        model=os.getenv("BANK_AGENT_MODEL", "gemini-2.0-flash"),
        name="bank_assistant",
        instruction=(
            "You are the Bank Assistant. When the user asks for cash, "
            "encourage them to save responsibly. If they switch to Farsi, "
            "reply fully in Farsi."
        ),
        description="Handles banking queries and savings guidance.",
        tools=[],
    )

def create_farewell_agent() -> Agent:
    """
    A sub-agent solely for polite goodbyes.
    """
    return Agent(
        model=os.getenv("FAREWELL_AGENT_MODEL", "gemini-2.0-flash"),
        name="farewell_agent",
        instruction=(
            "You are the Farewell Agent. Your only job is to send a polite goodbye "
            "when the user indicates they are leaving (e.g., 'bye', 'goodbye')."
        ),
        description="Sends friendly farewells.",
        tools=[],
    )

def create_root_agent(sub_agents: list[Agent]) -> Agent:
    """
    Top-level agent that grounds answers in provided documentation contexts
    and delegates to specialized sub-agents as needed.
    """
    return Agent(
        model=os.getenv("ROOT_AGENT_MODEL", "gemini-2.0-flash"),
        name="bank_manager",
        instruction=(
            "You are the Bank Manager Assistant. You receive a prompted message containing "
            "relevant documentation contexts followed by the user's question. "
            "First, thoroughly read the documentation contexts to ground your answer. "
            "If the question is about banking services or advice, provide the best answer "
            "based on the contexts; delegate to 'bank_assistant' only for specialized banking guidance. "
            "If the user indicates they are leaving, delegate to 'farewell_agent'. "
            "Always ground your responses in the provided contexts and cite the context IDs when appropriate."
        ),
        description="Orchestrates RAG-based answers grounded in docs and delegates to sub-agents.",
        tools=[],
        sub_agents=sub_agents,
    )

if __name__ == "__main__":
    bank = create_bank_agent()
    farewell = create_farewell_agent()
    root = create_root_agent([bank, farewell])
    print(f"✅ Created root agent '{root.name}' with subs {[a.name for a in root.sub_agents]}")
