from typing import List, Dict, Any, Optional

class AgentScratchpad:
    """
    A persistent working state for the agent. 
    Holds the current plan, sub-goals, and working state.
    Immune to transcript pruning to prevent context destruction.
    """
    
    def __init__(self):
        self.current_plan: str = ""
        self.sub_goals: List[Dict[str, str]] = []  # [{"goal": "Check db", "status": "pending"}]
        self.working_state: Dict[str, Any] = {}    # Variables like active claim_id, policy details
        
    def set_plan(self, plan: str) -> None:
        """Update the agent's main plan."""
        self.current_plan = plan
        
    def add_sub_goal(self, goal: str) -> None:
        """Add a new sub-goal."""
        self.sub_goals.append({"goal": goal, "status": "pending"})
        
    def complete_sub_goal(self, goal: str) -> None:
        """Mark a sub-goal as 'completed'."""
        for item in self.sub_goals:
            if item["goal"] == goal:
                item["status"] = "completed"
                
    def update_working_state(self, key: str, value: Any) -> None:
        """Update working state data (e.g., saving active claim ID)."""
        self.working_state[key] = value
        
    def remove_working_state(self, key: str) -> None:
        """Remove an item from the working state."""
        if key in self.working_state:
            del self.working_state[key]

    def get_state_summary(self) -> str:
        """
        Generates a text summary of the current agent state.
        Always sent as part of the System Prompt to keep the agent on track.
        """
        summary = f"=== AGENT SCRATCHPAD ===\n"
        summary += f"Current Plan: {self.current_plan or 'None'}\n"
        
        summary += "Sub-goals:\n"
        if not self.sub_goals:
            summary += "- None\n"
        else:
            for sg in self.sub_goals:
                mark = "[x]" if sg["status"] == "completed" else "[ ]"
                summary += f"- {mark} {sg['goal']}\n"
                
        summary += "Working Variables:\n"
        if not self.working_state:
            summary += "- None\n"
        else:
            for k, v in self.working_state.items():
                summary += f"- {k}: {v}\n"
                
        return summary
        
    def reset(self) -> None:
        """Reset the scratchpad when starting a completely new task."""
        self.current_plan = ""
        self.sub_goals = []
        self.working_state = {}