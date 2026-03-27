from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Action(BaseModel):
    action_type: str = Field(..., description="The action to perform: 'scan', 'exploit', 'escalate'.")
    target_ip: str = Field(..., description="The IP address to target.")
    port: Optional[int] = Field(None, description="The port to target (if applicable).")
    payload_type: Optional[str] = Field(None, description="The payload type for exploits (e.g., 'SQL_INJECTION').")
    method: Optional[str] = Field(None, description="The method for escalation (e.g., 'sudo_misconfig').")

class Observation(BaseModel):
    scan_results: Optional[Dict[str, Any]] = Field(None, description="Results from a scan action.")
    access_level: str = Field(..., description="Current access level: 'none', 'user', or 'root'.")
    output: str = Field(..., description="Message from the environment.")
    error: Optional[str] = Field(None, description="Error message if the action failed.")

class State(BaseModel):
    discovered_nodes: List[str] = Field(default_factory=list, description="IPs of nodes discovered so far.")
    active_vulnerabilities: Dict[str, List[str]] = Field(default_factory=dict, description="Known vulnerabilities per IP.")
    access_levels: Dict[str, str] = Field(default_factory=dict, description="Current access level per IP.")
    blue_team_alert_level: int = Field(0, description="Current alert level of the Blue Team.")
    action_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of actions taken.")
    current_task: int = Field(1, description="The current task being evaluated (1, 2, or 3).")
    is_blocked: bool = Field(False, description="Whether the Red Team IP is blocked by the Blue Team.")

class Reward(BaseModel):
    score: float = Field(..., description="The numerical reward value.")
    explanation: str = Field(..., description="An explanation of why this reward was given.")
