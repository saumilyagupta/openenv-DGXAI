from typing import Dict, Any, Tuple

class CyberSimulation:
    def __init__(self):
        # Define the network topology and vulnerabilities
        self.network = {
            "192.168.1.10": {
                "name": "Web Server",
                "ports": {
                    80: {"service": "http", "version": "Apache Struts 2", "vulnerable": True, "vuln_type": "SQL_INJECTION"}
                },
                "escalation_vulnerability": "sudo_misconfig"
            },
            "192.168.1.20": {
                "name": "Database",
                "ports": {
                    5432: {"service": "postgresql", "version": "13.4", "vulnerable": False}
                },
                "escalation_vulnerability": None
            },
            "192.168.1.30": {
                "name": "User Workstation",
                "ports": {
                    22: {"service": "ssh", "version": "OpenSSH 8.2p1", "vulnerable": False}
                },
                "escalation_vulnerability": None
            }
        }

        self.reset()

    def reset(self):
        self.access_levels = {ip: "none" for ip in self.network.keys()}
        self.discovered_nodes = set()
        self.blue_team_alert_level = 0
        self.failed_exploits = 0
        self.is_blocked = False

    def process_action(self, action_type: str, target_ip: str, port: int = None, payload_type: str = None, method: str = None) -> Tuple[Dict[str, Any], float, str, str]:
        """
        Processes an action and returns (observation_dict, reward, output_msg, error_msg)
        """
        if self.is_blocked:
            return {}, -1.0, "Action blocked.", "Your IP has been blocked by the Blue Team."

        if target_ip not in self.network:
            return {}, -0.1, "Target not found.", f"IP {target_ip} is unreachable."

        if action_type == "scan":
            return self._handle_scan(target_ip)
        elif action_type == "exploit":
            return self._handle_exploit(target_ip, port, payload_type)
        elif action_type == "escalate":
            return self._handle_escalate(target_ip, method)
        else:
            return {}, -0.1, "Invalid action.", f"Unknown action type: {action_type}"

    def _handle_scan(self, target_ip: str) -> Tuple[Dict[str, Any], float, str, str]:
        # Reward Hacking Fix: Only reward if it's a NEW discovery
        reward = 0.0
        if target_ip not in self.discovered_nodes:
            reward = 0.3
            self.discovered_nodes.add(target_ip)

        node_info = self.network[target_ip]

        scan_results = {
            "ip": target_ip,
            "open_ports": []
        }

        for p, details in node_info["ports"].items():
            scan_results["open_ports"].append({
                "port": p,
                "service": details["service"],
                "version": details["version"]
            })

        return {"scan_results": scan_results}, reward, f"Scan completed for {target_ip}", None

    def _handle_exploit(self, target_ip: str, port: int, payload_type: str) -> Tuple[Dict[str, Any], float, str, str]:
        if self.access_levels[target_ip] in ["user", "root"]:
            # Reward Hacking Fix: Don't reward exploiting a machine we already own
            return {}, 0.0, "Already have access.", "You already have access to this machine."

        node_info = self.network[target_ip]

        if port not in node_info["ports"]:
            self._trigger_blue_team()
            return {}, -0.1, "Exploit failed.", f"Port {port} is closed."

        port_info = node_info["ports"][port]

        if port_info.get("vulnerable") and port_info.get("vuln_type") == payload_type:
            # Success!
            self.access_levels[target_ip] = "user"
            return {}, 0.4, f"Exploit successful. Gained user access to {target_ip}.", None
        else:
            self._trigger_blue_team()
            return {}, -0.1, "Exploit failed.", "Payload ineffective or service not vulnerable."

    def _handle_escalate(self, target_ip: str, method: str) -> Tuple[Dict[str, Any], float, str, str]:
        if self.access_levels[target_ip] == "none":
            self._trigger_blue_team()
            return {}, -0.1, "Escalation failed.", "You need user access first before escalating."

        if self.access_levels[target_ip] == "root":
            # Reward Hacking Fix: Don't reward escalating a machine we already have root on
            return {}, 0.0, "Already root.", "You already have root access."

        node_info = self.network[target_ip]

        if node_info.get("escalation_vulnerability") == method:
            self.access_levels[target_ip] = "root"
            return {}, 0.3, f"Privilege escalation successful. Gained root access to {target_ip}.", None
        else:
            self._trigger_blue_team()
            return {}, -0.1, "Escalation failed.", "Method ineffective."

    def _trigger_blue_team(self):
        self.failed_exploits += 1
        self.blue_team_alert_level += 1
        if self.failed_exploits >= 3:
            self.is_blocked = True

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "discovered_nodes": list(self.discovered_nodes),
            "access_levels": self.access_levels.copy(),
            "blue_team_alert_level": self.blue_team_alert_level,
            "is_blocked": self.is_blocked
        }
