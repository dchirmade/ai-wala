"""
Ollama Advisor - AI-powered band optimization for ZTE routers
Uses local Ollama LLM to analyze signal data and recommend optimal band configurations
"""

import requests
import json
from typing import Dict, Any, Optional


class OllamaAdvisor:
    """AI advisor for cellular band optimization using local Ollama"""

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        """
        Initialize Ollama advisor
        
        Args:
            model: Ollama model name (llama3, mistral, qwen2, etc)
            host: Ollama API host URL
        """
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/generate"

    def _build_prompt(self, signal_data: Dict[str, Any]) -> str:
        """Build analysis prompt from signal data"""
        prompt = """You are a cellular network optimization expert. Analyze the following ZTE router signal data and recommend optimal band configurations for maximum performance.

## Current Signal Data:

"""
        # Network type
        prompt += f"**Network Type:** {signal_data.get('network_type', 'Unknown')}\n"
        prompt += f"**Active Bands:** {signal_data.get('band_info', 'Unknown')}\n"
        prompt += f"**Provider:** {signal_data.get('provider', 'Unknown')}\n\n"

        # LTE cells
        lte_cells = signal_data.get('lte_cells', [])
        if lte_cells:
            prompt += "### LTE Cells:\n"
            for i, cell in enumerate(lte_cells):
                prompt += f"- **Cell {i+1} ({cell.get('band', '?')})**: "
                prompt += f"RSRP={cell.get('rsrp1', '?')}dBm, "
                prompt += f"SINR={cell.get('sinr1', '?')}dB, "
                prompt += f"RSRQ={cell.get('rsrq', '?')}dB, "
                prompt += f"BW={cell.get('bandwidth', '?')}MHz, "
                prompt += f"PCI={cell.get('pci', '?')}\n"
            prompt += "\n"

        # 5G cells
        nr_cells = signal_data.get('nr_cells', [])
        if nr_cells:
            prompt += "### 5G NR Cells:\n"
            for i, cell in enumerate(nr_cells):
                prompt += f"- **Cell {i+1} ({cell.get('band', '?')})**: "
                prompt += f"RSRP={cell.get('rsrp1', '?')}dBm, "
                prompt += f"SINR={cell.get('sinr', '?')}dB, "
                prompt += f"RSRQ={cell.get('rsrq', '?')}dB, "
                prompt += f"BW={cell.get('bandwidth', '?')}MHz\n"
            prompt += "\n"

        # Current locks
        prompt += "### Current Configuration:\n"
        prompt += f"- LTE PCI Lock: {signal_data.get('lte_pci_lock', '0')}\n"
        prompt += f"- LTE EARFCN Lock: {signal_data.get('lte_earfcn_lock', '0')}\n"
        prompt += f"- SA Band Lock: {signal_data.get('nr5g_sa_band_lock', 'None')}\n"
        prompt += f"- NSA Band Lock: {signal_data.get('nr5g_nsa_band_lock', 'None')}\n\n"

        # Temperatures
        temps = signal_data.get('temperatures', {})
        if temps:
            prompt += "### Temperatures:\n"
            for sensor, val in temps.items():
                prompt += f"- {sensor.capitalize()}: {val}°C\n"
            prompt += "\n"

        prompt += """## Task:

Based on the signal metrics above, provide specific optimization recommendations for MAXIMUM DOWNLOAD SPEED.

1. **Signal Quality**: Briefly rate the metrics (RSRP, SINR).
2. **Recommended LTE Bands**: comma-separated (e.g., "1,3,7").
3. **Recommended 5G Bands**: comma-separated (e.g., "1,78").
4. **Network Mode**: Choose from `WL_AND_5G`, `Only_5G`, `LTE_AND_5G`.
5. **Logic**: Brief explanation.

Be concise. Use Markdown headings and bullet points."""

        return prompt

    def get_recommendation(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI recommendation for band optimization
        
        Args:
            signal_data: Signal info from ZTERouter.get_signal_info()
            
        Returns:
            Dict with recommendation and status
        """
        prompt = self._build_prompt(signal_data)

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower for more focused responses
                        "num_predict": 1000
                    }
                },
                timeout=120  # LLMs can take time
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "recommendation": result.get("response", ""),
                    "model": self.model,
                    "done": result.get("done", False)
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "details": response.text
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Make sure it's running on localhost:11434",
                "hint": "Start Ollama with: ollama serve"
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Ollama request timed out. The model may be loading or overloaded."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def check_connection(self) -> Dict[str, Any]:
        """Check if Ollama is available and model is loaded"""
        try:
            # Check Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                return {"available": False, "error": "Ollama not responding"}

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Smart matching: llama3 matches llama3:latest or llama3:8b
            match = None
            for full_name in model_names:
                if full_name == self.model or full_name.split(':')[0] == self.model:
                    match = full_name
                    break

            if match:
                return {
                    "available": True,
                    "model": match,
                    "all_models": model_names
                }
            else:
                return {
                    "available": False,
                    "error": f"Model '{self.model}' not found in Ollama library.",
                    "available_models": model_names,
                    "hint": f"Run 'ollama pull {self.model}' or select an available model from the dashboard."
                }

        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "error": "Ollama not running",
                "hint": "Start with: ollama serve"
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def parse_bands_from_recommendation(self, recommendation: str) -> Dict[str, Any]:
        """
        Try to extract band numbers from AI recommendation
        Returns parsed LTE and 5G bands if found
        """
        import re
        
        result = {
            "lte_bands": None,
            "nr_bands": None,
            "network_mode": None
        }
        
        # Try to find LTE bands pattern
        lte_patterns = [
            r"LTE [Bb]ands?[:\s]+([0-9,\s+]+)",
            r"enable LTE [Bb]ands?[:\s]+([0-9,\s+]+)",
            r"B([0-9]+(?:\s*[,+]\s*B?[0-9]+)*)"
        ]
        
        for pattern in lte_patterns:
            match = re.search(pattern, recommendation)
            if match:
                bands = re.findall(r'\d+', match.group(1))
                if bands:
                    result["lte_bands"] = "+".join(bands)
                    break
        
        # Try to find 5G bands pattern
        nr_patterns = [
            r"5G [Bb]ands?[:\s]+([0-9,\s+n]+)",
            r"NR [Bb]ands?[:\s]+([0-9,\s+n]+)",
            r"n([0-9]+(?:\s*[,+]\s*n?[0-9]+)*)"
        ]
        
        for pattern in nr_patterns:
            match = re.search(pattern, recommendation)
            if match:
                bands = re.findall(r'\d+', match.group(1))
                if bands:
                    result["nr_bands"] = "+".join(bands)
                    break
        
        # Try to find network mode
        modes = ["WL_AND_5G", "Only_5G", "LTE_AND_5G", "Only_LTE", "4G_AND_5G"]
        for mode in modes:
            if mode in recommendation or mode.lower() in recommendation.lower():
                result["network_mode"] = mode
                break
        
        return result
