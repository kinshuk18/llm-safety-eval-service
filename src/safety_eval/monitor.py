"""Rolling-window drift and flagged-rate monitor with real-time alerting."""
from collections import deque
from typing import Dict, Any
import yaml

class SafetyMonitor:
    def __init__(self, window_size: int = 100, alert_threshold: float = 0.20):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.history = deque(maxlen=window_size)
        self.total_requests = 0

    def record(self, flagged: bool) -> None:
        self.history.append(1 if flagged else 0)
        self.total_requests += 1

    def get_status(self) -> Dict[str, Any]:
        count = len(self.history)
        flagged_count = sum(self.history)
        flagged_rate = float(flagged_count / count) if count > 0 else 0.0
        alert = bool(flagged_rate > self.alert_threshold and count >= min(10, self.window_size))
        return {
            "total_requests": self.total_requests,
            "window_size": count,
            "max_window_size": self.window_size,
            "flagged_count": flagged_count,
            "flagged_rate": round(flagged_rate, 4),
            "alert_threshold": self.alert_threshold,
            "alert": alert,
        }
