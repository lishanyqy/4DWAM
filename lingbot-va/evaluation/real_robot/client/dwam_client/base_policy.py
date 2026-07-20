from typing import Dict


class BasePolicy:
    """Minimal policy interface used by the real-robot VA client."""

    def infer(self, obs: Dict) -> Dict:
        raise NotImplementedError

    def reset(self) -> None:
        pass
