"""Offline smoke check. See react_agent.py for the actual ReAct example."""


class Agent:
    def __init__(self):
        self.turn_count = 0

    def respond(self, message: str) -> str:
        self.turn_count += 1
        return f"Offline echo demo, turn {self.turn_count}: {message}"
