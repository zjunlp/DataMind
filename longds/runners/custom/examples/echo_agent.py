"""Offline integration check only: this agent does not solve the benchmark."""


class EchoAgent:
    def __init__(self, prefix='Echo'):
        self.prefix = prefix
        self.turn_count = 0

    def start_task(self, workspace, data_dir):
        self.workspace = workspace
        self.data_dir = data_dir

    def run_turn(self, context, question):
        self.turn_count += 1
        return f'{self.prefix} turn {self.turn_count}: {question}'
