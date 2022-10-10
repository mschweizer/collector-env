from gymnasium import register
from envs.collector import CollectorEnv

register(
    id="Collector-v0",
    entry_point="envs:CollectorEnv",
)
