from gym import register
from collector_env.collector import CollectorEnv

register(
    id="Collector-v0",
    entry_point="collector_env:CollectorEnv",
)
