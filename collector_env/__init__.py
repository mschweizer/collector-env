from gym import register
from collector_env.collector import *
from collector_env.single_object import SingleObjectEnv5x5

register(
    id="Collector-7x7-v0",
    entry_point="collector_env:CollectorEnv7x7",
)

register(
    id="Collector-5x5-v0",
    entry_point="collector_env:CollectorEnv5x5",
)

register(
    id="SingleObjectCollector-5x5-v0",
    entry_point="collector_env:SingleObjectEnv5x5",
)
