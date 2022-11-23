from collector_env import CollectorEnv


def test_updates_valuation_if_max_updates_not_set():
    interval_len = 10
    neg_rew = -1.
    pos_rew = 1.

    env = CollectorEnv(
        positive_object_reward=pos_rew,
        negative_object_reward=neg_rew,
        value_update_interval=interval_len,
        max_value_updates=None
    )

    old_valuations = [obj.value for obj in env.objects]

    env.reset()
    for _ in range(interval_len + 1):
        env.step(env.action_space.sample())

    new_values = [obj.value for obj in env.objects]

    assert old_valuations != new_values


def test_does_not_update_valuation_when_max_updates_is_reached():
    interval_len = 10
    max_updates = 1
    neg_rew = -1.
    pos_rew = 1.

    env = CollectorEnv(
        positive_object_reward=pos_rew,
        negative_object_reward=neg_rew,
        value_update_interval=interval_len,
        max_value_updates=max_updates
    )

    env.reset()
    updated_values = []
    for i in range(interval_len * (max_updates + 1) + 1):
        env.step(env.action_space.sample())
        if i == interval_len * max_updates - 1:
            updated_values = [obj.value for obj in env.objects]

    newest_values = [obj.value for obj in env.objects]

    assert updated_values == newest_values
