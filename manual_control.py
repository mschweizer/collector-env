import gym

from gym_minigrid.window import Window
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

# noinspection PyUnresolvedReferences
import collector_env  # import statement registers env with gym


def redraw(window, env):
    img = env.render('rgb_array', tile_size=32, highlight=False)
    if hasattr(env, "mission"):
        window.set_caption(env.mission)
    window.show_img(img)


def reset(env, window):
    env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    redraw(window, env)


def step(env, window, action):
    obs, reward, terminated, info = env.step(action)
    print(f"step={env.step_count}, reward={reward:.2f}")

    if terminated:
        print("terminated!")
        reset(env, window)
    else:
        redraw(window, env)


def key_handler(env, window, event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step(env, window, env.actions.left)
        return
    if event.key == "right":
        step(env, window, env.actions.right)
        return
    if event.key == "up":
        step(env, window, env.actions.forward)
        return

    # Spacebar
    if event.key == " ":
        step(env, window, env.actions.toggle)
        return
    if event.key == "pageup":
        step(env, window, env.actions.pickup)
        return
    if event.key == "pagedown":
        step(env, window, env.actions.drop)
        return

    if event.key == "enter":
        step(env, window, env.actions.done)
        return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-MultiRoom-N6-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    env = gym.make(args.env)

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window("minigrid - " + args.env)
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    reset(env, window)

    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    main()
