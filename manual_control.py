import gymnasium as gym
from minigrid.manual_control import key_handler, reset
from minigrid.utils.window import Window
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import envs

if __name__ == "__main__":
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

    env = gym.make(
        args.env,
        tile_size=args.tile_size,
    )

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window("minigrid - " + args.env)
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    seed = None if args.seed == -1 else args.seed
    reset(env, window, seed)

    # Blocking event loop
    window.show(block=True)
