import hailo_toolbox as ht
from hailo_toolbox.cli import parse_args
from hailo_toolbox.utils.config import Config
from hailo_toolbox.inference import InferenceEngine


def main():
    args = parse_args()
    print(vars(args))
    config = Config(vars(args))
    engine = InferenceEngine(config, args.callback)
    engine.run()


if __name__ == "__main__":
    main()
