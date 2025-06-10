import argparse
from hailo_tools.__version__ import version


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments with 'command' field indicating selected subcommand.
    """

    parser = argparse.ArgumentParser(description="DL Toolbox CLI")

    parser.add_argument(
        "--version",
        "-v",
        "-V",
        action="version",
        version=f"{version}",
    )

    # Add dest parameter to track which subcommand was selected
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    # Convert subcommand parser
    convert_parser = subparsers.add_parser("convert", help="convert model")

    convert_parser.add_argument(
        "--model", "-m", type=str, help="Path to the model file (ONNX format)"
    )

    convert_parser.add_argument(
        "--hw-arch",
        "--hw_arch",
        "-ha",
        type=str,
        help="Hardware architecture",
    )

    convert_parser.add_argument(
        "--calib-set-path",
        type=str,
        help="Path to the calibration set directory",
    )
    convert_parser.add_argument(
        "--use-random-calib-set",
        "--use_random_calib_set",
        "-urcs",
        action="store_true",
        help="Use random calibration set",
    )
    convert_parser.add_argument(
        "--calib-set-size",
        "-crs",
        type=int,
        help="Size of the calibration set",
    )
    convert_parser.add_argument(
        "--model-script",
        "--model_script",
        "-ms",
        type=str,
        help="Path to the model script",
    )

    # Inference subcommand parser
    infer_parser = subparsers.add_parser("infer", help="infer model")

    infer_parser.add_argument(
        "--model", "-m", type=str, help="Path to the model file (ONNX format)"
    )
    infer_parser.add_argument(
        "--source",
        "-s",
        type=str,
        help="Path to the source file (video, image, or camera)",
    )
    infer_parser.add_argument(
        "--save",
        "-sv",
        action="store_true",
        help="Save the output video",
    )
    infer_parser.add_argument(
        "--save-path",
        "-sp",
        type=str,
        help="Path to save the output video",
    )
    infer_parser.add_argument(
        "--show",
        "-sh",
        action="store_true",
        help="Show the output video",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Selected command: {args.command}")
    print("All arguments:", vars(args))
