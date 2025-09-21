import argparse
import enum
import os
import comfy.options


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()

parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0,::", help="Specify the IP address to listen on (default: 127.0.0.1).")
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")

parser.add_argument("--base-directory", type=str, default=None, help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")
parser.add_argument("--extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+', action='append', help="Load one or more extra_model_paths.yaml files.")
parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory. Overrides --base-directory.")
parser.add_argument("--temp-directory", type=str, default=None, help="Set the ComfyUI temp directory (default is in the ComfyUI directory). Overrides --base-directory.")
parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory. Overrides --base-directory.")
parser.add_argument("--auto-launch", action="store_true", help="Automatically launch ComfyUI in the default browser.")
parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")

class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.Auto, help="Default preview method for sampler nodes.", action=EnumAction)
parser.add_argument("--preview-size", type=int, default=512, help="Sets the maximum preview size for sampler nodes.")

cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument("--cache-classic", action="store_true", help="Use the old style (aggressive) caching.")
cache_group.add_argument("--cache-lru", type=int, default=0, help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")
cache_group.add_argument("--cache-none", action="store_true", help="Reduced RAM/VRAM usage at the expense of executing every node for each run.")

parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")
parser.add_argument("--deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap when loading ckpt/pt files.")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap when loading safetensors.")

parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")
parser.add_argument("--whitelist-custom-nodes", type=str, nargs='+', default=[], help="Specify custom node folders to load even when --disable-all-custom-nodes is enabled.")
parser.add_argument("--disable-api-nodes", action="store_true", help="Disable loading all api nodes.")

parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")

parser.add_argument("--verbose", default='INFO', const='DEBUG', nargs="?", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
parser.add_argument("--log-stdout", action="store_true", help="Send normal process output to stdout instead of stderr (default).")

# The default built-in provider hosted under web/
DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"

parser.add_argument(
    "--front-end-version",
    type=str,
    default=DEFAULT_VERSION_STRING,
    help="""
    Specifies the version of the frontend to be used. This command needs internet connectivity to query and
    download available frontend implementations from GitHub releases.

    The version string should be in the format of:
    [repoOwner]/[repoName]@[version]
    where version is one of: "latest" or a valid version number (e.g. "1.0.0")
    """,
)

def is_valid_directory(path: str) -> str:
    """Validate if the given path is a directory, and check permissions."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"The path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory.")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"You do not have read permissions for '{path}'.")
    return path

parser.add_argument(
    "--front-end-root",
    type=is_valid_directory,
    default=None,
    help="The local filesystem path to the directory where the frontend is located. Overrides --front-end-version.",
)

parser.add_argument("--user-directory", type=is_valid_directory, default=None, help="Set the ComfyUI user directory with an absolute path. Overrides --base-directory.")

parser.add_argument("--enable-compress-response-body", action="store_true", help="Enable compressing response body.")

parser.add_argument(
    "--comfy-api-base",
    type=str,
    default="https://api.comfy.org",
    help="Set the base URL for the ComfyUI API.  (default: https://api.comfy.org)",
)

database_default_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "user", "comfyui.db")
)
parser.add_argument("--database-url", type=str, default=f"sqlite:///{database_default_path}", help="Specify the database URL, e.g. for an in-memory database you can use 'sqlite:///:memory:'.")

if comfy.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.disable_auto_launch:
    args.auto_launch = False
