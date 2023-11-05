import logging
from ipdb import set_trace as debug
import pathlib
import subprocess
from rich.console import Console
from rich.logging import RichHandler

log = logging.getLogger(__name__)


def git_describe() -> str:
    label = subprocess.check_output(["git", "describe", "--always", "HEAD"]).strip().decode("utf-8")
    return label


def git_clean() -> bool:
    git_output = subprocess.check_output(["git", "status", "--porcelain"]).strip().decode("utf-8")
    is_clean = len(git_output) == 0

    return is_clean


def git_diff() -> str:
    diff = subprocess.check_output(["git", "diff", "HEAD"]).strip().decode("utf-8")
    if len(diff) == 0 or diff[-1] != "\n":
        diff = diff + "\n"
    return diff


def log_git_info(run_dir: pathlib.Path) -> None:
    label = git_describe()
    if git_clean():
        log.info("Working tree is clean! HEAD is {}".format(label))
        return

    diff_path = run_dir / "diff.patch"
    log.warning("Continuing with dirty working tree! HEAD is {}".format(label))
    log.warning("Saving the results of git diff to {}".format(diff_path))
    with open(diff_path, "w") as f:
        f.write(git_diff())

def setup_logger(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = open(log_dir / "log.txt", "w")
    file_console = Console(file=log_file, width=150)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[RichHandler(), RichHandler(console=file_console)],
    )