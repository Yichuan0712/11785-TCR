import sys
import datetime
import os
import shutil
from pathlib import Path


def printl(*args, log_path=None, **kwargs):
    """
    Enhanced print function that logs to a file if log_path is provided.

    Args:
    *args: Variable length argument list, passed to print.
    log_path (str, optional): The file path to log the output. If None, acts like the standard print.
    **kwargs: Arbitrary keyword arguments, passed to print.
    """
    # Perform the standard print operation
    print(*args, **kwargs)

    # If a log_path is provided, also write the output to the specified log file
    if log_path:
        # Capture the print output
        old_stdout = sys.stdout
        sys.stdout = open(log_path, 'a')
        print(*args, **kwargs)  # Output goes to the log file
        sys.stdout.close()
        sys.stdout = old_stdout


def printl_file(file_path, log_path, show_name=True):
    filename = os.path.basename(file_path)
    with open(file_path, 'r') as file:
        if show_name:
            printl(f"#### {filename} ####", log_path=log_path)
        for line in file:
            formatted_line = "    " + line.rstrip()
            printl(formatted_line, log_path=log_path)


def prepare_saving_dir(parse_args):
    """
    Prepare a directory for saving a training results.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
    curdir_path = os.getcwd()

    result_path = os.path.abspath(os.path.join(parse_args.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoint')
    log_path = os.path.join(result_path, "loginfo.log")

    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    config_path = shutil.copy(parse_args.config_path, result_path)

    # Return the path to the result directory.
    return curdir_path, result_path, checkpoint_path, log_path, config_path
