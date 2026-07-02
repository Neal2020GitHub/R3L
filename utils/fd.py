"""
File-descriptor-level stream control (a stdlib-only leaf).

Some libraries — notably Blender's bpy — write to the C-level stdout/stderr file
descriptors directly, so Python-level redirection (sys.stdout reassignment) cannot
silence them. `suppress_blender_output` redirects fds 1 and 2 to /dev/null for the
duration of the block and restores them afterwards. It lives alone here, away from
text/ANSI helpers, because file-descriptor control is its own responsibility.
"""

import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_blender_output():
    # Restore whatever stdout/stderr were current ON ENTRY — not sys.__stdout__.
    # This may be nested inside another redirection (e.g. contextlib.redirect_stdout
    # capturing a tqdm loop's chatter); hardcoding the real terminal would clobber
    # that outer redirect and leak later prints past it.
    prev_out, prev_err = sys.stdout, sys.stderr
    stdout_save = os.dup(1)
    stderr_save = os.dup(2)
    stdout_fd = open(os.devnull, 'w')
    stderr_fd = open(os.devnull, 'w')

    os.dup2(stdout_fd.fileno(), 1)
    os.dup2(stderr_fd.fileno(), 2)
    sys.stdout = stdout_fd
    sys.stderr = stderr_fd

    try:
        yield
    finally:
        os.dup2(stdout_save, 1)
        os.dup2(stderr_save, 2)
        os.close(stdout_save)
        os.close(stderr_save)
        sys.stdout = prev_out
        sys.stderr = prev_err
        stdout_fd.close()
        stderr_fd.close()
