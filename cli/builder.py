"""
Builder mode: launch the Scene Builder web app.

Unlike text/scene, this is NOT a batch synthesis run — the Scene Builder is a
long-running authoring tool (a Flask server) whose exported scene JSON later feeds
scene mode. So there are no numbered stages: just the launch screen and the server,
which blocks until Ctrl-C. The heavy import (the Flask app, which pulls CLIP/SBERT and
bpy) stays LAZY, inside run_builder, so importing this module stays cheap — and the
preflight runs (and can fail friendly) before any of that loads.
"""

import contextlib
import io
import sys

from colorama import Fore, Style

from utils.console import INDENT, section, spinner
from utils.log import print_error


def run_builder(args) -> None:
    """Load the retriever behind a spinner, print the launch screen, then serve the
    Scene Builder until Ctrl-C. `args` is unused (the builder is self-contained); it is
    kept for the uniform mode-flow signature shared with run_text / run_scene.

    The retriever's load chatter is captured (redirect_stdout) so it does not smear the
    spinner — the same call-site quieting cli.flow uses for the Holodeck planner. The
    port is resolved via builder.app.find_available_port() BEFORE the retriever loads
    (so a "no free port" failure is instant, not after the ~10s CLIP/SBERT load) and
    BEFORE the URL is printed, so the printed URL always matches the bound address."""
    from builder.app import (
        HOST, NoAvailablePortError, find_available_port, get_retriever, serve,
    )

    section("Scene Builder")
    try:
        port = find_available_port()
    except NoAvailablePortError as e:
        print_error(str(e))
        sys.exit(1)

    with spinner("starting server (loading CLIP/SBERT)"), contextlib.redirect_stdout(io.StringIO()):
        get_retriever()
    print(f"{INDENT}{Fore.GREEN}✓{Fore.RESET} serving at  "
          f"{Style.DIM}→{Style.RESET_ALL}  http://{HOST}:{port}")
    print()
    print(f"{INDENT}{INDENT}{Style.DIM}open the link in your browser to build{Style.RESET_ALL}")
    print(f"{INDENT}{INDENT}{Style.DIM}press Ctrl-C to stop{Style.RESET_ALL}")
    print()
    serve(HOST, port)
