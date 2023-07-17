"""Serve click based cli as a web app using streamlit"""

import logging
from collections import defaultdict
from collections.abc import Generator
from contextlib import closing, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, cast

import click
import streamlit as st
from typer.main import get_command

# Can't use relative import here because of the way streamlit works
from mrimagetools import cli2

logger = logging.getLogger(__name__)

Status = Literal["waiting", "running", "completed"]
Downloads = list[str]


@dataclass
class PageState:
    """State of a given page, corresponding to a click command"""

    status: Status = "waiting"
    """Which phase of the command is running"""

    temp_dir: Optional[TemporaryDirectory] = None
    """Location to save resources for this command invocation"""

    downloads: Downloads = field(default_factory=list)
    """Files to avail to user for download"""


@dataclass
class AppState:
    """Custom session info corresponding to a click cli"""

    command: click.Group
    """The click cli to render"""

    default_page: Optional[Callable] = None
    """The default page to render"""

    current_page: Optional[Callable] = None
    """The current page being rendered"""

    current_page_context: dict[click.Command, PageState] = field(default_factory=dict)
    """State for each page"""

    command_pages: list[Callable] = field(default_factory=list)
    """Callables to render the pages for the commands"""

    command_option_controls: dict[click.Command, dict[click.Parameter, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """Controls for each command option"""


class Stream(StringIO):
    """Stream to redicerct stdout/strerr to streamlit widgets"""

    def __init__(self, placeholder: Any, target: str):
        super().__init__()
        self.placeholder = placeholder
        self.target = target

    def write(self, s: str) -> int:
        res = super().write(s)
        getattr(self.placeholder, self.target)(self.getvalue())
        return res


@contextmanager
def st_stream_target(target: str) -> Generator[Stream, None, None]:
    """Context manager to redirect stdout/stderr to streamlit widgets

    :param target: name of the streamlit widget to use
    :returns: a stream to write to
    """

    placeholder = st.empty()
    with closing(Stream(placeholder, target)) as stream:
        yield stream


def save_file(tempdir: TemporaryDirectory, file: BytesIO) -> Path:
    """Helper to write a file in a temporary directory

    :param tempdir: Temporary Directory instance
    :param file: file-like object to write to disk
    :returns: The path to the file
    """
    p: Path = Path(tempdir.name) / file.name
    with open(p, "wb") as tempfile:
        tempfile.write(file.read())
    return p


def process_cmd(command: click.Group) -> dict[str, click.Command]:
    """Travarses a command tree and returns a mapping of all invocable commands.

    The keys in the returned mapping are are the command names prefixed with the
    names of all parent commands to ensure uniqueness

    :param command: The root command
    :return: A mapping of command names to commands.
    """
    register: dict[str, click.Command] = {}
    prefixmap: dict[click.Command, str] = {}
    cmds: list[click.Command] = [command]
    while cmds:
        cmd = cmds.pop()
        prefix = prefixmap.get(cmd, "")
        reg_key = f"{prefix}/{cmd.name}" if prefix else cmd.name or ""
        if hasattr(cmd, "callback") and cmd.callback is not None:
            register[reg_key] = cmd
        if hasattr(cmd, "commands"):
            subcommands = cast(click.Group, cmd).commands.values()
            for subcommand in subcommands:
                prefixmap[subcommand] = reg_key
                cmds.append(subcommand)
    return register


def set_up_command(command: click.Command, app_state: AppState) -> None:
    """Rendeer the widgets enabling user configuration/invocation of a command

    :param command: The command to render
    :param app_state: The application state
    """

    if command not in app_state.current_page_context:
        app_state.current_page_context[command] = PageState()

    logger.debug("Running page function for %s.", command.name)

    command_option_controls = app_state.command_option_controls

    st.markdown(command.help)

    # Step 1. Render the controls for the command parameters
    for option in command.params:
        logger.debug("Processing %s of type %s", option.name, option.type)

        desc = cast(Any, option).help if hasattr(option, "help") else option.name or ""
        key = f"{command.name}.{option.name}"

        if option.type == click.BOOL:
            command_option_controls[command][option] = st.checkbox(desc, key=key)
        elif option.type == click.STRING:
            command_option_controls[command][option] = st.text_input(desc, key=key)
        elif option.type in [click.INT, click.FLOAT]:
            command_option_controls[command][option] = st.number_input(desc, key=key)
        elif isinstance(option.type, click.Choice):
            c = cast(click.Choice, option.type)
            command_option_controls[command][option] = st.selectbox(
                desc, c.choices, key=key
            )
        elif isinstance(option.type, click.Path):
            p = cast(click.Path, option.type)
            if p.writable:
                command_option_controls[command][option] = st.checkbox(desc, key=key)
            else:
                multiple = option.nargs != 1 or option.multiple
                command_option_controls[command][option] = (
                    st.file_uploader(label=desc, key=key, accept_multiple_files=True)
                    if multiple
                    else st.file_uploader(label=desc, key=key)
                )
                # The if/else above is a workaround for type checking and the overloads
                # defined in the Streamlit API. The following is the correct code:
                # command_option_controls[command][option] = st.file_uploader(
                #     label=desc, key=key, accept_multiple_files=multiple
                # )

        else:
            logger.debug("Unsupported type: %s", option.type)
            command_option_controls[command][option] = None

    def update_status(command: click.Command, status: Status) -> None:
        app_state.current_page_context[command].status = status

    st.button(
        label="Run",
        key=f"run.{command.name}",
        on_click=update_status,
        kwargs={"command": command, "status": "running"},
    )


def execute_command(command: click.Command, app_state: AppState) -> None:
    """Invoke the command and communicate  stdout/stderr to the user

    :param command: The command to render
    :param app_state: The application state
    """

    page_state = app_state.current_page_context[command]
    if page_state.status != "running":
        return
    # Step 2. Gather the values for the params from the controls

    if page_state.temp_dir is None:
        # We're saving the temp directory instance in a field as opposed to using
        # a context manager because this needs to live across function invocations.
        page_state.temp_dir = TemporaryDirectory()

    download = partial(save_file, page_state.temp_dir)

    command_option_controls = app_state.command_option_controls

    kwargs: dict[str, Any] = {}
    downloads = page_state.downloads

    for option in command.params:
        if option.name is None:
            # Should't ever happen but just in case, skip it
            continue

        logger.debug("Processing option %s", option.name)
        value = command_option_controls[command].get(option)
        if value is None and not option.required:
            continue

        is_file = isinstance(option.type, click.Path)
        is_upload = is_file and cast(click.Path, option.type).readable
        is_multi = option.nargs != 1 or option.multiple

        if not is_file:
            kwargs[option.name] = value or option.default
        elif is_upload:
            if is_multi:
                kwargs[option.name] = [download(v) for v in (value or [])]
            elif value is not None:
                kwargs[option.name] = download(value)
            elif not option.required:
                kwargs[option.name] = option.default
        elif value:  # is download and user has requested it be computed
            downloads.append(option.name)
            kwargs[option.name] = Path(page_state.temp_dir.name) / option.name
    logger.debug("Option Values:\n%s", format(kwargs))

    # Step 3. Run the command
    logger.debug("Executing command")
    assert command.callback is not None

    with (
        st_stream_target("code") as out_stream,
        st_stream_target("code") as error_stream,
        redirect_stdout(out_stream),
        redirect_stderr(error_stream),
    ):
        command.callback(**kwargs)

    page_state.status = "completed"


def report_command_results(command: click.Command, app_state: AppState) -> None:
    """Report the results of a command to the user and offer any downloads.

    :param command: The command to render
    :param app_state: The application state
    """
    page_state = app_state.current_page_context[command]
    if page_state.status != "completed":
        return
    # Step 4. Display the output
    assert page_state.temp_dir is not None

    app_state.current_page_context[command].status = "completed"
    logger.debug("Displaying output")

    downloads = page_state.downloads
    if not downloads:
        logger.debug("No downloads to offer")
        return

    for item in downloads:
        item_path = Path(page_state.temp_dir.name) / item
        if not item_path.exists():
            logger.debug("Guessing file for %s", item)
            # let's guess, assuming a file extension was added to the name we provided
            for file in item_path.parent.glob(f"{item}.*"):
                item_path = file
                break
        logger.debug("File %s: %s: %s", item, item_path, item_path.exists())
        with open(item_path, "rb") as f:
            st.download_button(
                label=f"Download {item}",
                data=f,
                file_name=item,
                key=f"{command.name}.{item}.download",
            )


def command_page(command: click.Command) -> None:
    """Render the page for a command.

    :param command: The command to render
    """
    app_state = st.session_state.app_state
    set_up_command(command, app_state)
    execute_command(command, app_state)
    report_command_results(command, app_state)


def render_cli_as_streamlit_app(cli_app: click.Group) -> None:
    """Render a click application as a streamlit app.

    :param cli_app: The click application to render
    """
    if "app_state" not in st.session_state:
        st.session_state.app_state = app_state = AppState(command=cli_app)
        app_state.current_page = None
        command_pages = app_state.command_pages

    else:
        logger.debug("Command page functions already set up in session state.")
        return

    # for command in app_state.command.commands.values():
    for title, command in process_cmd(app_state.command).items():
        # This is nested as it is not used anywhere else
        def lambda_generator(command: click.Command) -> Callable:
            """Helper to work around late binding in for loops.

            Without this or something like it, all pages would render the same command.

            :param command: The command to render
            :return: A function that renders the command
            """
            return lambda: command_page(command)

        logger.debug("Prepping page function for %s.", command.name)
        page = lambda_generator(command)
        page.__name__ = title or command.name or "Unnamed Command"
        command_pages.append(page)


def set_and_show_page(page: Callable, from_sidebar: bool = False) -> None:
    """Set the current page and show it.

    When selecting a page from the sidebar link, the page is not rendered until the next
    time streamlit evaluates the script.

    :param page: Function to render the page
    :param from_sidebar: Whether the page was set from the sidebar
    """
    logger.debug("Setting current page to %s", page.__name__)
    state: AppState = st.session_state.app_state
    state.current_page = page
    if not from_sidebar:
        page()


def setup_sidebar_nav() -> None:
    """Setup the sidebar navigation.

    Each command gets an entry in the sidebar.
    """
    state: AppState = st.session_state.app_state
    st.sidebar.title(state.command.name or "Commands")
    page: Callable
    for page in state.command_pages:
        logger.debug("Adding %s to sidebar", page.__name__)

        def capture(page: Callable) -> None:
            st.sidebar.button(
                page.__name__, on_click=set_and_show_page, args=(page, True)
            )

        capture(page)


def show_current_or_default_command_page() -> None:
    """Show the current page or the default view if no page is set."""
    app_state: AppState = st.session_state.app_state
    page = app_state.current_page or app_state.default_page
    if page:
        set_and_show_page(page)
    else:
        st.write("Please select an option from the sidebar")


def style_sidebar_buttons() -> None:
    """Inject some CSS to style the sidebar buttons."""
    sidebar_css = """
    <style>
    [data-testid="stSidebar"] button {
        width: 100%;
        border: 0;
        text-align: left;
        justify-content: start;
    }
    [data-testid="stSidebar"] a {
        text-decoration: none;
        color: inherit;
        font-weight: bolder;
    }
    [data-testid="stSidebar"] a:hover {
        text-decoration: underline;
    }
    </style>
    """

    st.markdown(sidebar_css, unsafe_allow_html=True)


def add_feedback_links() -> None:
    """Add links to the sidebar for reporting bugs and giving feedback"""
    bug_report_link = (
        "https://goldstandardphantoms.atlassian.net/rest/collectors"
        "/1.0/template/form/da4d5e70?os_authType=none#"
    )
    feedback_report_link = (
        "https://goldstandardphantoms.atlassian.net/rest/collectors"
        "/1.0/template/form/26bf82ca?os_authType=none#"
    )

    report_bug = f"Report a [bug 🐛]({bug_report_link})"
    give_feedback = f"Give [feedback 📢]({feedback_report_link})"

    with st.sidebar:
        st.markdown(f"{report_bug} | {give_feedback}", unsafe_allow_html=True)


def main() -> None:
    """Main function"""
    logger.debug(
        (
            "\n"
            "============================================================\n"
            "Time: %s\n"
            "State:%s\n"
        ),
        datetime.now(),
        pformat(st.session_state.to_dict()),
    )

    cmd = get_command(cli2.app)
    cmd = cast(click.Group, cmd)

    if not cmd.name:
        cmd.name = "mrimagetools"

    style_sidebar_buttons()
    render_cli_as_streamlit_app(cmd)
    setup_sidebar_nav()
    show_current_or_default_command_page()
    add_feedback_links()


if __name__ == "__main__":
    from streamlit import runtime

    if not runtime.exists():
        import sys

        from streamlit.web import cli

        # If we're not running inside Streamlit, just run the app.
        sys.argv = ["streamlit", "run", sys.argv[0]]
        if not TYPE_CHECKING:
            # Bypass some typing issues in click. See https://github.com/pallets/click/issues/2227
            sys.exit(cli.main())
    else:
        main()
