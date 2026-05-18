








Can you change the advise for this:@

Traceback (most recent call last):
  File "/home/jammy/Code/PyAutoLabs/autolens_workspace/scripts/guides/modeling/slam_start_here.py", line 129, in <module>
    import autofit as af
  File "/home/jammy/Code/PyAutoLabs/PyAutoFit/autofit/__init__.py", line 147, in <module>
    check_version(__version__)
  File "/home/jammy/Code/PyAutoLabs/PyAutoConf/autoconf/workspace.py", line 133, in check_version
    raise WorkspaceVersionMismatchError(
autoconf.workspace.WorkspaceVersionMismatchError: Workspace version (2026.5.14.2) at /home/jammy/Code/PyAutoLabs/autolens_workspace does not match the installed library version (2026.5.8.2).

This usually means your installed library was upgraded but your workspace clone is from an older release tag. Re-clone the workspace at the matching tag:

    git clone --branch 2026.5.8.2 <workspace-repo-url>

To bypass this check, edit config/general.yaml:

    version:
      workspace_version_check: False

IMPORTANT: If you cloned the workspace from `main` rather than a release tag, you should set `workspace_version_check: False`. The `main` branch updates much more frequently than library releases, so version mismatches are expected and not actionable for `main`-branch users.

Can you instead say that if their workspace is more up to date than their source code, recommend they update the source code
(give the specific pip install, make sure its autolens if on autolens_workspace and autogalaxy if on autogalaxy_workspace).
Conversely if the source code is more up to date recommend they git pull the latest main branch of the workspace. lets not
do this giot clone --branch thing as we dont do version branches currently.