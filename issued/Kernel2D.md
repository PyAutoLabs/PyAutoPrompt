This error likes to crop up in autolens assistant can you work out why:

│ ... first 5 lines hidden (Ctrl+O to show) ...                                                                                                                                                            │
│ y than library releases, so version mismatches are expected and not actionable for `main`-branch users.                                                                                                  │
│                                                                                                                                                                                                          │
│ You can also set the environment variable PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 to disable temporarily.                                                                                                  │
│   warnings.warn(_missing_version_warning(root, library_version))                                                                                                                                         │
│ Traceback (most recent call last):                                                                                                                                                                       │
│   File "<string>", line 1, in <module>                                                                                                                                                                   │
│   File "/home/jammy/Code/PyAutoLabs/PyAutoLens/autolens/__init__.py", line 156, in __getattr__                                                                                                           │
│     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")                                                                                                                               │
│ AttributeError: module 'autolens' has no attribute 'Kernel2D'  

I wonder if it was built against old code base by my collaborator and thus we need to reuse the tools
that build some aspects of autolens assistant?

Heres a SLACK chat:

Jam  [10:23 AM]
Does autolens_assistant need any kind of rebuild or whatnot done when I update workspaces or do a release
Rich  [10:25 AM]
There's a skill that should check the API but we might want to nail it down better
[10:25 AM]It could possibly keep a hash of the AutoLens release and then diff so the agent can see API updates


Note that Kernel2D was removed like 6 months ago so this is a very strange old API issue.