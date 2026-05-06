Most examples now use Pathlib, but there are still some os.path.join uses from
legacy.

Can you update all source code and workspaces to not use path.join at all
and always use Pathlib?

This should aspan all source code, workspaces, etc, no more os.path!