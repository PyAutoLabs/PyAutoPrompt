A common problem is users pair a workspace with a different version to their installed software, leaidng to API
inconsistencies and config mismatches.

What do you think we can do about this? One option would be if they are running inside a workspace, it has its
version stored somewhere which is compared to their source code on import. However, you could still end up
with users doing work outside their workspace and copy and pasting old code and API and whatnot. This version number 
could be put in configs to be a bit more secure (e.g. even in their own drive they probably need a config)
but not clear cut.