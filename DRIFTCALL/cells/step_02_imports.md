# Consolidated imports

Pulls in the stdlib + third-party modules used throughout the notebook so each later cell can focus on its module logic. Heavy optional wheels (numpy, fastapi, soundfile, etc.) are loaded defensively — a missing wheel surfaces as `None` from `get_optional(...)` rather than aborting the notebook.
