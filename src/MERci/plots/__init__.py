# MERci/plots/__init__.py
"""
Plotting functions used by notebooks, one module per notebook/topic --
mirrors the sibling MERlin project's ``merlin/plots/`` convention (one file
per analysis task holding that task's plotting functions), adapted to
MERci's notebook-centric organization: the unit here is the notebook/topic
rather than a task class.

Non-plotting logic stays in the existing domain modules (``acquisition/``,
``analysis/``, ``common/``, etc.) -- only human-facing figure rendering
belongs here.
"""
