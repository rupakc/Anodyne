"""Task-metric providers (sub-system F). Importing this package registers every
provider module as a side effect -- each module calls `register_provider` at
import time, so callers only need `import anodyne_evaluation.judges.task_metrics`
(or trigger it transitively) before using `provider_for`/`catalog_for`.
"""

from __future__ import annotations

from anodyne_evaluation.judges.task_metrics import (
    chat,  # noqa: F401
    generic,  # noqa: F401
    graph_qa,  # noqa: F401
    media,  # noqa: F401
    qa,  # noqa: F401
    summarization,  # noqa: F401
    tabular,  # noqa: F401
    text_classification,  # noqa: F401
)
