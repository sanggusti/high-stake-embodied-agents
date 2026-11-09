"""L2 event recorder (docs/watcher.md): one JSONL file per sample."""

from dmb.watch.recorder import JsonlWatcher, NullWatcher, Watcher, record_score, watcher_from_env

__all__ = ["Watcher", "NullWatcher", "JsonlWatcher", "watcher_from_env", "record_score"]
