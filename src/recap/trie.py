"""A minimal, dependency-free trie for legal hierarchical SID paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

Token = Hashable


@dataclass
class _Node:
    children: dict[Token, "_Node"] = field(default_factory=dict)
    terminal: bool = False


class SIDTrie:
    """Catalog legality oracle used by RECAP at every repaired prefix."""

    def __init__(self, paths: Iterable[Sequence[Token]] = ()) -> None:
        self._root = _Node()
        for path in paths:
            self.add(path)

    def add(self, path: Sequence[Token]) -> None:
        if not path:
            raise ValueError("A SID path must contain at least one token.")
        node = self._root
        for token in path:
            node = node.children.setdefault(token, _Node())
        node.terminal = True

    def _node(self, prefix: Sequence[Token]) -> _Node:
        node = self._root
        for token in prefix:
            if token not in node.children:
                raise KeyError(f"Illegal SID prefix: {tuple(prefix)!r}")
            node = node.children[token]
        return node

    def legal_children(self, prefix: Sequence[Token]) -> tuple[Token, ...]:
        """Return the only tokens that can follow ``prefix`` in the catalog."""
        return tuple(self._node(prefix).children)

    def is_legal(self, path: Sequence[Token], *, require_terminal: bool = True) -> bool:
        try:
            node = self._node(path)
        except KeyError:
            return False
        return node.terminal if require_terminal else True
