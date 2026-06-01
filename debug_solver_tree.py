"""Print COMSOL solver tree properties for version-specific solver settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import mph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--allowed', action='store_true')
    return parser.parse_args()


def print_node(node, indent: int, show_allowed: bool) -> None:
    prefix = '  ' * indent
    try:
        name = node.name()
    except Exception:
        name = '<unnamed>'
    try:
        tag = node.tag()
    except Exception:
        tag = '<no-tag>'
    try:
        node_type = node.type()
    except Exception:
        node_type = '<no-type>'
    print(f'{prefix}{name} {tag} {node_type}')

    try:
        props = node.properties()
    except Exception:
        props = {}
    for key, value in sorted(props.items()):
        print(f'{prefix}  {key} = {value}')
        if show_allowed:
            try:
                allowed = node.java.getAllowedPropertyValues(key)
                if allowed is not None:
                    print(f'{prefix}  ALLOWED {key} = {list(allowed)}')
            except Exception:
                pass

    try:
        children = list(node.children())
    except Exception:
        children = []
    for child in children:
        print_node(child, indent + 1, show_allowed)


def main() -> None:
    args = parse_args()
    client = mph.start(cores=args.cores)
    model = client.load(args.model.resolve())
    try:
        for solution in model / 'solutions':
            print_node(solution, 0, args.allowed)
    finally:
        client.remove(model)


if __name__ == '__main__':
    main()
