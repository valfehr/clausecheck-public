"""Create a login for the local ClauseCheck web UI / API (NOT an Anthropic key).

Generates a random key, stores only its hash in the local database, saves the key in data/dev_users.json
(git-ignored) and prints it once.

  py -m uv run python scripts/add_user.py alice my-firm
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck import api  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: add_user.py USER_NAME FIRM_NAME   (users in the same firm see the same documents)")
    user, firm = sys.argv[1], sys.argv[2]
    key = secrets.token_urlsafe(24)
    api.add_user(user, firm, key)
    path = ROOT / "data" / "dev_users.json"
    keys = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    keys[user] = key
    path.write_text(json.dumps(keys, indent=1))
    print(f"user '{user}' in firm '{firm}' created. Sign-in key (also saved in data/dev_users.json):\n{key}")


if __name__ == "__main__":
    main()
