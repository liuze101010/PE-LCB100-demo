from __future__ import annotations

from redis import Redis
from rq import Queue, Worker

from .config import load_settings
from .db import init_database


def main() -> None:
    settings = load_settings()
    init_database()
    connection = Redis.from_url(settings.redis_url)
    Worker([Queue("polymer-agent", connection=connection)], connection=connection).work()


if __name__ == "__main__":
    main()
