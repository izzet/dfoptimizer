"""Runner script for DFOptimizer service."""

import argparse
import logging
import logging.config
import os
import sys

import structlog


def configure_logging(level: str = "info"):
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]

    logging_config = {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level.upper(),
                "formatter": "json",
            },
        },
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": pre_chain,
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": level.upper(),
                "propagate": False,
            }
        },
    }

    logging.config.dictConfig(logging_config)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def main():
    parser = argparse.ArgumentParser(description="DFOptimizer streaming service")
    parser.add_argument("--group-file", default=os.environ.get("MOFKA_GROUP_FILE", "mofka.group.json"))
    parser.add_argument("--input-topic", default=os.environ.get("DFOPTIMIZER_INPUT_TOPIC", "diagnosis.findings"))
    parser.add_argument("--output-topic", default=os.environ.get("DFOPTIMIZER_OUTPUT_TOPIC", "optimizer.plans"))
    parser.add_argument("--registry-topic", default=os.environ.get("DFOPTIMIZER_REGISTRY_TOPIC", "optimizer.registry"))
    parser.add_argument("--idle-timeout", type=int, default=int(os.environ.get("DFOPTIMIZER_IDLE_TIMEOUT_SEC", "0")))
    parser.add_argument("--pull-timeout-ms", type=int, default=int(os.environ.get("DFOPTIMIZER_PULL_TIMEOUT_MS", "1000")))
    parser.add_argument("--consumer-name", default=os.environ.get("DFOPTIMIZER_CONSUMER_NAME", ""))
    parser.add_argument("--debug", action="store_true", default=os.environ.get("DFOPTIMIZER_DEBUG", "") == "1")
    args = parser.parse_args()

    configure_logging(level="debug" if args.debug else "info")
    logger = structlog.get_logger()

    from dfoptimizer.optimizer import Optimizer

    logger.info(
        "optimizer.start",
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        registry_topic=args.registry_topic,
    )

    optimizer = Optimizer()

    optimizer.run_mofka(
        group_file=args.group_file,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        registry_topic=args.registry_topic,
        consumer_name=args.consumer_name,
        idle_timeout_sec=args.idle_timeout,
        pull_timeout_ms=args.pull_timeout_ms,
    )


if __name__ == "__main__":
    main()
