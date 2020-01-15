import os
import logging
import logging.config


LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def configure(dir_, file):
    logdict = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'verbose': {
                'format': "[%(asctime)s] %(levelname)s " \
                    "[%(threadName)s:%(lineno)s] %(message)s",
                'datefmt': "%Y-%m-%d %H:%M:%S"
            },
            'simple': {
                'format': '%(levelname)s %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': LOG_LEVEL,
                'class': 'logging.StreamHandler',
                'formatter': 'verbose'
            },
            'file': {
                'level': LOG_LEVEL,
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'verbose',
                'filename': os.path.join(dir_, file),
                'maxBytes': 10 * 10**6,
                'backupCount': 3
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': LOG_LEVEL,
            },
        }
    }

    logging.config.dictConfig(logdict)
