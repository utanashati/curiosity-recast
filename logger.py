import os
import logging
import logging.config
import time
import tensorboard_logger as tb


LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def setup_logs(args):
    args.sum_base_dir = ('runs/{}/{}({})').format(
        args.env_name, time.strftime('%Y.%m.%d-%H.%M.%S'),
        args.short_description)

    if not os.path.exists(args.sum_base_dir):
        os.makedirs(args.sum_base_dir)

    configure(args.sum_base_dir, 'rl.log')

    args_list = [f'{k}: {v}\n' for k, v in vars(args).items()]
    logging.info("\nArguments:\n----------\n" + ''.join(args_list))
    logging.info('Logging run logs to {}'.format(args.sum_base_dir))
    tb.configure(args.sum_base_dir)


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
