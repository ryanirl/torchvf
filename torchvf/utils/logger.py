# Copyright 2022 Ryan Peters
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import os


class CustomFormatter(logging.Formatter):
    """
    Custom formatter found here: 
        - https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/

    """
    grey     = "\x1b[38;21m"
    blue     = "\x1b[38;5;39m"
    yellow   = "\x1b[38;5;226m"
    red      = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset    = "\x1b[0m"

    def __init__(self, fmt):
        super(CustomFormatter, self).__init__()

        self.fmt = fmt
        self.FORMATS = {
            logging.INFO:     self.grey     + self.fmt + self.reset,
            logging.DEBUG:    self.blue     + self.fmt + self.reset,
            logging.WARNING:  self.yellow   + self.fmt + self.reset,
            logging.ERROR:    self.red      + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


def setup_logger(name, log_dir, filename = "log.txt"):
    if not os.path.exists(log_dir):
        print("LOG dir did not exist! Making LOG dir: '{log_dir}'")
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Only use custom formatter for the StreamHandler.
    color_formatter = CustomFormatter("[%(name)s]: %(message)s")
    basic_formatter = logging.Formatter("[%(name)s]: %(message)s")

    s_handler = logging.StreamHandler(stream = sys.stdout)
    s_handler.setFormatter(color_formatter)

    f_handler = logging.FileHandler(os.path.join(log_dir, filename), mode = 'w')
    f_handler.setFormatter(basic_formatter)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    return logger





