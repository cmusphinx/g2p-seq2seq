# Copyright 2016 AC Technologies LLC. All Rights Reserved.
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
# ==============================================================================

"""Python 2/3 compat helpers
"""
from __future__ import absolute_import, division, print_function

import sys

PY3 = sys.version_info[0] == 3

if PY3:
  text_type = str
else:
  text_type = unicode


def force_text(s, encoding="utf-8", errors="strict"):
  if issubclass(type(s), text_type):
    return s
  return s.decode(encoding, errors)
