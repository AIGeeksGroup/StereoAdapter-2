# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lazy imports to avoid dependency issues when only using submodules
# (e.g., for RAFT-Stereo which only needs DinoV2 + DPT)
def __getattr__(name):
    if name == "DepthAnything3Net":
        from depth_anything_3.model.da3 import DepthAnything3Net
        return DepthAnything3Net
    elif name == "NestedDepthAnything3Net":
        from depth_anything_3.model.da3 import NestedDepthAnything3Net
        return NestedDepthAnything3Net
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__export__ = [
    "NestedDepthAnything3Net",
    "DepthAnything3Net",
]
