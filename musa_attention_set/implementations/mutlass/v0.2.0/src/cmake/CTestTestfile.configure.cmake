# Copyright (c) 2024 - 2024 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Generated file

set(TEST_SETS_SUPPORTED @TEST_SETS_SUPPORTED@)

set(TEST_EXE_PATH @TEST_EXE_PATH@)
set(TEST_EXE_WORKING_DIRECTORY @TEST_EXE_WORKING_DIRECTORY@)
set(MUTLASS_USE_EXTENDED_ADD_TEST_FORMAT @TEST_USE_EXTENDED_FORMAT@)

if (DEFINED ENV{MUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_MUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{MUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_MUTLASS_TEST_EXECUTION_ENVIRONMENT @MUTLASS_TEST_EXECUTION_ENVIRONMENT@)
endif()

@_INLINE_PER_TEST_CODE@
