# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Temporal workflow, activities and models of the fit4dstem NOMAD action.

No NOMAD import at module level: Temporal re-imports workflow modules (and their parent packages) in its sandbox,
where NOMAD's imports are not allowed.  NOMAD is only used inside the activities (``backend.NomadBackend``).
The entry point lives in :mod:`fit4dstem_nomad.actions`.
"""
