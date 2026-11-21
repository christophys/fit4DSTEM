# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
from nomad.config.models.plugins import SchemaPackageEntryPoint


class FourDSTEMSchemaEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from .fourdstem import m_package

        return m_package


fourdstem_schema = FourDSTEMSchemaEntryPoint(
    name="fit4dstem 4D-STEM measurement",
    description="ELN of a 4D-STEM (CBED mapping) measurement analysed with fit4dstem, based on the "
                "MappingMeasurement of nomad-measurements.",
)
