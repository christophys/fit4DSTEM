# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""NOMAD schema of the 4D-STEM (CBED mapping) measurement ELN written by fit4dstem (tab 4).

``ELNFourDSTEMMeasurement`` extends ``MappingMeasurement`` of the nomad-measurements plugin (a ``Measurement`` with
mapping results) with the acquisition settings, the data sets (specimen scan, vacuum reference, rocking-curve
libraries) and the NXem results file; the processing steps of the fit4dstem workflow are its ``steps``.
"""
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import MeasurementResult
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection
from nomad.datamodel.data import ArchiveSection

try:  # nomad-measurements (FAIRmat-NFDI/nomad-measurements)
    from nomad_measurements.general import NOMADMeasurementsCategory
    from nomad_measurements.mapping.schema import MappingMeasurement as _Base
    _CATEGORIES = [NOMADMeasurementsCategory]
except Exception:  # noqa: BLE001  -- plain Measurement if nomad-measurements is not installed
    from nomad.datamodel.metainfo.basesections import Measurement as _Base
    _CATEGORIES = []

m_package = SchemaPackage(name="fit4dstem_nomad")


class FourDSTEMSettings(ArchiveSection):
    """Acquisition settings of the 4D-STEM scan."""
    m_def = Section(label="Acquisition settings")
    acceleration_voltage = Quantity(type=float, unit="V", a_eln=ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity, defaultDisplayUnit="kV"))
    convergence_semi_angle = Quantity(type=float, unit="mrad", a_eln=ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity))
    detector_sampling = Quantity(type=float, unit="mrad", description="angle per detector pixel",
                                 a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity))
    scan_step = Quantity(type=float, unit="nm", a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity))
    scan_shape = Quantity(type=int, shape=[2], description="scan positions (y, x)")
    detector_shape = Quantity(type=int, shape=[2], description="detector pixels (y, x)")
    zone_axis = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))
    dwell_time = Quantity(type=float, unit="ms", a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity))
    beam_current = Quantity(type=float, unit="pA", a_eln=ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity))
    detector = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))


class FourDSTEMDataset(ArchiveSection):
    """A data file of the analysis."""
    m_def = Section(label_quantity="name")
    name = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))
    file = Quantity(type=str, a_browser=dict(adaptor="RawFileAdaptor"),
                    a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity))
    role = Quantity(type=str, description="specimen | vacuum reference | library",
                    a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))


class FourDSTEMResult(MeasurementResult):
    """Summary of the maps (the maps themselves are in the NXem file)."""
    m_def = Section(label="Results")
    nxem_file = Quantity(type=str, a_browser=dict(adaptor="RawFileAdaptor"),
                         a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity))
    summary = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.RichTextEditQuantity))


class ELNFourDSTEMMeasurement(_Base, EntryData):
    """4D-STEM measurement of a ferroelectric superlattice analysed by CBED fitting (fit4dstem)."""
    m_def = Section(categories=_CATEGORIES, label="4D-STEM CBED mapping (fit4dstem)",
                    a_eln=ELNAnnotation(lane_width="600px"))
    settings = SubSection(section_def=FourDSTEMSettings)
    datasets = SubSection(section_def=FourDSTEMDataset, repeats=True)
    nxem_file = Quantity(type=str, description="NXem file with the maps",
                         a_browser=dict(adaptor="RawFileAdaptor"),
                         a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity))
    results_summary = Quantity(type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.RichTextEditQuantity))
    analysis_software = Quantity(type=str, default="fit4dstem",
                                 a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity))

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        if not self.method:
            self.method = "4D-STEM (CBED mapping)"


m_package.__init_metainfo__()
