"""Preserve optional CPML selection when generating inversion parameters."""
import configparser

from geopvi.forward.fwi2d.posterior import prepare_fwi_parameters
from test_cpml_api import aco2d


def test_parameter_writer_preserves_boundary_choice(tmp_path, aco2d):
    for boundary, pairs in [(None, 19), (0, 20), (1, 20)]:
        config = configparser.ConfigParser()
        config['FWI'] = dict(nx='19', nz='17', pml='8', Lc='3', laplace_slover='0',
                             ns='1', nt='81', ds='5', ns0='3', depths='3', depthr='2',
                             nr='5', dr='4', nr0='0', nt_interval='1', dx='10', dz='10',
                             dt='.0008', f0='25')
        if boundary is not None:
            config['FWI']['boundary_type'] = str(boundary)
        path = tmp_path/('params%s.txt' % boundary)
        prepare_fwi_parameters(str(path), config)
        lines = path.read_text().splitlines()
        assert len(lines) == 2*pairs
        if boundary is not None:
            assert int(lines[-1]) == boundary
        assert aco2d._boundary_parameters(lines) == (1 if boundary is None else boundary)
