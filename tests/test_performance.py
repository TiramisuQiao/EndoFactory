import polars as pl
import pytest
from endofactory.core import EndoFactoryEngine

pytestmark = pytest.mark.performance


def test_mix_performance(base_config, benchmark):
    engine = EndoFactoryEngine(base_config)
    engine.load_datasets()

    def do_mix():
        return engine.mix_datasets()

    mixed = benchmark(do_mix)
    assert mixed is not None
    assert isinstance(mixed, pl.DataFrame)
