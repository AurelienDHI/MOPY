
import mikeio
import numpy as np
import pandas as pd
import pytest
from mikeio import EUMType, EUMUnit, ItemInfo
from pandas.testing import assert_frame_equal
from mikeio import ItemInfo, EUMType, EUMUnit
from mikeoperationspy import ConnectionInfo
from mikeoperationspy import Application
from mikeoperationspy.modules import TimeSeriesManager


@pytest.fixture(scope="module")
def pandas_time_series():
    x = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    y = np.random.randn(len(x))
    series = pd.Series(y, x, name="MySeries")
    series.index.name = "Time"
    return series


@pytest.fixture()
def existing_time_series_path() -> str:
    return "/Models and Scenarios/HD_MF_CAL2/Model Setup/KhlongChamon - 0 - OpenSource/Input/Z21.dfs0 - 1 - QZ21"


@pytest.fixture(scope="module")
def connection_info():
    return ConnectionInfo(
        flavour="PostgreSQL",
        host="localhost",
        port=5436,
        user="admin",
        password="dssadmin",
        database="MOPY",
        workspace="workspace1",
    )


@pytest.fixture(scope="module")
def app(connection_info):
    return Application(connection_info)


@pytest.fixture(scope="module")
def time_series_manager(app):
    return app.time_series_manager


@pytest.fixture()
def new_time_series_path():
    return "/Testing/MySeries"


def test_get_existing_feature_path(
    time_series_manager: TimeSeriesManager, existing_time_series_path: str
):
    with pytest.raises(Exception):
        time_series_manager.get("Some path that doesn't exist")

    result = time_series_manager.get(existing_time_series_path)
    assert hasattr(result, "to_pandas")


def test_get_pandas(
    time_series_manager: TimeSeriesManager, existing_time_series_path: str
):
    result = time_series_manager.get(existing_time_series_path)
    df = result.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)

    assert len(df) == 1096
    assert list(df.columns) == ["Z21.dfs0 - 1 - QZ21"]

    row = df.iloc[5]
    assert str(row.name) == "2011-01-06 00:00:00"
    assert row["Z21.dfs0 - 1 - QZ21"] == pytest.approx(0.3000000119209289, abs=1e-3)

    assert df.max().max() == pytest.approx(62.049, abs=1e-3)


def test_get_mikeio(
    time_series_manager: TimeSeriesManager, existing_time_series_path: str
):
    result = time_series_manager.get(existing_time_series_path)
    da = result.to_mikeio()
    assert isinstance(da, mikeio.Dataset)
    assert da.n_items == 1
    item = da.items[0]
    assert item.type.name == "Discharge"
    assert item.unit.name == "meter_pow_3_per_sec"
    assert item.data_value_type.name == "Instantaneous"


def test_create_time_series_from_pandas_series(
    time_series_manager: TimeSeriesManager,
    pandas_time_series: pd.Series,
    new_time_series_path: str,
):
    # ensure it does not already exist
    with pytest.raises(Exception):
        time_series_manager.get(new_time_series_path)

    item = ItemInfo(
        itemtype=EUMType.Rainfall_Intensity,
        unit=EUMUnit.mm_per_hour,
        data_value_type="MeanStepBackward",
    )
    time_series_manager.create(new_time_series_path, pandas_time_series, item)

    qr = time_series_manager.get(new_time_series_path)

    assert qr._net_result.YAxisVariableCode == item.type.code
    assert qr._net_result.YAxisUnitCode == item.unit.code
    assert qr._net_result.ValueType.value__ == item.data_value_type.value

    validation_df = qr.to_pandas()
    pandas_time_series_df = pandas_time_series.to_frame()
    assert_frame_equal(pandas_time_series_df, validation_df, check_freq=False)


def test_delete_time_series(
    time_series_manager: TimeSeriesManager, new_time_series_path: str
):
    time_series_manager.delete(new_time_series_path)

    # ensure it no longer exists
    with pytest.raises(Exception):
        time_series_manager.get(new_time_series_path)