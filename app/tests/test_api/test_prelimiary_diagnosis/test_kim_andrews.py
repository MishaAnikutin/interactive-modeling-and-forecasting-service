import pytest

from tests.test_api.utils import process_variable


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('n', [1, 10, 20, 200, 300, 400, 1000])
@pytest.mark.parametrize('m', [1, 10, 20, 30, 40, 90])
@pytest.mark.parametrize('shift', [1, 10, 20, 200, 300, 400, 1000])
@pytest.mark.parametrize('trend', [True, False])
@pytest.mark.parametrize('const', [True, False])
def test_kim_andrews(
        n,
        m,
        shift,
        trend,
        const,
        client,
        balance
):
    ts = process_variable(balance)
    ts_len = len(balance.values)
    ts_nan = sum([(v is None) for v in balance.values])
    ts_eff = ts_len - shift - ts_nan
    result = client.post(
        url='/api/v1/preliminary_diagnosis/structure_shift_diagnosis/kim-andrews',
        json={
            "ts": ts,
            "n": n,
            "m": m,
            "shift": shift,
            "trend": trend,
            "const": const
        }
    )
    data = result.json()
    if (n <= m + 1) or (shift >= ts_len) or (n + m > ts_eff):
        assert result.status_code == 422
    else:
        assert result.status_code == 200, data