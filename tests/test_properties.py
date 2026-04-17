"""Property-based tests for invariants of classify() and noise_floor()."""

from hypothesis import given, settings
from hypothesis import strategies as st

from dictcollision import classify, noise_floor

_chars = st.text(alphabet="abcdefgh", min_size=1, max_size=6)
_token_list = st.lists(_chars, min_size=5, max_size=100)
_dict_list = st.lists(_chars, min_size=1, max_size=50)


@given(tokens=_token_list, dictionary=_dict_list)
@settings(max_examples=30, deadline=None)
def test_noise_floor_in_unit_interval(tokens, dictionary):
    r = noise_floor(tokens, set(dictionary))
    assert 0.0 <= r <= 1.0


@given(tokens=_token_list, dictionary=_dict_list)
@settings(max_examples=30, deadline=None)
def test_classify_fractions_in_unit_interval(tokens, dictionary):
    r = classify(tokens, set(dictionary), n_nulls=3)
    assert 0.0 <= r.signal <= 1.0
    assert 0.0 <= r.shared_hit <= 1.0
    assert 0.0 <= r.anti_signal <= 1.0
    assert 0.0 <= r.shared_miss <= 1.0
    assert 0.0 <= r.apparent_hit_rate <= 1.0 + 1e-9


@given(tokens=_token_list, dictionary=_dict_list)
@settings(max_examples=30, deadline=None)
def test_net_signal_bounded_by_apparent(tokens, dictionary):
    r = classify(tokens, set(dictionary), n_nulls=3)
    # net = signal - anti_signal;  apparent = signal + shared_hit.
    # anti_signal >= 0 and shared_hit >= 0, so net <= apparent always.
    assert r.net_signal <= r.apparent_hit_rate + 1e-9


@given(tokens=_token_list, dictionary=_dict_list)
@settings(max_examples=30, deadline=None)
def test_correction_nonnegative(tokens, dictionary):
    r = classify(tokens, set(dictionary), n_nulls=3)
    assert r.correction >= -1e-9


@given(tokens=_token_list, dictionary=_dict_list)
@settings(max_examples=30, deadline=None)
def test_apparent_equals_signal_plus_shared_hit(tokens, dictionary):
    r = classify(tokens, set(dictionary), n_nulls=3)
    assert abs(r.apparent_hit_rate - (r.signal + r.shared_hit)) < 1e-9
