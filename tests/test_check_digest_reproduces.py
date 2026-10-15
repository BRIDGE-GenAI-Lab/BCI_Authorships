from scripts.check_digest_reproduces import compare_digests


def test_identical_digests_have_no_mismatch():
    digest = {"a": {"b": 0.9143557487099897, "c": [1, 2]}, "d": "x"}
    assert compare_digests(digest, digest) == []


def test_changed_float_is_reported_with_its_path():
    expected = {"a": {"b": 0.9143557487099897}}
    actual = {"a": {"b": 0.9143557487099898 + 1e-6}}
    assert compare_digests(expected, actual) == ["a.b"]


def test_missing_and_extra_keys_are_reported():
    assert sorted(compare_digests({"a": 1}, {"b": 1})) == ["a", "b"]


def test_list_length_change_is_reported():
    assert compare_digests({"a": [1, 2]}, {"a": [1]}) == ["a"]
