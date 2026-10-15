import numpy as np
import pytest

from authorship.strong_decoders import (
    CV_SCHEMES,
    EEGNet,
    _grouped_cv_auc,
    fit_strong_decoder,
    resample_epochs,
    two_fold_file_split,
)

torch = pytest.importorskip("torch")
pytest.importorskip("pyriemann")

# numpy 2.0 on macOS Accelerate raises spurious floating-point flags on ordinary matmul
# (a plain random 50x50 product warns too); they are not numerical failures.
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")


def _synthetic(n_per_class=150, n_channels=16, sfreq=256.0, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq)  # 1.0 s, like the -200..800 ms epochs
    time = np.arange(n_samples) / sfreq
    p300 = 4e-6 * np.exp(-((time - 0.5) ** 2) / (2 * 0.05 ** 2))  # peak ~300 ms post-onset
    spatial = np.linspace(0.2, 1.0, n_channels)[:, None]
    noise = rng.normal(0, 5e-6, size=(2 * n_per_class, n_channels, n_samples))
    labels = np.r_[np.ones(n_per_class, int), np.zeros(n_per_class, int)]
    epochs = noise + labels[:, None, None] * spatial * p300
    groups = np.array(["f1", "f2"] * n_per_class)
    return epochs, labels, groups, sfreq


def test_resample_shape():
    epochs = np.zeros((3, 16, 256))
    assert resample_epochs(epochs, 256.0, 128.0).shape == (3, 16, 128)


def test_two_fold_split_alternates_sorted_file_names():
    groups = np.array(["c.edf", "a.edf", "b.edf", "a.edf", "d.edf", "c.edf"])
    # sorted: a -> 0, b -> 1, c -> 0, d -> 1
    np.testing.assert_array_equal(two_fold_file_split(groups), [0, 0, 1, 0, 1, 0])


def test_two_fold_split_keeps_each_file_whole():
    groups = np.repeat([f"f{index:02d}.edf" for index in range(5)], 4)
    folds = two_fold_file_split(groups)
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    assert set(folds) == {0, 1}


def test_single_calibration_file_gives_nan_auc():
    epochs, labels, _, _ = _synthetic(n_per_class=20)
    groups = np.full(len(labels), "only.edf")
    x = resample_epochs(epochs, 256.0)
    assert np.isnan(_grouped_cv_auc("xdawn_riemann", x, labels, groups, "two_fold_grouped"))


def test_cv_scheme_is_recorded_per_decoder():
    assert CV_SCHEMES == {"xdawn_riemann": "leave_one_file_out", "eegnet": "two_fold_grouped"}
    epochs, labels, groups, sfreq = _synthetic(n_per_class=40)
    decoder = fit_strong_decoder(epochs, labels, groups, classifier="xdawn_riemann", sampling_frequency=sfreq)
    assert decoder.cv_scheme == "leave_one_file_out"


def test_eegnet_forward_shape():
    model = EEGNet(n_channels=16, n_samples=128)
    out = model(torch.zeros(5, 16, 128))
    assert out.shape == (5,)


@pytest.mark.parametrize("classifier", ["xdawn_riemann", "eegnet"])
def test_decoder_separates_synthetic_p300(classifier):
    epochs, labels, groups, sfreq = _synthetic()
    decoder = fit_strong_decoder(epochs, labels, groups, classifier=classifier, sampling_frequency=sfreq)
    test_epochs, test_labels, _, _ = _synthetic(seed=1)
    scores = decoder.score_epochs(test_epochs)
    from sklearn.metrics import roc_auc_score
    assert scores.shape == (len(test_labels),)
    assert np.isfinite(scores).all()
    assert roc_auc_score(test_labels, scores) > 0.9


def test_eegnet_is_deterministic():
    epochs, labels, groups, sfreq = _synthetic(n_per_class=60)
    first = fit_strong_decoder(epochs, labels, groups, classifier="eegnet", sampling_frequency=sfreq)
    second = fit_strong_decoder(epochs, labels, groups, classifier="eegnet", sampling_frequency=sfreq)
    np.testing.assert_allclose(first.score_epochs(epochs[:10]), second.score_epochs(epochs[:10]))


def test_baseline_logistic_reproduces_selections():
    """The strong-decoder session loop, run with the ORIGINAL decoder, reproduces the
    baseline posteriors exactly, so the loop itself cannot be a source of difference."""
    import sys
    from pathlib import Path

    import pandas as pd

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "scripts"))
    import build_strong_decoder_selections as build

    baseline_path = build.OUTPUT / "selections.parquet"
    if not build.CACHE_ROOT.exists() or not baseline_path.exists():
        pytest.skip("raw bigP3BCI cache or baseline selections not available")
    import authorship.decoder as original

    baseline = pd.read_parquet(baseline_path)
    sizes = baseline.groupby(["study", "participant_id", "session_id"]).size().sort_values()
    session = sizes.index[0]
    selections, decoders, skipped = build.build_selections(
        original.fit_session_decoder, only_sessions=[session]
    )
    expected = baseline.set_index(["study", "participant_id", "session_id"]).loc[[session]].reset_index()
    expected = expected[baseline.columns]
    rebuilt = build.align_to_baseline(selections, expected[build.KEY_COLUMNS])
    assert len(decoders) == 1 and not skipped
    assert list(rebuilt.columns) == list(baseline.columns)
    assert len(rebuilt) == len(expected)
    other = [column for column in baseline.columns if column not in ("p_neural", "train_auc")]
    pd.testing.assert_frame_equal(rebuilt[other], expected[other])
    assert np.allclose(
        np.stack(rebuilt["p_neural"].to_numpy()), np.stack(expected["p_neural"].to_numpy()), atol=1e-10
    )
    assert np.allclose(rebuilt["train_auc"], expected["train_auc"], atol=1e-12)
