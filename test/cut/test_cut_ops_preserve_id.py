import pytest

# Note:
# Definitions for `cut1`, `cut2` and `cut_set` parameters are standard Pytest fixtures located in test/cut/conftest.py


# ########################################
# ############### PADDING ################
# ########################################


@pytest.mark.parametrize("direction", ["right", "left", "both"])
def test_pad_cut_preserve_id_false(cut1, direction: str):
    padded = cut1.pad(duration=300, direction=direction)
    assert padded.id != cut1.id


@pytest.mark.parametrize("direction", ["right", "left", "both"])
def test_pad_cut_preserve_id_true(cut1, direction: str):
    padded = cut1.pad(duration=300, direction=direction, preserve_id=True)
    assert padded.id == cut1.id


@pytest.mark.parametrize("direction", ["right", "left", "both"])
def test_pad_mixed_cut_preserve_id_false(cut1, direction: str):
    mixed = cut1.append(cut1)
    padded = mixed.pad(duration=300, direction=direction)
    assert padded.id != mixed.id


@pytest.mark.parametrize("direction", ["right", "left", "both"])
def test_pad_mixed_cut_preserve_id_true(cut1, direction: str):
    mixed = cut1.append(cut1)
    padded = mixed.pad(duration=300, direction=direction, preserve_id=True)
    assert padded.id == mixed.id


# ########################################
# ############## APPENDING ###############
# ########################################


def test_append_cut_preserve_id_none(cut1, cut2):
    appended = cut1.append(cut2)
    assert appended.id != cut1.id
    assert appended.id != cut2.id


def test_append_cut_preserve_id_left(cut1, cut2):
    appended = cut1.append(cut2, preserve_id="left")
    assert appended.id == cut1.id
    assert appended.id != cut2.id


def test_append_cut_preserve_id_right(cut1, cut2):
    appended = cut1.append(cut2, preserve_id="right")
    assert appended.id != cut1.id
    assert appended.id == cut2.id


def test_append_mixed_cut_preserve_id_none(cut1, cut2):
    premixed = cut1.append(cut1)
    appended = premixed.append(cut2)
    assert appended.id != premixed.id
    assert appended.id != cut2.id


def test_append_mixed_cut_preserve_id_left(cut1, cut2):
    premixed = cut1.append(cut1)
    appended = premixed.append(cut2, preserve_id="left")
    assert appended.id == premixed.id
    assert appended.id != cut2.id


def test_append_mixed_cut_preserve_id_right(cut1, cut2):
    premixed = cut1.append(cut1)
    appended = premixed.append(cut2, preserve_id="right")
    assert appended.id != premixed.id
    assert appended.id == cut2.id


# ########################################
# ############### MIXING #################
# ########################################


def test_mix_cut_preserve_id_none(cut1, cut2):
    mixed = cut1.mix(cut2)
    assert mixed.id != cut1.id
    assert mixed.id != cut2.id


def test_mix_cut_preserve_id_left(cut1, cut2):
    mixed = cut1.mix(cut2, preserve_id="left")
    assert mixed.id == cut1.id
    assert mixed.id != cut2.id


def test_mix_cut_preserve_id_right(cut1, cut2):
    mixed = cut1.mix(cut2, preserve_id="right")
    assert mixed.id != cut1.id
    assert mixed.id == cut2.id


def test_mix_mixed_cut_preserve_id_none(cut1, cut2):
    premixed = cut1.append(cut1)
    mixed = premixed.mix(cut2)
    assert mixed.id != premixed.id
    assert mixed.id != cut2.id


def test_mix_mixed_cut_preserve_id_left(cut1, cut2):
    premixed = cut1.append(cut1)
    mixed = premixed.mix(cut2, preserve_id="left")
    assert mixed.id == premixed.id
    assert mixed.id != cut2.id


def test_mix_mixed_cut_preserve_id_right(cut1, cut2):
    premixed = cut1.append(cut1)
    mixed = premixed.mix(cut2, preserve_id="right")
    assert mixed.id != premixed.id
    assert mixed.id == cut2.id


# ########################################
# ############ PERTURB SPEED #############
# ########################################


def test_cut_perturb_speed_affix_id_true(cut1):
    cut_sp = cut1.perturb_speed(1.1)
    assert cut_sp.id != cut1.id


def test_cut_perturb_speed_affix_id_false(cut1):
    cut_sp = cut1.perturb_speed(1.1, affix_id=False)
    assert cut_sp.id == cut1.id


def test_mixed_cut_perturb_speed_affix_id_true(cut1):
    premixed = cut1.append(cut1)
    cut_sp = premixed.perturb_speed(1.1)
    assert cut_sp.id != premixed.id


def test_mixed_cut_perturb_speed_affix_id_false(cut1):
    premixed = cut1.append(cut1)
    cut_sp = premixed.perturb_speed(1.1, affix_id=False)
    assert cut_sp.id == premixed.id


# ########################################
# ############ PERTURB TEMPO #############
# ########################################


@pytest.mark.xfail(
    reason="Torchaudio 2.2 dropped support for SoX, this effect may not be available."
)
def test_cut_perturb_tempo_affix_id_true(cut1):
    cut_tp = cut1.perturb_tempo(1.1)
    assert cut_tp.id != cut1.id


@pytest.mark.xfail(
    reason="Torchaudio 2.2 dropped support for SoX, this effect may not be available."
)
def test_cut_perturb_tempo_affix_id_false(cut1):
    cut_tp = cut1.perturb_tempo(1.1, affix_id=False)
    assert cut_tp.id == cut1.id


@pytest.mark.xfail(
    reason="Torchaudio 2.2 dropped support for SoX, this effect may not be available."
)
def test_mixed_cut_perturb_tempo_affix_id_true(cut1):
    premixed = cut1.append(cut1)
    cut_tp = premixed.perturb_tempo(1.1)
    assert cut_tp.id != premixed.id


@pytest.mark.xfail(
    reason="Torchaudio 2.2 dropped support for SoX, this effect may not be available."
)
def test_mixed_cut_perturb_tempo_affix_id_false(cut1):
    premixed = cut1.append(cut1)
    cut_tp = premixed.perturb_tempo(1.1, affix_id=False)
    assert cut_tp.id == premixed.id


# ########################################
# ########### PERTURB VOLUME #############
# ########################################


def test_cut_perturb_volume_affix_id_true(cut1):
    cut_vp = cut1.perturb_volume(1.1)
    assert cut_vp.id != cut1.id


def test_cut_perturb_volume_affix_id_false(cut1):
    cut_vp = cut1.perturb_volume(1.1, affix_id=False)
    assert cut_vp.id == cut1.id


def test_mixed_cut_perturb_volume_affix_id_true(cut1):
    premixed = cut1.append(cut1)
    cut_vp = premixed.perturb_volume(1.1)
    assert cut_vp.id != premixed.id


def test_mixed_cut_perturb_volume_affix_id_false(cut1):
    premixed = cut1.append(cut1)
    cut_vp = premixed.perturb_volume(1.1, affix_id=False)
    assert cut_vp.id == premixed.id


# ########################################
# ############## RESAMPLE ################
# ########################################


def test_cut_resample_affix_id_true(cut1):
    cut_rs = cut1.resample(44100, affix_id=True)
    assert cut_rs.id != cut1.id


def test_cut_resample_affix_id_false(cut1):
    cut_rs = cut1.resample(44100)
    assert cut_rs.id == cut1.id


def test_mixed_cut_resample_affix_id_true(cut1):
    premixed = cut1.append(cut1)
    cut_rs = premixed.resample(44100, affix_id=True)
    assert cut_rs.id != premixed.id


def test_mixed_cut_resample_affix_id_false(cut1):
    premixed = cut1.append(cut1)
    cut_rs = premixed.resample(44100)
    assert cut_rs.id == premixed.id
