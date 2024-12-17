from .adept import download_adept, prepare_adept
from .aishell import download_aishell, prepare_aishell
from .aishell3 import download_aishell3, prepare_aishell3
from .aishell4 import download_aishell4, prepare_aishell4
from .ali_meeting import download_ali_meeting, prepare_ali_meeting
from .ami import download_ami, prepare_ami
from .aspire import prepare_aspire
from .atcosim import download_atcosim, prepare_atcosim
from .babel import prepare_single_babel_language
from .baker_zh import download_baker_zh, prepare_baker_zh
from .bengaliai_speech import prepare_bengaliai_speech
from .broadcast_news import prepare_broadcast_news
from .but_reverb_db import download_but_reverb_db, prepare_but_reverb_db
from .bvcc import download_bvcc, prepare_bvcc
from .callhome_egyptian import prepare_callhome_egyptian
from .callhome_english import prepare_callhome_english
from .cdsd import prepare_cdsd
from .chime6 import download_chime6, prepare_chime6
from .cmu_arctic import download_cmu_arctic, prepare_cmu_arctic
from .cmu_indic import download_cmu_indic, prepare_cmu_indic
from .cmu_kids import prepare_cmu_kids
from .commonvoice import prepare_commonvoice
from .csj import prepare_csj
from .cslu_kids import prepare_cslu_kids
from .daily_talk import download_daily_talk, prepare_daily_talk
from .dihard3 import prepare_dihard3
from .dipco import download_dipco, prepare_dipco
from .earnings21 import download_earnings21, prepare_earnings21
from .earnings22 import download_earnings22, prepare_earnings22
from .ears import download_ears, prepare_ears
from .edacc import download_edacc, prepare_edacc
from .eval2000 import prepare_eval2000
from .fisher_english import prepare_fisher_english
from .fisher_spanish import prepare_fisher_spanish
from .fleurs import download_fleurs, prepare_fleurs
from .gale_arabic import prepare_gale_arabic
from .gale_mandarin import prepare_gale_mandarin
from .gigaspeech import prepare_gigaspeech
from .gigast import download_gigast, prepare_gigast
from .grid import download_grid, prepare_grid
from .heroico import download_heroico, prepare_heroico
from .hifitts import download_hifitts, prepare_hifitts
from .himia import download_himia, prepare_himia
from .icmcasr import prepare_icmcasr
from .icsi import download_icsi, prepare_icsi
from .iwslt22_ta import prepare_iwslt22_ta
from .kespeech import prepare_kespeech
from .ksponspeech import prepare_ksponspeech
from .l2_arctic import prepare_l2_arctic
from .libricss import download_libricss, prepare_libricss
from .librilight import prepare_librilight
from .librimix import download_librimix, prepare_librimix
from .librispeech import download_librispeech, prepare_librispeech
from .libritts import (
    download_libritts,
    download_librittsr,
    prepare_libritts,
    prepare_librittsr,
)
from .ljspeech import download_ljspeech, prepare_ljspeech
from .magicdata import download_magicdata, prepare_magicdata
from .mdcc import download_mdcc, prepare_mdcc
from .medical import download_medical, prepare_medical
from .mgb2 import prepare_mgb2
from .mls import prepare_mls
from .mobvoihotwords import download_mobvoihotwords, prepare_mobvoihotwords
from .mtedx import download_mtedx, prepare_mtedx
from .musan import download_musan, prepare_musan
from .nsc import prepare_nsc
from .peoples_speech import prepare_peoples_speech
from .radio import prepare_radio
from .reazonspeech import download_reazonspeech, prepare_reazonspeech
from .rir_noise import download_rir_noise, prepare_rir_noise
from .sbcsae import download_sbcsae, prepare_sbcsae
from .slu import prepare_slu
from .spatial_librispeech import (
    download_spatial_librispeech,
    prepare_spatial_librispeech,
)
from .speechcommands import download_speechcommands, prepare_speechcommands
from .speechio import prepare_speechio
from .spgispeech import download_spgispeech, prepare_spgispeech
from .stcmds import download_stcmds, prepare_stcmds
from .switchboard import prepare_switchboard
from .tedlium import download_tedlium, prepare_tedlium
from .tedlium2 import download_tedlium2, prepare_tedlium2
from .thchs_30 import download_thchs_30, prepare_thchs_30
from .this_american_life import download_this_american_life, prepare_this_american_life
from .timit import download_timit, prepare_timit
from .uwb_atcc import download_uwb_atcc, prepare_uwb_atcc
from .vctk import download_vctk, prepare_vctk
from .voxceleb import download_voxceleb1, download_voxceleb2, prepare_voxceleb
from .voxconverse import download_voxconverse, prepare_voxconverse
from .voxpopuli import download_voxpopuli, prepare_voxpopuli
from .wenet_speech import prepare_wenet_speech
from .wenetspeech4tts import prepare_wenetspeech4tts
from .xbmu_amdo31 import download_xbmu_amdo31, prepare_xbmu_amdo31
from .yesno import download_yesno, prepare_yesno

__all__ = [
    "download_adept",
    "prepare_adept",
    "download_aishell",
    "prepare_aishell",
    "download_aishell3",
    "prepare_aishell3",
    "download_aishell4",
    "prepare_aishell4",
    "download_ali_meeting",
    "prepare_ali_meeting",
    "download_ami",
    "prepare_ami",
    "prepare_aspire",
    "download_atcosim",
    "prepare_atcosim",
    "prepare_single_babel_language",
    "prepare_bengaliai_speech",
    "prepare_broadcast_news",
    "download_but_reverb_db",
    "prepare_but_reverb_db",
    "download_bvcc",
    "prepare_bvcc",
    "prepare_callhome_egyptian",
    "prepare_callhome_english",
    "download_chime6",
    "prepare_chime6",
    "download_cmu_arctic",
    "prepare_cmu_arctic",
    "download_cmu_indic",
    "prepare_cmu_indic",
    "prepare_cmu_kids",
    "prepare_commonvoice",
    "prepare_csj",
    "prepare_cslu_kids",
    "download_daily_talk",
    "prepare_daily_talk",
    "prepare_dihard3",
    "download_dipco",
    "prepare_dipco",
    "download_earnings21",
    "prepare_earnings21",
    "download_earnings22",
    "prepare_earnings22",
    "download_ears",
    "prepare_ears",
    "download_edacc",
    "prepare_edacc",
    "prepare_eval2000",
    "prepare_fisher_english",
    "prepare_fisher_spanish",
    "download_fleurs",
    "prepare_fleurs",
    "prepare_gale_arabic",
    "prepare_gale_mandarin",
    "prepare_gigaspeech",
    "download_gigast",
    "prepare_gigast",
    "download_grid",
    "prepare_grid",
    "download_heroico",
    "prepare_heroico",
    "download_hifitts",
    "prepare_hifitts",
    "download_himia",
    "prepare_himia",
    "prepare_icmcasr",
    "download_icsi",
    "prepare_icsi",
    "prepare_iwslt22_ta",
    "prepare_kespeech",
    "prepare_ksponspeech",
    "prepare_l2_arctic",
    "download_libricss",
    "prepare_libricss",
    "prepare_librilight",
    "download_librimix",
    "prepare_librimix",
    "download_librispeech",
    "prepare_librispeech",
    "download_libritts",
    "download_librittsr",
    "prepare_libritts",
    "prepare_librittsr",
    "download_ljspeech",
    "prepare_ljspeech",
    "download_magicdata",
    "prepare_magicdata",
    "download_medical",
    "prepare_medical",
    "prepare_mgb2",
    "prepare_mls",
    "download_mobvoihotwords",
    "prepare_mobvoihotwords",
    "download_mtedx",
    "prepare_mtedx",
    "download_musan",
    "prepare_musan",
    "prepare_nsc",
    "prepare_peoples_speech",
    "download_reazonspeech",
    "prepare_reazonspeech",
    "prepare_radio",
    "download_rir_noise",
    "prepare_rir_noise",
    "prepare_slu",
    "download_speechcommands",
    "prepare_speechcommands",
    "download_spgispeech",
    "prepare_spgispeech",
    "download_stcmds",
    "prepare_stcmds",
    "prepare_switchboard",
    "download_tedlium",
    "prepare_tedlium",
    "download_thchs_30",
    "prepare_thchs_30",
    "download_this_american_life",
    "prepare_this_american_life",
    "download_timit",
    "prepare_timit",
    "download_uwb_atcc",
    "prepare_uwb_atcc",
    "download_vctk",
    "prepare_vctk",
    "download_voxceleb1",
    "download_voxceleb2",
    "prepare_voxceleb",
    "download_voxconverse",
    "prepare_voxconverse",
    "download_voxpopuli",
    "prepare_voxpopuli",
    "prepare_wenet_speech",
    "download_xbmu_amdo31",
    "prepare_xbmu_amdo31",
    "download_yesno",
    "prepare_yesno",
]
