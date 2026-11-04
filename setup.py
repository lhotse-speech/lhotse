# coding=utf-8
#                `-:/++oooooooooooooooooooooooooooooooooooooooooooooo++/:-.
#            `-+sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssso:`
#          .+ssosyyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhyyosso-
#         /yosyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhysos+`
#       `osoyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhsoy.
#       osohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhsoy`
#      /h+hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhoso
#      yoyhhhhhhhhhhyooyhhhhhysyhhhhyhhhyysyhhhhyhhhhhyyhhysosyhhhysyyyyhhhhhhhhhhh+h`
#     `h+hhhhhhhhhhhs  +hhhhh.`:hh+.-yy+.```./y/.----..:o:`   -+y+```..-hhhhhhhhhhh+h:
#     -h+hhhhhhhhhhh+  shhhhy  /yh/  yo..--.` .+//.  /os/  /oosso  -::::hhhhhhhhhhh+h/
#     -h+hhhhhhhhhhh: `hhhhho  .-:. `y++yhhho  -hh/  yhho` .:/shy  /oosshhhhhhhhhhh+h/
#     -h+hhhhhhhhhhh. .hhhhh/  .--` `+ `shhhy` .hh- `hhhho:.. `/o  ```./hhhhhhhhhhh+h+
#     -h+hhhhhhhhhhy` `+sssy:  yhh- `y` ./+/. `ohh. .hhhosyhy- .+  ossssyhhhhhhhhhh+h+
#     -h+hhhhhhhhhh+   ````o:``hhh:``hs-`` ``-ohhh. .hhhs---.`.o+   ``  ohhhhhhhhhh+h+
#     -h+hhhhhhhhhhhooosssshysyhhhysyhhhysosyhhhhhsoshhhhyo++oyhyo++++ooyhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhyhhyhhhhhhhhyyhhhhyhhhyyyhhhhhhhhhhhyhhhhhyhyhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhsyhyyyyyyyyyyhhsyyyyyyyysyyyhhyyyyhhhyyyyyshhsyyyhhhhhhhhhhh+h+
#     -h+hhhhhhhhhyysyhoyyosyosyosyosyosyosyoyyosyoyysysohhoyyosssyyoyyoyshhhhhhhhh+h+
#     -h+hhhhhhyyysyosyoososy+ssosyyyyoss+ss+ssossoss/ssoyyoss+so+ss+sooyoyyyhhhhhh+h+
#     -h+hhhhhhhhhhhhhhsyhsyyyyyshhh:`-ohyyysyyyyyyhhsyyshhhyysyyyhhsyyshhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhyhhyhhyhhhy/`    /hhhhoshhhyhhhhhhhhhhhyhhyhhyhyhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhhhhhhho` /y/ `:-oo-  +s+yhhhhhy++yhhhhhhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhhhhhy/ `/hhh` .y/.     :::+o/`   `:ohhhhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhys+-` `ohhhy`syos/:.   sho:-.` `--  .yhhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhh-   `-shhhhssh+.`s:y- ./yssy--oy/` .s:yhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhs:  `-yyhhhhhhhy-` ohhyo:syyhosyh/  :yh+.ohhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhyo/`   oyhhhhhyso/`   yhh/oyo/`+/.so  :yhy-  .+yhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhs/`-    .yhhhhhh.`    `:hhhs`sho..-.-``-ss+hys:` .yyyhhhhhhhhhhh+h+
#     -h+hhhhhhhhy/.-:.  -oyhhhhhhh`     /yhhhy .hhy:. `+o+./yhhhhyo//-`/yhhhhhhhhh+h+
#     -h+hhhhhhho:/s+-`-oyhhhhhhhhh/     `-hhhy` yhh.  /-` -yhhhhhhhhy+  `+shhhhhhh+h+
#     -h+hhhhhs++ysososhhyosyhhhhhhy.    :shhhy  shh/  .  `yhhhhhhhhhhy:    -shhhhh+h/
#     .h+hhhysoyhsyhhhhhho.`./yhhhhhy- -syhhhhh-`hhhy.    -hhhhhhhhhhhhso :o:-/yhhh+h/
#     `h+hhhyyhhhhhhhhhhhhy+` :yhhhhhs-hhhhhhhh+-hhhh+  `+yhhhhhhhhhhhhhy/yhhysoyhh+h-
#      soshhhhhhhhhhhhhhhhhhs``-ohhhhhshhhhhhhhyohhhhh+.`-+shhhhhhhhhhhhhhhhhhhhhhh+h`
#      :h+hhhhhhhhhhhhhhhhhhhsss/ohhhhhhhhhhhhhhyhhhhhho.ohhhhhhhhhhhhhhhhhhhhhhhhoy+
#       oyohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhs+yhhhhhhhhhhhhhhhhhhhhhhhhsss`
#       `+soyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhyhhhhhhhhhhhhhhhhhhhhhhhhyoss`
#         :sooyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhysos/`
#          `/ssssyyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhyyssss+.
#             ./osssssssssssssssssssssssssssssssssssssssssssssssssssssssssss/-`
#                 .-:://++++++++++++++++++++++++++++++++++++++++++++///:-.`

"""Setuptools bridge for Lhotse's dynamic version; metadata is in pyproject.toml."""

from pathlib import Path
from runpy import run_path

from setuptools import setup

project_root = Path(__file__).parent
# Isolated PEP 517 builds do not put the source directory on sys.path.
version_helpers = run_path(str(project_root / "_lhotse_version.py"))
version = version_helpers["discover_lhotse_version"](project_root)
version_helpers["write_lhotse_version"](project_root, version=version)
setup(version=version)
