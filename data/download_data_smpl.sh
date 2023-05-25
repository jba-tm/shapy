#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
#
#read -p "Username:" username
#read -p "Password:" -s password
#
#username=$(urle $username)
#password=$(urle $password)

wget --post-data "username=eanepmf@mail.ru&password=12345678" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O 'SMPL_python_v.1.1.0.zip' --no-check-certificate --continue
unzip SMPL_python_v.1.1.0.zip
