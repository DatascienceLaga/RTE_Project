#! /bin/bash

# RUN THIS ONLY IN uvsw_part/test/ DIRECTORY

VENV=.venv
python3 -m venv $VENV
source $VENV/bin/activate

cd ..
python3 setup.py install

cd test
pip3 install -r requirements.txt

echo ""
echo -e "\e[1mto start the environment, type :\e[0m"
echo "source ${VENV}/bin/activate"
echo ""
echo -e "\e[1mto stop the environment, type :\e[0m"
echo "deactivate"
