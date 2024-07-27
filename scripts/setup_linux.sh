#!/bin/bash
# Make sure that conda is available

#hash conda 2>/dev/null || {
#    echo "Please install anaconda before continuing. You can download it at https://www.continuum.io/downloads. Please use the Python 2.7 installer."
#    exit 0
#}

echo "Installing system dependencies"
echo "You will probably be asked for your sudo password."
cp /etc/apt/sources.list /etc/apt/sources.list~
sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
apt-get update
apt-get install -y python-pip python-dev swig cmake build-essential
apt-get build-dep -y python-pygame
apt-get build-dep -y python-scipy

# Make sure that we're under the directory of the project
cd "$(dirname "$0")/.."

echo "Installing pip dependencies..."
python3 -m pip install requirements.txt
#conda env create -f environment.yml
#conda env update
#
#echo "Conda environment created! Make sure to run \`source activate rllab3\` whenever you open a new terminal and want to run programs under rllab."
