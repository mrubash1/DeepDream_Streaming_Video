#!/bin/bash

#Set up jupyterhub in tf-dd-stream environment
cd $LEXICONRN_HOME
source activate tf-dd-stream

#install necessary jupyterhub components
sudo apt-get -qq install npm nodejs-legacy
sudo npm install -g configurable-http-proxy
pip install jupyterhub
sudo pip install --upgrade notebook

#Create a new self signed ssl certificate
sudo openssl req -new -x509 -sha256 -newkey rsa:2048 -nodes \
    -keyout /etc/jupyterhub/ssl.key -days 365 -out /etc/jupyterhub/ssl.crt \
    -subj "/C=US/ST=CA/L=San Francisco/O=Self-Signed-By-Github-User/OU=R&D/CN=NA"

#create systemctl service to automatically start jupyterhub
sudo cp build/jupyterhub.service /etc/systemd/system/jupyterhub.service
sudo mkdir /etc/jupyterhub
sudo cp build/jupyterhub_config.py /etc/jupyterhub/jupyterhub_config.py
sudo systemctl daemon-reload
sudo systemctl start jupyterhub