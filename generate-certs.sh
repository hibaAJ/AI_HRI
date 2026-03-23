#!/bin/bash
# Generates a self-signed cert + key for local HTTPS (valid 1 year)
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem \
  -out cert.pem \
  -days 365 \
  -subj "/CN=HRI Local" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
echo "cert.pem and key.pem generated."
