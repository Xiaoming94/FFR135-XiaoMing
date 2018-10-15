#!/bin/bash

echo "Downloading the MNIST Data set"
echo "Make sure you have installed wget and gzip \n"

wget -A gz -m -p -E -k -K --no-parent --no-directories http://yann.lecun.com/exdb/mnist/
gunzip ./*.gz
