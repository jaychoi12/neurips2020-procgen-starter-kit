#!/bin/bash
set -e


# how to use
# ./submit.sh "<submission tag>"

echo "making submission: " $1

git tag -am "\"$1\"" $1
git push aicrowd master
git push aicrowd $1