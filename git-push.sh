#!/bin/bash

dating=`date`

git status
git add .
git commit -m "$dating $1"
git push origin master
git log 
