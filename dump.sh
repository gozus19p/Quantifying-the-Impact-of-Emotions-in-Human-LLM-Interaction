#!/bin/bash

mongodump --db thesis --collection Evaluation --out results
mongodump --db thesis --collection Case --out results