#!/usr/bin/env bash
set -e
mkdir -p data/sample_text
cat > data/sample_text/demo.txt <<'TXT'
Gas turbines compress air, inject fuel, and ignite the mixture to spin a turbine.
This tiny file exists only to bootstrap the tokenizer and dataset demo in this portfolio project.
Add more .txt files here to expand your toy dataset.
TXT
echo "Wrote data/sample_text/demo.txt"
