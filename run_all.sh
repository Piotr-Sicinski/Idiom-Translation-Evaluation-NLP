#!/bin/bash

# Try to detect the correct Python command
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
elif command -v py &>/dev/null; then
    PYTHON=py
else
    echo "Python interpreter not found!"
    exit 1
fi

echo "Using Python: $PYTHON"

echo "Running 1_make_sentences.py..."
$PYTHON 1_make_sentences.py

echo "Running 2_translate.py..."
$PYTHON 2_translate.py

echo "Running 3_check_translations.py..."
$PYTHON 3_check_translations.py

echo "Running 4_evaluate.py..."
$PYTHON 4_evaluate.py

echo "All scripts completed."
