#!/bin/bash

# Move existing source.txt to a backup file with date
if [ -f "source.txt" ]; then
    mv source.txt "source.txt.bak.$(date +%Y-%m-%d_%H-%M-%S)"
fi

# Create a new empty source.txt
touch source.txt

# Loop through all .cc and .h files in the current directory
for file in *.cc *.h; do
    # Check if the file exists to avoid wildcard issues
    if [ -f "$file" ]; then
        echo "FILE $file:" >> source.txt
        cat "$file" >> source.txt
        echo "" >> source.txt  # Add a newline for readability
    fi
done

echo "Process completed. source.txt has been updated."
