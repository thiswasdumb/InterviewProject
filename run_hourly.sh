#!/bin/bash
while true; do
    echo "Running the Python script at $(date)"
    python3 your_script_name.py  # Replace with your Python script's filename
    echo "Completed at $(date). Sleeping for 1 hour..."
    sleep 3600  # Wait for 1 hour (3600 seconds)
done

