#!/bin/bash


echo "Build 2d fwi solver..."
cd geopvi/forward/fwi2d
RESULT="$(python setup.py build_ext -i 2>&1)"
status=$?
if [ $status -eq 0 ]; then
    echo "Building 2d fwi solver succeeds"
else
    echo "Error: $RESULT"
fi
cd ../../../

echo "Build 2d FMM solver..."
cd geopvi/forward/tomo2d
RESULT="$(python setup.py build_ext -i 2>&1)"
status=$?
if [ $status -eq 0 ]; then
    echo "Building 2d FMM solver succeeds"
else
    echo "Error: $RESULT"
fi
cd ../../../

if [ "$1" = "install" ]; then
    echo "Install GeoPVI..."
    pip install -e .
fi
