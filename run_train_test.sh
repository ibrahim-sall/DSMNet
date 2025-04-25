#!/bin/bash

# To run the script, execute the following command: 
# chmod +x run_train_test.sh && ./run_train_test.sh


echo "Starting MTL training..."
sed -i 's/correction = .*/correction = False/' config.py
python train_mtl.py
if [ $? -ne 0 ]; then
    echo "MTL training failed"
    exit 1
fi

echo "Waiting 30 seconds for checkpoints and GPU cooldown..."
sleep 30

echo -e "\nStarting MTL testing..."
sed -i 's/correction = .*/correction = False/' config.py
python test_dsm.py
if [ $? -ne 0 ]; then
    echo "Testing failed"
    exit 1
fi

echo "Waiting 30 seconds for checkpoints and GPU cooldown..."
sleep 30

echo -e "\nStarting DAE training..."
sed -i 's/correction = .*/correction = True/' config.py
python train_dae.py
if [ $? -ne 0 ]; then
    echo "DAE training failed"
    exit 1
fi

echo "Waiting 30 seconds for checkpoints and GPU cooldown..."
sleep 30

echo -e "\nStarting DAE testing..."
sed -i 's/correction = .*/correction = True/' config.py
python test_dsm.py
if [ $? -ne 0 ]; then
    echo "Testing failed"
    exit 1
fi

echo "All tasks completed successfully"