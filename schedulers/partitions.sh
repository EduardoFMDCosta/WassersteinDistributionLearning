echo "START"
python -m experiments.partition --distribution Gaussian --num_dims 10 --setting 0
python -m experiments.partition --distribution Gaussian --num_dims 100 --setting 0
echo "FINNISHED"