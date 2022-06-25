python make.py --mode fs --run train --num_experiments 4 --round 16
python make.py --mode fs --run test --num_experiments 4 --round 16

python make.py --mode ps --run train --num_experiments 4 --round 16
python make.py --mode ps --run test --num_experiments 4 --round 16

python make.py --mode fl --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode fl --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode fl-alter --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode fl-alter --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode semi --run train  --num_experiments 4 --round 16 --split_round 1
python make.py --mode semi --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode semi-aug --run train  --num_experiments 4 --round 16 --split_round 1
python make.py --mode semi-aug --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode ssfl --run train --num_experiments 4 --round 8 --split_round 1 --split_round 1
python make.py --mode ssfl --run test --num_experiments 4 --round 8 --split_round 1 --split_round 1
