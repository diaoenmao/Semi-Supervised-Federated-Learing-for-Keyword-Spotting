python make.py --mode fs --run train --num_experiments 4 --round 16
python make.py --mode fs --run test --num_experiments 4 --round 16

python make.py --mode ps --run train --num_experiments 4 --round 16
python make.py --mode ps --run test --num_experiments 4 --round 16

python make.py --mode fl --run train --round 8 --split_round 1
python make.py --mode fl --run test --round 8 --split_round 1

python make.py --mode semi --run train --round 8 --split_round 1
python make.py --mode semi --run test --round 8 --split_round 1

python make.py --mode ssfl --run train --round 8 --split_round 1 --split_round 1
python make.py --mode ssfl --run test --round 8 --split_round 1 --split_round 1