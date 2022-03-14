python make.py --mode fs --round 8 --run train
python make.py --mode fs --round 8 --run test

python make.py --mode ps --round 8 --run train
python make.py --mode ps --round 8 --run test

python make.py --mode semi --round 8 --run train
python make.py --mode semi --round 8 --run test

python make.py --mode semi-aug --model wresnet28x2 --round 8 --run train
python make.py --mode semi-aug --model wresnet28x2 --round 8 --run test

python make.py --mode semi-loss --round 8 --run train
python make.py --mode semi-loss --round 8 --run test

python make.py --mode fl-cd --round 8 --run train
python make.py --mode fl-cd --round 8 --run test

python make.py --mode fl-ub --round 8 --run train
python make.py --mode fl-ub --round 8 --run test

python make.py --mode ssfl-cd --round 8 --run train --split_round 2
python make.py --mode ssfl-cd --round 8 --run test --split_round 2

python make.py --mode ssfl-ub --round 8 --run train --split_round 2
python make.py --mode ssfl-ub --round 8 --run test --split_round 2

python make.py --mode ssfl-loss --round 8 --run train