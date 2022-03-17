python make.py --mode fs --run train --num_experiments 4 --round 16
python make.py --mode fs --run test --num_experiments 4 --round 16

python make.py --mode ps --run train --num_experiments 4 --round 16
python make.py --mode ps --run test --num_experiments 4 --round 16

python make.py --mode ps-aug --run train --num_experiments 4 --round 16
python make.py --mode ps-aug --run test --num_experiments 4 --round 16

python make.py --mode semi --run train --round 16
python make.py --mode semi --run test --round 16

python make.py --mode semi-aug --model wresnet28x2 --run train --round 16
python make.py --mode semi-aug --model wresnet28x2 --run test --round 16

python make.py --mode semi-aug --model tcresnet18 --run train --round 16
python make.py --mode semi-aug --model tcresnet18 --run test --round 16

python make.py --mode semi-loss --model wresnet28x2 --run train --round 16
python make.py --mode semi-loss --model wresnet28x2 --run test --round 16

python make.py --mode semi-loss --model tcresnet18 --run train --round 16
python make.py --mode semi-loss --model tcresnet18 --run test --round 16

python make.py --mode fl-cd --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode fl-cd --run test --num_experiments 4  --round 16 --split_round 1

python make.py --mode fl-ub --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode fl-ub --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode ssfl --run train --round 16
python make.py --mode ssfl  --run test --round 16

python make.py --mode ssfl-loss --run train --round 16
python make.py --mode ssfl-loss --run test --round 16