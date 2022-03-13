python make.py --mode fs --model dscnn --round 10 --run train
python make.py --mode fs --model dscnn --round 10 --run test
python make.py --mode fs --model tcresnet18 --round 10 --run train
python make.py --mode fs --model tcresnet18 --round 10 --run test
python make.py --mode fs --model mhattrnn --round 10 --run train
python make.py --mode fs --model mhattrnn --round 10 --run test
python make.py --mode fs --model wresnet28x2 --round 10 --run train
python make.py --mode fs --model wresnet28x2 --round 10 --run test

python make.py --mode ps --model dscnn --round 10 --run train
python make.py --mode ps --model dscnn --round 10 --run test
python make.py --mode ps --model tcresnet18 --round 10 --run train
python make.py --mode ps --model tcresnet18 --round 10 --run test
python make.py --mode ps --model mhattrnn --round 10 --run train
python make.py --mode ps --model mhattrnn --round 10 --run test
python make.py --mode ps --model wresnet28x2 --round 10 --run train
python make.py --mode ps --model wresnet28x2 --round 10 --run test

python make.py --mode semi --model wresnet28x2 --round 8 --run train
python make.py --mode semi --model wresnet28x2 --round 8 --run test

python make.py --mode fl-cd --model wresnet28x2 --round 10 --run train
python make.py --mode fl-cd --model wresnet28x2 --round 10 --run test
python make.py --mode fl-ub --model wresnet28x2 --round 10 --run train
python make.py --mode fl-ub --model wresnet28x2 --round 10 --run test

python make.py --mode ssfl-cd --model wresnet28x2 --round 8 --run train
python make.py --mode ssfl-cd --model wresnet28x2 --round 8 --run test
python make.py --mode ssfl-ub --model wresnet28x2 --round 8 --run train
python make.py --mode ssfl-ub --model wresnet28x2 --round 8 --run test