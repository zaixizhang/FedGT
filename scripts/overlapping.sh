python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedgt'\
                --dataset 'Cora' \
                --mode 'overlapping' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients 30\
                --seed 42
