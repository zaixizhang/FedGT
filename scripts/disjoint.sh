python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedgt'\
                --dataset 'Cora' \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 200\
                --n-eps 1\
                --n-clients 10\
                --seed 42 
