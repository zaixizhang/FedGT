for dataset in 'PubMed' 'Computers' 'Photo'
do
    python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dataset $dataset \
                --mode 'overlapping' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients 30\
                --seed 17

    python3 main.py --gpu $1\
                    --n-workers $2\
                    --model 'fedpub'\
                    --dataset $dataset \
                    --mode 'overlapping' \
                    --frac 1.0 \
                    --n-rnds 100\
                    --n-eps 1\
                    --n-clients 30\
                    --clsf-mask-one\
                    --laye-mask-one\
                    --norm-scale 5\
                    --seed 16
done
