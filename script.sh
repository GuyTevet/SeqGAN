
# ZERO order experiments
#######################

python sequence_gan.py base --num_epochs 400  --gpu_inst 4
python sequence_gan.py 0.no_run --num_epochs 0 --dis_pretrain_epoch_num 0 --gen_pretrain_epoch_num 0  --gpu_inst 4
python sequence_gan.py 0.no_aversarial --num_epochs 0 --gpu_inst 4
python sequence_gan.py 0.no_pretrain --dis_pretrain_epoch_num 0 --gen_pretrain_epoch_num 0  --gpu_inst 4


# 1st order experiments
#######################

### 1.1 epochs

# epochs
python sequence_gan.py 1.1.num_epochs_400 --num_epochs 400  --gpu_inst 4
python sequence_gan.py 1.1.num_epochs_100 --num_epochs 100  --gpu_inst 4
python sequence_gan.py 1.1.num_epochs_50 --num_epochs 50  --gpu_inst 4
python sequence_gan.py 1.1.num_epochs_25 --num_epochs 25  --gpu_inst 4

# gen_pretrain_epoch_num
python sequence_gan.py 1.1.gen_pretrain_epoch_num_240 --gen_pretrain_epoch_num 240  --gpu_inst 4
python sequence_gan.py 1.1.gen_pretrain_epoch_num_60 --gen_pretrain_epoch_num 60  --gpu_inst 4

# dis_pretrain_epoch_num
python sequence_gan.py 1.1.dis_pretrain_epoch_num_100 --dis_pretrain_epoch_num 100  --gpu_inst 5
python sequence_gan.py 1.1.dis_pretrain_epoch_num_25 --dis_pretrain_epoch_num 25  --gpu_inst 5

### 1.2 dims
python sequence_gan.py 1.2.gen_emb_dim_4 --gen_emb_dim 4  --gpu_inst 5
python sequence_gan.py 1.2.gen_emb_dim_8 --gen_emb_dim 8  --gpu_inst 5
python sequence_gan.py 1.2.gen_emb_dim_16 --gen_emb_dim 16  --gpu_inst 5
python sequence_gan.py 1.2.gen_emb_dim_64 --gen_emb_dim 64  --gpu_inst 5
python sequence_gan.py 1.2.gen_emb_dim_128 --gen_emb_dim 128  --gpu_inst 5


python sequence_gan.py 1.2.dis_emb_dim_4 --dis_emb_dim 4  --gpu_inst 5
python sequence_gan.py 1.2.dis_emb_dim_8 --dis_emb_dim 8  --gpu_inst 5
python sequence_gan.py 1.2.dis_emb_dim_16 --dis_emb_dim 16  --gpu_inst 5
python sequence_gan.py 1.2.dis_emb_dim_32 --dis_emb_dim 32  --gpu_inst 5
python sequence_gan.py 1.2.dis_emb_dim_128 --dis_emb_dim 128  --gpu_inst 5


python sequence_gan.py 1.2.gen_hidden_dim_64 --gen_hidden_dim 64  --gpu_inst 4
python sequence_gan.py 1.2.gen_hidden_dim_128 --gen_hidden_dim 128  --gpu_inst 4
python sequence_gan.py 1.2.gen_hidden_dim_256 --gen_hidden_dim 256  --gpu_inst 4
python sequence_gan.py 1.2.gen_hidden_dim_512 --gen_hidden_dim 512  --gpu_inst 4
python sequence_gan.py 1.2.gen_hidden_dim_1024 --gen_hidden_dim 1024  --gpu_inst 5




### 1.3 seq_len

python sequence_gan.py 1.3.seq_len_10 --seq_len 10  --gpu_inst 4
python sequence_gan.py 1.3.seq_len_40 --seq_len 40  --gpu_inst 4

python sequence_gan.py 1.3.seq_len_80 --seq_len 80  --gpu_inst 4
python sequence_gan.py 1.3.seq_len_160 --seq_len 160  --gpu_inst 4


# 2st order experiments
#######################

### 2.1 pretrain

# gen_pretrain_epoch_num + dis_pretrain_epoch_num

python sequence_gan.py 1.1.gen_pretrain_epoch_num_240_dis_pretrain_epoch_num_100 --gen_pretrain_epoch_num 240 --dis_pretrain_epoch_num 100  --gpu_inst 5
python sequence_gan.py 1.1.gen_pretrain_epoch_num_60_dis_pretrain_epoch_num_25 --gen_pretrain_epoch_num 60 --dis_pretrain_epoch_num 25 --gpu_inst 5
python sequence_gan.py 1.1.gen_pretrain_epoch_num_240_dis_pretrain_epoch_num_25 --gen_pretrain_epoch_num 240 --dis_pretrain_epoch_num 25  --gpu_inst 5
python sequence_gan.py 1.1.gen_pretrain_epoch_num_60_dis_pretrain_epoch_num_100 --gen_pretrain_epoch_num 60 --dis_pretrain_epoch_num 100 --gpu_inst 5

python sequence_gan.py 2.1.gen_pretrain_epoch_num_60_dis_pretrain_epoch_num_10 --gen_pretrain_epoch_num 60 --dis_pretrain_epoch_num 10  --gpu_inst 4
python sequence_gan.py 2.1.gen_pretrain_epoch_num_120_dis_pretrain_epoch_num_10 --gen_pretrain_epoch_num 120 --dis_pretrain_epoch_num 10  --gpu_inst 4
python sequence_gan.py 2.1.gen_pretrain_epoch_num_240_dis_pretrain_epoch_num_10 --gen_pretrain_epoch_num 240 --dis_pretrain_epoch_num 10  --gpu_inst 4

python sequence_gan.py 2.1.gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_10 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 10  --gpu_inst 4
python sequence_gan.py 2.1.gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_25 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 25  --gpu_inst 4

python sequence_gan.py 2.1.gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_50 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 50  --gpu_inst 4
python sequence_gan.py 2.1.gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_100 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 100  --gpu_inst 4

python sequence_gan.py 2.3.gen_pretrain_epoch_num_960_dis_pretrain_epoch_num_50 --gen_pretrain_epoch_num 960 --dis_pretrain_epoch_num 50  --gpu_inst 4
python sequence_gan.py 2.3.gen_pretrain_epoch_num_960_dis_pretrain_epoch_num_100 --gen_pretrain_epoch_num 960 --dis_pretrain_epoch_num 100  --gpu_inst 4

python sequence_gan.py 2.3.gen_pretrain_epoch_num_960_dis_pretrain_epoch_num_200 --gen_pretrain_epoch_num 960 --dis_pretrain_epoch_num 200  --gpu_inst 4

python sequence_gan.py 2.3.gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_200 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 200  --gpu_inst 5
python sequence_gan.py 2.3.gen_pretrain_epoch_num_240_dis_pretrain_epoch_num_200 --gen_pretrain_epoch_num 240 --dis_pretrain_epoch_num 200  --gpu_inst 5



### 2.2 gen_arch

python sequence_gan.py 2.2.gen_emb_dim_8_gen_hidden_dim_128 --gen_emb_dim 8 --gen_hidden_dim 128 --gpu_inst 5
python sequence_gan.py 2.2.gen_emb_dim_8_gen_hidden_dim_256 --gen_emb_dim 8 --gen_hidden_dim 256 --gpu_inst 5

python sequence_gan.py 2.2.gen_emb_dim_8_gen_hidden_dim_512 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 5
python sequence_gan.py 2.2.gen_emb_dim_16_gen_hidden_dim_128 --gen_emb_dim 16 --gen_hidden_dim 128 --gpu_inst 5

python sequence_gan.py 2.2.gen_emb_dim_16_gen_hidden_dim_256 --gen_emb_dim 16 --gen_hidden_dim 256 --gpu_inst 5
python sequence_gan.py 2.2.gen_emb_dim_16_gen_hidden_dim_512 --gen_emb_dim 16 --gen_hidden_dim 512 --gpu_inst 5


# 3rd order experiments
#######################

python sequence_gan.py 3.1.gen_best_dis_emb_dim_16 --dis_emb_dim 16 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4
python sequence_gan.py 3.1.gen_best_dis_emb_dim_32 --dis_emb_dim 32 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4


python sequence_gan.py 3.1.gen_best_dis_emb_dim_64 --dis_emb_dim 64 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 5
python sequence_gan.py 3.1.gen_best_dis_emb_dim_128 --dis_emb_dim 128 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 5

python sequence_gan.py 3.1.gen_best_dis_emb_dim_256 --dis_emb_dim 256 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4

python sequence_gan.py 3.1.gen_best_dis_emb_dim_512 --dis_emb_dim 512 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 5

python sequence_gan.py 3.1.gen_best_dis_emb_dim_8 --dis_emb_dim 8 --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 5


##### RUNNING LINE ######

# 4th order experiments
#######################

python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_240_dis_pretrain_epoch_num_50 --gen_pretrain_epoch_num 240 --dis_pretrain_epoch_num 50  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 5
python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_50 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 50  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 5

python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_960_dis_pretrain_epoch_num_50 --gen_pretrain_epoch_num 960 --dis_pretrain_epoch_num 50  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4
python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_240_dis_pretrain_epoch_num_100 --gen_pretrain_epoch_num 240 --dis_pretrain_epoch_num 100  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4

python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_100 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 100  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4
python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_960_dis_pretrain_epoch_num_100 --gen_pretrain_epoch_num 960 --dis_pretrain_epoch_num 100  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4

python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_240_dis_pretrain_epoch_num_200 --gen_pretrain_epoch_num 240 --dis_pretrain_epoch_num 200  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4
python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_480_dis_pretrain_epoch_num_200 --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 200  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 4

python sequence_gan.py 4.1.best_arch_gen_pretrain_epoch_num_960_dis_pretrain_epoch_num_200 --gen_pretrain_epoch_num 960 --dis_pretrain_epoch_num 200  --gen_emb_dim 8 --gen_hidden_dim 512 --gpu_inst 6


# 5th order experiments
#######################

python sequence_gan.py 5.1.best_model --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 50  --gen_emb_dim 8 --gen_hidden_dim 512 --save_each_epochs 5 --num_epochs 400 --gpu_inst 5

# 6th order experiments
#######################

python sequence_gan.py 6.1.realy_best_model --gen_pretrain_epoch_num 480 --dis_pretrain_epoch_num 200  --gen_emb_dim 8 --gen_hidden_dim 512 --save_each_epochs 50  --gpu_inst 3





