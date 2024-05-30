# python test_kitti360.py --seq 2
python test_pred.py \
    --dir_src /hdd/datasets/waymo/seq1/image_0/ \
    --dir_tgt /hdd/datasets/waymo/seq1/intrinsic_0/shadow

python test_pred.py \
    --dir_src /hdd/datasets/waymo/seq2/image_0/ \
    --dir_tgt /hdd/datasets/waymo/seq2/intrinsic_0/shadow

python test_pred.py \
    --dir_src /hdd/datasets/waymo/seq3/image_0/ \
    --dir_tgt /hdd/datasets/waymo/seq3/intrinsic_0/shadow