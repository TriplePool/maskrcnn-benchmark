export NGPUS=2
#python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file "./configs/deep_fake/baseline_deep_fake_detection_mesonet_inc4.yaml"
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file "./configs/deep_fake/two_branch_merged_mesonet_inc4.yaml"
#python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file "./configs/deep_fake/baseline_celeb_mesonet_inc4.yaml"

