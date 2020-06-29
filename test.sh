export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/test_net.py --config-file "./configs/deep_fake/baseline_deep_fake_detection_mesonet_inc4.yaml"