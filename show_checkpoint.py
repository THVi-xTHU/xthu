from tensorflow.python.tools import inspect_checkpoint as chkp
model1='fcrn_depth_prediction/model/NYU_FCRN.ckpt'
model2='KittiSeg/RUNS/KittiSeg_pretrained/model.ckpt-15999'
#chkp.print_tensors_in_checkpoint_file(model1, tensor_name='depth', all_tensors=True)
chkp.print_tensors_in_checkpoint_file(model2, tensor_name='seg', all_tensors=True)


