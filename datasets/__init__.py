import torchvision


from .ytvos_T5 import build as build_ytvos_T5

from .concat_im_T5 import build as build_joint_im_T5

from .concat_dataset_im import build as build_joint_im

from .concat_dataset import build as build_joint

from .ytvos_clip import build as build_ytvos_clip


def build_dataset(dataset_file: str, image_set: str, args):
    
    if dataset_file == 'ytvos_T5' :
        return build_ytvos_T5(image_set, args)
    
    if dataset_file == 'joint_im_T5':
        return build_joint_im_T5(image_set, args)
    
    if dataset_file == 'joint_im':
        return build_joint_im(image_set, args)
    
    if dataset_file == 'joint':
        return build_joint(image_set, args)
    

    if dataset_file == 'ytvos_clip' :
        return build_ytvos_clip(image_set, args)

    

    raise ValueError(f'dataset {dataset_file} not supported')
