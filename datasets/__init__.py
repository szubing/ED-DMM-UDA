from datasets import hyperXDatas

def get_loader(name, split, batch_size=50, exp_dict=None):

    if name == 'indianSrc':
        return hyperXDatas.get_indian_src(split, batch_size, exp_dict)
    
    if name == 'indianTgt':
        return hyperXDatas.get_indian_tgt(split, batch_size, exp_dict)
        
    if name == 'paviaSrc':
        return hyperXDatas.get_pavia_src(split, batch_size, exp_dict)
    
    if name == 'paviaTgt':
        return hyperXDatas.get_pavia_tgt(split, batch_size, exp_dict)

    if name == 'shSrc':
        return hyperXDatas.get_sh_src(split, batch_size, exp_dict)
    
    if name == 'shTgt':
        return hyperXDatas.get_sh_tgt(split, batch_size, exp_dict)