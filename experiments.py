
def get_experiment_dict(args, exp_name):

#######################################################################################
    if exp_name == 'indianSrc2indianTgt':
        exp_dict = {'patch_size':5,
                    'n_classes':7,
                    'input_channels':220,
                    'disc_model':'DiscNetHyperX',
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'src_dataset':'indianSrc',
                    'src_train_size':180,
                    'src_model':'EmbeddingNetHyperX',
                    'src_epochs':1000,
                    'src_batch_size':49,
                    
                    'tgt_dataset':'indianTgt',
                    'tgt_train_size':100,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs':10,
                    'tgt_batch_size':49,
                    'tgt_disc_epochs':3,
                    'tgt_embedding_epochs':1,
                    
                    "options":{
                               "center":True,"disc":True,
                               },
                    "n_outputs":128}
    
    if exp_name == 'paviaSrc2paviaTgt':
        exp_dict = {'patch_size':5,
                    'n_classes':6,
                    'input_channels':72,
                    'disc_model':'DiscNetHyperX',
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'src_dataset':'paviaSrc',
                    'src_train_size':180,
                    'src_model':'EmbeddingNetHyperX',
                    'src_epochs':200,
                    'src_batch_size':36,
                    
                    'tgt_dataset':'paviaTgt',
                    'tgt_train_size':-1,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs':20,
                    'tgt_batch_size':36,
                    'tgt_disc_epochs':3,
                    'tgt_embedding_epochs':1,
                    
                    "options":{
                               "center":True,"disc":True,
                               },
                    "n_outputs":128}
    
    if exp_name == 'shSrc2shTgt':
        exp_dict = {'patch_size':1,
                    'n_classes':3,
                    'input_channels':198,
                    'disc_model':'DiscNetHyperX',
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'src_dataset':'shSrc',
                    'src_train_size':180,
                    'src_model':'EmbeddingNetHyperX',
                    'src_epochs':1000,
                    'src_batch_size':9*2,
                    
                    'tgt_dataset':'shTgt',
                    'tgt_train_size':-1,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs':10,
                    'tgt_batch_size':9*2,
                    'tgt_disc_epochs':3,
                    'tgt_embedding_epochs':1,
                    
                    "options":{
                               "center":True,"disc":True,
                               },
                    "n_outputs":128}
#########################################################################
    if exp_name == 'indianSrc_indianTgt':
        exp_dict = {'patch_size':5,
                    'n_classes':7,
                    'input_channels':220,
                    'disc_model':'DiscNetHyperX',
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'src_dataset':'indianSrc',
                    'src_train_size':180,
                    'src_model':'EmbeddingNetHyperX',
                    'src_epochs':1000,
                    'src_batch_size':49,
                    
                    'tgt_dataset':'indianTgt',
                    'tgt_train_size':200,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs':100,
                    'tgt_batch_size':49,
                    'tgt_disc_epochs':3,
                    'tgt_embedding_epochs':1,
                    
                    'tgt_supervised':True,
                    "tgt_epochs_supervised": 20,
                    "tgt_batch_size_supervised":7*5,
                    "tgt_train_size_supervised":20,
                    
                    "options":{
                               "center":False,"disc":False,
                               },
                    "n_outputs":128}
    
    if exp_name == 'paviaSrc_paviaTgt':
        exp_dict = {'patch_size':5,
                    'n_classes':6,
                    'input_channels':72,
                    'disc_model':'DiscNetHyperX',
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'src_dataset':'paviaSrc',
                    'src_train_size':180,
                    'src_model':'EmbeddingNetHyperX',
                    'src_epochs':200,
                    'src_batch_size':36,
                    
                    'tgt_dataset':'paviaTgt',
                    'tgt_train_size':200,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs':100,
                    'tgt_batch_size':36,
                    'tgt_disc_epochs':3,
                    'tgt_embedding_epochs':1,
                    
                    'tgt_supervised':True,
                    "tgt_epochs_supervised": 10,
                    "tgt_batch_size_supervised":6*5,
                    "tgt_train_size_supervised":20,
                    
                    "options":{
                               "center":False,"disc":False,
                               },
                    "n_outputs":128}
    
    if exp_name == 'shSrc_shTgt':
        exp_dict = {'patch_size':1,
                    'n_classes':3,
                    'input_channels':198,
                    'disc_model':'DiscNetHyperX',
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'src_dataset':'shSrc',
                    'src_train_size':180,
                    'src_model':'EmbeddingNetHyperX',
                    'src_epochs':1000,
                    'src_batch_size':9*2,
                    
                    'tgt_dataset':'shTgt',
                    'tgt_train_size':200,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs':20,
                    'tgt_batch_size':9*2,
                    'tgt_disc_epochs':3,
                    'tgt_embedding_epochs':1,
                    
                    'tgt_supervised':True,
                    "tgt_epochs_supervised": 10,
                    "tgt_batch_size_supervised":3*5,
                    "tgt_train_size_supervised":20,
                    
                    "options":{
                               "center":False,"disc":False,
                               },
                    "n_outputs":128}
#########################################################################
###################################
    if exp_name == 'indianTgt':
        exp_dict = {'patch_size':5,
                    'n_classes':7,
                    'input_channels':220,
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'tgt_dataset':'indianTgt',
                    'tgt_train_size':20,
                    'tgt_train_size_supervised':20,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs_supervised':1000,
                    'tgt_batch_size_supervised':7*5,
                    'only_supervised':True,
                    
                    "n_outputs":128}
    
    if exp_name == 'paviaTgt':
        exp_dict = {'patch_size':5,
                    'n_classes':6,
                    'input_channels':72,
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'tgt_dataset':'paviaTgt',
                    'tgt_train_size':20,
                    'tgt_train_size_supervised':20,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs_supervised':200,
                    'tgt_batch_size_supervised':6*5,
                    'only_supervised':True,
                    
                    "n_outputs":128}
    
    if exp_name == 'shTgt':
        exp_dict = {'patch_size':1,
                    'n_classes':3,
                    'input_channels':198,
                    'flip_aug':False,
                    'rotation_aug':False,
                    'sample_already':True,

                    'tgt_dataset':'shTgt',
                    'tgt_train_size':20,
                    'tgt_train_size_supervised':20,
                    'tgt_model':'EmbeddingNetHyperX',
                    'tgt_epochs_supervised':1000,
                    'tgt_batch_size_supervised':3*5,
                    'only_supervised':True,
                    
                    "n_outputs":128}
#########################################################################
    exp_dict["exp_name"] = exp_name
    exp_dict["path"]="checkpoints/{}/".format(exp_name)
    exp_dict["summary_path"] = "figures"
    
    if exp_name == 'paviaSrc2paviaTgt':
        exp_dict["path"]="checkpoints_final_pavia{}_center_{}_disc_{}/{}/".format(exp_dict['src_train_size'],exp_dict["options"]["center"],exp_dict["options"]["disc"],exp_name)
    return exp_dict