import matplotlib
matplotlib.use('Agg')
import json
import torch
import misc as ms
import models
import datasets
import test
import os 
from scipy import io

def set_gpu(gpu_id):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]="%d" % gpu_id


def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass 
    
def save_json(fname, data):
    create_dirs(fname)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)
    
def load_json(fname):
    with open(fname, "r") as json_file:
        d = json.load(json_file)
    
    return d

def copy_models(exp_dict, path_dst):
    history = load_history(exp_dict)

    ms.save_json(path_dst+"/history.json", history)

    print("copied...")

def test_latest_model(exp_dict, verbose=1):
    
    history = load_history(exp_dict)

    src_trainloader, _ = ms.load_src_loaders(exp_dict)
    _, tgt_valloader = ms.load_tgt_loaders(exp_dict)

    src_model, src_opt, _ = ms.load_model_src(exp_dict)
    tgt_model, tgt_opt,_, _, _,_  = ms.load_model_tgt(exp_dict)

    acc_tgt = test.validate(src_model, tgt_model, 
                            src_trainloader, 
                            tgt_valloader)
    if verbose:
        print("====================="
              "\nAcc of model at epoch {}: {}\n"
              "=====================".format(history["tgt_train"][-1]["epoch"],
                                        acc_tgt))
    return acc_tgt
    
def test_latest_model_bing(exp_dict, verbose=1):
    
    history = load_history(exp_dict)
    
    if ('only_supervised' in exp_dict) or ('tgt_supervised' in exp_dict):
        tgt_trainloader_supervised, tgt_testloader_supervised = ms.get_tgt_loader_supervised(exp_dict)
        # load models
        tgt_model, tgt_opt, tgt_scheduler, _,_,_ = ms.load_model_tgt(exp_dict)

        matrixErr, oa, aa, kappa = test.validate_bing(tgt_model, 
                                tgt_model, 
                                tgt_trainloader_supervised, 
                                tgt_testloader_supervised)
    else:
        src_trainloader, _ = ms.load_src_loaders(exp_dict)
        _, tgt_valloader = ms.load_tgt_loaders(exp_dict)

        src_model, src_opt, _ = ms.load_model_src(exp_dict)
        tgt_model, tgt_opt,_, _, _,_  = ms.load_model_tgt(exp_dict)

        matrixErr, oa, aa, kappa = test.validate_bing(src_model, tgt_model, 
                                src_trainloader, 
                                tgt_valloader)
    if verbose:
        print("====================="
              "\nOvearll Accuracy of model at epoch {}: {}\n"
              "=====================".format(history["tgt_train"][-1]["epoch"],
                                        oa))
        print("====================="
              "\nAverage Accuracy of model at epoch {}: {}\n"
              "=====================".format(history["tgt_train"][-1]["epoch"],
                                        aa))
        print("====================="
              "\nKappa of model at epoch {}: {}\n"
              "=====================".format(history["tgt_train"][-1]["epoch"],
                                        kappa))
    results={}
    results['confusion_matrix']=matrixErr
    results['OA']=oa
    results['AA']=aa
    results['Kappa']=kappa
    return results


def load_src_loaders(exp_dict):
    train_loader = datasets.get_loader(exp_dict["src_dataset"], "train", 
                        batch_size=exp_dict["src_batch_size"], exp_dict=exp_dict)
    val_loader = datasets.get_loader(exp_dict["src_dataset"], "val", 
                        batch_size=exp_dict["src_batch_size"], exp_dict=exp_dict)
    n_train = len(train_loader.dataset)
    n_test = len(val_loader.dataset)
    name = type(train_loader.dataset).__name__

    print("Source ({}): train set: {} - val set: {}".format(name, n_train, n_test))
    return train_loader, val_loader

def load_tgt_loaders(exp_dict):
    train_loader = datasets.get_loader(exp_dict["tgt_dataset"], "train", 
                        batch_size=exp_dict["tgt_batch_size"], exp_dict=exp_dict)
    val_loader = datasets.get_loader(exp_dict["tgt_dataset"], "val", 
                        batch_size=exp_dict["tgt_batch_size"], exp_dict=exp_dict)
    name = type(train_loader.dataset).__name__
    n_train = len(train_loader.dataset)
    n_test = len(val_loader.dataset)
    print("Target ({}): train set: {} - val set: {}".format(name, n_train, n_test))
    return train_loader, val_loader

def load_history(exp_dict):
    name_history = exp_dict["path"]+"/history_run{}.json".format(exp_dict['run'])

    if not os.path.exists(name_history) or (exp_dict["reset_src"] and exp_dict["reset_tgt"]):
        history = {"src_train":[{"epoch":0}]}
        history["tgt_train"] = [{"epoch":0, "acc_tgt":-1}]

        print("History from scratch...")
    else:
        history = ms.load_json(name_history)
        print("Loaded history {}".format(name_history))

    if exp_dict["reset_tgt"]:
        history["tgt_train"] = [{"epoch":0, "acc_tgt":-1}]

        print("Resetting target training...")

    return history

def save_model_src(exp_dict, history, model_src, opt_src):
    save_json(exp_dict["path"]+"/history_run{}.json".format(exp_dict['run']), history)
    torch.save(model_src.state_dict(), exp_dict["path"]+"/model_src_run{}.pth".format(exp_dict['run']))
    torch.save(opt_src.state_dict(), exp_dict["path"]+"/opt_src_run{}.pth".format(exp_dict['run']))
    print("Saved Source...")

def save_model_tgt(exp_dict, history, model_tgt, opt_tgt, disc=None, disc_opt=None):
    save_json(exp_dict["path"]+"/history_run{}.json".format(exp_dict['run']), history)
    torch.save(model_tgt.state_dict(), exp_dict["path"]+"/model_tgt_run{}.pth".format(exp_dict['run']))
    torch.save(opt_tgt.state_dict(), exp_dict["path"]+"/opt_tgt_run{}.pth".format(exp_dict['run']))

    if disc is not None:
        torch.save(disc.state_dict(), exp_dict["path"]+"/disc_run{}.pth".format(exp_dict['run']))
    if disc_opt is not None:
        torch.save(disc_opt.state_dict(), exp_dict["path"]+"/disc_opt_run{}.pth".format(exp_dict['run']))
    print("Saved Target...")

def load_model_src(exp_dict):
    src_model, src_opt, src_scheduler = models.get_model(exp_dict["src_model"], 
                                          exp_dict["n_outputs"],
                                          input_channels=exp_dict['input_channels'],
                                          patch_size=exp_dict['patch_size'],
                                          n_classes=exp_dict['n_classes'])
    
    name_model = exp_dict["path"]+"/model_src_run{}.pth".format(exp_dict['run'])
    name_opt = exp_dict["path"]+"/opt_src_run{}.pth".format(exp_dict['run'])

    if os.path.exists(name_model) and not exp_dict["reset_src"]:
      src_model.load_state_dict(torch.load(name_model))
      src_opt.load_state_dict(torch.load(name_opt))
      print("Loading saved {}".format(name_model))

    else:
      print("Loading source models from scratch..")
    
    return src_model, src_opt, src_scheduler
    
def load_model_tgt(exp_dict):
    tgt_model, tgt_opt, tgt_scheduler = models.get_model(exp_dict["tgt_model"], 
                                          exp_dict["n_outputs"],
                                          input_channels=exp_dict['input_channels'],
                                          patch_size=exp_dict['patch_size'],
                                          n_classes=exp_dict['n_classes'])
    if 'disc_model' in exp_dict:
        disc, disc_opt, disc_scheduler = models.get_model(exp_dict["disc_model"], 
                                          exp_dict["n_outputs"])
    else:
        disc = None
        disc_opt = None
        disc_scheduler = None
    
    name_model = exp_dict["path"]+"/model_tgt_run{}.pth".format(exp_dict['run'])
    name_opt = exp_dict["path"]+"/opt_tgt_run{}.pth".format(exp_dict['run'])

    name_disc = exp_dict["path"]+"/disc_run{}.pth".format(exp_dict['run'])
    name_disc_opt = exp_dict["path"]+"/disc_opt_run{}.pth".format(exp_dict['run'])

    if os.path.exists(name_model) and not exp_dict["reset_tgt"]:
      tgt_model.load_state_dict(torch.load(name_model))
      tgt_opt.load_state_dict(torch.load(name_opt))

      if 'disc_model' in exp_dict:
          disc.load_state_dict(torch.load(name_disc))
          disc_opt.load_state_dict(torch.load(name_disc_opt))

      print("Loading saved {}".format(name_model))

    else:
      print("Loading target models from scratch..")
    
    return tgt_model, tgt_opt, tgt_scheduler, disc, disc_opt, disc_scheduler

#######################################################################################
def get_tgt_loader_supervised(exp_dict):
    train_loader = datasets.get_loader(exp_dict["tgt_dataset"], "train_supervised", 
                        batch_size=exp_dict["tgt_batch_size_supervised"], exp_dict=exp_dict)
    test_loader = datasets.get_loader(exp_dict["tgt_dataset"], "test_supervised", 
                        batch_size=exp_dict["tgt_batch_size_supervised"], exp_dict=exp_dict)
    name = type(train_loader.dataset).__name__
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    print("Target Supervised ({}): train set: {} ---------- test set: {}".format(name, n_train, n_test))
    return train_loader, test_loader