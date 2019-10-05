import pandas as pd
import misc as ms
import test
import torch

import torch.nn as nn
import losses
from sklearn.cluster import KMeans

def train(exp_dict):
    history = ms.load_history(exp_dict)

    if 'only_supervised' in exp_dict:
        tgt_trainloader_supervised, tgt_testloader_supervised = ms.get_tgt_loader_supervised(exp_dict)
        # load models
        tgt_model, tgt_opt, tgt_scheduler, _,_,_ = ms.load_model_tgt(exp_dict)
        fit_source_supervised(tgt_model, tgt_opt, tgt_scheduler, tgt_trainloader_supervised, exp_dict)

        tgt_acc = test.validate(tgt_model, 
                              tgt_model, 
                              tgt_trainloader_supervised, 
                              tgt_testloader_supervised)

        print("{} TEST Accuracy Supervised =========== {:2%}\n".format(exp_dict["tgt_dataset"], 
                                                tgt_acc))
        ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt)
    else:
        # Source
        src_trainloader, src_valloader= ms.load_src_loaders(exp_dict)

        ####################### 1. Train source model
        src_model, src_opt, src_scheduler = ms.load_model_src(exp_dict)

        # Train Source  
        if exp_dict["reset_src"]:
            history = fit_source(src_model, src_opt, src_scheduler, src_trainloader, history, exp_dict)

        # Test Source
        src_acc = test.validate(src_model, 
                              src_model, 
                              src_trainloader, 
                              src_valloader)

        print("{} TEST Accuracy = {:2%}\n".format(exp_dict["src_dataset"], 
                                                src_acc))
        history["src_acc"] = src_acc

        ms.save_model_src(exp_dict, history, src_model, src_opt)

        ####################### 2. Train target model
        tgt_trainloader, tgt_valloader= ms.load_tgt_loaders(exp_dict)
        #load models
        tgt_model, tgt_opt, tgt_scheduler, disc_model, disc_opt, disc_scheduler = ms.load_model_tgt(exp_dict)
        tgt_model.load_state_dict(src_model.state_dict())
        if exp_dict["reset_tgt"]:
            history = fit_target(src_model, tgt_model, tgt_opt, tgt_scheduler, disc_model, disc_opt, 
                                disc_scheduler, src_trainloader, tgt_trainloader, tgt_valloader, history, exp_dict)


def fit_source(src_model, src_opt, src_scheduler, src_trainloader, history, exp_dict):
  # Train Source
  src_scheduler = torch.optim.lr_scheduler.MultiStepLR(src_opt, milestones=[100,500,800], gamma=0.1)
  for e in range(history["src_train"][-1]["epoch"], 
                 exp_dict["src_epochs"]):
    loss_sum = 0.
    for step, (images, labels) in enumerate(src_trainloader):
        # make images and labels variable
        images = images.cuda()
        labels = labels.squeeze_().cuda()

        # zero gradients for opt
        src_opt.zero_grad()

        # compute loss for critic
        loss = losses.triplet_loss(src_model, {"X":images,"y":labels})

        loss_sum += loss.item()

        # optimize source classifier
        loss.backward()
        src_opt.step()

    loss = loss_sum/step
    print("Source ({}) - Epoch [{}/{}] - loss={:.6f}".format(
                type(src_trainloader).__name__, e, 
                exp_dict["src_epochs"], loss))

    history["src_train"] += [{"loss":loss, "epoch":e}]

    src_scheduler.step()
    if e % 50 == 0:
      ms.save_model_src(exp_dict, history, src_model, src_opt)

  return history


def fit_target(src_model, tgt_model, tgt_opt, tgt_scheduler, disc_model, disc_opt, 
               disc_scheduler, src_trainloader, tgt_trainloader, tgt_valloader, history, exp_dict):
  acc_tgt_old = test.validate(src_model, tgt_model, src_trainloader, tgt_valloader)
  history["tgt_train"] += [{"epoch":0,
                             "acc_src":history["src_acc"], 
                             "acc_tgt":acc_tgt_old,
                             "n_train - "+ exp_dict["src_dataset"]:len(src_trainloader.dataset), 
                             "n_train - "+ exp_dict["tgt_dataset"]:len(tgt_trainloader.dataset), 
                             "n_test - " + exp_dict["tgt_dataset"]:len(tgt_valloader.dataset)}] 

  print("\n>>> Methods: {} - Source: {} -> Target: {} ---------- Originally".format(None, 
                                                                       exp_dict["src_dataset"], 
                                                                       exp_dict["tgt_dataset"]))
  print(pd.DataFrame([history["tgt_train"][-1]]))
  ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt, disc_model, disc_opt)
  
  if 'tgt_supervised' in exp_dict:
    tgt_trainloader_supervised, tgt_testloader_supervised = ms.get_tgt_loader_supervised(exp_dict)
    tgt_acc_old = 0
  ####
  tgt_scheduler = torch.optim.lr_scheduler.MultiStepLR(tgt_opt, milestones=[exp_dict['tgt_epochs']//2], gamma=0.1)
  disc_scheduler = torch.optim.lr_scheduler.MultiStepLR(disc_opt, milestones=[exp_dict['tgt_epochs']//2], gamma=0.1)
  ####
  for e in range(history["tgt_train"][-1]["epoch"]+1, exp_dict["tgt_epochs"]+1):
    # train supervised by target
    
    # 1. Train disc
    if exp_dict["options"]["disc"] == True:
      fit_discriminator(src_model, tgt_model, disc_model,
                    src_trainloader, tgt_trainloader, 
                    opt_tgt=tgt_opt,
                    opt_disc=disc_opt, 
                    epochs=exp_dict['tgt_disc_epochs'], verbose=0)
    
    print("\n>>> Source: {} -> Target: {} - Epochs: [{}/{}]".format(exp_dict["src_dataset"], 
                                                                    exp_dict["tgt_dataset"],
                                                                    e, exp_dict["tgt_epochs"]))
    
    
    #### 1.2. fit supervised training
    if 'tgt_supervised' in exp_dict:
        fit_source_supervised(tgt_model, tgt_opt, tgt_scheduler, tgt_trainloader_supervised, exp_dict)

    # save and test model
    test_epoch = 1 #exp_dict["tgt_epochs"] ##// 5
    if e % test_epoch == 0:
        if 'tgt_supervised' not in exp_dict:
            acc_tgt = test.validate(src_model, tgt_model, 
                                    src_trainloader, tgt_valloader)

            history["tgt_train"] += [{"epoch":e,
                             "acc_src":history["src_acc"], 
                             "acc_tgt":acc_tgt,
                             "n_train - "+ exp_dict["src_dataset"]:len(src_trainloader.dataset), 
                             "n_train - "+ exp_dict["tgt_dataset"]:len(tgt_trainloader.dataset), 
                             "n_test - " + exp_dict["tgt_dataset"]:len(tgt_valloader.dataset)}] 

            print("\n>>> Methods: {} - Source: {} -> Target: {}".format(None, 
                                                                       exp_dict["src_dataset"], 
                                                                       exp_dict["tgt_dataset"]))
            print(pd.DataFrame([history["tgt_train"][-1]]))
            
            ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt, disc_model, disc_opt)
            # if acc_tgt >= acc_tgt_old:
                # ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt,
                                # disc_model, disc_opt)
                # acc_tgt_old = acc_tgt
        
        else:
            tgt_acc = test.validate(tgt_model, 
                                    tgt_model, 
                                    tgt_trainloader_supervised, 
                                    tgt_testloader_supervised)

            print("{} TEST Accuracy Supervised =========== {:2%}\n".format(exp_dict["tgt_dataset"], 
                                                    tgt_acc))
            
            ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt, disc_model, disc_opt)
            # if tgt_acc >= tgt_acc_old:
                # ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt, disc_model, disc_opt)
                # tgt_acc_old = tgt_acc
    
    # 2. Train center-magnet
    if exp_dict["options"]["center"] == True:
      fit_center(src_model, tgt_model, 
                      src_trainloader, tgt_trainloader,
                      tgt_opt, epochs=exp_dict['tgt_embedding_epochs'])
    
    disc_scheduler.step()
    tgt_scheduler.step()
    
  return history

def fit_discriminator(src_model, tgt_model, disc,
              src_loader, tgt_loader,
              opt_tgt, opt_disc, 
              epochs=200,
              verbose=1):
    tgt_model.train()
    disc.train()

    # setup criterion and opt
    criterion = nn.CrossEntropyLoss()


    ####################
    # 2. train network #
    ####################

    for epoch in range(epochs):
        # zip source and target data pair
        
        data_zip = enumerate(zip(src_loader, tgt_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = images_src.cuda()
            images_tgt = images_tgt.cuda()

            # zero gradients for opt
            opt_disc.zero_grad()

            # extract and concat features
            feat_src = src_model.extract_features(images_src)
            feat_tgt = tgt_model.extract_features(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = disc(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long()
            label_tgt = torch.zeros(feat_tgt.size(0)).long()
            label_concat = torch.cat((label_src, label_tgt), 0).cuda()

            # compute loss for disc
            loss_disc = criterion(pred_concat, label_concat)
            loss_disc.backward()

            # optimize disc
            opt_disc.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for opt
            opt_disc.zero_grad()
            opt_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_model.extract_features(images_tgt)

            # predict on discriminator
            pred_tgt = disc(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            opt_tgt.step()

            



            #######################
            # 2.3 print step info #
            #######################
            if verbose and ((step + 1) % 20 == 0):
                print("Epoch [{}/{}] - "
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              epochs,
                              loss_disc.item(),
                              loss_tgt.item(),
                              acc.item()))



def fit_center(src_model, tgt_model, src_loader, tgt_loader,
                 opt_tgt, epochs=30):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    n_classes = tgt_model.n_classes
    # set train state for Dropout and BN layers
    src_model.train()
    tgt_model.train()


    src_embeddings, _ = losses.extract_embeddings(src_model, src_loader)


    src_kmeans = KMeans(n_clusters=n_classes)
    src_kmeans.fit(src_embeddings)



    #src_centers = torch.FloatTensor(src_kmeans.means_).cuda()
    src_centers = torch.FloatTensor(src_kmeans.cluster_centers_).cuda()

    
    ####################
    # 2. train network #
    ####################

    for epoch in range(epochs):
      for step, (images, _) in enumerate(tgt_loader):
        # make images and labels variable
        images = images.cuda()
        #labels = labels.squeeze_().cuda()

        # zero gradients for opt
        opt_tgt.zero_grad()

        # compute loss for critic
        loss = losses.center_loss(tgt_model, {"X":images,"y":None}, src_model, 
                                    src_centers, None, src_kmeans,
                                    None)
        # optimize source classifier
        loss.backward()
        opt_tgt.step()

##########################################################################################
def fit_source_supervised(tgt_model, tgt_opt, tgt_scheduler, tgt_trainloader_supervised, exp_dict):
  # Train Target supervised
  flag = False
  if tgt_scheduler is None:
    tgt_scheduler = torch.optim.lr_scheduler.MultiStepLR(tgt_opt, milestones=[100,500,800], gamma=0.1)
    flag = True
  for e in range(exp_dict["tgt_epochs_supervised"]):
    loss_sum = 0.
    for step, (images, labels) in enumerate(tgt_trainloader_supervised):
        # make images and labels variable
        images = images.cuda()
        labels = labels.squeeze_().cuda()

        # zero gradients for opt
        tgt_opt.zero_grad()

        # compute loss for critic
        loss = losses.triplet_loss(tgt_model, {"X":images,"y":labels})

        loss_sum += loss.item()

        # optimize source classifier
        loss.backward()
        tgt_opt.step()

    loss = loss_sum/step
    if flag:
        tgt_scheduler.step()
    print("Target Supervised ({}) - Epoch [{}/{}] - loss={:.6f}".format(
                type(tgt_trainloader_supervised).__name__, e, 
                exp_dict["tgt_epochs_supervised"], loss))