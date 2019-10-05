import torch
from sklearn import neighbors
import losses
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np


def validate(src_model, tgt_model, src_data_loader, tgt_data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    with torch.no_grad():
        X, y = losses.extract_embeddings(src_model, src_data_loader)
        Xtest, ytest = losses.extract_embeddings(tgt_model, tgt_data_loader)
        
        clf = neighbors.KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)
        y_pred = clf.predict(Xtest)

        acc = (y_pred == ytest).mean()
        # print(acc)
        

    return acc

def validate_bing(src_model, tgt_model, src_data_loader, tgt_data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    with torch.no_grad():
        X, y = losses.extract_embeddings(src_model, src_data_loader)
        Xtest, ytest = losses.extract_embeddings(tgt_model, tgt_data_loader)
        
        clf = neighbors.KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)
        y_pred = clf.predict(Xtest)

        acc = (y_pred == ytest).mean()
        # print(acc)
        
        matrixErr = confusion_matrix(ytest, y_pred)
        class_accuracy = np.diag(matrixErr)/np.sum(matrixErr,axis=1)
        average_accuracy = np.mean(class_accuracy)
        kappa = cohen_kappa_score(ytest, y_pred)
        
        print('\n\n ----------------length: {}\n\n'.format(len(y_pred)))

    return matrixErr, acc, average_accuracy, kappa
