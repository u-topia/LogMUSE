import sys
import torch
import torch.nn as nn
import numpy as np
from model.multi_Transformer_1 import LogClassifier
from model.dataloader import LogDataset
import time
import torch.optim as optim
from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

# set the parameters of Transformer
max_len = 20

d_k = d_v = 64
d_model = 768
d_ff = 512
n_layers =2
p_dropout = 0.1
n_heads = 12
# scale of different attention heads
scales = [max_len, max_len//2, max_len//4]

output_dir = 'output'
log_name = 'Thunderbird'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

batch_size = 64
epochs = 20
window_size = max_len
lr = 1e-4

# set random seed
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def get_data(mode_name):
    if mode_name == 'train':
        ratio = 0.875
        training_data_x = np.load(f'{output_dir}/Thunderbird/Thunderbird_train_x.npy', allow_pickle=True)
        training_data_y = np.load(f'{output_dir}/Thunderbird/Thunderbird_train_y.npy', allow_pickle=True)

        split_index = int(len(training_data_x) * ratio)
        x_data_train = training_data_x[:split_index]
        y_data_train = training_data_y[:split_index]

        del training_data_x, training_data_y

        dataset_train = LogDataset(x_data_train, y_data_train, window_size)
        data_loader = torch.utils.data.DataLoader(dataset_train, 
                                        batch_size=batch_size, shuffle=(mode_name == 'train'))
        
    elif mode_name == 'val':
        val_data_x = np.load(f'{output_dir}/Thunderbird/Thunderbird_test_x.npy', allow_pickle=True)
        val_data_y = np.load(f'{output_dir}/Thunderbird/Thunderbird_test_y.npy', allow_pickle=True)

        # get the index of [0,1]
        ab_indices = np.where(np.all(val_data_y == [0, 1], axis=1))[0]
        # get the index of [1,0]
        normal_indices = np.where(np.all(val_data_y == [1, 0], axis=1))[0]

        # random select 50% anomaly samples
        ab_sample_size = len(ab_indices) // 2
        selected_ab_indices = np.random.choice(ab_indices, size=ab_sample_size, replace=False)
        
        # select the same number of normal samples
        selected_normal_indices = np.random.choice(normal_indices, size=ab_sample_size, replace=False)

        # get data
        val_data_x_ab = val_data_x[selected_ab_indices]
        val_data_y_ab = val_data_y[selected_ab_indices]
        val_data_x_n = val_data_x[selected_normal_indices]
        val_data_y_n = val_data_y[selected_normal_indices]

        # combine
        val_data_x = np.concatenate([val_data_x_ab, val_data_x_n])
        val_data_y = np.concatenate([val_data_y_ab, val_data_y_n])

        dataset_val = LogDataset(val_data_x, val_data_y, window_size)
        data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        
    elif mode_name == 'test':
        testing_data_x = np.load(f'{output_dir}/Thunderbird/Thunderbird_test_x.npy', allow_pickle=True)
        testing_data_y = np.load(f'{output_dir}/Thunderbird/Thunderbird_test_y.npy', allow_pickle=True)

        x_data_test = testing_data_x
        y_data_test = testing_data_y

        del testing_data_x, testing_data_y

        dataset_test = LogDataset(x_data_test, y_data_test, window_size)
        data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return data_loader

def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    print("#"*50)
    print("Start Training ...")

    model.to(device)
    # Parallel computing
    model = torch.nn.DataParallel(model)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    # Early stop mechanism
    patience = 3 
    best_f1 = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        loss_all = []
        f1_all = []
        train_loss = 0
        train_pred = []
        train_true = []

        model.train()
        start_time = time.time()
        interval = 100
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            train_pred.extend(logits.argmax(dim=1).tolist())
            train_true.extend(y.argmax(dim=1).tolist())

            if batch_idx % interval == 0 and batch_idx > 0:
                cur_loss = train_loss / interval
                cur_f1 = f1_score(train_true, train_pred)
                time_cost = time.time() - start_time

                with open('result/train_thunderbird.txt', 'a', encoding='utf-8') as f:
                    f.write(f'| epoch: {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} | batches | '
                            f'loss {cur_loss:2.5f} |'
                            f'f1 {cur_f1:.5f} |'
                            f'time {time_cost: .5f} |'
                            f' lr {scheduler.get_last_lr()}\n')
                print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                  f'loss {cur_loss} |'
                  f'f1 {cur_f1}',
                  f'lr {scheduler.get_last_lr()}')
                
                loss_all.append(train_loss)
                f1_all.append(cur_f1)

                start_time = time.time()
                train_loss = 0
                train_acc = 0

        train_loss = np.sum(loss_all) / len(train_loader)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, train_loss))

        # Verification
        model.eval()
        val_pred = []
        val_true = []
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:  
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                val_pred.extend(logits.argmax(dim=1).tolist())
                val_true.extend(y.argmax(dim=1).tolist())
        

        val_f1 = f1_score(val_true, val_pred)
        val_loss = val_loss / len(val_loader)
        
        print(f"Validation - Epoch: {epoch}, Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"Found new best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            model.load_state_dict(best_model_state)
            break
            
        model.train()

    return model

def test_model(model, test_loader, criterion, device):
    print("#" * 50)
    print("Start Testing ...")
    model.eval()

    test_loss = 0
    test_pred = []
    test_true = []
    test_probs = []  

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            test_loss += loss.item()
            test_pred.extend(logits.argmax(dim=1).tolist())
            test_true.extend(y.argmax(dim=1).tolist())
            
            probs = torch.softmax(logits, dim=1)
            test_probs.extend(probs[:, 1].cpu().numpy())

            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"testing batch: {batch_idx}/{len(test_loader)}")
                print(f"loss: {test_loss:.4f}")
        
    avg_loss = test_loss / len(test_loader)

    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    accuracy = accuracy_score(test_true, test_pred)
    precision = precision_score(test_true, test_pred)
    recall = recall_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred)

    tp = sum((pred == 1) and (true == 1) for pred, true in zip(test_pred, test_true))
    fp = sum((pred == 1) and (true == 0) for pred, true in zip(test_pred, test_true)) 
    tn = sum((pred == 0) and (true == 0) for pred, true in zip(test_pred, test_true))
    fn = sum((pred == 0) and (true == 1) for pred, true in zip(test_pred, test_true))
    
    print("\nIndicators related to the confusion matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}") 
    print(f"False Negatives (FN): {fn}")

    fpr, tpr, _ = roc_curve(test_true, test_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"ROC AUC: {roc_auc:.4f}")

    print("Test Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    os.makedirs('result', exist_ok=True)
    
    plt.savefig('result/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('result/roc_curve.pdf', bbox_inches='tight')
    print("ROC has been saved at result/roc_curve.png 和 result/roc_curve.pdf")
    
    # plt.show()
    plt.close()

    with open('result/result_tbird.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"Test Result at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    return avg_loss, accuracy, precision, recall, f1, roc_auc

def train_and_test():
    train_loader = get_data('train')
    val_loader = get_data('val')
    model = LogClassifier(embed_dim=d_model, num_heads=n_heads, num_layers=n_layers, scales=scales)

    train_start = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, train_loader, val_loader, optimizer, criterion, device)
    train_end = time.time()
    train_time = train_end - train_start
    print("Training Time: " + str(train_time))

    torch.save(model.state_dict(), f'result/model_{log_name}.pth')

    test_start = time.time()
    test_loader = get_data('test')
    test_result = test_model(model, test_loader, criterion, device)
    test_end = time.time()
    test_time = test_end - test_start
    print("Testing Time: " + str(test_time))
    
    avg_loss, accuracy, precision, recall, f1, roc_auc = test_result
    print("\n" + "="*50)
    print("Results: ")
    print(f"avg_loss: {avg_loss:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("="*50)
    
    return test_result

if __name__=='__main__':
    train_and_test()
        
