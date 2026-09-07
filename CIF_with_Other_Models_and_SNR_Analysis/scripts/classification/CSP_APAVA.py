import argparse
import os
import json
import numpy as np

from natsort import natsorted
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stable_cov_torch(trial_ch_t: torch.Tensor, reg_eps: float) -> torch.Tensor:
    """
    trial_ch_t: (C, T) on device
    returns: (C, C)
    """
    x = trial_ch_t - trial_ch_t.mean(dim=1, keepdim=True)
    cov = (x @ x.transpose(0, 1)) / max(1, (x.shape[1] - 1))
    cov = cov / (torch.trace(cov) + 1e-12)
    cov = cov + reg_eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov


def compute_csp_binary_torch(
    X_ch_t: torch.Tensor,
    y01: torch.Tensor,
    n_components: int,
    reg_eps: float,
) -> torch.Tensor:
    """
    X_ch_t: (N, C, T) torch tensor
    y01: (N,) 0/1 torch tensor
    returns W: (C, n_components)
    """
    if X_ch_t.ndim != 3:
        raise ValueError(f"X must be (N,C,T), got {tuple(X_ch_t.shape)}")
    y01 = y01.to(dtype=torch.long).view(-1)
    if not torch.all((y01 == 0) | (y01 == 1)):
        raise ValueError("y01 must only contain 0/1")

    C = X_ch_t.shape[1]
    cov1 = torch.zeros((C, C), device=X_ch_t.device, dtype=torch.float64)
    cov0 = torch.zeros_like(cov1)
    n1 = 0
    n0 = 0
    for i in range(X_ch_t.shape[0]):
        c = _stable_cov_torch(X_ch_t[i].to(torch.float64), reg_eps)
        if int(y01[i].item()) == 1:
            cov1 += c
            n1 += 1
        else:
            cov0 += c
            n0 += 1

    if n1 == 0 or n0 == 0:
        raise ValueError("Need both positive and negative samples for CSP")

    cov1 = cov1 / float(n1)
    cov0 = cov0 / float(n0)
    cov_sum = cov1 + cov0

    evals, evecs = torch.linalg.eigh(cov_sum)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(evals, min=1e-12)))
    P = evecs @ inv_sqrt @ evecs.transpose(0, 1)

    S1 = P @ cov1 @ P.transpose(0, 1)
    d, B = torch.linalg.eigh(S1)
    order2 = torch.argsort(d, descending=True)
    B = B[:, order2]
    W_full = P.transpose(0, 1) @ B  # (C, C)

    k = int(n_components)
    if k <= 0 or k > W_full.shape[1]:
        raise ValueError(f"n_components must be in [1,{W_full.shape[1]}]")
    k_top = (k + 1) // 2
    k_bot = k // 2
    if k_bot == 0:
        W = W_full[:, :k_top]
    else:
        W = torch.cat([W_full[:, :k_top], W_full[:, -k_bot:]], dim=1)
    return W.to(dtype=torch.float32)


def csp_features_torch(X_ch_t: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    X_ch_t: (N,C,T)
    W: (C,K)
    returns: (N,K)
    """
    Z = torch.einsum("nct,ck->nkt", X_ch_t, W)  # (N,K,T)
    var = Z.var(dim=2, unbiased=False)
    var = var / (var.sum(dim=1, keepdim=True) + 1e-12)
    return torch.log(torch.clamp(var, min=1e-12))


def compute_multiclass_csp_features_torch(
    X_ch_t: torch.Tensor,
    y: torch.Tensor,
    n_components_per_class: int,
    reg_eps: float,
):
    y = y.view(-1)
    classes = torch.unique(y).detach().cpu().numpy().tolist()
    feats = []
    filters = {}
    for c in classes:
        y01 = (y == int(c)).to(dtype=torch.long)
        W = compute_csp_binary_torch(X_ch_t, y01, n_components=n_components_per_class, reg_eps=reg_eps)
        filters[int(c)] = W
        feats.append(csp_features_torch(X_ch_t, W))
    X_feat = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
    meta = {"classes": classes, "n_components_per_class": int(n_components_per_class)}
    return X_feat, {"meta": meta, "filters": filters}


def train_torch_logreg(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
):
    torch.manual_seed(seed)
    model = nn.Linear(X_train.shape[1], num_classes).to(X_train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = float(loss_fn(val_logits, y_val).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(X_train.device) for k, v in best_state.items()})
    return model


def _stable_cov(trial_ch_t: np.ndarray, reg_eps: float) -> np.ndarray:
    """
    trial_ch_t: (C, T)
    returns: (C, C)
    """
    x = trial_ch_t - trial_ch_t.mean(axis=1, keepdims=True)
    cov = (x @ x.T) / max(1, (x.shape[1] - 1))
    cov = cov / (np.trace(cov) + 1e-12)
    cov = cov + reg_eps * np.eye(cov.shape[0], dtype=cov.dtype)
    return cov


def compute_csp_binary(
    X_ch_t: np.ndarray,
    y01: np.ndarray,
    n_components: int,
    reg_eps: float,
) -> np.ndarray:
    """
    X_ch_t: (N, C, T)
    y01: (N,) values in {0,1}, where 1 is "class" and 0 is "rest"
    returns W: (C, n_components) spatial filters
    """
    if X_ch_t.ndim != 3:
        raise ValueError(f"X must be (N,C,T), got {X_ch_t.shape}")
    y01 = np.asarray(y01).astype(int).reshape(-1)
    if set(np.unique(y01)).difference({0, 1}):
        raise ValueError("y01 must only contain 0/1")

    cov1 = np.zeros((X_ch_t.shape[1], X_ch_t.shape[1]), dtype=np.float64)
    cov0 = np.zeros_like(cov1)
    n1 = 0
    n0 = 0
    for i in range(X_ch_t.shape[0]):
        c = _stable_cov(X_ch_t[i], reg_eps)
        if y01[i] == 1:
            cov1 += c
            n1 += 1
        else:
            cov0 += c
            n0 += 1

    if n1 == 0 or n0 == 0:
        raise ValueError("Need both positive and negative samples for CSP")

    cov1 /= n1
    cov0 /= n0
    cov_sum = cov1 + cov0

    # Solve generalized eigenvalue problem: cov1 v = λ (cov1+cov0) v
    # Do it via whitening of cov_sum.
    evals, evecs = np.linalg.eigh(cov_sum)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12)))
    P = evecs @ inv_sqrt @ evecs.T  # whitening

    S1 = P @ cov1 @ P.T
    d, B = np.linalg.eigh(S1)
    order2 = np.argsort(d)[::-1]
    B = B[:, order2]

    W_full = P.T @ B  # (C, C)

    k = int(n_components)
    if k <= 0 or k > W_full.shape[1]:
        raise ValueError(f"n_components must be in [1,{W_full.shape[1]}]")

    # Take both extremes (largest and smallest eigenvalues) as in standard CSP.
    k_top = (k + 1) // 2
    k_bot = k // 2
    if k_bot == 0:
        return W_full[:, :k_top]
    return np.concatenate([W_full[:, :k_top], W_full[:, -k_bot:]], axis=1)


def csp_features(X_ch_t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    X_ch_t: (N, C, T)
    W: (C, K)
    returns: (N, K) log-variance features
    """
    Z = np.einsum("nct,ck->nkt", X_ch_t, W)  # (N,K,T)
    var = Z.var(axis=2)  # (N,K)
    var = var / (var.sum(axis=1, keepdims=True) + 1e-12)
    return np.log(np.maximum(var, 1e-12))


def compute_multiclass_csp_features(
    X_ch_t: np.ndarray,
    y: np.ndarray,
    n_components_per_class: int,
    reg_eps: float,
):
    """
    One-vs-rest CSP; concatenate per-class CSP features.
    returns: X_feat (N, K_total), meta dict with filters per class.
    """
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    feats = []
    filters = {}
    for c in classes:
        y01 = (y == c).astype(int)
        W = compute_csp_binary(X_ch_t, y01, n_components=n_components_per_class, reg_eps=reg_eps)
        filters[int(c)] = W
        feats.append(csp_features(X_ch_t, W))
    X_feat = np.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]
    meta = {"classes": classes.tolist(), "n_components_per_class": int(n_components_per_class)}
    return X_feat, {"meta": meta, "filters": filters}


def load_apava_split(root_path: str, split: str, seed: int = 42):
    data_path = os.path.join(root_path, "Feature")
    label_path = os.path.join(root_path, "Label", "label.npy")
    subject_label = np.load(label_path)  # (num_subjects, 2): [label, id]

    all_ids = [int(i) for i in subject_label[:, 1].tolist()]
    val_ids = [15, 16, 19, 20]
    test_ids = [1, 2, 17, 18]
    train_ids = [i for i in all_ids if i not in (val_ids + test_ids)]

    if split.upper() == "TRAIN":
        ids = set(train_ids)
    elif split.upper() == "VAL":
        ids = set(val_ids)
    elif split.upper() == "TEST":
        ids = set(test_ids)
    else:
        raise ValueError("split must be TRAIN/VAL/TEST")

    filenames = natsorted([f for f in os.listdir(data_path) if f.endswith(".npy")])

    X_list = []
    y_list = []
    for j, fn in enumerate(filenames):
        subject_id = j + 1  # repo convention: id starts from 1, order matches label.npy
        if subject_id not in ids:
            continue
        lab = int(subject_label[j, 0])
        subj_trials = np.load(os.path.join(data_path, fn))  # (n_trials, 256, 16)
        for trial in subj_trials:
            # trial: (T=256, C=16) -> (C, T)
            X_list.append(trial.T)
            y_list.append(lab)

    X = np.asarray(X_list, dtype=np.float64)  # (N,C,T)
    y = np.asarray(y_list, dtype=np.int64)
    X, y = shuffle(X, y, random_state=seed)
    return X, y


def build_clf(name: str, seed: int):
    name = name.lower()
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    raise ValueError("clf must be one of: lda, logreg, svm, torch_logreg")


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    pred = probs.argmax(axis=1)
    n_class = probs.shape[1]
    y_onehot = np.eye(n_class, dtype=np.float64)[y_true]
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "AUROC": float(roc_auc_score(y_onehot, probs, multi_class="ovr")),
        "AUPRC": float(average_precision_score(y_onehot, probs, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser("APAVA classification with CSP only")
    parser.add_argument("--root_path", type=str, required=True, help="Path containing Feature/ and Label/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_components_per_class", type=int, default=4)
    parser.add_argument("--reg_eps", type=float, default=1e-6)
    parser.add_argument("--clf", type=str, default="lda", choices=["lda", "logreg", "svm", "torch_logreg"])
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu (used for torch_logreg/CSP)")
    parser.add_argument("--epochs", type=int, default=80, help="epochs for torch_logreg")
    parser.add_argument("--lr", type=float, default=1e-2, help="lr for torch_logreg")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay for torch_logreg")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for torch_logreg")
    parser.add_argument("--model_id", type=str, default="APAVA-Indep-CSP")
    parser.add_argument("--save_dir", type=str, default="./results/classification")
    args = parser.parse_args()

    set_seed(args.seed)

    X_train, y_train = load_apava_split(args.root_path, "TRAIN", seed=args.seed)
    X_val, y_val = load_apava_split(args.root_path, "VAL", seed=args.seed)
    X_test, y_test = load_apava_split(args.root_path, "TEST", seed=args.seed)

    if args.clf == "torch_logreg":
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
        print("Device:", device)
        Xtr = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)
        ytr = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
        Xva = torch.from_numpy(X_val).to(device=device, dtype=torch.float32)
        yva = torch.from_numpy(y_val).to(device=device, dtype=torch.long)
        Xte = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)
        yte = torch.from_numpy(y_test).to(device=device, dtype=torch.long)

        # CSP feature extraction (fit on train only) - on GPU
        Xtr_feat_t, csp_state = compute_multiclass_csp_features_torch(
            Xtr, ytr, n_components_per_class=args.n_components_per_class, reg_eps=args.reg_eps
        )
        classes = np.array(csp_state["meta"]["classes"], dtype=int)

        feat_val_t = []
        feat_test_t = []
        for c in classes:
            W = csp_state["filters"][int(c)]
            feat_val_t.append(csp_features_torch(Xva, W))
            feat_test_t.append(csp_features_torch(Xte, W))
        Xva_feat_t = torch.cat(feat_val_t, dim=1) if len(feat_val_t) > 1 else feat_val_t[0]
        Xte_feat_t = torch.cat(feat_test_t, dim=1) if len(feat_test_t) > 1 else feat_test_t[0]

        # Standardize features using train statistics (GPU)
        feat_mean = Xtr_feat_t.mean(dim=0, keepdim=True)
        feat_std = Xtr_feat_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xtr_feat_t = (Xtr_feat_t - feat_mean) / feat_std
        Xva_feat_t = (Xva_feat_t - feat_mean) / feat_std
        Xte_feat_t = (Xte_feat_t - feat_mean) / feat_std

        num_classes = int(len(np.unique(y_train)))
        model = train_torch_logreg(
            Xtr_feat_t,
            ytr,
            Xva_feat_t,
            yva,
            num_classes=num_classes,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
        )
        model.eval()
        with torch.no_grad():
            p_val = torch.softmax(model(Xva_feat_t), dim=1).detach().cpu().numpy()
            p_test = torch.softmax(model(Xte_feat_t), dim=1).detach().cpu().numpy()
    else:
        # CSP feature extraction (fit on train only) - CPU numpy
        Xtr_feat, csp_state = compute_multiclass_csp_features(
            X_train, y_train, n_components_per_class=args.n_components_per_class, reg_eps=args.reg_eps
        )

        # apply train-fitted filters to val/test
        classes = np.unique(y_train)
        feat_val = []
        feat_test = []
        for c in classes:
            W = csp_state["filters"][int(c)]
            feat_val.append(csp_features(X_val, W))
            feat_test.append(csp_features(X_test, W))
        Xva_feat = np.concatenate(feat_val, axis=1) if len(feat_val) > 1 else feat_val[0]
        Xte_feat = np.concatenate(feat_test, axis=1) if len(feat_test) > 1 else feat_test[0]

        scaler = StandardScaler()
        Xtr_feat = scaler.fit_transform(Xtr_feat)
        Xva_feat = scaler.transform(Xva_feat)
        Xte_feat = scaler.transform(Xte_feat)

        clf = build_clf(args.clf, seed=args.seed)
        clf.fit(Xtr_feat, y_train)

        # predicted probabilities
        p_val = clf.predict_proba(Xva_feat)
        p_test = clf.predict_proba(Xte_feat)

    val_metrics = metrics_from_probs(y_val, p_val)
    test_metrics = metrics_from_probs(y_test, p_test)

    out_dir = os.path.join(args.save_dir, args.model_id, "CSP")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result_classification.txt")
    state_path = os.path.join(out_dir, f"csp_state_seed{args.seed}.json")

    setting = (
        f"classification_{args.model_id}_CSP_APAVA_seed{args.seed}"
        f"_k{args.n_components_per_class}_reg{args.reg_eps}_{args.clf}"
    )

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "setting": setting,
                "args": vars(args),
                "csp_meta": csp_state["meta"],
                "classes": classes.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(result_path, "a", encoding="utf-8") as f:
        f.write(setting + "  \n")
        f.write(
            "Validation results --- "
            + ", ".join([f"{k}: {v:.5f}" for k, v in val_metrics.items()])
            + "\n"
        )
        f.write(
            "Test results --- "
            + ", ".join([f"{k}: {v:.5f}" for k, v in test_metrics.items()])
            + "\n\n"
        )

    print("Saved:", result_path)
    print("Validation:", val_metrics)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()

