import numpy as np, cv2, warnings
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]

def moments(px):
    """px: (N,C) -> mean, std, skew per channel"""
    if len(px) < 3: return np.zeros(px.shape[1] * 3)
    s = skew(px, axis=0); s = np.nan_to_num(s)
    return np.concatenate([px.mean(0), px.std(0), s])

def hsv_lab(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[..., 0] * 2 * np.pi / 180  # hue circular -> use cos/sin
    hsv2 = np.dstack([np.cos(h), np.sin(h), hsv[..., 1] / 255, hsv[..., 2] / 255])
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255; lab[..., 1:] -= 128
    return hsv2, lab

def glcm_feats(gray, levels=32):
    q = (gray.astype(np.float32) / 256 * levels).astype(np.uint8)
    g = graycomatrix(q, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=levels, symmetric=True, normed=True)
    return np.array([graycoprops(g, p).mean() for p in ["contrast", "correlation", "energy", "homogeneity"]])

def lbp_hist(gray, P=8, R=1):
    l = local_binary_pattern(gray, P, R, "uniform")
    h, _ = np.histogram(l, bins=P + 2, range=(0, P + 2), density=True)
    return h

def models():
    return {
        "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
        "RF": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0, n_jobs=-1)),
    }

def evaluate(X, y, groups, splitter_fn, names):
    """returns {model: {macroF1 list, cm}}"""
    res = {}
    for mn, mk in models().items():
        scores, cm = [], 0
        for seed in SEEDS:
            for tr, te in splitter_fn(seed):
                m = mk().fit(X[tr], y[tr]); p = m.predict(X[te])
                scores.append(f1_score(y[te], p, average="macro"))
                cm = cm + confusion_matrix(y[te], p, labels=range(len(names)))
        res[mn] = dict(f1=scores, mean=float(np.mean(scores)), std=float(np.std(scores)), cm=np.asarray(cm).tolist())
    return res
