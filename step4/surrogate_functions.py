import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, auc, roc_curve, confusion_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeClassifier


numerical = ['issue_d', 'loan duration', 'annual_inc', 'avg_cur_bal',
    'bc_open_to_buy', 'bc_util', 'delinq_2yrs', 'dti', 'fico_range_high', 'funded_amnt', 
    'inq_last_6mths', 'int_rate', 'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
    'mths_since_recent_bc', 'num_actv_bc_tl', 'num_bc_tl', 'num_il_tl',
    'num_rev_accts', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'revol_bal', 'revol_util',
    'tax_liens', 'zip_code', 'Pct_afro_american']

category = ['emp_length',
    'emp_title',
    'grade',
    'home_ownership',
    'purpose',
    'sub_grade',
    ]

def get_features(df: pd.DataFrame()) -> pd.DataFrame():
    """
    Get additional features.
    """

    df_with_features = df.assign(
        # Log transformations - reduce skewness in monetary values
        avg_cur_bal_log=np.log1p(df["avg_cur_bal"]),
        revol_bal_log=np.log1p(df["revol_bal"]),
        
        # Derived financial metrics
        cur_balance=df["avg_cur_bal"] * df["open_acc"],  # Total credit exposure across all accounts
        
        # Binary risk flags
        delinq_2yrs_flag=df["delinq_2yrs"] >= 1,  # Any recent delinquency is a strong risk signal
        tax_liens_flag=df["tax_liens"] >= 1,  # Tax problems indicate financial distress
        
        # Credit portfolio composition ratios
        s_actv_bc_tl=df["num_actv_bc_tl"] / (df["open_acc"] + 1e-6),  # Share of active bank cards vs total accounts
        s_bc_tl=df["num_bc_tl"] / (df["open_acc"] + 1e-6),  # Bank card concentration in credit profile
        s_il_tl=df["num_il_tl"] / (df["open_acc"] + 1e-6),  # Installment loan dependency ratio
        s_rev_accts=df["num_rev_accts"] / (df["open_acc"] + 1e-6),  # Revolving credit reliance measure
        
        # Risk interaction features
        revol_bal_income_ratio=df["revol_bal"]
        / (df["annual_inc"] + 1e-6),  # Credit card debt relative to earning capacity
    )

    df_with_features.drop(columns=["revol_bal", "avg_cur_bal"], inplace=True)
    return df_with_features

def cap_outliers(df, columns, percentile=99):
    df_capped = df.copy()

    for col in columns:
        if col in df_capped.columns:
            upper_bound = df_capped[col].quantile(percentile / 100)
            lower_bound = df_capped[col].quantile((100 - percentile) / 100)

            original_outliers = (
                (df_capped[col] > upper_bound) | (df_capped[col] < lower_bound)
            ).sum()

            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)

    return df_capped
	
def categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Improved categorical encoding."""

    df_encoded = df.copy()

    # Keep only sub_grade (more informative than grade)
    sg = df_encoded["sub_grade"].astype(str).str.upper().str.strip()
    letter = sg.str[0]
    number = pd.to_numeric(
        sg.str[1:].str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    letter_map = {ch: i + 1 for i, ch in enumerate("ABCDEFG")}
    base = letter.map(letter_map)
    df_encoded["sub_grade_num"] = ((base - 1) * 5 + number).astype("float32")

    # Employment length to numeric
    emp_length_map = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
    df_encoded["emp_length_num"] = (
        df_encoded["emp_length"].map(emp_length_map).astype("float32")
    )

    # Handle employment title differently - keep only frequent ones
    emp_title_counts = df_encoded["emp_title"].value_counts()
    frequent_titles = emp_title_counts[
        emp_title_counts >= 100
    ].index.tolist()  # Minimum 100 occurrences

    # Create 'other' category for infrequent titles
    df_encoded["emp_title_grouped"] = df_encoded["emp_title"].apply(
        lambda x: x if x in frequent_titles else "other"
    )

    # One-hot encode with manageable number of features
    onehot_cols = ["home_ownership", "purpose", "emp_title_grouped"]
    df_encoded = pd.get_dummies(
        df_encoded, columns=onehot_cols, prefix=onehot_cols, drop_first=True
    )

    # drop_first = False. This prevents the emp_title dimensionality explosion while preserving the signal from frequent job categories

    # Drop originals
    df_encoded = df_encoded.drop(
        columns=["grade", "sub_grade", "emp_length", "emp_title"]
    )

    return df_encoded

def transform_zip_code(df_engineered):
    # Create ZIP-based risk tiers instead of using individual ZIP codes
    zip_risk = df_engineered.groupby("zip_code")["target"].agg(["mean", "count"])
    zip_risk = zip_risk[zip_risk["count"] >= 50]  # Minimum sample size

    # Create risk tiers
    zip_risk["risk_tier"] = pd.qcut(
        zip_risk["mean"], q=5, labels=["Low", "Low-Med", "Medium", "Med-High", "High"]
    )

    # Map back to main dataset
    zip_risk_map = zip_risk["risk_tier"].to_dict()
    df_engineered["zip_risk_tier"] = (
        df_engineered["zip_code"].map(zip_risk_map).fillna("Medium")
    )

    # Create dummy variables and remove the original object column
    df_engineered = pd.get_dummies(
        df_engineered, columns=["zip_risk_tier"], prefix="zip_risk", drop_first=True
    )

    df_engineered.drop(columns=["zip_code"], inplace=True)
    return df_engineered

def get_data_step4():
    df = pd.read_csv("../dataproject2025.csv", index_col=0)
    df.drop(columns=['Predictions', 'Predicted probabilities'], inplace=True)
    df_dropped = df.dropna(axis=0)

    df_engineered = get_features(df_dropped)
    cap_variables = ["annual_inc", "open_acc", "fico_range_high", "inq_last_6mths"]
    df_engineered = cap_outliers(df_engineered, cap_variables, percentile=99)
    df_engineered["dti"] = df_engineered["dti"].clip(lower=0)
    df_engineered = transform_zip_code(df_engineered)
    df_engineered.drop(columns=["Pct_afro_american"], inplace=True)
    df_encoded = categorical_encoding(df_engineered)

    target_col = 'target'
    feature_cols = [col for col in df_encoded.columns if col not in ["target", "issue_d"]]
    X_encoded = df_encoded[feature_cols]
    y = df_encoded[target_col]

    return X_encoded, y

def print_evaluation_metrics(y, y_proba, y_preds, y_preds_proba):
    fpr, tpr, thresholds = roc_curve(y, y_preds)

    print(f"AUC: {auc(fpr, tpr)}")
    print(f"Accuracy score: {accuracy_score(y, y_preds)}")
    print(f"Confusion matrix: {confusion_matrix(y, y_preds)}")
    print(f"Mean squared error: {np.sqrt(mean_squared_error(y_proba, y_preds_proba))}")

def plot_odds_ratio(coefs, columns):
    summary = pd.DataFrame({
        'feature': columns,
        'coeff': coefs,
        'odds_ratio': np.exp(coefs)
    })
    summary['importance'] = np.abs(summary['coeff'])
    summary = summary.sort_values('importance', ascending=False)

    top_n = 15
    top_summary = summary.head(15)

    plt.figure(figsize=(10, 0.4 * top_n + 2))

    for i, (feature, or_val) in enumerate(zip(top_summary['feature'], top_summary['odds_ratio'])):
        plt.plot([1, or_val], [i, i], color='skyblue', linewidth=2)
        plt.scatter(or_val, i, color='navy', s=50)

    plt.axvline(x=1, color='red', linestyle='--', label='OR = 1')
    plt.yticks(range(len(top_summary)), top_summary['feature'])
    plt.xscale('log')
    plt.xlabel('Odds Ratio (log scale)')
    plt.title(f'Top {top_n} Most Important Features by Odds Ratio')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.show()

def longest_prefix_match(feat, categories):
    matches = [c for c in categories if feat.startswith(c)]
    if not matches:
        return None
    return max(matches, key=len)

def group_name(feat, numerical, categories):
    pref = longest_prefix_match(feat, categories)
    if pref is not None:
        return pref
    if feat in numerical:
        return feat
    return feat 

def plot_most_important_features(coefs, columns):
    top_n = 15

    summary = pd.DataFrame({
        'feature': columns,
        'coeff': coefs,
        'odds_ratio': np.exp(coefs)
    })
    summary['feature'] = summary['feature'].astype(str)
    summary['group'] = summary['feature'].apply(lambda f: group_name(f, numerical, category))

    rows = []
    for g, sub in summary.groupby('group'):
        n_levels = len(sub)
        importance = sub['coeff'].abs().mean()
        coeff_mean = sub['coeff'].mean()
        or_repr = np.exp(coeff_mean)
        typ = 'categorical' if (longest_prefix_match(sub['feature'].iloc[0], category) == g or n_levels > 1) else 'numerical'
        rows.append({
            'group': g,
            'type': typ,
            'n_levels': n_levels,
            'coeff_mean': coeff_mean,
            'importance': importance,
            'or_repr': or_repr
        })

    agg_df = pd.DataFrame(rows).sort_values('importance', ascending=False).reset_index(drop=True)
    top = agg_df.head(top_n).copy()
    top = top.reset_index(drop=True) 

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(top) + 1.2))
    y = np.arange(len(top))

    for i, row in top.iterrows():
        or_val = row['importance']
        xmin, xmax = (or_val, 1) if or_val < 1 else (1, or_val)
        ax.hlines(y=i, xmin=xmin, xmax=xmax, linewidth=2, color='skyblue')
        ax.plot(or_val, i, 'o', markersize=6, color='navy')

    ax.axvline(1, color='red', linestyle='--', linewidth=1)
    labels = top['group'] + top['n_levels'].apply(lambda n: ('' if n==1 else f' (n={int(n)})'))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale('log')
    ax.set_xlabel('Representative Odds Ratio (exp(mean(coef))) — log scale')
    ax.set_title(f'Top {top_n} Feature Groups by mean(|coef|) (OHE groups aggregated)')
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.08)
    plt.show()

class PLTR(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=80, max_depth=3, random_state=None, granularity=2, k=5, feature_names=None):
        self.granularity = granularity #how much the rounding should be
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.k = k
        self.feature_names = feature_names

    # def tree_fit(self, X, y):
    #     self.trees_ = []
    #     for i in range(self.n_estimators):
    #         tree = DecisionTreeClassifier(
    #             max_depth=self.max_depth,
    #             random_state=None if self.random_state is None else self.random_state + i
    #         )
    #         tree.fit(X, y)
    #         self.trees_.append(tree)
    #     return self
    
    def tree_fit(self, X, y):
        #boosting
        n_features = X.shape[1]
        n_hide = int(np.sqrt(n_features))
        self.trees_ = []
        self.hidden_features_ = []

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_estimators):
            # pick features to hide
            hide_idx = rng.choice(n_features, size=n_hide, replace=False)

            # make a copy and zero out hidden features
            X_mod = X.copy()
            X_mod.iloc[:, hide_idx] = 0  

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=None if self.random_state is None else self.random_state + i
            )
            tree.fit(X_mod, y)

            self.trees_.append(tree)
            self.hidden_features_.append(hide_idx)

        return self

    def tree_predict(self, X):
        preds = [tree.predict(X) for tree in self.trees_]
        return sum(preds) / len(preds)

    def identify(self, k=5, feature_names=None):
        """
        Identify the most common (root split, child split) pairs across trees.

        Parameters
        ----------
        k : int
            Number of most common split pairs to return.
        feature_names : list, optional
            Names of features. If None, indices are returned.
        """
        pair_counter = Counter()

        for tree in self.trees_:
            t = tree.tree_

            # Root node
            

            if t.feature[0] != -2:  # -2 means leaf
                f_root = t.feature[0]
                thr_root = round(t.threshold[0], self.granularity)
                f_root_name = feature_names[f_root] if feature_names is not None else f_root

                # Look at children
                for child in [t.children_left[0], t.children_right[0]]:
                    if child != -1 and t.feature[child] != -2:
                        f_child = t.feature[child]
                        thr_child = round(t.threshold[child], self.granularity)
                        f_child_name = feature_names[f_child] if feature_names is not None else f_child

                        pair_counter[((f_root_name, thr_root), (f_child_name, thr_child))] += 1

        #print(pair_counter)

        self.pairs_identified = pair_counter.most_common(k)
        self.pairs_identified = [pair for pair, count in self.pairs_identified]
        return self.pairs_identified
    

    def create_binary_vars(self, X, split_pairs, feature_names=None):
        """
        Create binary variables from root-child split pairs.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        split_pairs : list of tuples
            Each element is ((root_feature, root_thr), (child_feature, child_thr)).
        feature_names : list, optional
            Names of features. If None, indices are used.
            
        Returns
        -------
        df_bin : pd.DataFrame
            DataFrame with 2 binary variables per pair.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        df_bin = pd.DataFrame()
        
        for i, ((f_root, thr_root), (f_child, thr_child)) in enumerate(split_pairs):
            # Get feature indices if names are provided
            # f_root_idx = feature_names.index(f_root) if feature_names else f_root
            # f_child_idx = feature_names.index(f_child) if feature_names else f_child
            f_root_idx = f_root
            f_child_idx = f_child
            
            # Binary var 1: root split
            bin1 = (X[:, f_root_idx] > thr_root).astype(int)
            
            # Binary var 2: child split only where bin1 == 0
            bin2 = np.zeros(n_samples, dtype=int)
            mask = bin1 == 0
            bin2[mask] = (X[mask, f_child_idx] > thr_child).astype(int)
            
            df_bin[f"{feature_names[f_root]} > {thr_root}"] = bin1
            df_bin[f"{feature_names[f_root]} < {thr_root}x{feature_names[f_child]} > {thr_child}"] = bin2
        
        self.modified_df = df_bin
        return self.modified_df

    def fit(self, X, y, adaptive_lasso=False):

        if adaptive_lasso:
            self.tree_fit(X, y)
            self.identify(k=self.k)
            self.create_binary_vars(X, self.pairs_identified, feature_names=self.feature_names)
            modified_input_full = pd.concat([X, self.modified_df], axis=1)

            lr = LogisticRegression(penalty="l2", solver="liblinear", max_iter=3000)
            lr.fit(modified_input_full, y)

            theta0 = lr.coef_.flatten()
            self.weights_alasso = np.power(np.abs(theta0) + 1e-6, -1)  
            self.modified_input_full = modified_input_full / self.weights_alasso[np.newaxis, :]

            final_clf = LogisticRegression(penalty="l1", solver="liblinear",
                                        max_iter=3000)
            self.pltr = final_clf.fit(self.modified_input_full, y)

            return self
        
        else:
            self.pltr = LogisticRegression(max_iter=3000)
            self.tree_fit(X, y)
            self.identify(k=self.k)
            self.create_binary_vars(X, self.pairs_identified, feature_names=self.feature_names)
            self.modified_input_full = pd.concat([X, self.modified_df], axis=1)
            self.pltr.fit(self.modified_input_full, y)
            return self 
    
    def predict(self, X):
        return self.pltr.predict(X)

    def predict_proba(self, X):
        return self.pltr.predict_proba(X)