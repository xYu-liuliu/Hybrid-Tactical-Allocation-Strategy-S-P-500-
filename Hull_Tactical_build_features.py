import os
import pandas as pd
from Hull_Tactical_feature_engineering import build_features

IN_PATH  = r"E:\train.csv"
OUT_PATH = r"E:\kaggle_hull\all_feature_last_5000_no_nan.csv"
N_RAW = 5000


def main():
    print("Loading raw data...")
    df_raw = pd.read_csv(IN_PATH)

    print("Building features...")
    df_feat = build_features(df_raw, n_raw=N_RAW)

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Saving features...")
    df_feat.to_csv(OUT_PATH, index=False)

    print("Done.")
    print("Final shape:", df_feat.shape)


if __name__ == "__main__":
    main()