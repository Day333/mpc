import re
import argparse
import pandas as pd


VALID_PATCH = {2, 4, 8, 16, 24}
EXPECTED_PREDS = [96, 192, 336, 720]


def parse_line(header_line, metric_line):
    header_line = header_line.strip()
    metric_line = metric_line.strip()

    if "addloss" not in header_line:
        return None

    metric_match = re.search(r"mse:(.*?), mae:(.*)", metric_line)
    if metric_match is None:
        return None

    mse = float(metric_match.group(1))
    mae = float(metric_match.group(2))

    parts = header_line.split("_")
    dataset = parts[3]
    input_len = int(parts[4])
    pred_len = int(parts[5])

    alpha_match = re.search(r"alpha([0-9\.]+)", header_line)
    beta_match = re.search(r"beta([0-9\.]+)", header_line)
    patch_match = re.search(r"patch(?:len)?_?([0-9]+)", header_line)

    if not (alpha_match and beta_match and patch_match):
        return None

    patch = int(patch_match.group(1))
    if patch not in VALID_PATCH:
        return None

    beta = float(beta_match.group(1))

    return {
        "dataset": dataset,
        "input_len": input_len,
        "pred_len": pred_len,
        "patch": patch,
        "alpha": float(alpha_match.group(1)),
        "beta": beta,
        "mse": mse,
        "mae": mae,
    }


def parse_file(filepath):
    results = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("long_term_forecast"):
            header = lines[i]
            metric = lines[i + 1] if i + 1 < len(lines) else ""
            parsed = parse_line(header, metric)
            if parsed:
                results.append(parsed)
            i += 2
        else:
            i += 1

    return pd.DataFrame(results)


def analyze(df: pd.DataFrame, wide_mode: str = "full"):
    assert wide_mode in {"full", "avg"}, "wide_mode must be 'full' or 'avg'"

    for dataset in sorted(df["dataset"].unique()):

        print("\n" + "=" * 120)
        print(f"DATASET: {dataset}")
        print("=" * 120)

        df_dataset = df[df["dataset"] == dataset]

        for input_len in sorted(df_dataset["input_len"].unique()):

            print("\n" + "#" * 120)
            print(f"INPUT_LEN: {input_len}")
            print("#" * 120)

            df_input = df_dataset[df_dataset["input_len"] == input_len]

            pred_list = sorted(df_input["pred_len"].unique())

            # ============================================================
            # Global best per pred_len
            # ============================================================
            print("\nGlobal Best per pred_len (across patch & beta):")

            best_rows = []

            for pl in pred_list:
                df_pl = df_input[df_input["pred_len"] == pl]
                if df_pl.empty:
                    continue

                idx = df_pl["mse"].idxmin()
                best = df_pl.loc[idx]

                print(
                    f"pred_len={pl} → "
                    f"mse={best['mse']:.6f}, "
                    f"mae={best['mae']:.6f}, "
                    f"patch={best['patch']}, "
                    f"beta={best['beta']}"
                )

                best_rows.append(best)

            if best_rows:
                best_df = pd.DataFrame(best_rows)
                print("\nAverage of best MSE across pred_len:")
                print(f"avg_best_mse = {best_df['mse'].mean():.6f}")
                print(f"avg_best_mae = {best_df['mae'].mean():.6f}")

            # ============================================================
            # patch 
            # ============================================================
            for patch in sorted(df_input["patch"].unique()):

                print("\n" + "-" * 120)
                print(f"PATCH = {patch}")
                print("-" * 120)

                df_patch = df_input[df_input["patch"] == patch].copy()

                mse_wide = df_patch.pivot_table(
                    index="beta", columns="pred_len", values="mse", aggfunc="mean"
                )
                mae_wide = df_patch.pivot_table(
                    index="beta", columns="pred_len", values="mae", aggfunc="mean"
                )

                for pl in pred_list:
                    if pl not in mse_wide.columns:
                        mse_wide[pl] = pd.NA
                    if pl not in mae_wide.columns:
                        mae_wide[pl] = pd.NA

                mse_wide = mse_wide[pred_list]
                mae_wide = mae_wide[pred_list]

                mse_wide.columns = [f"mse_{pl}" for pl in pred_list]
                mae_wide.columns = [f"mae_{pl}" for pl in pred_list]

                combined = pd.concat([mse_wide, mae_wide], axis=1)

                combined["avg_mse"] = combined[
                    [f"mse_{pl}" for pl in pred_list]
                ].mean(axis=1, skipna=True)

                combined["avg_mae"] = combined[
                    [f"mae_{pl}" for pl in pred_list]
                ].mean(axis=1, skipna=True)

                combined["coverage"] = (
                    combined[[f"mse_{pl}" for pl in pred_list]]
                    .notna()
                    .sum(axis=1)
                    .astype(str)
                    + f"/{len(pred_list)}"
                )

                combined = combined.sort_index()

                if wide_mode == "avg":
                    to_print = combined[["avg_mse", "avg_mae", "coverage"]]
                else:
                    to_print = combined

                print("\nWide table:")
                print(to_print.to_string())

                # best beta per pred_len
                print("\nBest beta per pred_len (by MSE):")
                for pl in pred_list:
                    col = f"mse_{pl}"
                    series = combined[col].dropna()
                    if series.empty:
                        continue
                    best_beta = series.idxmin()
                    best_val = series.min()
                    print(f"pred_len={pl} → beta={best_beta}, mse={best_val:.6f}")

                best_global = combined["avg_mse"].idxmin()
                print("\nOverall best beta (by avg_mse):")
                print(
                    f"beta={best_global}, "
                    f"avg_mse={combined.loc[best_global, 'avg_mse']:.6f}, "
                    f"avg_mae={combined.loc[best_global, 'avg_mae']:.6f}, "
                    f"coverage={combined.loc[best_global, 'coverage']}"
                )

        print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument(
        "--wide_mode",
        type=str,
        default="avg",
        choices=["full", "avg"],
        help="Wide table output: 'full' prints per-pred columns; 'avg' prints only avg_mse/avg_mae/coverage.",
    )
    args = parser.parse_args()

    df = parse_file(args.file)

    if df.empty:
        print("No valid addloss experiments found.")
        return

    analyze(df, wide_mode=args.wide_mode)


if __name__ == "__main__":
    main()