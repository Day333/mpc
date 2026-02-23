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


def analyze(df: pd.DataFrame):

    for dataset in sorted(df["dataset"].unique()):
        print("\n" + "=" * 120)
        print(f"DATASET: {dataset}")
        print("=" * 120)

        df_dataset = df[df["dataset"] == dataset]

        for patch in sorted(df_dataset["patch"].unique()):
            print("\n" + "-" * 120)
            print(f"PATCH = {patch}")
            print("-" * 120)

            df_patch = df_dataset[df_dataset["patch"] == patch].copy()

            # Pivot
            mse_wide = df_patch.pivot_table(
                index="beta",
                columns="pred_len",
                values="mse",
                aggfunc="mean"
            )

            mae_wide = df_patch.pivot_table(
                index="beta",
                columns="pred_len",
                values="mae",
                aggfunc="mean"
            )

            # Ensure all expected pred_len columns exist
            for pl in EXPECTED_PREDS:
                if pl not in mse_wide.columns:
                    mse_wide[pl] = pd.NA
                if pl not in mae_wide.columns:
                    mae_wide[pl] = pd.NA

            mse_wide = mse_wide[EXPECTED_PREDS]
            mae_wide = mae_wide[EXPECTED_PREDS]

            # Rename columns
            mse_wide.columns = [f"mse_{pl}" for pl in EXPECTED_PREDS]
            mae_wide.columns = [f"mae_{pl}" for pl in EXPECTED_PREDS]

            combined = pd.concat([mse_wide, mae_wide], axis=1)

            # Compute avg over available preds
            combined["avg_mse"] = combined[
                [f"mse_{pl}" for pl in EXPECTED_PREDS]
            ].mean(axis=1, skipna=True)

            combined["avg_mae"] = combined[
                [f"mae_{pl}" for pl in EXPECTED_PREDS]
            ].mean(axis=1, skipna=True)

            # Coverage info
            combined["coverage"] = (
                combined[[f"mse_{pl}" for pl in EXPECTED_PREDS]]
                .notna()
                .sum(axis=1)
                .astype(str)
                + f"/{len(EXPECTED_PREDS)}"
            )

            combined = combined.sort_index()

            print("\nWide table (per beta, each pred_len + avg):")
            print(combined.to_string())

            # Best per pred_len
            print("\nBest beta per pred_len (by MSE):")
            for pl in EXPECTED_PREDS:
                col = f"mse_{pl}"
                if col not in combined.columns:
                    continue
                series = combined[col].dropna()
                if series.empty:
                    print(f"pred_len={pl} → no result")
                    continue
                best_beta = series.idxmin()
                best_val = series.min()
                print(f"pred_len={pl} → beta={best_beta}, mse={best_val:.6f}")

            # Overall best beta
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
    args = parser.parse_args()

    df = parse_file(args.file)

    if df.empty:
        print("No valid addloss experiments found.")
        return

    analyze(df)


if __name__ == "__main__":
    main()