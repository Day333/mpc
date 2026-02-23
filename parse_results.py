import re
import argparse
import pandas as pd


def parse_line(header_line, metric_line):

    header_line = header_line.strip()
    metric_line = metric_line.strip()

    metric_match = re.search(r"mse:(.*?), mae:(.*)", metric_line)
    if metric_match is None:
        return None

    mse = float(metric_match.group(1))
    mae = float(metric_match.group(2))

    parts = header_line.split("_")

    dataset = parts[3]
    input_len = int(parts[4])
    pred_len = int(parts[5])
    model = parts[6]

    addloss_match = re.search(r"addloss(.*?)_", header_line)
    addloss = addloss_match.group(1) if addloss_match else "None"

    numpairs_match = re.search(r"numpairs(.*?)_", header_line)
    numpairs = numpairs_match.group(1) if numpairs_match else None

    alpha_match = re.search(r"alpha([0-9\.]+)", header_line)
    alpha = float(alpha_match.group(1)) if alpha_match else None

    beta_match = re.search(r"beta([0-9\.]+)", header_line)
    beta = float(beta_match.group(1)) if beta_match else None

    return {
        "dataset": dataset,
        "input_len": input_len,
        "pred_len": pred_len,
        "model": model,
        "addloss": addloss,
        "numpairs": numpairs,
        "alpha": alpha,
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


def print_grouped_results(df, include_addloss=False):

    for dataset in sorted(df["dataset"].unique()):

        print("\n" + "=" * 80)
        print(f"DATASET: {dataset}")
        print("=" * 80)

        df_dataset = df[df["dataset"] == dataset]

        for input_len in sorted(df_dataset["input_len"].unique()):

            print(f"\n---- Input Length: {input_len} ----")

            df_input = df_dataset[df_dataset["input_len"] == input_len]
            df_input = df_input.sort_values("pred_len")

            if not include_addloss:
                display_cols = ["pred_len", "model", "mse", "mae"]
            else:
                display_cols = [
                    "pred_len",
                    "model",
                    "addloss",
                    "numpairs",
                    "alpha",
                    "beta",
                    "mse",
                    "mae",
                ]

            print(df_input[display_cols].to_string(index=False))

            avg_mse = df_input["mse"].mean()
            avg_mae = df_input["mae"].mean()

            print("\nAVG over pred_len:")
            print(f"MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")
            print("-" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-addloss",
        action="store_true",
        help="Include addloss experiments",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="./TimeFilter/baseline.txt",
    )

    args = parser.parse_args()

    df = parse_file(args.file)

    if not args.include_addloss:
        df = df[df["addloss"] == "None"]

    print_grouped_results(df, include_addloss=args.include_addloss)


if __name__ == "__main__":
    main()