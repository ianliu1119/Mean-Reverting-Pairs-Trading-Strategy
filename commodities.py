import pandas as pd
import numpy as np
from stationarity_test import johansen_test

commodities_tbl = pd.read_csv("commodities.csv")
commodities_list = commodities_tbl["Ticker_Symbol"].tolist()

test_ticker_list = ["SLV", "GLD", "USO", "NVDA", "WFC"]
start_date = "2023-10-01"
end_date = "2024-10-01"


def commodities_combination(ticker_list):
    table = pd.DataFrame(columns=["Asset 1", "Asset 2", "Test Stat", "Critical Value"])

    for i in range(len(ticker_list)):
        for j in range(i + 1, len(ticker_list)):
            pair = [ticker_list[i], ticker_list[j]]

            result, _, _ = johansen_test(pair, start_date, end_date)

            # Handle "Not applicable" case
            if isinstance(result, str):
                test_stat = np.nan
                critical_value = np.nan
            else:
                test_stat, critical_value = result

            # Append to table
            table.loc[len(table)] = [pair[0], pair[1], test_stat, critical_value]

    return table


final = commodities_combination(commodities_list)
final_sorted = final.sort_values(by="Test Stat", ascending=False)
print(final_sorted)
