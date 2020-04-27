import pandas as pd

TIME_COLUMN = "ds"


class DataFrameBuilder:
    def __init__(self, regressors):
        self.regressors = regressors

    def create_df(self, df, column, train=False):
        train_df = pd.DataFrame({"ds": df[TIME_COLUMN].values})

        if train:
            train_df["y"] = df[column].values

        if self._contains_columns(df, f"cap_{column}"):
            train_df["cap"] = df[f"cap_{column}"].values

        if self._contains_columns(df, f"floor_{column}"):
            train_df["floor"] = df[f"floor_{column}"].values

        for regressor in self.regressors.get(column, []):
            train_df[regressor] = df[regressor]

        return train_df

    def add_regressor(self, name, columns):
        for column in columns:
            self._append_regressor(name, column)

    def _contains_columns(self, df, column):
        return column in df.columns

    def _append_regressor(self, name, column):
        self.regressors[column] = self.regressors.get(column, []) + [name]
