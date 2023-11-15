import random
import time
from pathlib import Path
from typing import Literal, Tuple

from howso.engine import Trainee
from howso.utilities import infer_feature_attributes
from pandas import DataFrame, Series
from scipy.io import arff
from sklearn.model_selection import train_test_split


def howsoOversampling(
    X_train: DataFrame,
    y_train: Series,
    *,
    desired_conviction: float = 5,
    generate_new_cases: Literal["always", "attempt", "no"] = "no",
) -> Tuple[float, DataFrame, Series]:
    """
    Synthesize a balanced verison of an imbalanced dataset using Howso Engine.

    Parameters
    ----------
    X_train : DataFrame
        The predictors to train on.
    y_train : Series
        The predictand to train on.
    desired_conviction : float, default 5
        The desired conviction to use when synthesizing the dataset.
    generate_new_cases : {"always", "attempt", "no"}, default "no"
        Whether to enforce privacy constraints when generating.
    """
    start_time = time.time()
    action_features = X_train.columns.values.tolist()
    target_feature = y_train.name
    X_train[target_feature] = y_train.values

    features = infer_feature_attributes(X_train)
    for attributes in features.values():
        if attributes["type"] == "nominal":
            attributes["non_sensitive"] = True

    context_shape = int(X_train.shape[0] / 2)
    contexts = [[1]] * context_shape + [[0]] * context_shape
    context_features = [target_feature]

    t = Trainee(
        features=features,
        overwrite_existing=True
    )

    t.train(X_train)
    t.analyze()

    reaction = t.react(
        desired_conviction=desired_conviction,
        generate_new_cases=generate_new_cases,
        contexts=contexts,
        context_features=context_features,
        action_features=action_features,
        num_cases_to_generate=len(contexts)
    )

    rt = time.time() - start_time
    X_new = reaction['action']
    print(X_new)
    X_train_new = reaction['action'].iloc[:, :-1]
    y_train_new = reaction['action'].iloc[:, -1]

    return rt, X_train_new, y_train_new


if __name__ == "__main__":
    proj_path = Path(__file__).parent.parent.resolve()
    data_path = proj_path / "data" / "Vulnerable_Files" / "moodle-2_0_0-metrics.arff"
    data, _ = arff.loadarff(data_path)

    df = DataFrame(data).astype({"IsVulnerable": str})
    vulnerable_map = {'b\'yes\'': 1, 'b\'no\'': 0}
    df["IsVulnerable"] = df["IsVulnerable"].map(vulnerable_map).fillna(df["IsVulnerable"])

    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    rs = random.randint(0, 100_000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=rs
    )
    rt, X_train_new, y_train_new = howsoOversampling(X_train=X_train, y_train=y_train)

    print(X_train_new)
    print(y_train_new)
    print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))
