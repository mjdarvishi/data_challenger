import json
import os

def _load_data(path: str = "output/dashboard_data.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            if not content:
                return [], [], {}

            payload = json.loads(content)

            if isinstance(payload, dict):
                return (
                    payload.get("records", []),
                    payload.get("grid_search_history", []),
                    payload.get("config", {})
                )

            return payload, [], {}

    except FileNotFoundError:
        return [], [], {}

    except json.JSONDecodeError as e:
        print(f"[JSON ERROR] file corrupted: {path}")
        print(f"Error: {e}")
        return [], [], {}


def _get_data_for_source(source_name: str):
    data_path = os.path.join("output", source_name)
    data, grid_search_history, config_dict= _load_data(data_path)
    return {
        "data": data,
        "grid_search_history": grid_search_history,
        "config": config_dict,
    }

def _get_x_features_and_forcaster_name(source_name: str):
    split = source_name.split(".")
    xfeatures_and_forcaster_name = split[0]
    forcaster_name = xfeatures_and_forcaster_name.split("_")[-1]
    x_features = xfeatures_and_forcaster_name.split("_")[:-2]
    return "_".join(x_features), forcaster_name

def splitter(source_name: str)-> tuple[str, str, float]:
    data_bundle = _get_data_for_source(source_name)
    data = data_bundle["data"]
    highest_test_mse = 0.0
    for step_data in data:
        test_mse = step_data.get("test_eval_mse", 0.0)
        if test_mse >highest_test_mse:
            highest_test_mse = test_mse
    x_features, forcaster_name = _get_x_features_and_forcaster_name(source_name)
    return x_features, forcaster_name, highest_test_mse
    
def create_result_table(source_names: list[str]):
    table = {}  # {x_features: {model: value}}

    all_models = set()

    for source_name in source_names:
        x_features, model, highest_test_mse = splitter(source_name)

        all_models.add(model)

        if x_features not in table:
            table[x_features] = {}

        # if already exists, keep max (or overwrite if you prefer)
        if model in table[x_features]:
            table[x_features][model] = max(
                table[x_features][model],
                highest_test_mse
            )
        else:
            table[x_features][model] = highest_test_mse

    # ensure full matrix (fill missing with "-")
    all_models = sorted(all_models)
    result_table = []

    for x_feat in sorted(table.keys()):
        row = {"x_features": x_feat}

        for model in all_models:
            row[model] = table[x_feat].get(model, "-")

        result_table.append(row)

    return result_table

    
if __name__ == "__main__":
    source_names = [
        "X2_X5_CHRONOLOGICAL_DLinearForcaster.json",
        "X2_X5_CHRONOLOGICAL_ITransformerForcaster.json",
        "X11_X13_X14_CHRONOLOGICAL_AutoformerForcaster.json",
        "X11_X13_X14_CHRONOLOGICAL_ITransformerForcaster.json",
        "X11_X13_X14_CHRONOLOGICAL_NLinearForcaster.json",
        "X11_X13_X14_CHRONOLOGICAL_DLinearForcaster.json",
        
    ]
    result_table = create_result_table(source_names)
    print(json.dumps(result_table, indent=2))