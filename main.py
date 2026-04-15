from core.config import Config, XFeature

from forcast_model.grid_search import GridSearchEngine
from training.pipeline import BasePipeline
from forcast_model.itransformer import ITransformerWrapper
from data_generator.x_feature_registery import XFeatureRegistery
from data_generator.generator_model import GeneratorModel
from core.setup import setup_models
from visualization.visualizer import plot_ground_truth_vs_prediction, plot_loss, plot_b0_heatmap, print_mse_table,  print_grid_results


def main():
    # setup_models()
    x_registery = XFeatureRegistery()
    x_registery.select_generators([XFeature.X5, XFeature.X2])
    gen_model = GeneratorModel()
    grid_engine = GridSearchEngine(model_class=ITransformerWrapper)
    pipe = BasePipeline(
        name="base_pipeline",
        x_registery=x_registery,
        gen_model=gen_model,
        grid_search_engine=grid_engine,
    )

    pipe.run()
    print_mse_table(pipe.tracker)
    print_grid_results(tracker=pipe.tracker)
    plot_ground_truth_vs_prediction(tracker=pipe.tracker)
if __name__ == "__main__":
    main()
