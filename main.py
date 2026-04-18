from core.config import Config, XFeature

from forcast_model.grid_search import GridSearchEngine
from training.pipeline import BasePipeline
from forcast_model.itransformer import ITransformerForcaster
from data_generator.x_feature_registery import XFeatureRegistery
from data_generator.generator_model import GeneratorModel
from core.setup import setup_models


def main():
    # setup_models()
    x_registery = XFeatureRegistery()
    features=[XFeature.X5, XFeature.X2]
    features_str = "_".join([f.name for f in features])
    x_registery.select_generators(features)
    gen_model = GeneratorModel()
    grid_engine = GridSearchEngine(model_class=ITransformerForcaster)
    pipe = BasePipeline(
        name=f"{features_str}_{ITransformerForcaster.__name__}",
        x_registery=x_registery,
        gen_model=gen_model,
        grid_search_engine=grid_engine,
    )

    pipe.run()
    pipe.tracker.export(path=pipe.name)

if __name__ == "__main__":
    main()
