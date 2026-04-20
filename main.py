from core.config import Config, SplitMode, XFeature

from forcast_model.base_forcast_model import BaseForecastModel
from forcast_model.grid_search import GridSearchEngine
from training.pipeline import BasePipeline
from forcast_model.itransformer import ITransformerForcaster
from forcast_model.auto_former import AutoformerForcaster
from forcast_model.dlinear import DLinearForcaster
from forcast_model.nlinear import NLinearForcaster
from data_generator.x_feature_registery import XFeatureRegistery
from data_generator.generator_model import GeneratorModel

def main(
    features: list[XFeature],
    forcaster_cls: type[BaseForecastModel],
):
    # setup_models()

    x_registery = XFeatureRegistery()
    features_str = "_".join([f.name for f in features])
    x_registery.select_generators(features)
    num_features = len(x_registery.selected_generators)
    Config.set_input_dim(features)

    gen_model = GeneratorModel(num_features=num_features)
    grid_engine = GridSearchEngine(
        model_class=forcaster_cls,
        model_fixed_kwargs={"input_dim": Config.input_dim},
    )
    pipe = BasePipeline(
        name=f"{features_str}_{forcaster_cls.__name__}",
        x_registery=x_registery,
        gen_model=gen_model,
        grid_search_engine=grid_engine,
    )

    pipe.run()
    pipe.tracker.export(name=pipe.name)

if __name__ == "__main__":
    # main(
    #     [XFeature.X9, XFeature.X5, XFeature.X1, XFeature.X2, XFeature.X3, XFeature.X4, XFeature.X11],
    #     ITransformerForcaster,
    # )
    main(
        [ XFeature.X5, XFeature.X2],
        DLinearForcaster,
    )
    # main([XFeature.X7, XFeature.X5], NLinearForcaster)
    # main([XFeature.X5, XFeature.X4], AutoformerForcaster)
