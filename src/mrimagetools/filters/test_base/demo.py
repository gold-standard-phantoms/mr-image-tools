"""Demo script"""
from .pipelines.complex_pipeline import complex_pipeline
from .pipelines.data_combiner_pipeline import data_combiner_pipeline
from .pipelines.data_massager_pipeline import data_massager_pipeline
from .pipelines.simple_pipeline import simple_pipeline

if __name__ == "__main__":
    print("Running demo")

    print("Simple pipeline")
    simple_pipeline(use_optional_input=False).visualise()

    print("Simple pipeline with optional input")
    simple_pipeline(use_optional_input=True).visualise()

    print("Data massager pipeline")
    data_massager_pipeline().visualise()

    print("Data combiner pipeline")
    data_combiner_pipeline().visualise()

    print("Complex pipeline")
    complex_pipeline().visualise()
