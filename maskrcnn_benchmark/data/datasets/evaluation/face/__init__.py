import logging

from .face_eval import do_face_evaluation


def face_evaluation(dataset, predictions, output_folder, vis, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    return do_face_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        visualize_scores=vis
    )
