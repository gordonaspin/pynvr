from ultralytics import YOLO

from context import Context

class Model:

    def __init__(self, ctx: Context):
        self._model = YOLO(ctx.model)
        self._c2i = {v: k for k, v in self._model.names.items()}

    def class_to_index(self, names):
        return [self._c2i[n] for n in names]

    @property
    def model(self):
        return self._model

