from data.windowing import make_windows
from pipeline.metrics import compute_metrics


class ExperimentRunner:
    """Orchestrates: per-experiment windowing → preprocessor fit/transform
    → model fit → predict → metrics.

    Windowing lives here (not in a preprocessor) because it changes sample count
    and must happen with per-experiment input_len/horizon overrides. Preprocessors
    are shape-preserving transformations provided by experiments (scaling,
    differencing, target-aware transforms, etc.) — this application does not
    define preprocessing logic itself.

    Validation data, if provided, is windowed and transformed with the same
    fitted preprocessor, then forwarded to model.fit() via kwargs. Models that
    need it (NNs with early stopping) consume X_val/y_val; baselines ignore
    them via **kwargs.
    """

    def __init__(self, input_len=168, horizon=48):
        self.input_len = input_len
        self.horizon = horizon

    def run(self, experiment, X_train, y_train, X_test, y_test,
            X_val=None, y_val=None, **fit_kwargs):
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both be None.")

        il = experiment.input_len if experiment.input_len is not None else self.input_len
        hz = experiment.horizon   if experiment.horizon   is not None else self.horizon

        Xw_tr, yw_tr = make_windows(X_train, y_train, il, hz)
        Xw_te, yw_te = make_windows(X_test,  y_test,  il, hz)

        experiment.preprocessor.fit(Xw_tr, yw_tr)
        Xw_tr_t = experiment.preprocessor.transform(Xw_tr)
        Xw_te_t = experiment.preprocessor.transform(Xw_te)

        if X_val is not None:
            Xw_val, yw_val = make_windows(X_val, y_val, il, hz)
            fit_kwargs = {
                **fit_kwargs,
                "X_val": experiment.preprocessor.transform(Xw_val),
                "y_val": yw_val,
            }

        experiment.model.fit(Xw_tr_t, yw_tr, **fit_kwargs)
        predictions = experiment.model.predict(Xw_te_t)

        return {
            "experiment_name": experiment.name,
            "predictions": predictions,
            "y_test": yw_te,
            "input_len": il,
            "horizon": hz,
            "metrics": compute_metrics(yw_te, predictions),
            "model_config": experiment.model.get_config(),
            "preprocessor_config": experiment.preprocessor.get_config(),
        }

    def run_all(self, experiments, X_train, y_train, X_test, y_test,
                X_val=None, y_val=None, **fit_kwargs):
        return [self.run(exp, X_train, y_train, X_test, y_test,
                         X_val=X_val, y_val=y_val, **fit_kwargs)
                for exp in experiments]