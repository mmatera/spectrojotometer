"""Worker thread that runs the Monte Carlo simulation without blocking the GUI."""

import threading
import traceback


class MCWorker(threading.Thread):
    """Runs run_ising_mc in the background.

    It communicates with the GUI through out_queue:
      ('start', n_steps)
      ('progress', done, total, message)
      ('done', results_dict)
      ('error', traceback_string)
      ('cancelled',)
    """

    def __init__(self, params, out_queue):
        super().__init__(daemon=True)
        self.params = params
        self.q = out_queue
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def is_cancelled(self):
        return self._cancel.is_set()

    def run(self):
        from ..monte_carlo import SimulationCancelled, run_ising_mc

        try:
            p = self.params
            self.q.put(("start", p["n_temps"]))

            def progress(done, total, msg):
                # run_ising_mc lets SimulationCancelled through (any other
                # exception of the callback was ignored, so the simulation
                # used to go on after Cancel).
                if self.is_cancelled():
                    raise SimulationCancelled()
                self.q.put(("progress", done, total, msg))

            results = run_ising_mc(
                p["spins"], p["bonds"], p["J_vals"], p["E0"],
                T_min=p["T_min"], T_max=p["T_max"], n_temps=p["n_temps"],
                n_equil=p["n_equil"], n_sweeps=p["n_sweeps"],
                algorithm=p["algorithm"],
                n_restarts=p["n_restarts"],
                seed=p["seed"],
                progress_callback=progress,
            )
            self.q.put(("done", results))

        except SimulationCancelled:
            self.q.put(("cancelled",))
        except Exception:
            self.q.put(("error", traceback.format_exc()))
    pass
