from __future__ import division
import numpy as np
import timeit

from icecube import icetray, dataclasses


class BaseBiasedSelection(icetray.I3ConditionalModule):
    """Class to perform a biased event selection

    This class can be used as a basis to apply a biased selection of events.
    Use cases include selection of high-energy events or certain event
    topologies. The selection is performed based on a `bias_function` that
    returns a probability to keep a certain event. This probability is
    compared against a random number sampled uniformly between [0, 1], to
    decide whether or not to keep the given event.
    Due to this biased sampling, an additional factor must be multiplied
    to the event weights to obtain proper event rates. This factor
    is written to the `bias_meta_key` frame key.

    """

    def __init__(self, context):
        """Initialize class

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter(
            "lower_probability_bound",
            "A lower bound of this value is applied to the computed keep "
            "probability.",
            1e-4,
        )
        self.AddParameter(
            "keep_all_events",
            "If True, all events are kept and the bias results "
            "are only written to the frame",
            False,
        )
        self.AddParameter(
            "verbose_output",
            "If True, additional bias info is written to the " "output key.",
            True,
        )
        self.AddParameter(
            "output_key",
            "The output base key to which bias weights will be saved.",
            "BiasedSelectionWeight",
        )
        self.AddParameter(
            "random_service",
            "The random service or seed to use. If this is an "
            "integer, a numpy random state will be created with "
            "the seed set to `random_service`",
            42,
        )
        self.AddParameter(
            "run_on_p_frames",
            "True: run on P-frames; False: run on Q-Frames. "
            "Note: if running on Q-Frames, it is expected that no P-frames "
            "exist in the i3-file.",
            True,
        )

    def Configure(self):
        """Configures MuonLossProfileFilter."""
        self.lower_probability_bound = self.GetParameter(
            "lower_probability_bound"
        )
        self.keep_all_events = self.GetParameter("keep_all_events")
        self.verbose_output = self.GetParameter("verbose_output")
        self.output_key = self.GetParameter("output_key")
        self.random_service = self.GetParameter("random_service")
        self.run_on_p_frames = self.GetParameter("run_on_p_frames")

        if isinstance(self.random_service, int):
            self.random_service = np.random.RandomState(self.random_service)

    def Physics(self, frame):
        """Process physics frames

        Parameters
        ----------
        frame : I3Frame
            The current physics frame.

        Raises
        ------
        ValueError
            If run_on_p_frames is set to False and physics frames exist.
        """
        if self.run_on_p_frames:
            self.bias_selection(frame)
        else:
            raise ValueError(
                "Setting run_on_p_frames is set to False, but P-frames exist!"
            )

    def DAQ(self, frame):
        """Process DAQ frames

        Parameters
        ----------
        frame : I3Frame
            The current Q-frame.
        """
        if self.run_on_p_frames:
            self.PushFrame(frame)
        else:
            self.bias_selection(frame)

    def bias_selection(self, frame):
        """Bias events based on the specified bias function."""

        # start timer
        t_0 = timeit.default_timer()

        # compute keep probability
        keep_prob, additional_bias_info = self.bias_function(frame)

        keep_prob = float(keep_prob)
        assert keep_prob > 0.0 and keep_prob <= 1.0, keep_prob
        keep_prob = np.clip(keep_prob, self.lower_probability_bound, 1.0)

        passed = self.random_service.uniform(0.0, 1.0) <= keep_prob

        bias_weights = dataclasses.I3MapStringDouble(
            {
                "weight_multiplier": 1.0 / keep_prob,
                "passed": float(passed),
            }
        )

        # stop timer
        t_1 = timeit.default_timer()

        # add verbose output if desired
        if self.verbose_output:
            bias_weights["runtime_total"] = t_1 - t_0
            for key, value in additional_bias_info.items():
                bias_weights[key] = float(value)

        frame[self.output_key] = bias_weights

        # push frame to next modules
        if self.keep_all_events:
            self.PushFrame(frame)
        else:
            if passed:
                self.PushFrame(frame)

    def bias_function(self, frame):
        """Compute probability to keep the current frame

        This method is to be overwritten by inherited classes.
        This example would keep all frames.

        Parameters
        ----------
        frame : I3Frame
            The current I3Frame for which to compute the probability,
            whether or not it is kept.

        Returns
        -------
        float
            The probability with which to keep this frame.
        dict
            A dictionary with additional information that will be
            saved to the bias meta key in the frame if `verbose_output`
            is set to True.
        """
        additional_bias_info = {}
        keep_prob = 1.0
        return keep_prob, additional_bias_info
