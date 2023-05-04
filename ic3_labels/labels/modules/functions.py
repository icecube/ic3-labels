from icecube import dataclasses
from ic3_labels.labels.utils import general


def add_charge_to_frame(frame, OutputKey='ChargeAndDomHits', pulse_key='InIceDSTPulses'):
    n_dom_hits, total_charge = general.get_charge(frame, pulse_key=pulse_key)
    frame.Put(OutputKey, dataclasses.I3MapStringDouble())
    frame[OutputKey]['n_dom_hits'] = n_dom_hits 
    frame[OutputKey]['total_charge'] = total_charge 

    return True

