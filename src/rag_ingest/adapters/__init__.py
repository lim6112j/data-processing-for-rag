"""Adapters translate a concrete parser into the narrow ``docmodel`` contract.

Keeping adapters in their own package is what lets the pipeline, the chunker,
the gate and the gold set stay free of heavy parser dependencies.
"""
